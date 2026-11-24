# Kiến trúc end-to-end cho đồ án chấm điểm và phản hồi bài C++ bằng LLM

Quy ước dùng trong tài liệu:
- **[GT]** là giả thuyết về dữ liệu, chưa kiểm chứng.
- **D#** là điểm quyết định, liệt kê ở §4.7.
- **S#** là thành phần dùng chung, liệt kê ở §1.
- Mọi con số GPU-giờ và $ là ước lượng, dựa trên giả định khoảng 5k bài train và khoảng 3k token mỗi bài. Phải đo lại trong tuần 1.

---

## 0. Sáu nguyên tắc thiết kế

Các nguyên tắc này rút ra từ 5 bản phân tích đầu vào, có điều chỉnh.

1. **Phần tất định làm trước, LLM làm sau.**
   - Những gì suy ra được từ biên dịch, chạy thử hoặc AST thì dùng rule hoặc feature.
   - LLM chỉ làm phần cần hiểu ngữ nghĩa: logic của multi_problem khi không có test, edge_case, code_quality, chẩn đoán lỗi, sinh phản hồi.
2. **Tách phần ước lượng khỏi phần quyết định.**
   - Model chỉ trả về xác suất hoặc E-value. Một lớp decoder riêng cho từng metric quyết định output cuối.
   - Lý do: không model nào tự học được những điều sau chỉ qua cross-entropy:
     - QWK bằng Lin's CCC, nên phải giãn dự đoán ra (de-shrink).
     - Macro-F1 cần ngưỡng riêng cho từng nhãn, với ngưỡng tối ưu bằng F\*/2 khi xác suất đã calibrate.
     - Compliance cần ràng buộc cứng.
3. **Chính sách chấm viết thành code, không đặt trong prompt.** Policy engine viết bằng Python, tất định, có unit test.
4. **Hai hướng bắt buộc chạy trên cùng một backbone open-weight, cục bộ.**
   - Prompting arm (P): Qwen3-8B chưa fine-tune, dùng zero-shot, few-shot, CoT, self-consistency, RAG.
   - Fine-tune arm (F): cùng Qwen3-8B cộng LoRA.
   - Điều này điều chỉnh căng thẳng "API chỉ để huấn luyện" mà bản spec-compliance nêu ở §3 dòng 2. Không gọi API lúc suy luận thì:
     - hết căng thẳng đó;
     - hết vấn đề tái lập khi dùng API;
     - so sánh P với F trở thành so sánh có kiểm soát, vì hai bên chỉ khác trọng số.
   - API thương mại, nếu giảng viên cho phép, chỉ đóng vai teacher và tham chiếu cận trên. Mọi lời gọi API đi qua cache-replay.
5. **Chặn rò rỉ bằng kiểu dữ liệu.**
   - `BlindSample` dùng cho T1/T2, không có trường `feedback`, `level` hay `labels`. `CondSample` dùng cho T3.
   - Loader chỉ đọc các trường trong whitelist của từng task.
   - CI tự kiểm tra rằng prompt T1/T2 không chứa chuỗi feedback nào.
6. **Mỗi module là một flag trong config.** Bảng ablation khi đó là tích các flag, không phải viết lại từng thí nghiệm.

---

## 1. Thành phần dùng chung (S1–S9)

Mọi phương án ở §2 đều dùng các thành phần này.

| Mã | Thành phần | Việc làm | Output (cache theo khoá) | Dùng ở |
|---|---|---|---|---|
| S1 | Ingest và chuẩn hoá | Nối JSON với `code_file`. Decode lần lượt utf-8 → cp1258 → latin-1 và log số file phải fallback. Chuẩn hoá NFC. Giữ bản gốc để đếm dòng giống giảng viên, bản LF dùng để parse. Lấy `pid` từ path. Gán `dup_cluster` bằng MinHash 5-gram trên token đã chuẩn hoá | `samples.parquet` | Tất cả |
| S2 | Problem compiler (chạy 1 lần mỗi đề) | Trích policy gồm câu, trọng số, tiên quyết, chữ ký hàm, io_kind: regex trước, sau đó LLM structured output với SC=5, rồi validate (Σw đúng, đồ thị tiên quyết không có chu trình). Generator input do LLM viết, có seed. Oracle là output đa số của các bài train đạt điểm tối đa cùng đề; với đề mới thì dùng lời giải LLM đã pass test công khai | `problem_card/<pid>.json` | T1, T2, T3 |
| S3 | Sandbox và evidence engine | Docker `--network none`, đặt ulimit. Biên dịch với cờ khớp compile_log gốc. Thêm `-Wall -Wextra`, ASan/UBSan và `_GLIBCXX_DEBUG`. Chạy test công khai với 2 comparator (strict và normalized) để ra `io_gap`. Chạy test sinh và peer-oracle. Chạy cppcheck, lizard, tree-sitter | `evidence/<code_sha>.json` kèm bản text gọn | T1, T2, T3 |
| S4 | Bộ tách câu multi_problem | Thử lần lượt: comment đánh dấu, tên hàm, `switch/case` cộng call graph, khớp chữ ký lấy từ S2, LLM (validate bằng AST). Không cách nào thành công thì dùng cả file kèm cờ `split_failed` | Trường `seg` trong evidence | T1, T3 |
| S5 | Policy engine | `aggregate(q_scores, policy)` có bản gating cứng và bản mềm. Ngưỡng τ fit trên OOF | `policy_total_{hard,soft}` | T1 (và làm feature cho T2) |
| S6 | Index retrieval | BM25 trên token đã chuẩn hoá, `Qwen3-Embedding-0.6B`, và Jaccard trên vector verdict test. Lọc cùng `pid`. **Loại chính bài đó, cả `dup_cluster` của nó, và fold hiện tại** | Top-k và feature kNN | T1/T2 (feature), P-arm (few-shot), T3 (exemplar) |
| S7 | Tường lửa, validator, submit | Loader theo whitelist. Schema pydantic. Validator chính thức. `make submit` ghi log quota | `predictions.json` kèm sha | Tất cả |
| S8 | Hạ tầng tái lập | Config YAML. Seed ở mọi tầng. Pin `revision` của model. Cache API trong SQLite (replay offline). `env.json`. Commit `folds.csv` | `run_id` | Tất cả |
| S9 | Bản sao metric và judge nội bộ | QWK có `labels=range(11)` và không có. Macro-F1 theo 3 quy ước `zero_division`. Ensemble proxy cho compliance. Judge chẩn đoán chạy local, cố định | `reports/*.csv` | Chọn model, ablation, bonus |

**Evidence card** là biểu diễn trung gian chung cho cả 3 task. Mọi model, P hay F, đều đọc cùng card này:

```json
{"sid":"…","pid":"…","type":"single_problem","code_sha":"…","toolchain":"g++ (pin) -std=c++17",
 "compile":{"ok_given":true,"ok_ours":true,"first_error":null,"warn":{"Wreturn-type":0,"Wsign-compare":2}},
 "tests":{"public":{"pass":7,"n":10,"verdicts":"AAAWAAWAAT","io_gap":0},
          "gen":{"pass":31,"n":40,"edge_fail":["n=0","dup"]},"san":{"oob":1,"ovf":0}},
 "static":{"loop_depth":2,"recursion":false,"cppcheck":["uninitvar@12"],"ccn_max":7},
 "seg":[{"q":1,"lines":[5,30],"how":"comment"}],"near_empty":false,"dup_cluster":"c_0412",
 "text":"[COMPILE:OK][PUBLIC 7/10 WA@4,7 TLE@10][GEN 31/40 edge:n=0,dup][ASAN:OOB×1][LOOP 2]"}
```

---

## 2. Bốn phương án kiến trúc

### A. Evidence-first Hybrid (EFH)

**Triết lý:** kết quả thực thi là nguồn sự thật rẻ và tất định. LLM có head học phần còn lại mà thực thi không giải thích được. LLM sinh văn bản chỉ dùng ở chỗ cần ngữ nghĩa.

```
sample ─► S1 ─► S3 Evidence ──────────────────┬─► rules (compilable, CE, io_gap)
           └──► S2/S4/S5 policy ──────────────┤
                S6 kNN feats ─────────────────┤
  P-arm: Qwen3-8B prompt (EV/logprob) ────────┼─► Stacker/task (ridge|GBDT|LR, OOF)
  F-arm: Qwen3-8B LoRA + heads (E, P) ────────┘          │
                                                         ▼
                       metric decoders (QWK / per-label τ) ─► S7
  T3: evidence + nhãn ─► LLM render (prompt|SFT) ─► regex guard ─► S7
```

- **Yêu cầu ≥2 hướng:** P-arm và F-arm cùng nhận evidence card. So sánh 3 cấu hình: P, F, và P+F+stack.
- **Leaderboard kỳ vọng:**
  - **T1 cao.** QWK phạt bình phương, nên điểm mất nhiều nhất ở các ca lệch xa: bài không biên dịch, bài gần rỗng, câu tiên quyết sai. Tín hiệu tất định bắt đúng những ca này.
  - **T2 khá đến cao.** Rule xử lý nhãn biên dịch; xác suất đã calibrate cộng ngưỡng theo nhãn xử lý phần còn lại. Yếu ở các nhãn thuần ngữ nghĩa như "Lỗi hàm" nếu head LLM yếu.
  - **T3 khá.** Chẩn đoán bám bằng chứng nên tốt, nhưng renderer chỉ có prompt cộng regex guard, chưa có học ưu tiên (preference learning).
- **Chi phí:**
  - CPU sandbox: N × (1 lần biên dịch + 10–50 lần chạy), vài giờ CPU.
  - GPU: khoảng 15–30 GPU-giờ (3 adapter cho cross-fit, cộng inference của P-arm).
  - API: gần 0.
- **Rủi ro:**
  - Harness cho multi_problem tốn công trong tuần 1–3.
  - Toolchain lệch (Windows/MinGW) làm kết quả biên dịch tái lập lệch với nhãn.
  - Nếu LLM chỉ còn là một feature thì đồ án dễ bị chấm là "chưa đủ LLM". Phải giữ head LLM làm model chính và stacker thật mỏng.
- **Novelty:** policy engine, peer-oracle, decoding theo metric. Phân tích lỗi theo tín hiệu rất minh bạch.

### B. Một LLM fine-tune đa nhiệm (MTL)

**Triết lý:** một Qwen3-8B với 1 LoRA học cả 3 task từ dữ liệu trộn. Hạ tầng tối thiểu.

```
input + task tag + evidence text tối thiểu
      ▼
Qwen3-8B + 1 LoRA đa nhiệm (SFT trộn T1:T2:T3, cân bằng theo token)
      ├─ T1: JSON 6 chiều → đọc logprob → EV
      ├─ T2: slot-template 10 nhãn (teacher-forced P(Y))
      └─ T3: feedback điều khiển bằng token <Lk>
      ▼
hậu xử lý nhẹ (làm tròn, ngưỡng, regex) → S7
P-arm = cùng model chưa fine-tune
```

- **Leaderboard kỳ vọng:**
  - **T1 trung bình.** Decode argmax trên token kéo dự đoán về giữa; EV từ logprob cứu được một phần.
  - **T2 trung bình.** Nhãn hiếm bị chìm vì gradient bị T3 (sinh văn bản dài) lấn át.
  - **T3 trung bình đến khá.** SFT sẽ học luôn các phản hồi gold L1/L2 có chứa code nếu không lọc; không có guard.
- **Chi phí:** thấp nhất, khoảng 10–20 GPU-giờ.
- **Rủi ro:**
  - Các task tranh gradient của nhau.
  - Không có policy engine, tức bỏ qua yêu cầu cốt lõi của multi_problem.
  - Hộp đen khi phân tích lỗi, ablation nghèo.
- **Novelty:** thấp. Đây là lựa chọn mặc định của nhiều nhóm.

### C. Teacher–Student Distillation (TSD)

**Triết lý:** model lớn sinh giám sát giàu thông tin, rồi chưng cất vào student ≤10B.

```
train (đề, code, log, rubric/nhãn gold — T1/T2 KHÔNG có feedback)
      ▼
Teacher (API frontier | Qwen 27–32B 4-bit local) ──cache──►
  (i)  rationale chấm có điều kiện theo gold     (ii) chẩn đoán JSON
  (iii) feedback level-swap L1–L4                 (iv) cặp DPO: leak/mơ hồ/hallucination
  (v)  bug tổng hợp cho nhãn hiếm
      ▼ lọc: verify bằng compile/test, so teacher-blind với gold, leak detector
Student Qwen3-8B: SFT (rationale→scores); T3: SFT → DPO
      ▼ inference local, greedy/seeded → hậu xử lý → S7
P-arm = teacher chạy như hệ thống (cận trên, qua cache) + P-arm local
```

- **Leaderboard kỳ vọng:**
  - **T1 khá đến cao.** Rationale giúp student học "vì sao", nhưng vẫn cần decoder QWK.
  - **T2 khá đến cao.** Bug tổng hợp bổ sung dữ liệu cho nhãn hiếm.
  - **T3 cao nhất trong 4 phương án.** Level-swap cộng DPO tối ưu thẳng vào trade-off giữa compliance và chẩn đoán.
- **Chi phí:**
  - API $ tính theo công thức ở models-infra §2.3. Với giá giả định, khoảng $50–300 cho 3 task với R=1–3.
  - GPU khoảng 30–60 giờ.
  - Nếu API bị cấm: teacher là một model 27–32B lượng tử 4-bit chạy local. Rẻ về tiền nhưng chậm, và VRAM cần kiểm chứng.
- **Rủi ro:**
  1. Giảng viên có thể không cho gửi code sinh viên ra API (mục 9 của đề). Nếu vậy mọi phần cần teacher API bị chặn.
  2. Bias của teacher truyền sang student.
  3. Vùng xám quanh feedback.
  4. Tái lập phụ thuộc hoàn toàn vào cache.
- **Novelty:** trung bình đến cao. Distilling step-by-step và DPO cho tính sư phạm đều có tài liệu đi trước.

### D. Retrieval-heavy (RAG/kNN)

**Triết lý:** [GT] bài cùng đề giống nhau, nên nhãn của các bài láng giềng là predictor mạnh. LLM chấm theo mẫu.

```
code ─► normalize ─► BM25 + dense + verdict-vector ─► top-k cùng đề (loại self/dup/fold)
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
   kNN regression (T1)     kNN vote nhãn (T2)    exemplar cùng level+nhãn (T3)
            └────────► LLM few-shot local có nhãn láng giềng ◄────────┘
                                   ▼ calibrate → S7
F-arm = retrieval-augmented SFT (prompt chứa k láng giềng kèm nhãn)
```

- **Leaderboard kỳ vọng:** phụ thuộc hoàn toàn vào việc dev/test có đề mới hay không (D2).
  - Nếu cùng đề với train: T1 cao, T2 khá.
  - Nếu có đề mới: T1 và T2 tụt về mức zero-shot.
  - T3 trung bình. Chép được văn phong giảng viên, nhưng exemplar lấy từ L3/L4 dễ kéo theo code.
- **Chi phí:** thấp.
- **Rủi ro:**
  - Near-dup làm điểm CV lạc quan.
  - Copy bias: LLM chép điểm láng giềng.
  - Prompt k× độ dài làm multi_problem vượt context.
- **Novelty:** thấp đến trung bình.

---

## 3. Ma trận so sánh và khuyến nghị

Thang điểm 1–5. Trọng số lấy theo cơ cấu điểm đồ án:
- RANK 30% chia đều cho 3 task;
- METH 25, ABL 15, REP 20, REPRO 10;
- BONUS tính như trọng số 10.

| Tiêu chí (trọng số) | A. EFH | B. MTL | C. TSD | D. RAG |
|---|---|---|---|---|
| LB T1 (10) | **5**: bắt các ca lệch xa | 3: co về giữa | 4 | 3 (5 nếu cùng đề, 1–2 nếu đề mới) |
| LB T2 (10) | 4 | 3 | 4 | 3 |
| LB T3 (10) | 4 | 3 | **5**: DPO, level-swap | 3 |
| Phương pháp và thiết kế thí nghiệm (25) | 4 (rủi ro "LLM phụ") | 3 | 4 | 2 |
| Ablation và phân tích lỗi (15) | **5**: module bật tắt được | 3 | 4 | 3 |
| Báo cáo, giải thích, triển khai (20) | **5** | 3 | 4 | 3 |
| Tái lập (10) | **5**: phần lớn tất định | 4 | 3: phụ thuộc cache teacher | 4 |
| Đồng vận với bonus (10) | **5**: evidence sinh ra luôn bằng chứng dữ liệu | 2 | 3: bất đồng teacher và gold là tín hiệu nhiễu nhãn | 4: index near-dup |
| **Điểm có trọng số /5** | **4.59** | 3.00 | 3.91 | 2.95 |
| *Khả thi trong 8 tuần (không tính trọng số)* | 3 | 4 | 3 (2 nếu API bị cấm) | 5 |
| *Bền khi gặp đề mới (không tính trọng số)* | 4 | 3 | 4 | 1–2 |

**Khuyến nghị: "Evidence-first, Metric-aware Hybrid" (EMH)**, lấy A làm xương sống và ghép các phần sau:

- **Từ C, chỉ dùng ở chỗ có lợi rõ:**
  - T3: chẩn đoán JSON từ teacher, level-swap, cặp DPO.
  - T2: bug tổng hợp cho nhãn hiếm.
  - T1/T2: rationale chỉ là một dòng ablation.
- **Từ D:** dùng S6 làm feature kNN và nguồn exemplar few-shot, không làm cơ chế dự đoán chính.
- **Từ B, ở tầng hạ tầng:** một backbone và 2 adapter, phục vụ qua multi-LoRA:
  - `A12`: head cho T1 và T2, đa nhiệm vì rubric là giám sát dày, giúp regularize cho nhãn hiếm.
  - `A3`: renderer T3.
- **Thứ tự triển khai để giảm rủi ro tiến độ:**
  - Tuần 1–4: lõi A.
  - Tuần 4–6: các phần ghép từ C.
  - Tuần 6–7: DPO, chỉ làm nếu còn thời gian.

---

## 4. Thiết kế chi tiết EMH

### 4.1 Sơ đồ tổng

```
┌──────────────────────── OFFLINE (chỉ train) ─────────────────────────┐
│ train ─► S1 ─► S2 problem_card ─► S3 evidence ─► S6 index (fold-aware) │
│                 │                                                      │
│      Teacher (cache) ─► diag JSON, level-swap, DPO pairs, mutants      │
│                 ▼                                                      │
│ Qwen3-8B: A12 (heads T1+T2, cross-fit K=3)   A3 (T3: SFT → DPO)        │
│ Stacker + ngưỡng + rounder fit trên OOF (nested)                       │
└──────────────────────────────────────────────────────────────────────┘
ONLINE (dev/test, `make predict`, không người can thiệp):
sample ─► S1 ─► S3 evidence ─┬─► rules ───────────────────────────────┐
           S2 problem_card ──┼─► P-arm Qwen3-8B (vLLM, EV/logprob) ───┤
           S6 kNN feats ─────┼─► F-arm A12×3 fold (HF+PEFT, heads) ───┼─► stacker ─► decoders
                             └─► GBDT trên evidence ──────────────────┘
decoders: T1 policy → de-shrink/thresholds → MAP 6 chiều │ T2 calib → τ_l → joint consistency
T3: cond. diagnoser → masked renderer A3 → constrained decode → guard → retry ≤N → template
──► S7 firewall/validator ─► predictions.json + sha + commit
```

### 4.2 Phân bổ model

| Vai trò | Mặc định | Biến thể dùng cho ablation | Lý do |
|---|---|---|---|
| Backbone cho cả P và F | `Qwen/Qwen3-8B` (Apache-2.0, text-only, có thinking và non-thinking) | `Qwen2.5-Coder-7B-Instruct` (T1/T2), `Qwen3.5-9B` (T3) | TRL, PEFT, Unsloth, vLLM-LoRA đều hỗ trợ ổn định, và có sẵn `Qwen3ForSequenceClassification`. Qwen3.5 small (ra 02/03/2026, Apache-2.0) là model multimodal native với kiến trúc hybrid; theo bản phân tích models-infra, Unsloth không khuyến nghị QLoRA cho nó. Vì vậy chỉ dùng làm biến thể |
| A12 | LoRA r=16–32. Head: CORN cho logic (K=5) và edge_case (K=3); BCE cho 4 chiều nhị phân; Huber cho total; 10 sigmoid cho nhãn | Generative JSON, slot-template | Có xác suất thì mới chạy được decoder theo metric |
| A3 | LoRA SFT rồi DPO, `rpo_alpha` (cần tune) | ORPO, KTO | |
| Encoder rẻ | `answerdotai/ModernBERT-large` | — | 5-fold OOF rẻ, làm baseline F-arm |
| Embedding | `Qwen3-Embedding-0.6B` cộng BM25 | UniXcoder | |
| Judge nội bộ | Rule cộng Qwen3-8B (thinking), hoặc 27–32B bản 4-bit local | API (cache) | Chỉ dùng để chọn model; meta-eval trên train/dev |

- **Inference head:** dùng HF transformers + PEFT, batch size cố định, bf16. Head CORN tự viết không chạy sẵn được trên vLLM. vLLM mới có hỗ trợ head multi-label ở dạng thử nghiệm (cần kiểm chứng trên phiên bản đã pin).
- **Inference sinh văn bản:** dùng vLLM với multi-LoRA, prefix caching và seed.

### 4.3 Luồng huấn luyện, cross-fitting, tường lửa

- **`folds.csv`:**
  - Dùng `StratifiedGroupKFold`.
  - Group theo `pid` nếu dev/test có đề mới (D2); ngược lại group theo `dup_cluster`.
  - Stratify theo `type` × bin total.
  - Sau khi chia, kiểm mỗi fold có đủ positive cho từng nhãn hiếm.
- **Cross-fit A12 với K=3:**
  - Train 3 adapter, mỗi adapter trên 2/3 dữ liệu, dự đoán 1/3 còn lại để có OOF. Stacker, ngưỡng và rounder đều fit trên OOF này.
  - Dự đoán test là trung bình của 3 adapter, tức có ensemble miễn phí.
  - Lý do: nếu fit stacker và ngưỡng trên dự đoán in-sample của LLM đã fine-tune thì sẽ quá tự tin, ngưỡng lệch khi sang test. Các bản phân tích đầu vào chưa tính chi phí này. Chi phí train tăng gấp 3.
  - Nếu thiếu GPU: dùng blending holdout 70/30 thay cho cross-fit.
- **P-arm không cần train**, nên tự nhiên là OOF. Chỉ cần S6 loại chính bài đó, cả `dup_cluster` và fold hiện tại.
- **Dữ liệu tổng hợp** (mutant, teacher) chỉ sinh trên train và **nằm cùng fold với bài gốc**.
- **Tường lửa:**
  - T1/T2 chỉ nhận `BlindSample`.
  - Test CI render 200 prompt T1/T2 và assert không prompt nào chứa substring dài ≥30 ký tự của bất kỳ feedback nào.
  - Pipeline T2 không bao giờ mở file input của T3.
  - Không dùng feedback L4 làm lời giải tham chiếu (bác đề xuất D4 của bản data-metric). Thay bằng peer-oracle hoặc lời giải LLM.

### 4.4 Task 1

```
evidence + problem_card
 ├─ R: compilable ← rule trên compile_log GỐC (override cứng nếu khớp ≥99% train, D5)
 ├─ multi: S4 tách câu → evidence từng câu (compile riêng, gen-test, peer-oracle)
 │         → P-arm chấm từng câu {đúng/một phần/sai/không làm} → EV s_q
 │         → S5 policy: total_hard, total_soft (τ fit OOF)
 ├─ F-arm A12: E[dim] cho 6 chiều, E[total], P(dim=k)
 ├─ P-arm: JSON 6 chiều, structured output enum, đọc logprob → EV (+ CoT/SC trong ablation)
 ├─ kNN: total trung bình có trọng số của láng giềng cùng đề (OOF)
 └─ GBDT trên evidence (monotone: tăng theo pass_rate, giảm theo số error)
       ▼
Stacker tách theo type: ridge không âm (multi, ít mẫu) | LightGBM nhỏ (single)
       ▼
Decoder QWK: rescale theo μ/σ của train (2 tham số) vs OptimizedRounder (median bootstrap);
             chọn bằng nested CV; fit riêng single/multi nếu mỗi loại ≥ vài trăm mẫu
       ▼
total trên lưới hợp lệ ─► MAP trên 240 tổ hợp 6 chiều:
   argmax Σ log P(dim) với ràng buộc: luật cứng đã kiểm (<0.5% vi phạm), Σ = total (nếu D1)
       ▼ validator
```

- **Mấu chốt:**
  - Tổng điểm đi theo nhánh tối ưu QWK (giá trị liên tục rồi áp ngưỡng).
  - Từng chiều đi theo nhánh MAP có ràng buộc, tối ưu exact-match.
  - Hai nhánh gặp nhau ở ràng buộc Σ = total.

### 4.5 Task 2

```
evidence ─► rule flags: CE (pattern error:), TLE, ASan OOB/overflow, io_gap, -W* warnings
         ─► P(label) từ 4 nguồn:
             (1) A12 sigmoid heads (đa nhiệm với rubric)
             (2) P-arm: hỏi yes/no từng nhãn, đọc logprob; 10 câu dùng chung prefix
                 (vLLM prefix caching nên chi phí chỉ ~1.x lượt); định nghĩa + phản ví dụ nhãn
             (3) GBDT trên evidence      (4) kNN vote nhãn (OOF)
         ─► stacking theo nhãn: LR L2 trên OOF → Platt/temperature
         ─► chính sách ngưỡng theo nhãn:
             nhãn thường: argmax F1 trên OOF, median bootstrap, kiểm τ ≈ F̂/2
             nhãn hiếm (n_pos nhỏ): shrinkage về τ chung; nếu D6 = zero_division 1 thì dùng
                expected-F1 top-k (Σp nhỏ → không đoán)
             fallback: dùng hằng số nếu OOF F1 < 2π/(1+π)
         ─► CE: override cứng nếu tam giác compile nhất quán (D5)
         ─► joint consistency với output T1 (chỉ các ràng buộc conf ≥99% trên train)
         ─► map về đúng chuỗi NFC trong vocabulary; [] là hợp lệ
```

### 4.6 Task 3

```
evidence + nhãn gold + level
 ├─ Diagnoser (P-arm local hoặc A3 ở chế độ diag): JSON issues {label, câu, line_span,
 │   identifiers, symptom, root_cause, failing_input, fix_sketch(chỉ nội bộ)}, SC k=3–5
 │   kiểm cứng: dòng tồn tại, định danh ∈ AST, mọi nhãn được phủ, khẳng định khớp log
 ├─ Repair nội bộ (single: sửa và chạy public test, chỉ giữ bản pass) → diff
 ├─ Renderer A3 với information masking:
 │     L1: khái niệm + vùng (hàm/câu) + câu hỏi gợi mở  │ L2: + dòng/định danh/hiện tượng/input gây sai
 │     L3: + hướng sửa bằng lời/pseudo một phần          │ L4: + code đầy đủ đã pass test
 │     (renderer L1/L2 KHÔNG nhận fix_sketch/diff ⇒ không thể lộ thứ nó không thấy)
 ├─ Constrained decoding: L1 cấm cứng (```, `, ;, {}, #include, cout, cin, ::); L2 phạt mềm
 ├─ Guard: regex + parse tree-sitter + n-gram trùng với dòng thêm vào trong diff
 │         + classifier compliance + grounding check (dòng, định danh, giá trị test)
 └─ Vi phạm → sinh lại với danh sách seed cố định (N=3) → fallback template từ JSON chẩn đoán
```

- Độ dài theo median ± IQR của từng level, đo trên train.
- Đại từ xưng hô và thuật ngữ theo tần suất dùng của giảng viên.

### 4.7 Điểm quyết định (kiểm trong tuần 1–2)

| D | Câu hỏi | Kiểm bằng | Nếu Có | Nếu Không |
|---|---|---|---|---|
| D1 | Total luôn bằng Σ 6 chiều? | `(df[DIMS].sum(1)-df.total).abs().gt(1e-6).groupby(df.type).mean()` | MAP ràng buộc theo tổng | Head total riêng, policy làm feature |
| D2 | Dev/test có đề mới? | `(~dev.pid.isin(train.pid)).mean()` | Group CV theo `pid`, oracle do LLM sinh, kNN giảm vai trò | Group theo `dup_cluster`, kNN và peer-oracle mạnh |
| D3 | Multi có điểm từng câu? | Đọc 5 JSON thô | Train head từng câu có giám sát | Coi điểm câu là biến ẩn, dùng N1 (§6) |
| D4 | Scorer QWK làm tròn và truyền `labels` thế nào? Validator có bắt Σ = total? | Script phát kèm; 1 lượt nộp dò | Theo lưới đó | Xuất số nguyên, Σ nhất quán |
| D5 | Tam giác compile (log, compilable, nhãn CE) khớp ≥99%? | Bảng crosstab | Override cứng | Chỉ dùng làm feature |
| D6 | `zero_division` bằng bao nhiêu, không gian nhãn cố định 10? | Hỏi giảng viên, unit test | Chiến lược nhãn hiếm tương ứng | Bảo thủ: không đoán nhãn có Σp quá nhỏ |
| D7 | Gold L1/L2 có chứa code? | Regex và đọc tay 30 mẫu train | Lọc khỏi SFT, ghi vào hồ sơ bonus | Giữ nguyên |
| D8 | Detector compliance là gì? L2 có được trích code sinh viên? | Hỏi giảng viên, nộp dò template | Đặt mức cấm ở L2 | Mặc định phạt mềm |
| D9 | Có được gửi code sinh viên lên API? | Hỏi bằng văn bản | Teacher API, gói không train trên dữ liệu | Teacher 27–32B local |
| D10 | p95 token của multi có vượt 8k? | Tokenizer thật | Bắt buộc tách câu | Dùng cả file |
| D11 | Toolchain trong log là Windows? | Grep `mingw`, `C:\` | Ưu tiên log gốc, Docker MinGW (cần kiểm chứng) | g++ Linux |
| D12 | Có được viết tay cấu hình policy cho từng đề? | Hỏi giảng viên | Viết tay `policy.json` | S2 tự động, chỉ audit tay trên train |

### 4.8 Baseline tuần 1–2 và kế hoạch nộp

| ID | Task | Cấu hình | Mục đích |
|---|---|---|---|
| B0-T1 | T1 | Mean train; mean theo đề (LOO); luật compile (0 nếu không biên dịch được, ngược lại median) | Sàn QWK; đo mức thổi phồng do phương sai giữa các đề (B4) |
| B1-T1 | T1 | LightGBM trên feature regex từ log và test_report, độ dài, type; regression rồi rescale | Baseline mạnh và rẻ; **nộp LB trong tuần 1** |
| B2-T1 | T1 | Qwen3-8B qua vLLM, `enable_thinking=False`, temperature 0, schema enum, đọc top-logprobs ra EV, calibrate tuyến tính | Prompting arm v0 |
| B0-T2 | T2 | Hằng số theo nhãn / toàn tập rỗng | Sàn F1; kiểm hành vi `zero_division` |
| B1-T2 | T2 | Rule CE + LR/GBDT từng nhãn + ngưỡng theo nhãn trên OOF | **Nộp LB trong tuần 1** |
| B2-T2 | T2 | Qwen3-8B zero-shot JSON enum, SC n=5, tỷ lệ phiếu làm xác suất | Prompting arm v0 |
| B0-T3 | T3 | Template tất định theo nhãn × level | Trần compliance, sàn chẩn đoán; **nộp để dò metric** (bằng chứng cho bonus F5) |
| B1-T3 | T3 | Qwen3-8B few-shot cùng level và nhãn, regex guard, fallback template | Prompting arm v0 |
| FT0 | T1+T2 | Qwen3-8B QLoRA nf4, r=16, α=32, lr 2e-4, 1 epoch, max_len 4096, batch hiệu dụng 16, seed 42, các head như A12, `pos_weight` ≤10; chạy trên 1 fold | Smoke test fine-tune; đo VRAM, tốc độ; thắng B1 hay chưa |
| Infra | — | S1, S7 (validator, `make submit`), S8 (seed, cache, `folds.csv`), S9 (bản sao metric), `env.json` | Bảo vệ điều kiện tái lập ngay từ đầu |

- **Kế hoạch nộp:**
  - Tuần 1 nộp B1-T1, B1-T2 và B0-T3 để có điểm sàn và đối chiếu bản sao metric với scorer thật.
  - Sau đó chỉ nộp các điểm nằm trên đường Pareto của CV nội bộ.
  - Ghi tương quan CV với public LB sau mỗi lượt nộp.

### 4.9 Phân công theo lớp (6 người) và mốc

| Người | Sở hữu | Mục báo cáo |
|---|---|---|
| 1 | S1, S9, EDA, hồ sơ bonus | Dữ liệu và EDA, bonus |
| 2 | S3 sandbox, S4 tách câu | Tín hiệu thực thi |
| 3 | S2, S5, decoder T1 | T1 và policy |
| 4 | P-arm, S6, decoder T2 | Prompting, T2 |
| 5 | A12/A3, S8 tái lập | Fine-tune, phần cứng, tái lập |
| 6 | Renderer, guard và judge T3 | T3, đánh giá nội bộ |

Mốc thời gian:
- **Tuần 1–2:** baseline và hạ tầng.
- **Tuần 3–4:** evidence đầy đủ, policy, cross-fit A12, guard T3.
- **Tuần 5:** stacker và decoder, SFT A3.
- **Tuần 6:** distill, DPO nếu còn thời gian, ablation.
- **Tuần 7:** freeze, tái lập trên máy sạch.
- **Tuần 8:** chạy private test, slide.

---

## 5. Trục ablation sinh ra từ kiến trúc

Mỗi dòng báo cáo:
- metric chính kèm CI 95% (paired bootstrap so với cấu hình mặc định);
- metric phụ;
- tách theo single/multi;
- chi phí: $/1000 bài, GPU-giờ, độ trễ.

| Lớp | Trục | Mức | Task | Câu hỏi trả lời |
|---|---|---|---|---|
| Evidence | E1 Tín hiệu đầu vào | code → +log → +test_report → +tự chạy và io_gap → +test sinh/peer-oracle → +sanitizer/static | 1, 2, 3 | Mỗi tín hiệu đóng góp bao nhiêu |
| | E2 Log gốc vs log tái lập | 2 mức | 1, 2 | Toolchain lệch ảnh hưởng thế nào (gắn với bonus D1) |
| Problem | P1 Policy engine | tắt / cứng / mềm / cả hai làm feature | 1 (multi) | Giá trị của yêu cầu tiên quyết |
| | P2 Trích policy | regex / LLM / LLM-SC | 1 | Độ chính xác extractor (đo trên train) |
| | P3 Tách câu | cả file / heuristic / LLM | 1, 3 | |
| Retrieval | R1 k và khoá | k ∈ {0, 2, 4, 8}; BM25 / dense / verdict; cùng đề vs bất kỳ | 1, 2, 3 | Giá trị của RAG, copy bias |
| | R2 Chống rò rỉ | loại self/dup/fold vs ngây thơ | 1, 2 | Mức lạc quan của CV ngây thơ |
| Model | M1 Chế độ P-arm | zero / few-shot RAG / thinking-CoT / SC k ∈ {1, 5} | 1, 2, 3 | Chi phí so với chất lượng |
| | **M2 P vs F cùng backbone** | 3 mức (P, F, P+F stack) | 1, 2, 3 | **So sánh bắt buộc của đề, có kiểm soát** |
| | M3 Dạng đầu ra của F | CORN / head MSE / SFT JSON / slot-template | 1, 2 | Loss biết thứ tự có giúp không |
| | M4 Backbone và lượng tử | ModernBERT / Coder-7B / Qwen3-8B / Qwen3.5-9B (T3); 4-bit vs bf16 | tất cả | Kích thước, chi phí triển khai |
| | M5 Head đa nhiệm T1+T2 vs tách riêng | 2 mức | 1, 2 | Rubric có regularize được cho nhãn hiếm |
| | M6 Distill | ± rationale; ± bug tổng hợp (tỷ lệ 0.5/1/2×); AUC của domain classifier | 1, 2 | Giá trị của teacher |
| Decoder | Dα Decoder QWK | argmax / EV × round / rescale / OptimizedRounder / quantile | 1 | De-shrink theo CCC; đánh đổi với MAE |
| | Dβ Cách tính tổng | Σ argmax / Σ EV / total trực tiếp / stack | 1 | |
| | Dγ Ngưỡng nhãn | 0.5 / chung / theo nhãn / shrinkage / expected-F1 | 2 | Kèm mức optimism đo bằng nested CV |
| | Dδ Joint consistency và MAP 240 | bật / tắt | 1, 2 | Exact-match, mức nhất quán giữa T1 và T2 |
| | Dε Stacker vs model đơn tốt nhất | 2 mức | 1, 2 | |
| T3 | G1 Số bước | 1 bước vs chẩn đoán rồi render | 3 | |
| | G2 Information masking | bật / tắt | 3 | |
| | G3 Constrained decoding | không / chỉ fence / cấm cứng / phạt mềm | 3 | Kèm độ trôi chảy |
| | G4 Guard | N ∈ {0, 1, 3, 5} ± fallback; tín hiệu regex / AST / overlap / classifier | 3 | Kèm tỷ lệ phải fallback |
| | G5 Học | SFT / +DPO theo từng loại rejected / ± level-swap | 3 | |
| | G6 Lọc gold L1/L2 vi phạm | bật / tắt | 3 | Định lượng hệ quả của vấn đề dữ liệu |
| Giao thức | X1 Cách chia | random vs group | tất cả | Phương pháp luận |
| | X2 Nhiễu nhãn | nguyên vs khử cụm near-dup mâu thuẫn | 1, 2 | |
| | X3 Độ bền | chèn comment injection ± bỏ comment | 1, 2 | Số liệu cho model card |

Mỗi đường Pareto (T3 dùng compliance × chẩn đoán; T1/T2 dùng metric × $) là một hình trong báo cáo.

---

## 6. Ý tưởng novelty

Độ khó chấm 1–5. Ưu tiên P1 là làm chắc chắn, P3 là làm nếu còn thời gian.

| # | Ý tưởng | Khó | Giá trị LB | Giá trị báo cáo | Ưu tiên |
|---|---|---|---|---|---|
| N1 | Policy engine cộng **học điểm từng câu ẩn qua gating mềm khả vi** (multi-instance, giám sát mức file) | 4 | Cao (multi) | Cao: đúng điểm đề nhấn mạnh | P1 (bản cứng), P2 (bản khả vi) |
| N2 | **Decoding theo metric có cơ sở lý thuyết**: QWK = CCC nên de-shrink; F1 có τ\* = F\*/2; expected-F1 tuỳ theo `zero_division` | 2 | Cao | Cao: một mục lý thuyết gọn | P1 |
| N3 | **Sinh văn bản có chặn rò rỉ ở T3**: masking, ràng buộc logit, guard nhiều tầng, fallback, kèm đường Pareto | 3 | Rất cao | Rất cao | P1 |
| N4 | **Joint decoding T1+T2 có ràng buộc nhất quán**; ràng buộc khai phá bằng association rules, chỉ áp khi conf ≥99% | 3 | Thấp đến trung bình | Trung bình đến cao, cộng bonus nếu gold tự vi phạm | P2 |
| N5 | Differential testing bằng peer-oracle | 3 | Cao nếu cùng đề | Cao | P2 |
| N6 | **LeakIG**: `[log p(diff│code,fb) − log p(diff│code)]/│diff│` dưới một code LLM cố định | 4 | Trung bình | Cao (mới) | P3 |
| N7 | **Trần nhiễu nhãn**: κ giữa các cặp near-dup cộng bất đồng teacher-blind với gold, để báo cáo "đạt x% trần" | 2 | 0 | Cao, cộng bonus | P1 |
| N8 | **Định tuyến theo độ bất định**: entropy của EV quyết định có chuyển sang nhánh đắt (CoT/SC/teacher) không, cho ra đường chi phí–chất lượng | 2 | Trung bình | Cao (mục triển khai thực tế) | P2 |
| N9 | Slot-template teacher-forced: lấy xác suất 10 nhãn trong 1 lượt sinh | 2 | Trung bình | Trung bình | P3 |
| N10 | Grounding checker cho T3 (dòng, định danh, giá trị test phải có thật) | 1 | Trung bình (b) | Trung bình | P1 |
| N11 | Đo độ bền trước prompt injection, đưa vào model card | 1 | 0 | Trung bình | P2 |

**N1, chi tiết.**
- Với multi_problem không có điểm từng câu (nhánh Không của D3):
  `logiĉ = 4·Σ_q w_q · s_q · Π_{p∈pre(q)} σ((s_p − τ)/T)`
  - `s_q` là xác suất câu q đúng, do head chạy trên đoạn code của câu q.
  - Train end-to-end bằng loss Huber hoặc ordinal so với nhãn mức file.
  - Gradient chảy qua cấu trúc chính sách, nên head từng câu tự học mà không cần nhãn từng câu.
- Sản phẩm phụ: τ và T học được cho biết giảng viên có thực sự áp chính sách hay không. Đây là bằng chứng định lượng cho giả thuyết A4/GT-10, dùng được cho bonus.
- Ablation: không có policy / gating cứng / gating mềm cố định / gating mềm học được.

**N2, chi tiết.**
- Một trang báo cáo dẫn ra κ = 2cov/(σ²_y + σ²_ŷ + Δμ²).
- Hệ quả: hồi quy MSE cho κ = 2ρ²/(1+ρ²) < ρ.
- Ablation Dα kiểm chứng điều này bằng thực nghiệm.
- Tương tự cho F1: τ\* = F\*/2 dùng làm phép kiểm calibration.
- Chi phí thấp, nhưng phần "phương pháp" (25%) nhận được một luận cứ toán học thay cho "thử thấy tốt".

**N3, chi tiết.**
- Đóng góp chính là một bảo đảm theo nghĩa thông tin: renderer L1/L2 không được nhìn thấy cách sửa, nên không thể lộ cách sửa.
- Báo cáo chi phí của bảo đảm này: đường Pareto theo λ và N, cùng tỷ lệ phải fallback.

---

## 7. Phản biện và điều chỉnh so với các bản phân tích đầu vào

1. **Prompting arm chạy local** (nguyên tắc 4). Cách này giải căng thẳng ở spec-compliance §3 dòng 2 và 3. Vì Anthropic không trả logprobs, cách hỏi yes/no từng nhãn và đọc logprob của bản task2 chỉ khả thi đầy đủ trên model local; trên API còn tốn gấp 10 lượt gọi.
2. **Cross-fitting LLM fine-tune là chi phí bị bỏ sót.** Stacker ở các bản task1 và task2 cần OOF của model 8B, nghĩa là phải train K adapter (§4.3).
3. **Bản task1 đề xuất 9 hướng, quá nhiều cho 8 tuần.**
   - Giữ: (a) có EV, (b) chỉ làm feature, (d), (e), (h).
   - (c) SFT JSON chỉ còn là một dòng ablation M3.
   - Bỏ (i) so sánh cặp.
4. **Qwen3.5-9B không làm nền.** Lý do: multimodal native, kiến trúc hybrid, rủi ro QLoRA và rủi ro với head classification. Model này vẫn là biến thể đáng thử cho T3.
5. **Head chạy bằng HF + PEFT, không chạy trên vLLM.** Head CORN tự viết không được vLLM hỗ trợ sẵn; vLLM chỉ dùng cho phần sinh văn bản.
6. **Stacker cho multi_problem nên là ridge hoặc LR** nếu số mẫu multi nhỏ [GT]. LightGBM dễ overfit.
7. **Thứ tự ưu tiên ở T3:** guard (tuần 3) → SFT (tuần 4–5) → DPO (tuần 6, tuỳ chọn). Guard mới là thứ bảo đảm compliance; DPO chỉ cải thiện chất lượng.
8. **LeakIG và mô phỏng sinh viên cần một diff tham chiếu**, tức chỉ có ở những bài repair đã pass test. Dùng làm reranker hoặc phân tích, không dùng làm cổng cứng.
9. **Bác việc dùng feedback L4 làm lời giải tham chiếu** cho feature T1/T2 (đề xuất D4 của bản data-metric). Vi phạm R15 theo câu chữ là quá rủi ro so với lợi ích, nhất là khi peer-oracle và lời giải LLM đã thay thế được.

---

## 8. Câu hỏi mở ảnh hưởng trực tiếp tới kiến trúc

1. **D9:** có được gửi code sinh viên lên API không? Câu trả lời quyết định teacher là API hay model local.
2. **D4 và D6:** scorer QWK làm tròn và truyền `labels` thế nào; `zero_division` của macro-F1 bằng bao nhiêu; validator có bắt total = Σ chiều không.
3. **D8:** detector compliance dùng cơ chế gì; L2 có được trích code của sinh viên không; L4 thiếu lời giải có bị phạt không.
4. **D2:** dev/test có đề mới không?
5. **D3 và D12:** có điểm từng câu không; có được viết tay cấu hình policy cho từng đề không.
6. "Chạy cục bộ" có tính VM thuê theo giờ không? Câu trả lời ảnh hưởng tới kế hoạch cross-fit K=3.
7. Test T2 và test T3 có dùng chung bài không (rò rỉ chéo)? Nếu có thì tường lửa ở S7 là bắt buộc, và phát hiện này cần báo giảng viên.

Sources:
- [Qwen 3.5 Small Model Series announcement (X)](https://x.com/Alibaba_Qwen/status/2028460046510965160)
- [Unsloth Qwen3.5 collection](https://huggingface.co/collections/unsloth/qwen35)
- [vLLM Classification Usages (pooling models)](https://docs.vllm.ai/en/latest/models/pooling_models/classify/)
- [vLLM PR #30315: multi-task multi-label classification heads](https://github.com/vllm-project/vllm/pull/30315)
- [vLLM multi-LoRA blog (2026-02-26)](https://vllm.ai/blog/2026-02-26-multi-lora)

Status: DONE
Summary: Đề xuất 4 kiến trúc (A Evidence-first Hybrid, B Multitask LLM, C Teacher–Student, D Retrieval-heavy) kèm ma trận có trọng số (A 4.59, C 3.91, B 3.00, D 2.95). Khuyến nghị EMH: A làm xương sống, ghép C (cho T3 và nhãn hiếm), D (làm feature) và B (một backbone Qwen3-8B, 2 adapter). Có thiết kế chi tiết từng task, 12 điểm quyết định, baseline tuần 1–2, 28 trục ablation và 11 ý tưởng novelty.
Concerns/Blockers: Chưa có dữ liệu; mọi [GT] và ước lượng GPU/$ phải đo lại. Còn cần kiểm chứng: hỗ trợ head multi-label trên phiên bản vLLM sẽ pin, VRAM của teacher 27–32B bản 4-bit, Docker MinGW cho trường hợp toolchain Windows.