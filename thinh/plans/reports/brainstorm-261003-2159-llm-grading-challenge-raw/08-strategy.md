# Kế hoạch chiến lược cho đồ án LLM chấm điểm và phản hồi bài lập trình C++

Mọi nhận định về dữ liệu trong tài liệu này là **[GT] (giả thuyết)**. Cách kiểm chứng định lượng đã có trong các bản phân tích đầu vào, ký hiệu ví dụ A2 hay H2 trỏ về đó. Chưa có số liệu thật nào ở đây.

---

## 0. Tám quyết định chiến lược (có chỗ khác các bản phân tích đầu vào)

| # | Quyết định | Lý do và chỗ phản bác |
|---|---|---|
| D1 | **Hệ thống nộp private chạy hoàn toàn bằng model local.** Hướng prompting dùng LLM open-weight chạy local, ví dụ Qwen2.5-Coder-32B-Instruct bản AWQ hoặc Qwen3-14B/32B. Giới hạn ≤10B chỉ khuyến nghị cho model fine-tune, không áp cho model chỉ dùng prompting. API thương mại chỉ dùng để distill hoặc sinh dữ liệu, và chỉ khi giảng viên đồng ý bằng văn bản. | Một quyết định gỡ được 3 rủi ro: không phát tán dữ liệu (mục 9), tái lập (API không tất định), và câu chữ "API để huấn luyện". Các bản phân tích đầu vào mặc định prompting đi qua API; tôi đề xuất không làm vậy. |
| D2 | **Giới hạn số hướng.** Mỗi task có: 1 baseline phi-LLM, 1 hướng prompting, 1 hướng fine-tune, sau đó stacking. Cắt bỏ: so cặp với anchor (T1-i), GRPO, Label Powerset/RAkEL, LeakIG, mô phỏng học sinh, và Qwen3.5-9B làm backbone chính. Qwen3.5-9B chỉ đưa vào ablation nếu stack chạy ổn trước W4. | Bản phân tích T1 có 9 hướng, T3 có 6 lớp guardrail cộng DPO/GRPO. Với 8 tuần thì không làm hết được. ABL cần ablation sạch, không cần nhiều hướng. |
| D3 | **Một backbone fine-tune, 2 adapter.** Backbone là Qwen2.5-Coder-7B-Instruct hoặc Qwen3-8B, chọn bằng pilot ở W2. Adapter (a) là head đa nhiệm T1+T2 (6 head ordinal và 10 sigmoid). Adapter (b) là generative cho T3 (SFT rồi DPO). | Chỉ cần một stack, một model card và một cấu hình phần cứng để báo cáo. Head cho xác suất nên tune được ngưỡng QWK và macro-F1. |
| D4 | **Phần tất định viết bằng code, không giao cho LLM:** compilable và nhãn Lỗi biên dịch (CE) lấy từ log (nếu A2 đúng), policy engine cho multi_problem, guardrail L1/L2, validator, ràng buộc nhất quán T1↔T2. | Đây đúng là hai điểm đề nhấn mạnh (câu tiên quyết, compliance). Code tuân thủ 100%, ablation sạch, và dễ bảo vệ ở Q&A. |
| D5 | **Chọn mọi thứ theo CV (OOF).** Validation protocol khoá cuối W2. Public LB chỉ dùng để xác nhận. | Xếp hạng chính thức lấy theo private. Quota 2 lượt/ngày quá ít để chọn model theo LB. |
| D6 | **Trường feedback chỉ được chạm ở hai chỗ:** làm target cho T3 và làm EDA/audit. Không đưa vào input, exemplar, index RAG, feature hay prompt của teacher cho T1/T2. | **Bác bỏ** đề xuất D4 của bản data-metric (dùng feedback L4 làm lời giải tham chiếu để sinh feature T1/T2). Đề xuất này rơi đúng vào "trường feedback làm input". Thay vào đó dùng peer-oracle (bài train đạt điểm tối đa) hoặc lời giải do LLM sinh rồi kiểm bằng test công khai. |
| D7 | **L1/L2 mặc định không trích code của sinh viên.** Chỉ nới cho L2 (trích định danh hoặc biểu thức sai ngắn) khi EDA A5 đo được rằng giảng viên trích ở mức có hệ thống. | Bản spec-compliance và bản T3 mâu thuẫn ở điểm này. Dữ liệu sẽ quyết định. Trong lúc chờ, chọn phương án an toàn cho metric (a). |
| D8 | **CI dùng cluster bootstrap theo `problem_id`** nếu dev/test có đề mới (H2). | Bootstrap iid như các bản đầu vào đề xuất sẽ cho CI hẹp giả, vì các bài cùng đề tương quan với nhau. |

---

## 1. Kế hoạch 8 tuần

Cột "Mốc kiểm tra" là gate. Không qua gate thì xử lý theo cột cuối, không đi tiếp.

| Tuần | Mục tiêu | Sản phẩm cụ thể | Mốc kiểm tra (gate) | Nếu trượt gate |
|---|---|---|---|---|
| **W1** | Nền móng, điểm sàn cho cả 3 task, gửi câu hỏi cho giảng viên | Repo private và skeleton (§7). Loader và tài liệu schema. Notebook `audit/00_sanity` chạy A0, A1, A2, B1–B3, H1 (token), H2 (đề trùng), A5 (feedback L1/L2 có code). `eval/metrics.py` mô phỏng scorer, có unit test (QWK `labels=range(11)`, 3 quy ước `zero_division`). `make submit` (validator chính thức cộng kiểm chặt). **Baseline 3 task** (§7). `splits/folds.csv`. Protocol v0. Email câu hỏi cho giảng viên | Cả 3 task có điểm public. `make baseline` chạy được trên máy của một thành viên khác. CV baseline đã ghi log | Dừng mọi việc khác, cả nhóm dồn vào cho tới khi 3 task có điểm |
| **W2** | Tín hiệu tất định, prompting v1, **khoá protocol** | Sandbox Docker compile/run và `features/*.parquet` (log, verdict, warning, sanitizer, io_gap). Chỉ mục near-dup (C1/C2). LightGBM T1/T2 trên feature. Prompting local v1: T1 dùng EV decoding, T2 hỏi yes/no và đọc logprob, T3 làm diagnose rồi render cộng regex guard. Pilot backbone (2 model × 200 mẫu, đo QWK/F1 và chất lượng tiếng Việt chấm mù). **Protocol v1.0 có chữ ký cả nhóm** | Mỗi task có ít nhất 1 hướng thắng baseline, CI của Δ không chứa 0. Có ít nhất 3 phát hiện cho bonus kèm số liệu | Thu hẹp: giữ hướng nào đang thắng, hoãn hướng còn lại |
| **W3** | Fine-tune v1, xử lý đặc thù từng task | Head QLoRA T1+T2 v1 (seed 0). T3 SFT v1 (đã lọc L1/L2 vi phạm). Policy extractor và aggregator cho multi_problem có unit test. Ngưỡng theo nhãn cho T2 (bootstrap cộng shrinkage). Rescale hoặc OptimizedRounder cho T1. Guardrail T3 2 tầng kèm fallback template. **Khung báo cáo**: mục lục và các hình EDA | Pipeline fine-tune chạy từ config tới predictions. Inference chạy 2 lần cho diff = 0. Có số QWK multi_problem khi bật và tắt policy engine | Nếu fine-tune chưa chạy được: chuyển người từ T1/T2 sang hỗ trợ trong 3 ngày |
| **W4** | Mở rộng, **review giữa kỳ quyết định giữ hay bỏ** | kNN/RAG cùng đề (chỉ khi H2 đúng). Stacking v1 trên OOF. Detector T3, best-of-N, judge nội bộ đã qua meta-eval. Peer assessment lần 1. `ablation.csv` đủ ≥50% số hàng | Mỗi hướng có Δ CV kèm CI. Hướng nào sau 2 tuần chưa đạt ngưỡng thực tế (§4) thì **dừng**. **Chốt danh sách thành phần final; sau W4 không mở hướng mới** | — |
| **W5** | Nâng cấp có chọn lọc, hoàn thiện hồ sơ bonus | T3: DPO (chỉ khi SFT và guard đã xong). T2: augmentation cho nhãn hiếm (chỉ giữ nhãn có OOF thật tăng). T1: tách câu và test sinh cho multi_problem. **Hồ sơ bonus 2–4 phát hiện** theo mẫu đầy đủ, có `make audit-*`, gửi giảng viên kèm timestamp. Viết bản nháp mục EDA và Phương pháp | Đã gửi hồ sơ. Ablation đủ ≥80% | Nếu DPO chưa xong: bỏ, giữ SFT cộng guard |
| **W6** | Train bản final, kiểm tái lập | Model final chạy 3 seed. Ensemble/stack final. Ablation đủ 100% kèm CI. Các thẻ phân tích lỗi (§4.3). **Chạy thử `make reproduce` lần 1 trên máy sạch.** Đo throughput, ước thời gian chạy private test | Reproduce khớp metric trong ε. Mọi con số trong báo cáo truy được về run_id | Sửa tái lập trước mọi việc khác, vì đây là GATE của 30% RANK |
| **W7** | **Feature freeze**, viết báo cáo, cũng là tuần đệm | Tag `v1.0-rc`. Báo cáo bản nháp đầy đủ. Model card. Chọn cấu hình final theo quy tắc §3.4. Chạy thử reproduce lần 2. Slide v1 | Mỗi mục báo cáo được 2 người không viết mục đó review. Checklist sản phẩm nộp đều xanh | Cắt bớt các ablation phụ, không cắt phần phân tích lỗi |
| **W8** | Chạy private test, trình bày | `predictions.json` private sinh từ tag `v1.0` bằng 1 lệnh; lưu hash file và commit. Slide final. Diễn tập 2 lần. Luyện Q&A. Bảng đóng góp có chữ ký | Diễn tập ≤14 phút. Mỗi thành viên trả lời được top 15 câu Q&A | — |

**Validation protocol (bản v1.0 khoá cuối W2) gồm:**
1. **Split.** Nếu H2 cho thấy dev/test có đề mới: dùng `StratifiedGroupKFold`, group theo cụm near-dup lồng trong problem_id. Nếu không: group theo cụm near-dup, stratify theo (đề × nhóm total) cho T1 và iterative stratification cho T2. Mã sinh viên ẩn danh chỉ dùng làm group, không làm feature. `folds.csv` được commit.
2. **Metric.** Dùng bản mô phỏng scorer (§0 D5), có unit test bám theo câu trả lời của giảng viên. Khi giảng viên chưa trả lời thì báo cáo song song các quy ước.
3. **Thống kê.** Theo §4.1.
4. **Quy tắc thay đổi.** Sau khi khoá, muốn sửa protocol cần cả nhóm đồng ý, ghi vào `docs/protocol-changelog`, và chạy lại các hàng ablation bị ảnh hưởng.
5. **Vùng cấm.** Không ai đọc input của test để chỉnh prompt, rule hay ngưỡng. Phân tích lỗi chỉ làm trên OOF của train và trên dev.

---

## 2. Phân vai

### 2.1 Nhóm 5 người

| Vai | Sở hữu (module, quyết định) | Mục báo cáo / slide | Người dự phòng |
|---|---|---|---|
| **P1 Trưởng nhóm, Eval, Repro** | Repo, CI, `core/` (config, seed, log), `eval/` (metric, bootstrap, ablation harness), `scripts/submit.py`, nhật ký nộp, cache API, `make reproduce`, tracking | Thiết lập thí nghiệm, Kết quả, Tái lập; biên tập toàn bộ báo cáo | P4 |
| **P2 Data và Tín hiệu** | Loader, EDA, audit và hồ sơ bonus, sandbox, `features/`, chỉ mục near-dup, `folds.csv` | Dữ liệu và EDA, Vấn đề dữ liệu/metric | P3 |
| **P3 Chủ T1** | Policy extractor và aggregator, tách câu, LightGBM/kNN T1, calibration và làm tròn, prompting T1 | Phương pháp T1, phân tích lỗi T1 | P2 |
| **P4 Chủ T2, LLM platform** | vLLM serving, pipeline QLoRA, head đa nhiệm T1+T2, ngưỡng theo nhãn, rule T2, stacking | Phương pháp T2, Fine-tune, Phần cứng | P1 |
| **P5 Chủ T3** | Diagnose JSON, renderer, guardrail và detector, SFT/DPO T3, judge nội bộ và meta-eval | Phương pháp T3, phân tích lỗi T3, model card (phần rủi ro) | P4 |

### 2.2 Nhóm 7 người

| Vai | Sở hữu | Người dự phòng |
|---|---|---|
| **A Trưởng nhóm, Tích hợp** | Kế hoạch, postprocess và submission writer, nhật ký nộp, biên tập báo cáo, slide | B |
| **B Eval, Repro, Infra** | `eval/`, bootstrap, ablation harness, CI, cache API, log chi phí, env report, `make reproduce`, chạy thử trên máy sạch | A |
| **C Data audit, Bonus** | EDA, audit A–F, hồ sơ bonus, phần privacy (chỉ đếm) | D |
| **D Tín hiệu thực thi** | Sandbox, compile/run, sanitizer, test sinh, peer-oracle, static analysis, `features/` | C |
| **E Chủ T1** | Policy engine, tách câu, calibration, stack T1 | F |
| **F Chủ T2, Fine-tune platform** | QLoRA head T1+T2, vLLM, ngưỡng theo nhãn, augmentation | E |
| **G Chủ T3** | Pipeline T3, guardrail, SFT/DPO, judge | B |

### 2.3 Ranh giới và điểm tích hợp (hợp đồng dữ liệu)

- `data/processed/{split}.parquet`: Data sở hữu, schema có version. Cột tối thiểu: `sample_id, pid, type, course, code, compile_log, test_report, labels…`.
- `features/{name}.parquet`: Tín hiệu sở hữu. Khoá theo `sample_id`, không chứa nhãn.
- `oof/{task}/{run_id}.parquet`: **mọi model** xuất cùng một định dạng `sample_id, fold, seed, pred_*` (E-value hoặc xác suất). Eval sở hữu định dạng này. Stacker chỉ đọc từ đây.
- `policy/{pid}.json`: T1 sở hữu. `diagnosis/{sample_id}.json`: T3 sở hữu. T1/T2 chỉ được dùng bản chẩn đoán sinh **không có** nhãn và **không có** feedback.
- **Nhịp tích hợp.** Thứ Sáu hằng tuần là ngày tích hợp: merge vào `main`, chạy `make dev-eval` từ đầu đến cuối, cập nhật `ablation.csv`. Chạy hỏng thì sửa trước khi làm việc mới.
- Không ai nộp tay. Chỉ người giữ vai Tích hợp (P1/A) chạy `make submit`.

### 2.4 Bảng đóng góp minh bạch

- **Sổ tuần**, mỗi người điền 10 phút vào thứ Sáu. Cột: Tuần | Thành viên | Việc hoàn thành | Bằng chứng (PR, commit, run_id, mục báo cáo) | Giờ ước tính | Việc tuần sau.
- **Bảng cuối** (nộp kèm báo cáo). Cột: Thành viên | Vai | Module sở hữu | Đóng góp chính (3–5 dòng, có link PR/run_id) | Mục báo cáo/slide | Tổng giờ | % đóng góp | Xác nhận (chữ ký).
- Việc không phải code (EDA, viết báo cáo, slide, gán tay train để đo detector) cũng có dòng và có bằng chứng.
- Số dòng code trong `git log` chỉ là bằng chứng phụ, không dùng để tính %.
- **Peer assessment ẩn danh** ở W4 và W8. Mỗi người chia 100 điểm cho những người còn lại. Chênh lệch lớn ở W4 thì xử lý ngay (phân lại việc), không để đến cuối kỳ.
- % cuối cùng do cả nhóm đồng thuận, ghi lại lý do nếu lệch so với peer assessment.

---

## 3. Chiến lược leaderboard

### 3.1 Quota thực tế
- 2 lượt/ngày × khoảng 56 ngày cho trần 112 lượt. Thực tế dùng được khoảng **35–50 lượt**: tuần đầu còn đang dựng hạ tầng, W7–W8 đã freeze, nhiều ngày không có ứng viên đạt gate.
- Nếu quota tính chung cho 3 task, mỗi task chỉ còn khoảng 12–17 lượt. Nếu một file chứa cả 3 task thì mỗi lượt đánh giá được cả 3. Phải hỏi giảng viên (§8).
- **Phân bổ.** Slot A mỗi ngày dành cho ứng viên đã qua gate CV. Slot B để dự phòng (sửa lỗi, đối chiếu scorer), không dùng thì bỏ. Mỗi task tối đa 3–4 lượt/tuần. Gom nhiều cải tiến nhỏ đã kiểm trên CV vào một lượt nộp.

### 3.2 Quy tắc nộp (phải thoả **một** trong ba)
1. **Cải thiện thật.** Δ CV ≥ max(ngưỡng thực tế, 1 SE của paired bootstrap), và đã qua validator.
2. **Hiệu chỉnh CV↔LB.** Trong W1–W3 nộp 3–5 cấu hình có CV trải rộng để fit `LB ≈ a + b·CV`. Nếu b xa 1 hoặc phần dư lớn thì nghi distribution shift; khi đó chạy adversarial validation (B5).
3. **Đối chiếu scorer.** Nộp baseline một lần để so điểm public với `eval/metrics.py`. Lệch nhau thì sửa bản mô phỏng.

**Không làm:** dò nhãn dev; nộp hệ suy biến (output rỗng, template) lên public để chứng minh metric có thể bị game khi giảng viên chưa đồng ý (phần này chứng minh cục bộ); tune ngưỡng theo điểm public.

### 3.3 Nhật ký nộp (`submissions/log.csv`)
Cột: # | ngày giờ | task | commit | config_hash | run_id | thay đổi so với lần trước (1 dòng) | CV mean [CI95] | Δ CV so với best | public | Δ public | public dự đoán từ fit CV↔LB | phần dư | người nộp | quyết định (giữ/bỏ) | hash file.
- Trước mỗi lượt, ước độ nhiễu của LB bằng bootstrap OOF với cỡ mẫu bằng |dev|. Δ public nhỏ hơn mức nhiễu này thì coi là không có thông tin.

### 3.4 Chọn bản nộp private cuối
- Điểm chọn = CV mean − 0.5·SE (bảo thủ). Hoà thì chọn cấu hình đơn giản hơn, rẻ hơn, tất định hơn.
- Ưu tiên ensemble hoặc stack nhiều seed vì phương sai thấp hơn. Không chọn cấu hình chỉ đứng đầu trên public.
- CV và LB mâu thuẫn: tin CV, trừ khi adversarial validation hoặc kiểm tra đề mới cho thấy dev giống test hơn train. Khi đó dùng CV có trọng số theo tỷ lệ đề mới.
- Nếu dev có nhãn công khai sau này và giảng viên cho phép: train lại trên train+dev với **số epoch và ngưỡng đã chốt từ CV**, không tune lại.

### 3.5 Bảng tổng là trung bình thứ hạng: phân bổ công sức thế nào
- **Không task nào được rơi xuống đáy.** Đứng cuối ở một task kéo hạng tổng xuống nhiều hơn mọi cải thiện biên ở hai task kia.
- **Độ co giãn thứ hạng.** Với task t: `e_t` = số đội có điểm trong khoảng [s_t, s_t + δ_t] trên public LB, chia cho số giờ công ước tính để tăng δ_t. Từ W3 mỗi tuần dồn người vào task có `e_t` lớn nhất. Task có nhiều đội sát điểm nhau thì một cải thiện nhỏ cũng vượt được nhiều đội.
- **Giả thuyết ưu tiên ban đầu** (cập nhật khi có LB):
  1. **T3, guardrail compliance.** Rẻ nhất, khoảng cách giữa các đội có lẽ lớn nhất vì nhiều đội sẽ để lọt code ở L1/L2.
  2. **T2, ngưỡng theo nhãn và rule cho các nhãn tất định.** Macro-F1 rất nhạy với nhãn hiếm. Nhưng cũng nhiễu nhất, nên một hạng đạt được ở đây kém bền.
  3. **T1, policy engine cho multi_problem và calibration QWK (de-shrink).** Các đội có lẽ sẽ sát điểm nhau sau khi calibrate; điểm khác biệt nằm ở bài multi_problem và các ca lệch xa.

---

## 4. Ablation và phân tích lỗi

### 4.1 Giao thức thống kê
- **Seed.** Fine-tune LLM: 3 seed, báo mean ± std, CI của Δ tính trên dự đoán đã trung bình qua seed. Encoder và LightGBM: 5 seed × 5 fold. Prompting greedy local: 1 lần chạy (đã tất định). Self-consistency và sampling: 3 seed.
- **Bootstrap.** B = 2000. **Cluster theo problem_id** nếu có đề mới (lấy mẫu lại theo đề, kéo theo mọi bài của đề), nếu không thì theo sample. **Paired**: A và B dùng cùng chỉ số resample. Báo CI percentile 95%.
- **Kiểm định ý nghĩa.**
  - Δ QWK và Δ macro-F1 (metric không phân rã theo mẫu): paired bootstrap, kiểm lại bằng approximate randomization (10k hoán vị đổi chỗ dự đoán A/B theo từng mẫu; theo Dror et al., ACL 2018).
  - Exact-match từng chiều và tỷ lệ vi phạm compliance (nhị phân, ghép cặp): McNemar.
  - Điểm judge T3: Wilcoxon signed-rank.
  - Nhiều so sánh trong cùng một họ ablation của một task: hiệu chỉnh Holm–Bonferroni.
- **Ngưỡng thực tế (ghi trong protocol):** ΔQWK ≥ 0.01; Δmacro-F1 ≥ 0.01; Δ tỷ lệ vi phạm ≥ 1 điểm %; Δ điểm judge ≥ 0.05 trên thang 0–1. Cải thiện chỉ "có ý nghĩa thống kê" mà nhỏ hơn ngưỡng này thì không giữ nếu làm tăng độ phức tạp.
- **Hai góc nhìn bắt buộc:** cộng dần từ baseline (build-up) và bỏ từng thành phần khỏi hệ final (leave-one-out). Chỉ xét tương tác cho 2–3 cặp quan trọng, ví dụ policy engine × tách câu.

### 4.2 Bảng ablation mẫu (sinh tự động từ tracking)

| ID | Task | Họ | Cấu hình (diff so với ref) | n | Metric chính [CI95] | Δ so với ref [CI95] | p (Holm) | Metric phụ | Seed | Chi phí ($ / GPU-h) | Độ trễ (s/bài) | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1-07 | T1 | Hậu xử lý | round → OptimizedRounder | | QWK | | | MAE, EM/chiều (kèm baseline majority) | 3 | | | |
| T2-02 | T2 | Ngưỡng | 0.5 → theo nhãn + shrinkage | | macro-F1 | | | micro-F1, F1/nhãn | 5×5 | | | |
| T3-05 | T3 | Guard | không guard → regex + sinh lại N=3 + fallback | | vi phạm % ; judge | | McNemar | tỷ lệ fallback | 3 | | | |

**Các hàng lõi** (chọn từ các bản đầu vào, không làm hết):
- **T1:** baseline rule/mean theo đề; cộng dần tín hiệu (log → test → sandbox); argmax vs EV decoding; round vs rescale vs OptimizedRounder; Σ chiều vs total trực tiếp vs stack; policy engine tắt / gating cứng / gating mềm (chỉ multi); head CORN vs SFT JSON; bỏ cụm near-dup có nhãn mâu thuẫn; chèn injection vào comment.
- **T2:** cộng dần tín hiệu; 5 quy tắc ngưỡng; BCE vs pos_weight vs ASL; rule tắt / làm feature / override cứng; augmentation có/không (kèm AUC của domain classifier); head rubric đa nhiệm có/không; random split vs group split (đo độ lạc quan của random split).
- **T3:** 1 bước vs 2 bước; information masking; các mức guard; N lần sinh lại; SFT vs SFT+DPO; lọc gold vi phạm có/không; cách điều khiển level; vẽ **đường Pareto** (compliance, chẩn đoán).

### 4.3 Phân tích lỗi theo "kiểu bài nào và vì sao"
**Quy trình 5 bước** (chỉ trên OOF của train và trên dev):
1. **Bảng lát cắt.** Theo type, môn, đề đã gặp/đề mới, nhóm điểm gold, decile token (có cắt context hay không), trạng thái biên dịch, có test_report hay không, số nhãn (0/1/2/3+), từng nhãn, level (T3), có comment tiếng Việt không dấu, có mẫu prompt injection. Mỗi lát gồm n, metric, CI và **phần đóng góp vào loss**.
   - T1: tử số của QWK phân rã được theo mẫu. Phần lát chiếm trong Σ(y−ŷ)² chính là phần lát đó làm mất QWK.
2. **Xếp hạng lát** theo (phần loss − phần n). Lát nào chiếm loss nhiều hơn tỷ lệ số mẫu của nó thì xem trước.
3. **Đọc mẫu.** Lấy 20–30 lỗi từ mỗi lát trong top 3–5. Hai người gán nguyên nhân độc lập theo codebook, báo κ giữa hai người. Codebook:
   - cắt context;
   - toolchain lệch;
   - trích policy sai;
   - tách câu sai;
   - test công khai yếu (pass nhưng sai);
   - lỗi IO bị đọc thành lỗi logic;
   - bias của LLM (độ dài, comment);
   - nhiễu nhãn gold (near-dup mâu thuẫn);
   - nhãn hiếm thiếu dữ liệu;
   - bịa dòng hoặc định danh (T3);
   - lộ cách sửa bằng lời (T3).
4. **Oracle-fix** để chứng minh "vì sao". Thay từng thành phần nghi ngờ bằng bản đúng (policy gold, tách câu tay trên train, tín hiệu biên dịch đúng, không cắt context) rồi đo Δ metric. Δ đó là trần lợi ích của việc sửa thành phần ấy.
5. **Thẻ lỗi** (mỗi thẻ khoảng ¼ trang trong báo cáo): Kiểu bài | n (% dữ liệu) | % loss | metric lát so với toàn bộ | nguyên nhân kèm bằng chứng (κ, oracle-fix Δ) | ví dụ đã che | sửa đã thử và Δ | phần còn lại không sửa được (nhiễu nhãn, đo bằng trần từ near-dup).

---

## 5. Báo cáo, model card, slide, Q&A

### 5.1 Báo cáo (tổng khoảng 14.5 trang; hỏi giảng viên phụ lục có tính trang không)
| Mục | Trang |
|---|---|
| 1. Bài toán, đóng góp (3–4 gạch đầu dòng) | 0.75 |
| 2. Dữ liệu và EDA: phân bố, nhãn suy biến, lệch phân bố, độ dài, đề trùng | 2.0 |
| 3. Vấn đề dữ liệu và metric phát hiện được (bonus): 2–4 phát hiện, mỗi phát hiện có claim, số liệu kèm CI, tác động lên metric, đề xuất sửa | 1.75 |
| 4. Phương pháp: pipeline chung (hình) 0.5; T1 1.0; T2 0.75; T3 1.0 | 3.25 |
| 5. Thiết lập thí nghiệm: split, metric, seed, thống kê, phần cứng, phiên bản | 0.75 |
| 6. Kết quả chính: bảng 3 task × các hướng, CI, public và CV | 1.0 |
| 7. Ablation: bảng gộp cộng đường Pareto T3 | 1.5 |
| 8. Phân tích lỗi: 4–6 thẻ lỗi | 1.5 |
| 9. Chi phí và triển khai: $ / 1000 bài, GPU-h, độ trễ, người kiểm tra trong vòng lặp | 0.75 |
| 10. Hạn chế, đạo đức, privacy | 0.5 |
| 11. Kết luận; khai báo công cụ LLM (tách công cụ hỗ trợ viết khỏi thành phần của hệ thống) | 0.5 |
| Phụ lục: bảng đóng góp, prompt, siêu tham số, ablation đầy đủ | (ngoài trang) |

### 5.2 Model card (1–2 trang, theo Mitchell et al. 2019)
- **Chi tiết model:** base model và revision hash, adapter và checksum, phần cứng, ngày.
- **Mục đích:** hỗ trợ giảng viên, có người duyệt. **Ngoài phạm vi:** chấm điểm cuối cùng không có người duyệt; đề mới khác môn.
- **Dữ liệu train:** nguồn, kích thước, phân bố, những gì đã lọc (L1/L2 vi phạm, near-dup).
- **Metric theo nhóm:** 2 môn × 2 loại đề × level; đề đã gặp vs đề mới.
- **Giới hạn:** code dài, đề mới, chất lượng tiếng Việt, test công khai yếu.
- **Rủi ro khi chấm thật:** bài điểm sát ngưỡng; thiên lệch theo phong cách code hoặc comment không dấu; prompt injection (có số đo); lộ lời giải; người dùng quá tin máy; privacy khi gửi code ra ngoài.
- **Khuyến nghị:** bài có entropy cao hoặc điểm sát ngưỡng thì chuyển cho người chấm.

### 5.3 Slide 15 phút (3–4 người trình bày, chuyển người theo kế hoạch)
| Phút | Nội dung |
|---|---|
| 0:00–1:00 | Bài toán, vì sao khó (tính sư phạm, 3 metric) |
| 1:00–2:30 | Dữ liệu: 3 con số EDA quan trọng nhất |
| 2:30–4:00 | Hai phát hiện bonus mạnh nhất kèm số liệu |
| 4:00–5:00 | Kiến trúc chung: tầng tất định, LLM, hậu xử lý theo metric |
| 5:00–8:00 | T1 (policy engine, de-shrink), T2 (ngưỡng theo nhãn, rule), T3 (diagnose rồi render, guardrail) |
| 8:00–10:30 | Kết quả cộng ablation chính, đường Pareto T3 |
| 10:30–12:15 | Phân tích lỗi: 2–3 kiểu bài kèm oracle-fix |
| 12:15–13:15 | Chi phí, triển khai, clip 20 giây chạy `make reproduce` |
| 13:15–14:30 | Hạn chế, rủi ro, điểm chính của model card |
| 14:30–15:00 | Kết luận, phân công |

Slide dự phòng: ablation đầy đủ, chi tiết từng phát hiện bonus, cấu hình phần cứng và phiên bản, prompt, bảng chi phí.

### 5.4 Chuẩn bị Q&A (15 câu, có hướng trả lời)
1. **Làm sao chắc là không rò feedback vào T1/T2?** Chỉ ra vị trí trong code nơi các trường bị loại, unit test kiểm prompt không chứa feedback, và ablation "không dùng feedback ở đâu cả".
2. **OptimizedRounder có phải là gian lận metric không?** Ngưỡng fit trên OOF theo kiểu nested. Báo cáo cả MAE ở hai điểm vận hành. Giải thích QWK tương đương Lin's CCC nên co về giữa thì bị phạt.
3. **Vì sao fine-tune 7B thắng (hoặc thua) prompting 32B?** Trả lời bằng bảng chi phí và hiệu năng cùng phân tích theo lát (đề đã gặp vs đề mới).
4. **Chạy lại trên GPU khác có ra đúng như vậy không?** Hai mức tái lập: inference khớp argmax; train lại nằm trong mean ± std của 3 seed. Cache, phiên bản đã pin, môi trường đã ghi.
5. **Viết tay cấu hình policy cho từng đề có phải là can thiệp không?** Policy được trích tự động. Phần audit tay chỉ làm trên đề của train để đo độ chính xác của extractor.
6. **Detector compliance của nhóm khác của ban tổ chức thì sao?** Thiết kế cho trường hợp xấu nhất (vượt cả kiểm tra bề mặt lẫn kiểm tra ngữ nghĩa). Đưa precision/recall của detector, đo bằng mẫu gán tay trên train.
7. **L2 không trích code thì chẩn đoán có kém đi không?** Đưa đường Pareto. Ranh giới được chọn dựa trên số liệu A5.
8. **Hệ thống chấm đề hoàn toàn mới thế nào?** CV group theo đề; khoảng cách giữa đề đã gặp và đề mới; với đề mới thì dùng lời giải do LLM sinh và kiểm bằng test công khai.
9. **Làm sao biết đó là nhiễu nhãn chứ không phải model sai?** Trần tính từ near-dup và κ giữa các cặp near-dup. Thẻ lỗi tách phần không sửa được.
10. **Cải thiện macro-F1 có ý nghĩa thống kê không, khi nhãn hiếm chỉ có vài mẫu?** Cluster bootstrap, approximate randomization, Holm. Thừa nhận cụ thể nhãn nào có CI rộng.
11. **Prompt injection trong comment thì sao?** Đưa ablation chèn comment, delimiter, và kết quả có/không bỏ comment.
12. **Có gửi code sinh viên ra API không?** Hệ thống final chạy local. Nếu có distill thì nêu sự đồng ý của giảng viên, gói dịch vụ, scrub PII, và log chi phí.
13. **Có công bằng với sinh viên viết code theo phong cách khác không?** Phân tích theo lát (độ dài, comment không dấu), kèm phần rủi ro trong model card.
14. **Chi phí cho 1000 bài là bao nhiêu? Có thay được giảng viên không?** Đưa $, GPU-h, độ trễ. Khẳng định chỉ là công cụ hỗ trợ, bài bất định cao chuyển cho người.
15. **Thành viên X đã làm cụ thể gì?** Ai cũng trả lời được về module của mình và về pipeline tổng (luyện chéo ở W8).

---

## 6. Sổ rủi ro

Ký hiệu: T = thấp, TB = trung bình, C = cao.

| Rủi ro | Xác suất | Tác động | Phòng ngừa | Phương án dự phòng |
|---|---|---|---|---|
| File nộp sai định dạng, mất quota | TB | TB | Validator nằm cứng trong `make submit`, cấm nộp tay | Giữ lại file baseline hợp lệ để nộp thay |
| Scorer khác bản mô phỏng (làm tròn, `zero_division`, NFC) | C | C | Hỏi giảng viên ở W1. Đối chiếu bằng 1 lượt baseline | Báo cáo song song các quy ước; chọn chiến lược bền với cả hai |
| Vi phạm lệnh cấm feedback một cách gián tiếp (exemplar, RAG, teacher, `level`) | TB | **Rất C** (huỷ kết quả) | Quy định D6, unit test grep prompt, review checklist ở mỗi PR | Bỏ thành phần nghi ngờ, chạy lại từ cache |
| Không tái lập được | TB | **Rất C** (mất 30% RANK) | Seed, pin phiên bản, cache, lockfile, chạy thử ở W6 và W7 | Nộp predictions kèm checksum. Giữ thêm một cấu hình fallback tất định (head, greedy) |
| Rò rỉ trong CV (near-dup, cùng đề) làm CV lạc quan | C | C | Group theo cụm near-dup/đề, adversarial validation | Hạ kỳ vọng, chọn bản nộp theo CV − 0.5·SE |
| Overfit public dev | TB | C | Quy tắc §3.2, nhật ký nộp | Chọn bản cuối chỉ theo CV |
| Test có đề mới nên kNN và peer-oracle mất tác dụng | TB | C | Kiểm H2 ở W1; CV group theo đề | Dùng lời giải do LLM sinh và tín hiệu không phụ thuộc đề |
| Bài multi_problem quá dài, bị cắt context | C | TB | Đo H1; tách câu; max_len theo type | Giữ đầu và cuối file kèm marker; gắn cờ `truncated` làm feature |
| Thiếu GPU hoặc hết quota Colab/Kaggle | C | C | Chốt nguồn GPU ở W1; QLoRA; dùng model 1.5–4B để debug | Thuê A100 theo giờ (ghi chi phí); bỏ DPO, giữ SFT |
| Stack model mới lỗi (Qwen3.5, transformers v5) | TB | TB | D2: backbone chính là model đã ổn định | Quay về Qwen2.5-Coder-7B / Qwen3-8B |
| Chi phí API vượt ngân sách | TB | TB | Pilot 50 mẫu, đặt trần ngân sách, cache, Batch API | Dừng distill, dùng teacher local |
| Giảng viên không cho gửi code lên API | TB | TB | D1: hệ thống không phụ thuộc API | Teacher open-weight chạy local |
| Model T3 lộ code hoặc lời giải ở L1/L2 | C | C | Masking, guard 2 tầng, fallback template | Ép L1/L2 dùng template cho nhóm bài có rủi ro cao |
| Code sinh viên độc hại khi chạy | T | C | Docker `--network none`, ulimit, không mount secret | Tạm dừng thực thi, chỉ dùng tín hiệu tĩnh |
| Rò dữ liệu (repo public, adapter public, log chứa code) | T | **Rất C** | Repo private, `.gitignore`, gitleaks, không log mẫu | Thu hồi ngay, báo giảng viên |
| Prompt injection trong code | TB | TB | Delimiter, chỉ dẫn đặt ở system prompt, ablation bỏ comment | Bỏ comment trước khi đưa vào LLM ở T1/T2 |
| Thành viên rời nhóm hoặc không đóng góp | TB | C | Người dự phòng cho mỗi vai, peer assessment ở W4 | Phân lại việc; cắt phạm vi theo D2 |
| Tích hợp dồn cuối kỳ (big-bang) | C | C | Ngày tích hợp thứ Sáu, hợp đồng `oof/` | Freeze sớm ở W7 |
| Báo cáo viết muộn | C | C | Khung báo cáo từ W3, hình sinh tự động, mỗi người sở hữu một mục | W7 dành riêng cho viết |
| Chạy private test không kịp | T | **Rất C** | Đo throughput ở W6; chạy thử với dev | Cấu hình nhanh (head, N nhỏ) đã kiểm sẵn |
| Nhóm khác công bố phát hiện bonus trước | TB | TB | Gửi giảng viên kèm timestamp ở W5; trên diễn đàn chỉ bàn ý tưởng | Đi sâu hơn: đưa thí nghiệm sửa và thí nghiệm đổi hạng |
| Mất cache API hoặc checkpoint | T | C | Backup cache và adapter lên storage private, có checksum | Train lại nếu cache còn đủ |

---

## 7. Quick wins trong 3 ngày đầu

**Ngày 1**
- Gửi giảng viên 8 câu hỏi (§8).
- Tải dữ liệu, tạo `MANIFEST.sha256`.
- Repo private: `.gitignore` chặn `data/`, `*.cpp`, `cache/`; pre-commit với gitleaks; `uv.lock`; `core/seeding.py`; `report_env.py`.
- Loader nối JSON với `code_file` (decode utf-8 → cp1258, đếm số file phải fallback), xuất tài liệu schema.
- Chạy validator chính thức trên file dummy cho cả 3 task, kể cả trường hợp T2 trả `[]`.

**Ngày 2**
- Notebook `audit/00_sanity`, mỗi kiểm tra chỉ vài dòng: A0 (biến thể chuỗi nhãn, NFC), A1 (total so với Σ chiều, điểm lẻ), A2 (crosstab compilable × log × CE), B1–B3 (prevalence, chiều suy biến, phân bố total), H1 (token p50/p95/p99 theo type), H2 (tỷ lệ đề dev/test trùng train), A5 (feedback L1/L2 có code).
- **Khoá `folds.csv` v0** theo kết quả H2.
- `eval/metrics.py` kèm unit test.

**Ngày 3: baseline nộp được (sàn điểm)**
- **T1.** compilable lấy từ log (rule). Các chiều lấy theo mode của (đề, compilable) trên train. Total = Σ các chiều. Áp `compilable=0 ⇒ logic=edge=0` nếu A2/C3 xác nhận.
- **T2.** CE lấy từ log. Các nhãn khác: TF-IDF char 3–5gram trên code và token của log, LR one-vs-rest `class_weight='balanced'`, ngưỡng theo nhãn tune trên OOF. Nhãn nào có F1 OOF thấp hơn mốc hằng số `2π/(1+π)` thì dùng hằng số.
- **T3.** L1/L2 dùng template theo (nhãn × level), luôn tuân thủ vì không chứa code. L3/L4 dùng LLM local zero-shot.
- `make submit` cho cả 3 task, ghi vào nhật ký nộp. Hệ quả: có điểm sàn, kiểm được scorer, có điểm đầu tiên cho đường CV↔LB.

---

## 8. Câu hỏi mở cần giảng viên trả lời trước cuối W1
1. Quota 2 lượt/ngày tính chung hay theo từng task? Một file chứa 3 task hay 3 file riêng? Bản nào được tính cho private?
2. Phạm vi lệnh cấm feedback: có gồm `level`, exemplar/RAG, rationale distill, target phụ không? Test của T2 và T3 có dùng chung bài không?
3. Chi tiết scorer: cách làm tròn, `labels` của QWK, scorer đọc `total` hay tự cộng 6 chiều, `zero_division` và biến thể macro-F1, chuẩn hoá Unicode.
4. Định nghĩa Level 1–4. Detector compliance dùng gì? Metric (b) đo bằng gì, có cố định seed không?
5. Có được gửi code sinh viên lên API không? API có được dùng lúc suy luận không?
6. Chính sách tiên quyết được mã hoá ở đâu? Nhãn gắn theo câu hay theo file?
7. Có được dùng input của test hoặc dev (không có nhãn) không? Dev có công bố nhãn không?
8. Mục 7 của đề (bị thiếu) có nội dung gì: mốc thời gian, checkpoint, hỗ trợ GPU? Thứ hạng quy ra 30% điểm theo cách nào?