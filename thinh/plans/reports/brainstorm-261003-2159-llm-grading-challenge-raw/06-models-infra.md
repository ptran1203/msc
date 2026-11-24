# Kế hoạch mô hình open-weight, fine-tune, hạ tầng và tái lập cho đồ án chấm điểm C++ bằng LLM

> Mọi nhận định về dữ liệu dưới đây là **GIẢ THUYẾT**, vì nhóm chưa tải dataset. Tên cột trong các đoạn pandas (`problem_id`, `problem_type`, `total`, `level`, `feedback`…) cũng là giả định, cần đổi theo schema JSON thật. Thông tin model đã được kiểm chứng bằng WebSearch đến 10/2026. Chỗ nào chưa chắc đều ghi "(cần kiểm chứng)".

---

## 0. Ba kiểm tra dữ liệu cần làm trước khi chọn kiến trúc (tuần 1)

Kết quả ba kiểm tra này quyết định trực tiếp max_len, RAG và cách xử lý nhãn.

| Giả thuyết | Ảnh hưởng thiết kế | Kiểm chứng |
|---|---|---|
| **H1** — file multi_problem dài gấp nhiều lần single_problem, p99 > 8K token | max_len, cắt hay tách câu | `df["n_tok"]=df.code_file.map(lambda p: len(tok(open(ROOT/p,encoding="utf-8",errors="replace").read()).input_ids)); df.groupby("problem_type").n_tok.describe(percentiles=[.5,.9,.95,.99])` |
| **H2** — mã đề ở test/dev trùng với train | Nếu trùng thì RAG/kNN theo cùng đề rất mạnh. Nếu không trùng thì phải tổng quát hoá sang đề mới | `len(set(dev.problem_id)&set(train.problem_id))/dev.problem_id.nunique()` |
| **H3** — có chiều/nhãn gần như hằng số. Ví dụ `compilable` ≈ 1 và suy ra được từ compile_log | Chiều đó dùng rule thay vì model. Rủi ro cho macro-F1 | `df[dims].apply(lambda s:s.value_counts(normalize=True).max())`; `Y.mean().sort_values()`; `pd.crosstab(df.compile_log.str.contains(r"error",case=False), df.compilable)` |

Một số kiểm tra phụ, rẻ nhưng hữu ích, kiêm luôn ứng viên cho điểm thưởng:
- Điểm có lẻ không: `df[dims].map(lambda x: float(x).is_integer()).mean()`.
- Tổng có bằng tổng các chiều không: `(df.total-df[dims].sum(1)).abs().gt(1e-6).mean()`.
- Bài trùng code nhưng nhãn lệch: chuẩn hoá code (bỏ comment và khoảng trắng) rồi lấy hash, sau đó `groupby(hash).total.std()`.
- Tỷ lệ phản hồi có code theo từng level: `df.groupby("level").feedback.str.contains(r"```|#include|cout\s*<<").mean()`. Kết quả này cho biết chuẩn "vi phạm" thực tế trong dữ liệu.

---

## 1. Bảng ứng viên backbone

VRAM ghi trong bảng là ước lượng, ở seq 2–4K, batch 1, có gradient checkpointing. Khi chạy thật phải đo bằng `torch.cuda.max_memory_allocated()`. Quy tắc thô cho inference: bf16 ≈ 2 byte/param + KV cache; 4-bit ≈ 0,6 byte/param + overhead.

### (a) Generator cho Task 3 (tiếng Việt, hiểu C++)

| Model (HF id) | Size / License / Ctx | Điểm mạnh | Rủi ro | VRAM: LoRA bf16 / QLoRA / Infer bf16 / 4-bit |
|---|---|---|---|---|
| `Qwen/Qwen3.5-9B` | 9B dense; Apache-2.0; 262K native | Ra 02/03/2026, hỗ trợ 201 ngôn ngữ, có thinking/non-thinking, code tốt | Kiến trúc hybrid (Gated DeltaNet + Gated Attention), có vision encoder. Cần transformers v5. Unsloth **không khuyến nghị QLoRA** vì sai số lượng tử cao. vLLM cần bản mới (≥0.17 theo Unsloth). Kernel trên T4 và batch-invariance (cần kiểm chứng) | 22GB / không khuyến nghị / ~20GB / ~7GB |
| `Qwen/Qwen3.5-4B` | 4B; Apache-2.0; 262K | Vừa với 1×24GB khi LoRA bf16. Hợp làm ablation về kích thước | Yếu hơn 9B về chẩn đoán. Rủi ro hệ sinh thái giống 9B | 10GB / – / ~9GB / ~3GB |
| `Qwen/Qwen3-8B` | 8.2B; Apache-2.0; 32K native (131K YaRN) | Fallback an toàn nhất: TRL, Unsloth, vLLM (cả LoRA), llama.cpp và `Qwen3ForSequenceClassification` đều đã hỗ trợ ổn định | Thế hệ cũ hơn 3.5 | ~20–24GB / ~9–12GB / ~17GB / ~5,5GB |
| `google/gemma-4-E4B-it` | "Effective 4B"; Apache-2.0; 128K | Ra 04/2026, đa ngôn ngữ tốt, fine-tune được trên 16GB bằng QLoRA | Tổng tham số thực lớn hơn "4B" (cần kiểm chứng). Năng lực C++ kém Qwen (cần kiểm chứng) | ~17–18GB / ~10GB / (cần kiểm chứng) |
| `mistralai/Ministral-3-8B-Instruct-2512` | 8B; Apache-2.0; 256K | Ra 12/2025, có thêm biến thể Reasoning | Tiếng Việt chưa được đánh giá rõ (cần kiểm chứng). Có vision encoder | ~20GB / ~9GB / ~17GB / ~5,5GB |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B; Llama 3.1 Community License; 128K | Hệ sinh thái lớn nhất | Tiếng Việt kém Qwen. License không phải OSI | ~20GB / ~9GB / ~16GB / ~5GB |
| `SeaLLMs/SeaLLMs-v3-7B-Chat`, `Viet-Mistral/Vistral-7B-Chat` | ~7B; license riêng (cần kiểm chứng) | Tối ưu cho tiếng Việt | Cũ (2024), code yếu. Chỉ nên dùng làm baseline tiếng Việt | ~19GB / ~8GB / ~15GB / ~5GB |

### (b) Scorer/classifier cho Task 1–2

| Model | Size / License / Ctx | Điểm mạnh | Rủi ro | VRAM |
|---|---|---|---|---|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7.6B; Apache-2.0; 32K (128K YaRN) | Mạnh nhất về code trong nhóm ≤10B dense có hỗ trợ ổn định. Gắn head được | Đã cũ (2024) | LoRA ~19–22GB / QLoRA ~8–10GB / infer ~16GB |
| `ByteDance-Seed/Seed-Coder-8B-Instruct` | 8B; MIT; 32K | Chuyên code, có bản Reasoning | Tiếng Việt yếu, nhưng đề bài chỉ là input nên ít ảnh hưởng | Tương đương Qwen3-8B |
| `Qwen/Qwen3-8B` hoặc `Qwen3.5-9B` + head | như trên | Đọc được cả đề tiếng Việt lẫn code | Qwen3.5 là lớp VLM nên hỗ trợ `SequenceClassification` chưa chắc (cần kiểm chứng) | như trên |
| `answerdotai/ModernBERT-large` | 395M; Apache-2.0; 8192 | Được huấn luyện có code. Rẻ, đủ để chạy 5-fold CV và lấy OOF. Kết quả gần như tất định | Không hiểu tiếng Việt sâu | Full fine-tune ~8–16GB ở 8K |
| `microsoft/unixcoder-base` | 125M; license (cần kiểm chứng); ctx ~1K (cần kiểm chứng) | Baseline encoder code kinh điển | Ctx ngắn, phải cắt | <8GB |

### (c) Judge nội bộ (không bị giới hạn ≤10B vì chỉ dùng để đánh giá và rerank)

| Model | Ghi chú |
|---|---|
| Bộ kiểm tra rule-based (regex/AST code detector) | Tầng 1, tất định, chi phí bằng 0. Bắt buộc phải có |
| `Qwen/Qwen3.5-9B` (chế độ thinking) | Chạy cục bộ được. Dùng chấm "độ chính xác chẩn đoán" |
| `Qwen/Qwen3.6-27B` | Apache-2.0, 262K, ra 22/04/2026. bf16 khoảng 55GB nên cần A100 80GB, hoặc bản 4-bit chừng 16–18GB (cần kiểm chứng). Bản `Qwen3.8-27B` (08/2026) mới thấy trên một nguồn thứ cấp (cần kiểm chứng) |
| LLM thương mại qua API | Dùng hiệu chỉnh judge trên train/dev. Phải cache và báo cáo chi phí |

### (d) Embedding code cho RAG

| Model | Size / License / Ctx | Ghi chú |
|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` (và 4B/8B) | Apache-2.0; 32K | Điểm MTEB-Code: 75,41 (0.6B), 80,06 (4B), 80,68 (8B). Đa ngôn ngữ nên embed được cả đề tiếng Việt |
| `jinaai/jina-code-embeddings-0.5b` / `1.5b` | CC-BY-NC-4.0 | Chuyên code. Dùng học thuật được, không dùng thương mại |
| `Salesforce/codet5p-110m-embedding` | BSD-3 (cần kiểm chứng); ~512 tok | Rẻ, làm baseline |
| BM25 trên token code đã chuẩn hoá (bỏ comment, đổi định danh) | – | Rất mạnh khi tìm bài **cùng đề** (H2). Nên dùng hybrid với dense |

**Shortlist đề xuất:**
- T1+T2: Qwen2.5-Coder-7B (hoặc Qwen3-8B) + LoRA + head. Baseline rẻ là ModernBERT-large.
- T3: Qwen3.5-9B non-thinking, SFT rồi DPO, nếu có ≥24GB bf16. Nếu không thì dùng Qwen3-8B QLoRA.
- Judge: rule-based + Qwen3.6-27B hoặc API.
- RAG: Qwen3-Embedding-0.6B + BM25.

---

## 2. Chiến lược fine-tune

### 2.1 So sánh ba kiến trúc

| Phương án | Ưu | Nhược | Dùng cho |
|---|---|---|---|
| **A. Một model đa nhiệm**, prompt riêng theo task | Một adapter duy nhất, chia sẻ được hiểu biết về code | Các task tranh gradient. Task 3 sinh dài lấn át T1/T2 | Ablation |
| **B. Mỗi task một LoRA adapter**, chung backbone | Tách biệt, tune riêng được. vLLM chạy multi-LoRA với `enable_lora=True` và chọn adapter theo request | Train nhiều lần | **T3**: adapter riêng, đi qua SFT rồi DPO |
| **C. Backbone + LoRA + head** | Một forward pass ra cả 6 điểm và 10 xác suất nhãn. Có xác suất đã hiệu chỉnh nên tune ngưỡng được. Tất định, nhanh | Không sinh được rationale. Phải tự viết module | **T1+T2 chung một model** |

Chi tiết phương án C:
- Pooling: dùng last token với decoder, mean pooling với encoder.
- Head: 6 head ordinal (CE theo từng chiều, hoặc CORN/CORAL) và 10 head sigmoid.
- Loss: `L = Σ_dim CE + λ·BCE(pos_weight = min(neg/pos, 10))`.
- Hậu xử lý:
  - (i) Điểm kỳ vọng `E[s]=Σ p_k·k` cho từng chiều.
  - (ii) Tổng = Σ các chiều, sau đó áp **policy layer tất định** (xem 2.4).
  - (iii) Tối ưu ngưỡng làm tròn cho QWK (kiểu OptimizedRounder) trên OOF/dev.
  - (iv) Tune ngưỡng riêng cho từng nhãn để tối đa macro-F1.

Phương án C vẫn tính là fine-tune open-weight. Ablation bắt buộc là C so với generative (prompt xuất JSON), cùng backbone.

Biến thể generative có xác suất: cho model xuất JSON 10 key boolean cố định (Task 2) hoặc chữ số điểm (Task 1). Đọc **logprob** của token `true`/`false` hoặc `0..4` tại từng key để lấy xác suất. Cách này giữ được lợi thế tune ngưỡng mà không cần head.

### 2.2 Task 3: SFT rồi DPO

**SFT.** Input gồm đề, code, nhãn lỗi cho sẵn và level. Target là phản hồi của giảng viên. Chỉ tính loss trên phần completion (`assistant_only_loss`/`completion_only_loss` trong `SFTConfig`). Level đưa vào cả system prompt lẫn một control token kiểu `<level_2>`.

**DPO.** Cặp preference sinh tự động, không cần người gán nhãn:
- Chosen: phản hồi gốc đúng level, hoặc mẫu do model SFT sinh ra có judge cao và qua được rule checker.
- Rejected, ba nguồn:
  - Phản hồi Level 4 (có code) của **cùng bài** khi yêu cầu là Level 1/2. Nếu cùng bài không có nhiều level thì dùng mẫu SFT sinh ở Level 4.
  - Mẫu SFT sinh ra mà rule checker bắt có code.
  - Phản hồi đúng level nhưng chẩn đoán sai, lấy từ bài khác có tập nhãn lệch.
- Cấu hình: `beta=0.1`, thêm `rpo_alpha` (thành phần SFT) để giữ độ trôi chảy. Biến thể thay thế là ORPO/SimPO, đưa vào ablation.

**Guard khi suy luận** (tự động hoàn toàn, nên hợp lệ):
- Với Level 1/2, chặn token "```" và `#include` qua `bad_words`/logit bias. Cần kiểm chứng tham số `bad_words` trên phiên bản vLLM đang pin.
- Best-of-N với N=4, seed cố định theo `hash(sample_id,k)`. Lọc bằng rule checker, sau đó rerank bằng judge.

### 2.3 Distill rationale từ LLM thương mại

**Cách làm:** theo "Distilling step-by-step" (Hsieh et al., 2023).
- Teacher nhận đề, code, compile_log, test_report, **điểm/nhãn gold** và giải thích ngắn gọn vì sao (gold-conditioned rationalization).
- Student học đa nhiệm: `[rationale] → [scores]`, kèm task phụ chỉ `[scores]`. Khi suy luận có thể bỏ rationale cho nhanh. Ablation: có và không có rationale.

**Lọc:**
- Cho teacher dự đoán điểm khi chưa thấy gold. Nếu lệch quá ngưỡng thì đánh dấu là mẫu khó, không bỏ.
- Loại rationale mâu thuẫn với gold.

**Ràng buộc:** không đưa trường `feedback` vào prompt sinh rationale cho T1/T2. Như vậy rationale không bị nghi là rò feedback. Có nên hỏi giảng viên thêm không, xem mục câu hỏi mở.

**Công thức chi phí:**
```
C = Σ_task N_t · R · [ T_in·(1−h)·p_in + T_in·h·p_cache + T_out·p_out ] / 1e6 · (1 − d_batch)
```
- N_t: số mẫu train của task. R: số rationale mỗi mẫu.
- T_in, T_out: số token vào/ra trung bình, đo bằng pilot 50 mẫu, đọc `usage` từ response.
- h: tỷ lệ prompt cache hit (system prompt và đề dùng chung).
- d_batch: chiết khấu Batch API. Mức phổ biến khoảng 50% (cần kiểm chứng theo nhà cung cấp).

**Ví dụ minh hoạ** (giá **giả định**, phải tra bảng giá hiện hành): N=5.000, 3 task, R=1, T_in=3.000, T_out=600, p_in=$1/M, p_out=$5/M, h=0.
- Mỗi lệnh gọi ≈ $0,006, tổng ≈ $90.
- Qua Batch API ≈ $45. Nếu self-consistency R=3 thì gấp ba.
- Ghi `usage` của từng call vào cache, `scripts/cost_report.py` cộng lại để đưa vào báo cáo.

### 2.4 Độ dài sequence với multi_problem

1. Đo phân vị độ dài (H1). Đặt `max_len` ở p95–p99 theo từng loại đề, ví dụ 4096 cho single và 8192 cho multi.
2. **Tách câu**: dùng `tree-sitter-cpp` lấy function/khối, cộng regex comment kiểu `Câu|Cau|Bai|Bài \d`. Map từng đoạn với câu trong đề.
3. **Chấm theo câu, gộp tất định:** LLM chỉ đánh giá từng câu. **Policy layer bằng Python** (`policy/multi_problem.py`) áp trọng số và luật tiên quyết (câu tiên quyết sai thì các câu sau bằng 0). Đây là phần đề bài coi là cốt lõi, và rule code bảo đảm tuân thủ 100%.
   - Giả thuyết: nhãn chỉ có ở mức file, không có theo câu. Kiểm tra JSON có điểm từng câu không. Nếu không, train ở mức file và chỉ dùng policy layer như ràng buộc hậu xử lý cùng feature.
4. Nếu vẫn dài: đặt đề trước, giữ head và tail của code, chèn marker `/* ... truncated ... */` ở giữa. **Không xoá comment**, vì có thể ảnh hưởng `code_quality`. Chỉ gộp khoảng trắng.
5. Train với padding-free/packing (Unsloth/TRL) để không tốn compute cho padding.

### 2.5 Mất cân bằng nhãn trong SFT

- **Oversample theo nhiệt độ:** trọng số mẫu = `max_{label ∈ y} (1/freq_label)^0.5`, cộng thêm cân bằng theo bin tổng điểm. Mỗi mẫu nhân bản tối đa 3–5 lần.
- **Augmentation giữ nguyên ngữ nghĩa** cho mẫu nhãn hiếm: clang-format theo style khác, đổi tên biến cục bộ. Chỉ augment **sau khi split** để không rò sang fold khác.
- Phương án C có `pos_weight`/focal loss. Tune ngưỡng theo nhãn trên OOF.
- **Nhãn suy biến** (H3): nếu suy ra được từ compile_log/test_report thì dùng rule. Riêng `zero_division` của metric chính thức (0 hay 1) quyết định có nên **không bao giờ đoán** một nhãn gần như không có positive hay không. Mỗi nhãn chiếm 10% macro-F1, nên phải nhân bản đúng metric chính thức trong `eval/metrics.py`.

---

## 3. Công cụ và cấu hình xuất phát

| Lớp | Lựa chọn | Lý do |
|---|---|---|
| Train | **TRL** (`SFTTrainer`, `DPOTrainer`) + **PEFT**, tăng tốc bằng **Unsloth** khi model được hỗ trợ | Chỉ một stack. Unsloth nhanh khoảng 1,5× và đỡ VRAM trên 1 GPU. Head tự viết bằng HF `Trainer` + PEFT |
| Thay thế | LLaMA-Factory (UI/YAML, có DPO) · Axolotl (YAML, multi-GPU) | Chỉ dùng khi cần multi-GPU hoặc người không viết code |
| Serving | **vLLM** (batch, multi-LoRA, structured output) · **llama.cpp/GGUF** để chứng minh chạy cục bộ trên laptop/24GB | |
| Structured output | vLLM `StructuredOutputsParams(json=schema)` với backend xgrammar. Các trường `guided_json`… đã bị gỡ từ v0.12. Đường HF transformers dùng Outlines | Giữ schema **phẳng** vì xgrammar từng bỏ qua `allOf` nhiều nhánh. Không dựa vào `uniqueItems` mà khử trùng ở hậu xử lý. Với scoring, tắt thinking: `chat_template_kwargs={"enable_thinking": False}` |
| Validate cuối | Pydantic model cho từng task output, rồi chạy script kiểm tra định dạng chính thức | Xem mục 6 |

Schema gợi ý:
- T1: object 6 key, mỗi key là `enum` các giá trị hợp lệ (như `logic ∈ {0..4}`, đổi theo kết quả kiểm tra điểm lẻ). **Tổng không để model sinh** mà tính ở hậu xử lý.
- T2: object 10 key boolean cố định. Tên nhãn chuẩn hoá Unicode NFC.

### Cấu hình xuất phát

| | Generative SFT (T3 / baseline T1-2) | DPO (T3) | LLM + head (T1+T2) | ModernBERT-large |
|---|---|---|---|---|
| LoRA | r=16, α=16–32, dropout 0, target mọi linear (q,k,v,o,gate,up,down) | kế thừa adapter SFT | r=16–32, α=2r | full fine-tune |
| LR | 1e-4 → 2e-4, cosine, warmup 3–5% | 5e-6 → 2e-5 (LoRA), β=0.1 | LoRA 1e-4, head 1e-3 | 2e-5 → 5e-5 |
| Epochs | 2–3, eval mỗi 0,5 epoch, early stop | 1–2 | 3–5 | 5–10 |
| Batch hiệu dụng | 16 (1×16 grad accum) | 16 | 16 | 16–32 |
| max_len | 4096 / 8192 (multi) | như SFT | 4096 / 8192 | 8192 |
| Khác | bf16, gradient checkpointing "unsloth", completion-only loss | `rpo_alpha` ≈ 0.5–1 (cần tune) | λ BCE = 1, `pos_weight` ≤ 10 | FA2 |

CV: model nhỏ (ModernBERT) chạy 5-fold `StratifiedGroupKFold`, group theo `problem_id`, và lưu OOF. LLM 7–9B chỉ dùng holdout cố định cho đỡ tốn. Gộp các model bằng stacking logistic regression trên OOF/dev.

---

## 4. Phần cứng

**Ước lượng thời gian train:**
```
T ≈ 6 · N_params · tokens_per_epoch · epochs / (MFU · peak_FLOPs)
```
Hệ số 6 là xấp xỉ cho LoRA có gradient checkpointing. Cần đo thật bằng khoảng 50 step thử.

Giả định: 5.000 mẫu × 3.000 token ≈ 15M token/epoch, model 8B.

| Phương án | Peak bf16 (dense) | MFU giả định | ~giờ/epoch, 8B | Gợi ý |
|---|---|---|---|---|
| 1× RTX 4090 24GB | ~165 TFLOPs | 0,35 | ~3,5h (QLoRA ~4,5h) | QLoRA 7–8B cho T1/T2/T3. LoRA bf16 với Qwen3.5-4B. Qwen3.5-9B bf16 (22GB) chỉ vừa ở seq ngắn |
| 1× RTX 3090 24GB | ~71 TFLOPs | 0,35 | ~8h | Như trên nhưng chậm hơn khoảng 2× |
| Colab Pro L4 24GB | ~121 TFLOPs | 0,30 | ~5,5h | Có bf16 (sm89). Hợp cho QLoRA 8B |
| Colab/Kaggle T4 16GB | ~65 TFLOPs fp16 | 0,25 | ~12h+ (không khả thi với 8B) | **Không có bf16, không có FA2.** Chỉ chạy model 4B/ModernBERT với seq 2K. Session Kaggle khoảng 12h, quota tuần khoảng 30h (cần kiểm chứng). Kernel Gated DeltaNet của Qwen3.5 trên sm75 (cần kiểm chứng) |
| Kaggle P100 | – | – | – | **Tránh dùng**: sm60, Unsloth/bitsandbytes/FA hỗ trợ kém |
| A100 80GB thuê theo giờ | 312 TFLOPs | 0,35 | ~1,8h | LoRA bf16 Qwen3.5-9B ở 8K, DPO, judge 27B. Chi phí = GPU-giờ × đơn giá hiện hành (tra RunPod/Lambda/Vast…) |

**Inference** (khoảng 1.000 mẫu dev/test, giả thuyết):
- Head T1+T2 trên 4090: dưới 15 phút.
- T3 qua vLLM, khoảng 300 token ra mỗi mẫu: vài phút trên A100, 10–20 phút trên 4090.
- GGUF Q4_K_M qua llama.cpp chạy được trên laptop để demo. Lượng tử hoá làm output khác đi, nên báo cáo cả số liệu bf16 (bản chính thức) lẫn 4-bit.

**Kế hoạch ngân sách gợi ý:**
- Dev và debug trên 4090/L4 hoặc Kaggle (ModernBERT, model 4B).
- Khoảng 20–40 GPU-giờ A100 cho các run chính (SFT, DPO, head 8B ×3 seed).
- API cho distill, tính theo công thức 2.3.

**"Chạy cục bộ" và báo cáo cấu hình:** `scripts/report_env.py` ghi ra `reports/env.json`, gồm `nvidia-smi`, driver, CUDA, `torch.version.cuda`, GPU, VRAM, CPU, RAM, OS và `pip freeze`/lock hash. Mỗi run lưu một bản kèm theo.

---

## 5. Tái lập

### 5.1 Seed ở mọi tầng
- `transformers.set_seed(SEED)`: gồm python `random`, numpy và torch, cả CUDA.
- Tất định hoá:
  - `torch.use_deterministic_algorithms(True, warn_only=True)`
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  - `cudnn.benchmark=False`
  - `DataLoader(generator=g, worker_init_fn=...)`
  - `SFTConfig(seed=, data_seed=)`
- **Split**: `splits/folds.csv` (chỉ gồm sample_id và fold) được commit. Không split ngẫu nhiên lúc chạy.
- **Sampling**: T1/T2 greedy hoặc head nên không có sampling. T3 dùng greedy, hoặc best-of-N với `SamplingParams(seed=hash(sample_id,k))`. `LLM(seed=SEED)`.
- Train lại sẽ **không bit-exact**, vì kernel Unsloth/FA và atomics không tất định. Nên định nghĩa hai mức:
  - **L1**: inference từ adapter đã phát hành phải bit-exact, hoặc trùng 100% argmax.
  - **L2**: train lại cho metric trong khoảng mean ± std của 3 seed. Báo cáo con số này.

### 5.2 Nguồn bất tất định và cách xử lý

| Nguồn | Cách xử lý |
|---|---|
| vLLM batching, prefill chunking | `VLLM_BATCH_INVARIANT=1` (yêu cầu NVIDIA sm ≥ 8.0; hỗ trợ cho kiến trúc hybrid Qwen3.5 cần kiểm chứng). Nếu không dùng được thì cố định `max_num_seqs`, thứ tự request và kích thước batch, hoặc chạy HF bs=1 cho run chính thức |
| Khác GPU hoặc khác phiên bản kernel | Ghi GPU dùng cho run chính thức. Phát hành kèm `predictions.json` và checksum. Ưu tiên head cho T1/T2 vì argmax bền hơn sinh văn bản |
| API thương mại | **Cache toàn bộ**: key = `sha256(model_snapshot + prompt_template_hash + messages + params)`, lưu vào SQLite/JSONL gồm response, `usage` và timestamp. Script tái lập chạy ở chế độ `--offline-cache`, cache miss thì báo lỗi. Ghi model ID có ngày/snapshot, không dùng alias "latest" |
| Model trên HF Hub bị cập nhật | `from_pretrained(..., revision="<commit sha>")` cho mọi backbone và embedding |
| Thư viện | `uv.lock` (hoặc pip-tools có hash). Docker image pin theo **digest** của base CUDA image |
| Dữ liệu | `data/MANIFEST.sha256` được commit, script kiểm hash trước khi chạy |

### 5.3 Một lệnh tái lập
- `make reproduce` chạy `scripts/reproduce_best.sh` theo thứ tự:
  1. Kiểm hash dữ liệu.
  2. Kiểm môi trường.
  3. Tải adapter đã pin (private).
  4. Infer với `configs/best.yaml`.
  5. Validate.
  6. So với `reports/expected_dev_metrics.json` theo dung sai.
- `make reproduce FULL=1` thì train lại từ đầu, chạy từ cache API.

### 5.4 Sandbox cho code sinh viên (code không tin cậy)
- Docker: `--network none --read-only --tmpfs /tmp:size=64m,noexec=off --memory 512m --memory-swap 512m --pids-limit 64 --cpus 1 --cap-drop ALL --security-opt no-new-privileges --user 65534`.
- Bên trong: `timeout -s KILL 5s`, `ulimit -t 5 -v 524288 -f 10240`.
- Thay thế: `isolate` (IOI), nsjail hoặc firejail.
- Cờ biên dịch lấy theo compile_log gốc (`-std=c++??`). Kiểm cờ này trong compile_log trước.
- Không bao giờ chạy code sinh viên trên máy có token API hay mount dữ liệu.

---

## 6. Cấu trúc repo, tracking và validator

```
llm-grading/                      # PRIVATE
├─ README.md  Makefile  pyproject.toml  uv.lock  .gitignore  .pre-commit-config.yaml
├─ docker/{train.Dockerfile, sandbox.Dockerfile}
├─ configs/                       # 1 YAML / thí nghiệm (model+revision, data, train, infer, seed)
├─ data/            (gitignored)  # raw/ interim/ processed/ ; chỉ commit MANIFEST.sha256
├─ splits/folds.csv               # sample_id → fold (không chứa code)
├─ src/grader/
│  ├─ core/        config.py (pydantic-settings) exceptions.py logging.py seeding.py
│  ├─ data/        loader.py (JSON ↔ code_file) splitter.py token_stats.py
│  ├─ features/    compile_log.py test_report.py cpp_parse.py (tree-sitter)
│  ├─ policy/      multi_problem.py   # trọng số + luật tiên quyết, tất định, có unit test
│  ├─ prompts/     *.jinja (versioned, hash vào cache key)
│  ├─ models/      heads.py
│  ├─ train/       sft.py dpo.py heads.py
│  ├─ infer/       vllm_runner.py api_client.py (cache) postprocess.py
│  ├─ eval/        metrics.py (nhân bản metric chính thức) level_checker.py judge.py
│  └─ shared/      hashing.py text_norm.py (NFC)
├─ sandbox/run_cpp.py
├─ scripts/        reproduce_best.sh validate_predictions.py submit.py report_env.py cost_report.py
├─ cache/api/      (gitignored; bàn giao riêng)   outputs/runs/<run_id>/ (gitignored)
├─ reports/        ablation.csv env.json expected_dev_metrics.json figures/
└─ tests/          # fixture tổng hợp tự viết, không dùng dữ liệu thật
```

**Experiment tracking:**
- Dùng MLflow local file store (dữ liệu không ra ngoài) hoặc W&B project private. **Không log mẫu code/feedback**, chỉ log metric.
- `run_id = date + git_sha + config_hash`.
- Mỗi run log: config, data hash, lock hash, GPU, seed, metric theo từng chiều và từng nhãn, token/chi phí API.
- `reports/ablation.csv` được sinh tự động từ tracking.

**Validator** (`make submit TASK=k`, chạy trước MỌI lần nộp):
1. Chạy script kiểm định dạng chính thức.
2. Kiểm tra chặt hơn:
   - JSON UTF-8, không BOM, `ensure_ascii=False`.
   - Mỗi phần tử có `sample_id` và output. `sample_id` **duy nhất** và **bằng đúng tập id** của split.
   - T1: từng chiều nằm trong khoảng và đúng bước giá trị. Tổng bằng tổng các chiều, nằm trong [0,10], và khớp policy layer.
   - T2: nhãn là tập con của taxonomy **sau chuẩn hoá NFC**. Lỗi NFC/NFD với "Lỗi biên dịch" là loại lỗi im lặng. Tập rỗng hợp lệ.
   - T3: chuỗi không rỗng. Level 1/2 phải qua được `level_checker`.
3. Tính lại metric trên dev bằng nhãn công khai nếu có.
4. Ghi `submissions/log.csv`: ngày, task, commit, điểm CV, điểm public. Dùng để quản quota 2 lượt/ngày và tránh overfit public LB.
5. Đặt lại thành pre-push hook.

---

## 7. Bảo mật dữ liệu (khuyến nghị)

**Repo và lưu trữ:**
- Repo **private**, chỉ thành viên nhóm có quyền.
- `.gitignore` chặn `data/`, `*.cpp`, `cache/`, `outputs/`.
- Pre-commit chạy `gitleaks` cộng hook chặn file lớn hơn 1MB hoặc file `.cpp`/`.json` trong `data/`.
- Adapter đã fine-tune có thể nhớ code sinh viên, nên đẩy lên HF Hub **private** hoặc Drive nội bộ, không public.
- Kaggle dataset để private. Drive không chia sẻ bằng link công khai. Xoá dữ liệu sau khi kết thúc môn.

**API thương mại:**
- Ưu tiên tài khoản hoặc gói có **zero data retention**, hoặc cam kết không dùng dữ liệu API để huấn luyện (cần kiểm chứng theo nhà cung cấp và gói). Không dùng gói miễn phí có điều khoản dùng dữ liệu để cải thiện model.
- Chỉ gửi những gì cần: đề, code, log. Không gửi metadata.
- Lọc PII còn sót trong comment (dãy số giống MSSV, họ tên) bằng regex rồi thay bằng `<MASK>`. Đây là **che**, **không** điều tra danh tính. Nếu phát hiện PII còn sót thì báo giảng viên, vì đó đồng thời là một vấn đề dữ liệu cho điểm thưởng.
- Log lại những gì đã gửi đi: số mẫu và hash, không log nội dung.
- Không dùng ID gợi ý từ dữ liệu làm tên file hay URL bên ngoài.

---

## Câu hỏi mở (cần hỏi giảng viên hoặc thống nhất trong nhóm)

1. Gửi code sinh viên qua API thương mại có trái với điều "không phát tán ra ngoài lớp" không? Có bắt buộc dùng ZDR không?
2. Train T1/T2 với target là rationale do LLM sinh (có hoặc không tham chiếu feedback) có vi phạm luật cấm feedback không? Retrieve bài train lân cận kèm feedback cho T1/T2 cũng là vùng xám, mặc định là không dùng.
3. QWK tính trên tổng làm tròn thế nào nếu điểm lẻ? Macro-F1 dùng `zero_division` bằng 0 hay 1?
4. Quota "2 lượt/ngày/nhóm" tính chung cho cả ba task hay riêng từng task?
5. "Chạy cục bộ" có tính VM thuê theo giờ không, hay bắt buộc máy của nhóm?
6. Tập test có đề (mã đề) mới không? Câu này quyết định độ mạnh của RAG theo cùng đề.
7. Level compliance được ban tổ chức đo bằng gì: regex, classifier hay LLM judge? Level 3 có được phép đưa pseudo-code không?

Sources:
- [Qwen/Qwen3.5-9B · Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3.5 Fine-tuning Guide (Unsloth)](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Qwen3.5 small models — Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- [Qwen 3.5 to 3.8: Alibaba Open-Weights Guide (Codersera)](https://codersera.com/blog/qwen-3-5-complete-guide-2026/)
- [Qwen3.6-27B blog](https://qwen.ai/blog?id=qwen3.6-27b)
- [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 Fine-tuning Guide (Unsloth)](https://unsloth.ai/docs/models/gemma-4/train)
- [mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
- [ByteDance-Seed/Seed-Coder-8B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct)
- [Qwen3-Embedding (GitHub)](https://github.com/QwenLM/Qwen3-Embedding)
- [jina-code-embeddings-0.5b](https://jina.ai/models/jina-code-embeddings-0.5b/)
- [SeaLLMs 3 paper](https://arxiv.org/html/2407.19672v1)
- [Viet-Mistral/Vistral-7B-Chat](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat)
- [vLLM Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)
- [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [vLLM issue #56556 (xgrammar allOf)](https://github.com/vllm-project/vllm/issues/56556)