# Red-team: phản biện tài liệu phân tích và kế hoạch

Ký hiệu: **[GT]** là giả thuyết về dữ liệu, luôn kèm cách kiểm. Tên trong ngoặc vuông là bản phân tích bị phản biện: spec, T1, T2, T3, data, infra, arch, strat.

---

## 1. Nhận định sai hoặc đáng ngờ

### 1.1 Metric

| # | Nhận định (nguồn) | Vì sao sai hoặc đáng ngờ | Sửa |
|---|---|---|---|
| M1 | Baseline "luật compile" là `np.where(df.compilable==0, 0, …)` [T1 §1.4] | Dùng nhãn gold `compilable` để dự đoán, tức là oracle và rò nhãn T1. Dùng con số này để kết luận "metric bị thổi phồng" là kết luận sai | Dự đoán từ `compile_log` (regex `error:`). Báo oracle riêng, ghi rõ đó là cận trên |
| M2 | Cross-fit K=3: dự đoán test lấy trung bình 3 adapter, còn stacker, ngưỡng QWK và ngưỡng nhãn đều fit trên OOF của từng adapter đơn lẻ [arch §4.3]. T2 §2.5 ghi "tune lại trên OOF của ensemble" | Lấy trung bình làm phương sai dự đoán trên test nhỏ hơn trên OOF. Đây chính là cái bẫy "co về giữa" mà mục N2 cảnh báo, và nó làm ngưỡng nhãn lệch. Cross-fit cũng không cho ra OOF của ensemble | Kiểm `np.std(test_avg)/np.std(oof)`. Cách xử lý: (a) chuẩn hoá moment của dự đoán test về OOF, một thao tác tự động nhưng transductive nên phải khai báo; (b) dùng ngưỡng theo quantile; (c) bỏ cross-fit, train 1 adapter và dùng holdout |
| M3 | "κ=ρ khi σ_ŷ=σ_y", "de-shrink" [T1 §1.2, arch N2] | Định lý chỉ đúng với dự đoán liên tục. Khi phải làm tròn lên lưới, clip vào [0,10] và phân bố đa đỉnh (0 và 8–10), giãn tuyến tính đẩy dự đoán ra biên rồi bị clip | Gọi đây là heuristic, không gọi là định lý. Chọn giữa rescale và OptimizedRounder bằng nested CV (đã có) |
| M4 | `E[F1] ≈ 2Σ_{i≤k}p_(i)/(k+Σp)` [T2 §2.3] | Đây là tỉ số hai kỳ vọng, không phải E[F1]. Sai lệch lớn nhất ở nhãn hiếm, đúng chỗ định dùng công thức | Tính E[F1] bằng Monte Carlo trên các p_i (rẻ, tất định khi có seed), hoặc bằng DP Poisson-binomial |
| M5 | Lấy EV từ logprob của vLLM khi dùng structured output [T1 (g), arch §4.4] | Mặc định V1 là `logprobs_mode=raw_logprobs`, tức logprob tính trước grammar mask và logits processor (đã kiểm). Một phần xác suất rơi vào token không hợp lệ, và top-k có thể thiếu chữ số → EV lệch. Tokenizer Qwen tách mỗi chữ số thành 1 token, nên phải xác định đúng vị trí token | Chấm theo kiểu forced-choice: tính logprob của từng giá trị 0..K bằng `prompt_logprobs` trên các continuation cố định, sau đó chuẩn hoá trên tập giá trị hợp lệ. Viết unit test cho đoạn này |
| M6 | Compliance chỉ phạt L1/L2 [T3 §1, strat D7] | "Level compliance" có thể đo theo kiểu "level của output = level yêu cầu", tức một classifier hai chiều. Khi đó template fallback viết kiểu L1 cho yêu cầu L2, L3 quá kín, hay L4 thiếu lời giải đều bị tính là vi phạm | Mỗi level có template riêng (L2 phải nêu vị trí). Tự train classifier level trên gold, kiểm template có được xếp đúng level yêu cầu. Hỏi giảng viên |
| M7 | Renderer L1/L2 được "bảo đảm theo nghĩa thông tin" là không lộ lời giải [T3 §3, arch N3] | Sai. Renderer vẫn thấy code, nhãn, `root_cause` và `failing_input`, và một code LLM tự suy ra được cách sửa. Masking chỉ giảm xác suất lộ | Không dùng chữ "bảo đảm" trong báo cáo. Đo tỷ lệ leakage bằng thực nghiệm |
| M8 | Diagnoser phải phủ mọi nhãn cho sẵn [T3 §3] | Nếu nhãn gold nhiễu thì quy tắc này ép model bịa lỗi, làm hỏng (b) | Cho phép issue có `confidence` thấp và diễn đạt mềm. Đo tỷ lệ nhãn "không xác nhận được" |
| M9 | Trần inter-rater ước bằng κ trên các cặp near-dup [T1 §2.7, data C1] | Hai vấn đề. (1) Các cặp near-dup là tập con có miền giá trị hẹp, chủ yếu là bài chép hoặc bài template, nên κ không phải trần của toàn tập. (2) **[GT]** Cặp gần trùng mà điểm lệch có thể do **xử phạt đạo văn** (cho 0 điểm), không phải nhiễu. Nếu đúng, claim bonus "nhiễu nhãn" sẽ sai | Dùng tỷ lệ bất đồng và MAE giữa các cặp, rồi mô phỏng nhiễu đó lên gold để ra QWK kỳ vọng tối đa. Kiểm đạo văn bằng `pair.query("total_a==0 & compilable_a==1 & pass_a>=0.9")`, và grep feedback (chỉ để EDA) với `sao chép\|giống bài` |
| M10 | Bootstrap theo cụm `problem_id` [strat D8] | Hướng đúng, nhưng nếu dev/test chỉ có dưới khoảng 20–30 đề thì CI bootstrap theo cụm hẹp hơn thực tế và không ổn định | Báo thêm phân bố metric theo từng đề. Nếu ít đề thì ghi rõ thành hạn chế |
| M11 | Bonus "F1 không thưởng đoán đúng tập rỗng", "QWK tính theo chỉ số lớp" [spec, data F1/F4] | Đây là tính chất đã biết của metric. Chỉ thành "vấn đề thật" khi chứng minh được trên **scorer thật** rằng nó làm đổi thứ hạng | Hạ ưu tiên. **Không tối ưu theo bug của scorer**: nếu giảng viên sửa scorer giữa kỳ, hệ thống tối ưu theo bug sẽ sụp. Chọn cấu hình bền dưới mọi quy ước |

### 1.2 Quy chế

| # | Nhận định | Vấn đề | Sửa |
|---|---|---|---|
| Q1 | "Nếu được: viết tay `policy.json` cho từng đề" [arch D12, T1 §3.1] | Với đề chỉ xuất hiện ở dev/test, việc này nghĩa là có người đọc input test rồi tạo cấu hình. Vừa là can thiệp thủ công, vừa vi phạm "vùng cấm" của chính strat §1 | Chỉ cho các đề có trong train, hoặc khi có văn bản đồng ý của giảng viên. Mặc định dùng extractor tự động có fallback. Không ai sửa tay generator hay oracle cho đề test |
| Q2 | Phân tích lỗi trên dev, tính QWK trên dev (B4), "λ tune trên dev", "chú thích tay 100 mẫu train/dev" [T3 §7, data §0, strat §1, §4.3] | Mâu thuẫn với T1 §5.2 ("nhãn dev bị ẩn"). Nếu dev không có nhãn thì không phân tích lỗi trên dev được. Gán tay dev là gán nhãn chính tập chấm public LB, vừa là vùng xám vừa overfit | Kiểm `dev.columns`. Mặc định mọi phép đo làm trên OOF của train và chỉ gán tay train. Hỏi giảng viên |
| Q3 | Chẩn đoán T3 dùng lại làm feature cho T1/T2, với diagnoser là "P-arm hoặc A3" [T3 §3, arch §4.6]. Kiến trúc B dùng 1 LoRA cho cả 3 task | A3 đã SFT với target là feedback. Output của nó vào T1/T2 nghĩa là thông tin feedback chui vào dự đoán T1/T2. Đây là xám đậm | Lập bảng "model nào đã thấy feedback". Đường T1/T2 chỉ dùng model chưa từng thấy feedback: P-arm gốc, A12, GBDT. Loại kiến trúc B |
| Q4 | Dùng từ khoá trong feedback để tìm nhãn gán thiếu [T2 §6.3, data A3] | Nếu sau đó sửa nhãn train T2 theo kết quả này thì feedback đã ảnh hưởng tới model T2 | Chỉ dùng cho hồ sơ bonus. Không sửa nhãn train |
| Q5 | Nộp B0-T3 "để dò metric" [arch §4.8] vs "không nộp hệ suy biến" [strat §3.2] | Hai bản mâu thuẫn nhau | Template đúng level là baseline hợp lệ, được nộp. Chuỗi rỗng hay hệ chỉ nhắc lại nhãn thì chỉ chạy cục bộ |
| Q6 | Chỉ coi API là rủi ro phát tán dữ liệu [spec §3, infra §7] | Thuê GPU (đặc biệt host cộng đồng kiểu Vast.ai), Kaggle private dataset, HF Hub private, W&B cloud đều đưa dữ liệu sinh viên (hoặc adapter có thể đã nhớ code) cho bên thứ ba | Xem §2 S1 |
| Q7 | Bỏ sót: AI coding assistant | Copilot, Cursor hay Claude Code chạy trong repo có thư mục `data/` sẽ đọc file `.cpp` rồi gửi cho vendor. Đề cho phép dùng LLM viết code, nhưng không cho phát tán dữ liệu | Để dữ liệu ngoài workspace, cấu hình ignore/deny cho thư mục dữ liệu, cấm dán mẫu vào chatbot |
| Q8 | "Inference từ adapter phải bit-exact" [infra §5.1] | Khác GPU, driver hoặc batch thì không bit-exact được. Với T3 greedy, lệch 1 token là lệch cả đoạn văn | Định nghĩa tái lập là metric trong ±ε. Ship predictions kèm checksum và cache các bước trung gian. Hỏi giảng viên tiêu chí |

### 1.3 Kỹ thuật, tái lập, ước lượng

| # | Nhận định | Vấn đề | Sửa |
|---|---|---|---|
| K1 | Slope wall-time, TLE "chạy 3 lần lấy đa số", `perf stat` trong Docker [T1 §2.5, T2 §4.5] | Tín hiệu phụ thuộc máy nên dự đoán không tái lập trên máy chấm. Seccomp mặc định của Docker và Colab/Kaggle thường chặn `perf_event_open` (cần kiểm chứng). Code có UB (biến chưa khởi tạo) cộng ASLR cho output ngẫu nhiên | **Cache evidence theo `code_sha` và coi cache là artifact tái lập.** Chạy với `setarch -R`. Ưu tiên TLE và thời gian trong `test_report` gốc. Bỏ slope thời gian |
| K2 | Sandbox dùng `ulimit -v 524288` + ASan, và `--tmpfs /tmp:…,noexec=off` [infra §5.4] | ASan cần reserve hàng TB bộ nhớ ảo cho shadow memory nên không khởi động được dưới RLIMIT_AS. `noexec=off` không phải option hợp lệ, phải dùng `exec` (cần kiểm cú pháp) | Giới hạn bộ nhớ bằng cgroup (`--memory`). Bản build ASan không đặt `-v` |
| K3 | 1 container cho mỗi lần chạy | **[GT]** khoảng 5k bài × 10–50 test, overhead khởi động container nhân lên thành hàng chục giờ | Mỗi bài 1 container chạy tất cả test. Benchmark ở W2 |
| K4 | `df.code.map(norm).map(hash)` [data C1] | `hash()` của str trong Python bị salt theo process, nên cụm near-dup và `folds.csv` không tái lập. Regex xoá comment cũng xoá cả `//` nằm trong string literal | Dùng `hashlib.sha1(s.encode()).hexdigest()`. Dùng tree-sitter để bỏ comment |
| K5 | `str.contains('error')` trên log [spec, T1, data] | Thiếu `na=False`. Khớp nhầm `-Werror` và "0 errors" | Dùng `r'^\S+:\d+:\d+: (fatal )?error:'` kèm `na=False` |
| K6 | "15–30 GPU-giờ" [arch A], "~3.5h/epoch" [infra §4] | Bỏ sót các hệ số nhân. Cross-fit K=3 nhân 3 seed (strat §4.1) là 9 lần train mỗi cấu hình. Cộng thêm A3 SFT, DPO, các dòng ablation M3/M4/M5 (mỗi dòng là 1 lần train) và P-arm OOF trên toàn train với SC k=5–10 (hàng chục triệu token output). MFU 0.35 cho QLoRA trên GPU consumer là lạc quan. Ước thực tế 150–300 GPU-giờ cỡ 4090 | Ablation fine-tune chạy trên 1 fold, 1 seed, model 1.5–4B. Chỉ cấu hình final chạy đủ. Chốt nguồn GPU và ngân sách giờ ngay W1 |
| K7 | Ví dụ chi phí distill $1/M input, $5/M output [infra §2.3] | Đây là giá tier rẻ. Teacher frontier đắt hơn 3–15 lần, chưa kể level-swap ×4, rejected cho DPO, R=3 | Pilot 50 mẫu với đúng model sẽ dùng, đặt trần ngân sách |
| K8 | P-arm là 32B AWQ chạy local [strat D1] vs P-arm là Qwen3-8B [arch] | 32B AWQ chiếm khoảng 18–20GB trên GPU 24GB, ít chỗ cho KV cache khi prompt multi_problem 8k+. So 32B prompting với 8B fine-tune thì lẫn hiệu ứng kích thước, không phải so sánh có kiểm soát | P-arm chính dùng cùng backbone 8B. 32B chỉ là tham chiếu phụ nếu có GPU 48GB trở lên |
| K9 | Ma trận chấm điểm 4 kiến trúc (A=4.59) [arch §3] | Tác giả tự chấm, và khả thi bị loại khỏi trọng số trong khi đó là rủi ro lớn nhất | Không đưa vào báo cáo như bằng chứng |
| K10 | Harness `-Dmain=student_main` gọi hàm theo chữ ký đề [T1 §2.3] | Sinh viên hay đặt sai chữ ký hoặc dùng menu `switch` trong `main`, nên compile harness sẽ fail hàng loạt. Công làm rất lớn | Đưa vào danh sách cắt đầu tiên (§4) |

### 1.4 Tên model, paper, thư viện

- **Đã kiểm bằng WebSearch, có thật:**
  - arXiv 2608.00515 có thật, nhưng là preprint một tác giả, xếp mục cs.CR. Không dùng làm luận cứ chính.
  - sklearn 1.9 có `replace_undefined_by` trong `cohen_kappa_score`.
  - Gemma 4 E4B ra 04/2026, giấy phép Apache-2.0.
  - Qwen3.6-27B ra 22/04/2026, Apache-2.0, nhưng là **model multimodal**. Bản Q4_K_M khoảng 16.8GB theo nguồn thứ cấp.
  - vLLM mặc định trả raw logprobs.
- **Không dùng:**
  - Qwen3.8-27B: chỉ thấy trên blog thứ cấp.
  - "vLLM PR #30315 multi-label head": chưa kiểm.
  - "vLLM ≥0.17 cho Qwen3.5": chưa kiểm.
  - `VLLM_BATCH_INVARIANT`: có thể chỉ hỗ trợ một số GPU/backend (cần kiểm trên 4090 sm89 và kiến trúc hybrid).
- **Mâu thuẫn giữa các bản:** T2 viết "guided_json deprecated", infra viết "gỡ ở v0.12". Cách xử lý: pin một phiên bản vLLM ngay W1 và viết smoke test cho structured output, logprobs, `bad_words` và multi-LoRA.
- **Trích dẫn có thể phóng đại:** Hellas et al. (ICER 2023) được dẫn là "LLM gần như luôn chèn code". Tóm tắt của paper chỉ nêu tỷ lệ tìm lỗi (90% / 57% / 40% false positive). Đọc lại để lấy đúng con số trước khi trích.
- **Có thể đã lỗi thời:** "Qwen2.5-Coder-7B mạnh nhất về code trong nhóm ≤10B" có lẽ không còn đúng ở 10/2026. Quyết định bằng pilot W2.

---

## 2. Rủi ro tuân thủ, xếp theo mức độ

### Mức S: huỷ kết quả hoặc kỷ luật

1. **Dữ liệu ra khỏi phạm vi được phép.** Các đường rò: API, GPU thuê, Kaggle, HF Hub, W&B, AI assistant, repo hoặc slide bị public.
   - Phòng: văn bản quy định "dữ liệu được đi đâu" gửi giảng viên duyệt trong W1. Pre-commit chặn file `.cpp`/`.json` dữ liệu, chạy gitleaks. Ignore thư mục dữ liệu cho mọi assistant. Báo cáo và slide chỉ trích đoạn code đã che.
2. **Feedback rò vào T1/T2, trực tiếp hoặc gián tiếp.** Các đường: `level`, exemplar RAG, prompt của teacher, model đã SFT trên feedback (Q3), sửa nhãn theo feedback (Q4), nhãn gold của input T3 khi test T2 và T3 dùng chung bài.
   - Phòng: loader theo whitelist. CI kiểm prompt không chứa feedback, nhưng phải trừ phần trùng với chính code sinh viên, vì feedback L2 hay trích nguyên dòng code. Lập đồ thị lineage "model và artifact nào đã chạm feedback".
3. **Chạm vào test.** Gồm: viết policy cho đề test, đọc test để chỉnh bất cứ thứ gì, chạy lại có chọn lọc những bài lỗi, sửa tay output, pseudo-label trên test.
   - Phòng: input test chỉ được giải nén bên trong `make predict-private`. Retry và fallback phải nằm sẵn trong code. Log lưu hash.
4. **Truy danh tính.** PII scan chỉ đếm, không in ra. Không dùng mã sinh viên làm feature.

### Mức A: mất khoảng 30% RANK

5. **Không tái lập được.** Nguồn: tín hiệu phụ thuộc thời gian (K1), API, phiên bản không pin, `hash()` (K4), máy chấm thiếu Docker hoặc GPU.
   - Phòng: lệnh `reproduce` mặc định chạy từ cache (evidence và output LLM). Thêm `reproduce FULL=1`. Docker image pin theo digest. Chạy thử trên máy sạch ở W6 và W7. Hỏi giảng viên họ chạy lại trên phần cứng nào.
6. **predictions private sai định dạng hoặc không kịp chạy.** Phòng: chạy thử trên dev đúng bằng lệnh đó, đo throughput, chuẩn bị sẵn cấu hình nhanh đã kiểm.

### Mức B: trừ điểm từng phần

7. **Yêu cầu bắt buộc về hướng tiếp cận:**
   - Thiếu một trong hai hướng (prompting, fine-tune) hoặc so sánh không công bằng (khác split, khác backbone).
   - Model fine-tune không chạy được cục bộ.
   - Thiếu báo cáo phần cứng. Nếu train trên cloud thì ghi rõ và chứng minh inference chạy local.
8. **Thiếu phiên bản hoặc seed** của teacher, judge, embedding, toolchain g++ và Docker image.
9. **Thiếu log chi phí API**, kể cả thí nghiệm hỏng và phần sinh dữ liệu.
10. **Báo cáo:** thiếu mục bắt buộc, vượt 15 trang, thiếu khai báo công cụ LLM.
11. **Bảng đóng góp** lệch với commit history và với khả năng trả lời Q&A.

---

## 3. Những gì còn thiếu

### 3.1 Góc nhìn chưa ai đề cập

- **Kích thước dữ liệu chưa biết, nhưng cả kế hoạch ngầm giả định khoảng 5k bài.** Cần một cổng ở W1:
  - [GT] Nếu `len(train) < ~1500` hoặc có nhãn với dưới 10 positive thì bỏ head 8B cross-fit cho T1/T2, dùng GBDT kết hợp P-arm, và dồn fine-tune vào T3.
  - Kiểm: `len(train)`, `Y.sum(0)`.
- **Có thể phải đáp ứng yêu cầu fine-tune bằng một task duy nhất.** Đề chỉ đòi "ít nhất một model fine-tune". Cần hỏi: phần so sánh ≥2 hướng có bắt buộc cho từng task không. Câu trả lời quyết định có cắt được A12 hay không.
- **Input private test có thể chỉ được phát ở cuối kỳ.** Khi đó B5 và H2 không chạy sớm trên test được. Protocol phải chịu được việc này, ví dụ báo cả CV group theo đề lẫn CV theo near-dup.
- **Dữ liệu hoặc scorer có thể bị sửa giữa kỳ** sau khi các nhóm báo lỗi. Cần manifest có version, pipeline chạy lại được, và bảng so sánh v1 với v2.
- **Đề bài có thể không có trong JSON** (chỉ có mã đề). Khi đó policy extractor và phần sinh test vô dụng. Cần fallback.
- **[GT] Encoding output của chương trình.** Chuỗi tiếng Việt in ra, nếu biên dịch trên Windows, có thể ra CP1258 thay vì UTF-8. Comparator sẽ báo `io_gap` giả.
  - Kiểm: `df.code.str.contains(r'cout\s*<<\s*"[^"]*[À-ỹ]').mean()`, rồi crosstab với `io_format`.
- **Fairness theo toolchain.** Bài dùng `conio.h` hay Dev-C++ có thể bị chấm thấp một cách hệ thống. Đưa thành một lát cắt trong phân tích lỗi và model card.
- **Nút thắt là giờ công người, không phải GPU.** Học viên cao học thường đi làm (xem §4).

### 3.2 Tín hiệu chưa khai thác

- **Thời gian và bộ nhớ trong `test_report`, so với các bài cùng đề.** Proxy cho `complexity`, tất định, không phải tự đo. [GT] Kiểm: parse `test_report` xem có trường time/mem không.
- **Phát hiện câu "không làm" trong multi_problem** (stub, `// TODO`, hàm `return 0;`) bằng AST. Rẻ hơn LLM và bắt đúng ca lệch xa.
- **Nhiều bài nộp của cùng sinh viên hoặc cùng đề.** Đếm bằng `code_file.str.extract(r'-([^-/]+)\.cpp$')[0].value_counts().describe()`.
  - Chỉ dùng để chia group CV.
  - Không dùng làm feature: rủi ro profiling.
- **Boilerplate chép từ template** (macro `#define`, `bits/stdc++.h`) làm feature cho `code_quality`.

### 3.3 Thí nghiệm then chốt chưa có

1. **Learning curve.** Train với 10/25/50/100% dữ liệu, so F-arm với P-arm, chạy trên model 1.5–4B cho rẻ. Kết quả này biện minh cho việc fine-tune và có giá trị báo cáo cao.
2. **Chuyển giao giữa 2 môn.** Train trên một môn, test trên môn kia. Bổ sung cho lát cắt đề đã gặp/đề mới.
3. **Ma trận bền vững theo quy ước scorer.** Chọn cấu hình final sao cho tốt dưới mọi quy ước (`labels` của QWK, cách làm tròn, `zero_division`).
4. **Chế độ chỉ dùng tín hiệu được cấp.** Không tự chạy code, chỉ dùng `compile_log` và `test_report`. Vừa là một dòng ablation, vừa là fallback tái lập.
5. **Kiểm detector compliance của nhóm so với kết quả chính thức.** Dùng các lượt nộp T3 làm điểm hiệu chỉnh.
6. **Kiểm tra giả thuyết đạo văn** cho các cặp near-dup lệch điểm (M9), làm trước khi nộp claim bonus về nhiễu nhãn.
7. **Phân tích chi phí lỗi bất đối xứng** cho mục triển khai: chấm thiếu điểm so với chấm thừa điểm, và ngưỡng chuyển bài cho người chấm.

### 3.4 Câu hỏi cần thêm cho giảng viên

1. Giảng viên có chạy lại pipeline không? Trên phần cứng nào, dung sai bao nhiêu? Nộp weights và cache ở đâu khi không được public?
2. Khi nào phát input private test, cửa sổ chạy dài bao lâu?
3. Dev có nhãn không? Được gán tay dev không?
4. Những nơi nào được xử lý dữ liệu: Colab, Kaggle, GPU thuê, HF private, W&B, AI coding assistant?
5. So sánh ≥2 hướng có bắt buộc cho **từng task** không?
6. Compliance chỉ phạt L1/L2, hay đo khớp level hai chiều? Metric (a) và (b) gộp thế nào để xếp hạng T3?
7. Nếu dữ liệu hoặc scorer được sửa giữa kỳ, leaderboard có chấm lại không? Nhiều nhóm cùng báo một lỗi thì bonus tính thế nào?
8. Bài gần trùng mà điểm khác có phải do xử phạt đạo văn hay quy định nộp muộn không?
9. Có code khung không? Đề bài có nằm trong dữ liệu không?

---

## 4. Khả thi trong 8 tuần

**Năng lực [GT]:** 5–7 người × 8–12 giờ/tuần × 8 tuần, khoảng **320–670 giờ công**.

**Khối lượng EMH đầy đủ, ước thô:**

| Hạng mục | Giờ công |
|---|---|
| Hạ tầng, validator, tái lập | 60 |
| EDA và bonus | 80 |
| Sandbox và evidence | 100 |
| Multi_problem (policy, tách câu, harness) | 100 |
| P-arm cho 3 task | 60 |
| A12 cross-fit | 60 |
| A3 SFT + DPO | 70 |
| Guard và judge T3 | 60 |
| Stacker và decoder | 40 |
| Báo cáo, model card, slide, diễn tập | 110 |
| Tích hợp và debug | 80 |
| **Tổng** | **khoảng 800–900** |

Kế hoạch hiện tại vượt năng lực 1.5–3 lần.

**MVP phải xong trước cuối W4** (đây là hệ thống nộp được, đủ mọi điều kiện):
- **Hạ tầng:** validator và `reproduce` chạy từ cache.
- **T1:**
  - Rule từ `compile_log`.
  - LightGBM trên các tín hiệu được cấp.
  - P-arm EV.
  - Policy engine cứng: regex/LLM extractor, tách câu heuristic có fallback.
  - Rescale hoặc OptimizedRounder.
- **T2:**
  - Rule cho nhãn Lỗi biên dịch.
  - LR hoặc GBDT.
  - P-arm hỏi yes/no và đọc logprob.
  - Ngưỡng theo nhãn có shrinkage.
- **T3:**
  - Diagnose rồi render bằng P-arm.
  - Regex guard.
  - Template fallback **theo từng level**.
  - SFT cho A3.
- **Fine-tune T1/T2:** A12 chạy 1 lần, ngưỡng fit trên holdout 80/20.
- **Bonus:** 2 phát hiện có số liệu.

**Thứ tự cắt nếu trễ** (cắt mục đầu trước):
1. N1 bản khả vi, LeakIG, slot-template, mô phỏng sinh viên (phần còn sót).
2. Harness `-Dmain` và compile riêng từng câu.
3. Peer-oracle và test sinh. Giữ chạy lại test công khai và `io_gap`; nếu trễ nặng thì chỉ dùng tín hiệu được cấp.
4. Augmentation bằng mutation hoặc LLM sinh bug cho T2.
5. DPO và level-swap. Giữ SFT cộng guard.
6. Teacher API, distill rationale.
7. Cross-fit K=3 × 3 seed, thay bằng 1 adapter cộng holdout. Phương sai theo seed chỉ báo trên model nhỏ.
8. Ablation backbone (M4) và ablation dạng output (M3). Giữ 1 cấu hình mỗi loại.
9. Stacker phức tạp, MAP 240 tổ hợp, joint T1–T2. Thay bằng trung bình có trọng số và hard rule.
10. Meta-eval judge đầy đủ, classifier compliance. Giữ regex, tree-sitter và 50 mẫu train gán tay.

**Tuyệt đối không cắt:**
- Validator và `make submit`.
- Tái lập: một lệnh, chạy thử trên máy sạch, seed, pin phiên bản, cache.
- Tường lửa feedback kèm test tự động. Quy định xử lý dữ liệu.
- Ít nhất 1 hướng prompting và 1 model ≤10B fine-tune chạy local, có báo phần cứng, so sánh trên cùng split.
- Policy engine cứng cho multi_problem.
- Guard L1/L2 cộng fallback đúng level.
- Decoder theo metric (QWK, ngưỡng theo nhãn).
- CV chia group đúng.
- EDA và ít nhất 2 phát hiện bonus.
- Ablation các flag lõi, phân tích lỗi theo lát cắt.
- Đủ các mục báo cáo bắt buộc, model card, bảng đóng góp, khai báo LLM, log chi phí.

---

## 5. Top 10 điều quyết định thành bại

1. **Tuân thủ cứng.** Dữ liệu chỉ ở nơi được phép, kể cả GPU thuê và AI assistant. Feedback không chạm vào T1/T2, kể cả qua model đã train trên feedback. Không chạm test. Một vi phạm xoá mọi thứ khác.
2. **Tái lập một lệnh trên máy sạch, chạy từ cache evidence và output LLM.** Đây là cổng của 30% RANK. Tín hiệu phụ thuộc thời gian và `hash()` là các bẫy cụ thể.
3. **Câu trả lời của giảng viên trong W1** về scorer (làm tròn, `labels`, `zero_division`), cơ chế phát hiện compliance, nhãn dev và quy định xử lý dữ liệu. Kèm 1 lượt nộp baseline để đối chiếu bản mô phỏng scorer.
4. **Validation phản ánh đúng test.** Group theo đề hoặc near-dup, chọn model bằng OOF, không tune theo public LB.
5. **Kiểm soát MVP và giờ công.** MVP xong trước W4, cắt theo thứ tự ở §4, freeze ở W7. Giờ công là ràng buộc thật.
6. **Guard compliance T3 đúng theo cả hai chiều level, có fallback theo level.** Rẻ nhất mà có thể tạo khoảng cách hạng lớn nhất.
7. **Decoding theo metric.** De-shrink hoặc tối ưu ngưỡng cho QWK. Ngưỡng theo nhãn và chiến lược nhãn hiếm phù hợp với `zero_division`. Kiểm hiện tượng co phương sai khi ensemble (M2).
8. **Bắt các ca lệch xa của T1** bằng tín hiệu tất định và policy engine cho multi_problem. QWK phạt theo bình phương nên các ca này chi phối điểm.
9. **2–3 phát hiện bonus có số liệu và thí nghiệm sửa**, gửi giảng viên sớm kèm timestamp. Loại trừ giả thuyết đạo văn trước khi claim nhiễu nhãn.
10. **Báo cáo, ablation và phân tích lỗi sinh tự động từ log**, mỗi người làm chủ một mục. Mọi thành viên trả lời được Q&A về toàn pipeline. Bảng đóng góp khớp commit.

Sources:
- [arXiv 2608.00515 – Auditable Release Control for Pedagogical Leakage in LLM Tutors](https://arxiv.org/abs/2608.00515)
- [scikit-learn 1.9.1 – cohen_kappa_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)
- [scikit-learn What's new 1.9](https://scikit-learn.org/stable/whats_new/v1.9.html)
- [Gemma 4: Expanding the Gemmaverse with Apache 2.0](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html)
- [The Decoder – Gemma 4 Apache 2.0](https://the-decoder.com/googles-gemma-4-is-now-available-with-apache-2-0-licensing-for-the-first-time/)
- [Qwen3.6-27B (NYU Shanghai RITS)](https://rits.shanghai.nyu.edu/ai/qwen3-6-27b-a-dense-27b-model-that-beats-a-397b-moe-on-coding)
- [Qwen3.8-27B (nguồn thứ cấp, chưa xác nhận)](https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/)
- [vLLM Engine Arguments (logprobs-mode)](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [vLLM V1 guide (raw logprobs mặc định)](https://docs.vllm.ai/en/v0.10.2/usage/v1_guide.html)
- [Hellas et al. ICER 2023 (Aalto portal)](https://research.aalto.fi/en/publications/exploring-the-responses-of-large-language-models-to-beginner-prog/)

Status: DONE
Summary: Phản biện có kèm kiểm chứng web: 11 lỗi về metric, 8 về quy chế, 10 về kỹ thuật, ước lượng và tái lập; rủi ro tuân thủ xếp theo mức; các phần còn thiếu; MVP và thứ tự cắt giảm; Top 10.
Concerns: Các ước lượng giờ công và GPU-giờ là [GT]. Hỗ trợ `VLLM_BATCH_INVARIANT` trên 4090 và cú pháp tmpfs của Docker cần kiểm chứng thêm.