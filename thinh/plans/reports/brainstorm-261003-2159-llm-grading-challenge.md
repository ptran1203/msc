# Phân tích yêu cầu và brainstorm: đồ án LLM Challenge (chấm điểm, phân loại lỗi, phản hồi bài C++)

- **Ngày:** 2026-10-03.
- **Nguồn:** file đề PDF (4 trang).
- **Chưa có dữ liệu.** Mọi nhận định về dữ liệu đều đánh dấu **[GT]** (giả thuyết) và đi kèm cách kiểm.
- **Cách làm:**
  - 6 agent phân tích song song: đặc tả/quy chế, T1, T2, T3, dữ liệu và metric, model và hạ tầng.
  - 2 agent thiết kế: kiến trúc, chiến lược.
  - 1 agent red-team.
  - Bản này là tổng hợp, **đã sửa theo phản biện của red-team**.
- **Bản gốc chi tiết** (có code pandas cho từng phép kiểm) nằm ở [brainstorm-261003-2159-llm-grading-challenge-raw/](brainstorm-261003-2159-llm-grading-challenge-raw/).

---

## 0. Tóm tắt

1. **Đây là ba bài toán tối ưu metric khác nhau**, không phải một bài LLM: QWK, macro-F1, và compliance cộng chẩn đoán. Lớp **giải mã theo metric** (rescale, ngưỡng, guard) quan trọng ngang model mà lại rẻ hơn nhiều.
2. **Phần tất định làm trước, LLM làm sau.**
   - `compile_log` cho ra `compilable` và nhãn "Lỗi biên dịch".
   - `test_report` cho ra `io_format` và `logic`.
   - **Chính sách tiên quyết của multi_problem viết thành code Python** (policy engine), không giao cho LLM tự nhớ.
3. **QWK về bản chất là Lin's CCC**, nên tuyệt đối **không co dự đoán về giữa**. QWK phạt theo bình phương độ lệch, nên điểm nằm ở các ca cực: bài không biên dịch, bài rỗng, câu tiên quyết sai.
4. **Macro-F1 tách được theo nhãn**, nên đặt **ngưỡng riêng cho từng nhãn**.
   - Một mẫu của nhãn hiếm có giá trị khoảng 60 lần một mẫu của nhãn phổ biến.
   - Chiến lược cho nhãn cực hiếm phụ thuộc tham số `zero_division` của scorer, phải hỏi giảng viên.
5. **T3 làm theo luồng Diagnose → Render.**
   - Chẩn đoán có cấu trúc trước.
   - Render theo level: L1/L2 **không được thấy cách sửa**.
   - Guard (regex + AST); vi phạm thì sinh lại, cuối cùng dùng **template đúng level** làm fallback.
   - **Compliance là ràng buộc cứng; chẩn đoán là mục tiêu tối ưu.**
6. **Hai hướng bắt buộc chạy trên cùng một backbone local**, ví dụ Qwen3-8B: một bên prompting, một bên LoRA. So sánh khi đó có kiểm soát, đồng thời gỡ được rủi ro tái lập và quyền riêng tư của API.
7. **Cơ cấu điểm:**
   - 60% (phương pháp 25, ablation 15, báo cáo 20) không phụ thuộc thứ hạng.
   - Tái lập (10%) là **cổng** của 30% thứ hạng, vì đề ghi "không tái lập được sẽ không được chấm".
   - Bonus +10% là khoản rẻ nhất, cần EDA sớm.
8. **Rủi ro lớn nhất là tuân thủ, không phải kỹ thuật:**
   - feedback lọt vào T1/T2, kể cả gián tiếp;
   - dữ liệu ra ngoài qua API, GPU thuê, **AI coding assistant**;
   - chạm vào tập test.
9. **Khối lượng vượt sức người.** Ước khoảng 800–900 giờ công, trong khi năng lực nhóm khoảng 320–670 giờ [GT]. Phải có **MVP trước cuối W4** và thứ tự cắt giảm định sẵn (§9).

---

## 1. Bài toán thực sự là gì

| Task | Input | Output | Metric chính | Bản chất | Cái khó ẩn |
|---|---|---|---|---|---|
| T1 Chấm rubric | đề, code (+compile_log, +test_report) | 6 chiều + tổng 0–10 | QWK trên tổng | Hồi quy thứ bậc đa đầu ra có ràng buộc: 2·2·5·3·2·2 = **240 tổ hợp**; tổng có thể khác Σ6 chiều ở multi_problem [GT] | Chính sách tiên quyết; QWK nhạy với phân bố biên; scorer làm tròn thế nào |
| T2 Lỗi đa nhãn | như T1 | tập con của 10 nhãn (rỗng hợp lệ) | macro-F1 | Về metric là **10 bài nhị phân độc lập** | Nhãn hiếm hoặc suy biến; `zero_division`; TN không được thưởng |
| T3 Phản hồi | đề, code, **nhãn gold**, **level** | đoạn văn tiếng Việt | (a) compliance, (b) chẩn đoán | Sinh văn bản có ràng buộc, đánh đổi Pareto giữa (a) và (b) | Chưa biết detector; gold có thể tự vi phạm; chưa biết (b) đo bằng gì |

**Hai loại đề:**
- **multi_problem:** nhiều câu, có trọng số, có câu tiên quyết, chỉ có `compile_log`.
- **single_problem:** stdin/stdout, có testcase công khai và `test_report`.

**Đề nhấn mạnh 3 điểm**, cũng là chỗ giám khảo chấm phần "phương pháp":
1. Phát hiện và xử lý vấn đề dữ liệu.
2. Chính sách tiên quyết, "không phải chi tiết phụ".
3. Tính sư phạm ở T3.

---

## 2. Checklist yêu cầu và các bẫy ngầm

### 2.1 Bắt buộc (MUST), rút gọn

- **Nộp bài**
  - Đủ cả 3 task. Bảng tổng là trung bình thứ hạng, nên thiếu một task là gần chót bảng tổng.
  - `predictions.json` gồm `{sample_id, output}` và qua được script kiểm tra. File sai vẫn bị tính lượt.
  - Tối đa 2 lượt/ngày.
- **Output**
  - T1: điểm đúng miền (0–1, 0–1, 0–4, 0–2, 0–1, 0–1).
  - T2: chuỗi nhãn khớp đúng taxonomy (chú ý Unicode NFC); `[]` hợp lệ.
  - T3: tiếng Việt, L1/L2 không có code hay lời giải.
- **Phương pháp**
  - Ít nhất 1 hướng prompting và ít nhất 1 model open-weight được fine-tune (khuyến nghị ≤10B), **chạy cục bộ**, có báo cáo phần cứng.
  - So sánh các hướng với nhau.
  - Báo cáo chi phí API nếu có dùng.
- **Quy trình**
  - Suy luận tự động 100%.
  - Cố định seed, ghi phiên bản model.
  - **Không dùng feedback làm input cho T1/T2.**
- **Liêm chính**
  - Không phát tán dữ liệu, repo private.
  - Không gán nhãn tay cho test. Không truy danh tính.
  - Không chia sẻ code hay dự đoán giữa các nhóm.
  - Khai báo công cụ LLM ở cuối báo cáo.
- **Sản phẩm nộp**
  - Repo có lệnh tái lập 1 lệnh.
  - Báo cáo 10–15 trang gồm: EDA, các hướng tiếp cận, ablation, phân tích lỗi "kiểu bài nào, vì sao", chi phí và triển khai, hạn chế.
  - Model card. Slide 15 phút + 10 phút Q&A.
  - Predictions cho private test. Bảng đóng góp.

### 2.2 Bẫy ngầm

**Phạm vi lệnh cấm feedback.** Đề chỉ ghi "input", nhưng nên hiểu rộng:

| Cách dùng | Đánh giá | Khuyến nghị |
|---|---|---|
| Feedback của chính bài đó đưa vào prompt hoặc feature T1/T2 | Cấm | Không dùng |
| Feedback của bài train trong exemplar few-shot hoặc index RAG cho T1/T2 | Theo câu chữ vẫn là input | Loại khỏi exemplar và index |
| `level`, độ dài feedback, embedding feedback | Dẫn xuất từ feedback | Cấm cho T1/T2 |
| Feedback làm target phụ khi fine-tune T1/T2 | Vùng xám | Hỏi giảng viên; giữ ablation "không dùng feedback ở đâu cả" |
| Teacher đọc feedback rồi sinh rationale cho T1/T2 | Xám đậm | Teacher chỉ thấy code và nhãn rubric/taxonomy |
| **Model T3 (đã SFT trên feedback) sinh output làm feature cho T1/T2** | Xám đậm: thông tin feedback chui vào T1/T2 | **Cấm.** Lập bảng "model/artifact nào đã chạm feedback" |
| Dùng feedback để EDA hoặc làm hồ sơ bonus | An toàn | Được, nhưng **không** sửa nhãn train theo kết quả đó |

**Rò chéo giữa T2 và T3.** Input của T3 chứa nhãn gold.
- Nếu test T2 và test T3 dùng chung bài thì nhãn T3 chính là đáp án T2.
- Kiểm: `len(set(t2_test.code_file) & set(t3_test.code_file))`.
- Nếu có giao nhau: **không khai thác**, báo giảng viên. Đây cũng là ứng viên bonus.

**Input test nằm trong tay nhóm.**
- Không ai đọc input test để chỉnh prompt, rule hay ngưỡng.
- Không chạy lại có chọn lọc những bài "trông sai". Không sửa tay output hỏng.
- Retry và fallback phải nằm sẵn trong code.

**"Không người can thiệp".** **Không viết tay `policy.json` cho đề chỉ xuất hiện ở dev/test.** Chính sách phải trích tự động; chỉ được audit tay trên đề của train để đo độ chính xác của bộ trích.

**API thương mại.**
- Không tất định kể cả khi temperature=0, nên **cache-replay** là nguồn sự thật.
- Anthropic API không trả logprobs, nên các kỹ thuật cần xác suất (EV, yes/no theo nhãn) chỉ làm đầy đủ được trên model local.

**Các điểm khác.**
- **Đề không có mục 7** (đánh số nhảy từ 6 sang 8). Có thể đã sót mốc thời gian hoặc hỗ trợ GPU, cần hỏi.
- **Hai môn dùng chung một rubric** nhưng có thể hiểu khác nhau. Ví dụ `complexity` có thể gần hằng ở Cơ sở lập trình [GT].
- **Mã SV ẩn danh trong tên file:** dùng làm `group` khi chia CV là hợp lệ; dùng làm feature "năng lực sinh viên" thì **không** (profiling).

---

## 3. Câu hỏi gửi giảng viên trong tuần 1 (xếp theo tác động)

1. **Scorer.**
   - T1: QWK làm tròn thế nào? Có truyền `labels=range(11)` không? Tính trên trường `total` hay tự cộng 6 chiều?
   - T2: macro-F1 dùng `zero_division` nào? Không gian nhãn cố định 10 nhãn không?
   - Có chuẩn hoá Unicode không? Có phát scorer không, hay chỉ phát script kiểm định dạng?
2. **Phạm vi cấm feedback.** Có bao gồm `level`, exemplar/RAG, target phụ, rationale distill không? Test T2 và T3 có dùng chung bài không?
3. **T3.**
   - Định nghĩa Level 1–4.
   - Detector compliance là regex, classifier hay LLM-judge? **Chỉ phạt L1/L2 lộ code, hay đo khớp level theo cả hai chiều** (ví dụ L4 thiếu lời giải cũng bị phạt)?
   - Metric (b) đo bằng gì? (a) và (b) gộp lại thế nào để xếp hạng?
4. **Dữ liệu được phép đi đâu:**
   - API thương mại (lúc train, lúc suy luận);
   - GPU thuê, Colab/Kaggle, HF Hub private, W&B cloud;
   - **AI coding assistant** (Copilot, Cursor, Claude Code) chạy trong repo có dữ liệu.
5. **Multi_problem.**
   - Chính sách tiên quyết nằm ở đâu? "Sai" nghĩa là gì?
   - Có điểm theo từng câu không? 6 chiều tính ở mức file hay gộp từ các câu?
   - Đề bài có nằm trong JSON không? Có code khung không?
6. **Dev.** Dev có nhãn không? Được dùng input dev/test (không nhãn) cho self-training không? Có được gán tay dev không?
7. **Quota.** 2 lượt/ngày tính chung hay theo task? Một file hay ba file? Bản nào được tính cho private? Khi nào phát input private test, cửa sổ chạy dài bao lâu?
8. **Tái lập.** Giảng viên có chạy lại không, trên phần cứng nào? Dung sai bao nhiêu? Nộp weights và cache ở đâu khi repo không được public?
9. **"Chạy cục bộ"** có tính VM thuê không? "≤10B" là tổng tham số hay tham số active?
10. **So sánh ≥2 hướng** có bắt buộc cho **từng task** không?
11. **Nhãn.** Có file taxonomy chính thức kèm định nghĩa 10 nhãn không?
12. **Dữ liệu ngoài, test sinh thêm.** Có được dùng dữ liệu hoặc model ngoài không? Có được tự biên dịch, chạy code sinh viên và sinh testcase lúc suy luận không?
13. **Báo cáo.** Mục 7 bị thiếu là gì? Thứ hạng quy ra 30% điểm thế nào? 10–15 trang có tính phụ lục không?
14. **Bonus.** Tiêu chí của "vấn đề thật" là gì? Nhiều nhóm cùng báo một lỗi thì tính thế nào? Nếu scorer hoặc dữ liệu được sửa giữa kỳ thì leaderboard có chấm lại không?

---

## 4. Metric: những điều quyết định chiến lược

### 4.1 QWK (T1)

**Hệ thức gốc.** Với trọng số bậc hai và kỳ vọng tính từ tích hai phân bố biên:

**κ = 2·cov(y, ŷ) / (σ²_y + σ²_ŷ + (μ_y − μ_ŷ)²)**, đúng bằng Lin's CCC.

Hệ quả:
- **Khi tương quan ρ cố định, κ đạt cực đại khi σ_ŷ = σ_y và μ_ŷ = μ_y.** Khi đó κ = ρ.
- **Hồi quy tối ưu MSE tự co về giữa** (σ_ŷ = ρσ_y), cho κ = 2ρ²/(1+ρ²).
  - ρ = 0.8 cho 0.78. ρ = 0.6 cho 0.53.
  - Vì vậy phải **giãn dự đoán ra (de-shrink)**: rescale theo μ và σ, hoặc OptimizedRounder fit trên OOF.
  - Lưu ý: đây là heuristic. Sau khi làm tròn, clip về [0, 10] và với phân bố đa đỉnh, hệ thức không còn đúng chính xác. Chọn giữa rescale và OptimizedRounder bằng nested CV.
- **Lệch trung bình bị phạt.** LLM zero-shot thường chấm rộng tay, nên **bắt buộc calibrate bias**.
- **Lệch xa bị phạt bình phương.** Lệch 5 điểm nặng gấp 25 lần lệch 1 điểm. Ưu tiên số 1 là bắt đúng các ca cực bằng tín hiệu tất định.
- **MAE (metric phụ) lại ưa co về median.** Tối ưu cho QWK và báo cáo MAE ở cả hai điểm vận hành.

**Bẫy của scorer:**
- sklearn `cohen_kappa_score` khi không truyền `labels` sẽ tính trọng số theo **chỉ số lớp** chứ không theo giá trị. Thiếu mức 3 điểm thì khoảng 2↔4 chỉ còn bằng 1 bước.
- Điểm lẻ (7.5) trở thành một lớp riêng. Nếu truyền `labels=range(11)` thì các mẫu 7.5 **bị bỏ âm thầm**.
- `round`/`np.round` dùng banker's rounding: `round(2.5) = 2`.
- **Cách phòng:** luôn xuất tổng nằm trên lưới hợp lệ. Tự viết bản sao scorer có unit test, rồi dùng 1 lượt nộp baseline để đối chiếu.

**Tách 2 nhánh output:**
- Tổng lấy từ E-value, sau đó rescale hoặc áp ngưỡng (nhánh tối ưu QWK).
- Sáu chiều dùng MAP (argmax) có ràng buộc trên 240 tổ hợp (nhánh tối ưu exact-match).
- Hai nhánh gặp nhau ở ràng buộc Σ = tổng, **nếu** dữ liệu cho thấy tổng luôn bằng Σ6 chiều (D1).

**[GT] Phân bố tổng đa đỉnh** (đỉnh ở 0 và ở 8–10), cộng phương sai lớn giữa các đề.
- Một baseline "trung bình theo đề" hoặc "luật biên dịch" có thể đã cho QWK đáng kể. Đây là ứng viên bonus kiểu "metric bị thổi phồng".
- **Chú ý:** baseline luật biên dịch phải dự đoán từ `compile_log`, không dùng nhãn gold `compilable` (đó là oracle).

### 4.2 Macro-F1 (T2)

- **F1_l = 2TP / (2TP + FP + FN).** Không có TN, nên đoán đúng một bài sạch (`[]`) không được cộng gì; tập rỗng chỉ có giá trị vì giúp giảm FP.
- **Mỗi nhãn chiếm 0.1 điểm macro-F1 bất kể support.** Ví dụ số học:
  - Nhãn hiếm 5 positive: đúng thêm 1 mẫu thì macro tăng +0.02.
  - Nhãn phổ biến 300 positive: đúng thêm 1 mẫu thì macro tăng +0.0003.
  - **→ Dồn công sức cho nhãn hiếm.**
- **Mốc hằng số:** đoán toàn 1 cho F1 = 2π/(1+π). Model phải thắng mốc này **trên từng nhãn**; nhãn nào thua thì fallback về hằng số.
- **Ngưỡng tối ưu cho F1 bằng F\*/2 khi xác suất đã calibrate** (Lipton et al. 2014). Ngưỡng vì vậy luôn ≤ 0.5. Dùng hệ thức này làm sanity check cho calibration.
- **Nhãn không có positive nào trong test.** Xác suất xảy ra là (1−π)^N; với π = 0.5% và N = 400 thì khoảng 13%. Khi đó:
  - `zero_division=0`: mọi đội đều 0 ở nhãn đó, trần macro-F1 còn 0.9.
  - `zero_division=1`: không đoán gì thì được trọn 0.1; **chỉ 1 FP là mất cả 0.1.**
  - `zero_division=nan`: nhãn bị loại, các nhãn còn lại nặng 1/9 mỗi nhãn.
  - → **Hỏi giảng viên**, và chọn cấu hình bền với mọi quy ước. Không tối ưu theo bug của scorer.
- **Bẫy cài đặt:** nếu scorer dựng không gian nhãn từ `y_true ∪ y_pred` thay vì cố định 10 nhãn, chỉ một lần đoán nhầm cũng *thêm* một cột F1 = 0 vào mẫu số.
- **Bẫy Unicode:** nhãn tiếng Việt lẫn NFC và NFD làm so khớp chuỗi thất bại. Map mọi output về đúng chuỗi trong vocabulary.
- **Phương sai:** F1 của nhãn hiếm trên dev nhảy ±0.2 chỉ vì 1 mẫu, nên **không tune theo public LB**.

### 4.3 Compliance và chẩn đoán (T3)

| Detector có thể dùng | Bắt được | Bỏ lọt |
|---|---|---|
| Regex (```` ``` ````, `;`, `{}`, `cout<<`, `#include`) | Code nguyên văn | Pseudo-code, lệnh sửa viết bằng lời |
| Classifier học từ feedback giảng viên | Phong cách "lời giải" | Dạng code chưa thấy khi train |
| LLM-judge | "Đổi `<=` thành `<`", liệt kê đủ các bước thuật toán | Nhiễu, phụ thuộc prompt |
| Độ tương đồng với lời giải | Đúng thay đổi cần sửa | Diễn đạt lại |

**Thiết kế cho trường hợp xấu nhất.** Ở L1/L2 phải tránh cả "code trá hình":
- pseudo-code;
- lệnh sửa bằng lời ("đổi … thành …");
- biểu thức của bản đã sửa;
- liệt kê đủ các bước thuật toán.

Phép thử thực dụng: **nếu sinh viên làm theo phản hồi mà không cần nghĩ vẫn ra code đúng, thì phản hồi đó là lộ lời giải.**

**Ranh giới đề xuất** (phải kiểm trên dữ liệu):
- L1: khái niệm và vùng lỗi (hàm hoặc câu), kèm câu hỏi gợi mở.
- L2: **lỗi là gì và nằm ở đâu** (dòng, định danh, hiện tượng, input gây sai), **không nói sửa thế nào**.
- L3: hướng sửa bằng lời hoặc pseudo-code một phần.
- L4: code đầy đủ, **đã chạy qua test**.

**Compliance có thể được đo theo cả hai chiều** (L4 thiếu lời giải, hay L3 quá kín, cũng bị phạt).
- Vì vậy **mỗi level cần template fallback riêng**. Không được lấy template kiểu L1 làm fallback cho mọi level.

**Metric này có thể bị game.**
- Phản hồi mơ hồ hoặc rỗng gần như chắc chắn compliant, nhưng chẩn đoán kém.
- Nhắc lại tên nhãn có thể thổi phồng (b) nếu (b) đo bằng nhãn.
- **→ Đo cục bộ, vẽ đường Pareto (compliance × chẩn đoán).** Đây là ứng viên bonus về metric.

---

## 5. Brainstorm theo task

### 5.1 Task 1: chấm rubric

**Tín hiệu theo chiều** [GT về mức tất định]:

| Chiều | Nguồn | Mức tất định |
|---|---|---|
| compilable | `compile_log` (regex `^\S+:\d+:\d+: (fatal )?error:`, có `na=False`), biên dịch lại với toolchain khớp log gốc | Gần tất định |
| io_format | `io_gap` = pass khi so normalized − pass khi so strict; feature tĩnh (in prompt "Nhap n", `freopen` sót, "Ket qua:") | Cao (single), trung bình (multi) |
| logic | pass rate trên test công khai, test sinh, test peer-oracle; đánh giá của LLM | Tương quan mạnh |
| edge_case | test biên (n=0, 1, max, âm, trùng), sanitizer (ASan/UBSan) | Trung bình |
| complexity | độ sâu vòng lặp (AST), đệ quy không memo, sort tự viết; thời gian/bộ nhớ trong `test_report` nếu có | Có thể suy biến |
| code_quality | lizard, cppcheck, clang-tidy, tỷ lệ tên biến 1 ký tự, biến toàn cục | Thấp, chủ quan. [GT] **Halo effect**: bài đúng thì hay được cho code_quality=1 |

**Pipeline cho multi_problem** (cũng là chỗ đề nhấn mạnh):
1. **Trích chính sách** (chạy 1 lần mỗi đề, có cache).
   - Regex trước, sau đó LLM structured output với self-consistency 3–5 lần.
   - Validate: Σ trọng số đúng; đồ thị tiên quyết không có chu trình.
   - Thất bại thì fallback về trọng số đều, gắn cờ `policy_failed`.
2. **Tách câu.** Thử lần lượt:
   - comment đánh dấu;
   - tên hàm `cau1`/`bai2`;
   - `switch/case` trong menu, cộng call graph (tree-sitter);
   - khớp chữ ký hàm lấy từ đề;
   - LLM, có validate bằng AST.

   Không cách nào thành công thì dùng cả file, gắn cờ `split_failed`.
3. **Chấm từng câu** bằng một trong {đúng, một phần, sai, không làm}, kèm E-value. Phát hiện câu "không làm" (stub, `// TODO`, chỉ `return 0;`) bằng AST thì rẻ và bắt đúng các ca lệch xa.
4. **`aggregate()` viết bằng Python, tất định, có unit test.**
   - Bản cứng: câu tiên quyết sai thì câu sau bị 0 hoặc bị chặn trần.
   - Bản mềm: s_q × Π P(pass_p).
   - Đưa **cả hai** làm feature. Ngưỡng τ fit trên OOF.
5. [GT] **Nhãn chỉ có ở mức file.** Khi đó điểm từng câu là biến ẩn. Ý tưởng novelty: **gating mềm khả vi** để học điểm từng câu chỉ với giám sát ở mức file. τ và T học được còn cho biết giảng viên có thật sự áp chính sách hay không (bằng chứng cho bonus).

**Các hướng mô hình hoá:**

| Hướng | Vai trò | Ghi chú |
|---|---|---|
| (a) **P-arm:** LLM local, rubric, evidence card, JSON enum, **EV từ xác suất** | Bắt buộc (prompting) | Lấy xác suất theo kiểu **forced-choice**: tính logprob cho từng giá trị 0..K bằng `prompt_logprobs` rồi chuẩn hoá. **Không** đọc logprob khi đang dùng structured output của vLLM, vì mặc định là raw logprobs tính trước grammar mask nên EV bị lệch |
| (b) kNN/RAG theo cùng đề | Feature + exemplar few-shot | Loại chính bài đó, cụm near-dup của nó và fold hiện tại; exemplar **không kèm feedback** |
| (d) **F-arm:** cùng backbone + LoRA + head thứ bậc (CORN cho logic/edge_case, BCE cho 4 chiều nhị phân) | Bắt buộc (fine-tune) | Head cho ra xác suất, nên chạy được decoder theo metric |
| (e) LightGBM trên evidence | Baseline mạnh, rẻ | Ràng buộc đơn điệu: điểm tăng theo pass_rate, giảm theo số lỗi |
| (h) Stacking | Kết hợp | Ridge hoặc LR cho multi (ít mẫu), LightGBM nhỏ cho single. Fit trên OOF |
| (c) SFT sinh JSON | Chỉ một dòng ablation | Thường co về giữa |
| (i) So cặp với bài anchor | Cắt | Không đáng công |

**Validation.** Kiểm trước, rồi mới chọn sơ đồ chia.
- Kiểm: `(~dev.pid.isin(train.pid)).mean()`, tức dev/test có đề mới không.
  - Nếu **có**: group theo đề.
  - Nếu **không**: group theo cụm near-dup.
- Stratify theo (type × khoảng điểm tổng).

### 5.2 Task 2: phân loại lỗi

- **Rule:**
  - "Lỗi biên dịch" lấy từ log. Chỉ override cứng nếu tam giác (`compilable`, log, nhãn CE) khớp ≥99% trên train.
  - TLE, RE, ASan OOB, `io_gap`, cảnh báo `-W*` dùng làm cờ.
- **Bốn nguồn xác suất:**
  1. Head sigmoid của F-arm (đa nhiệm cùng rubric T1; rubric là giám sát dày nên giúp regularize nhãn hiếm).
  2. P-arm hỏi **yes/no từng nhãn và đọc logprob**. 10 câu hỏi dùng chung prefix, nhờ prefix caching chi phí chỉ khoảng 1.x lượt. Kèm định nghĩa và phản ví dụ cho từng nhãn. Thêm câu "**có thể không có lỗi nào**" để chống bias "tìm lỗi bằng được".
  3. GBDT trên evidence.
  4. kNN vote.
- **Stacking theo nhãn** bằng LR L2 trên OOF, sau đó Platt hoặc temperature scaling.
- **Chính sách ngưỡng:**
  - Nhãn thường: argmax F1 trên OOF, lấy median qua bootstrap.
  - Nhãn hiếm: shrink về ngưỡng chung.
  - Nếu biết `zero_division = 1`: dùng expected-F1. **Tính bằng Monte Carlo trên p_i**, không dùng công thức xấp xỉ (công thức đó là tỉ số hai kỳ vọng, không phải E[F1]).
  - Thua mốc hằng số thì dùng hằng số.
- **Tăng dữ liệu cho nhãn hiếm:**
  - Mutation có kiểm soát: chèn off-by-one, sai điều kiện biên, sai chữ ký hàm vào bài đúng của train.
  - LLM sinh bug.
  - Dữ liệu sinh ra phải **nằm cùng fold với bài gốc**. Đo AUC của domain classifier để biết mức lệch phân phối.
- **Nhất quán với T1** (ví dụ có nhãn CE thì `compilable=0`; 10/10 thì tập lỗi rỗng): chỉ áp ràng buộc nào đúng ≥99% trên train, có ablation bật/tắt.
- **Encoder rẻ** (ModernBERT 8k ctx): chunk theo hàm rồi **max-pool logit theo nhãn**. Đây đúng là ngữ nghĩa multi-instance: một hàm có lỗi thì cả file mang nhãn.

### 5.3 Task 3: phản hồi có kiểm soát

```
evidence (compile/test/AST) + nhãn gold + level
 ├─ Diagnoser → JSON {label, câu, line_span, identifiers, symptom, root_cause,
 │                    failing_input, fix_sketch (chỉ nội bộ), confidence}
 │   kiểm cứng: dòng tồn tại, định danh ∈ AST, khẳng định khớp log
 │   (cho phép issue confidence thấp; KHÔNG ép bịa lỗi khi nhãn gold có vẻ nhiễu)
 ├─ Repair nội bộ (single: sửa → chạy test công khai → chỉ giữ bản pass) → diff
 ├─ Renderer theo level, che bớt thông tin:
 │     L1: khái niệm + vùng + câu hỏi gợi mở   │ L2: + dòng/định danh/hiện tượng/input sai
 │     L3: + hướng sửa bằng lời/pseudo một phần │ L4: + code đầy đủ đã pass test
 │     (L1/L2 không nhận fix_sketch/diff)
 ├─ Constrained decoding: L1 cấm cứng (```, `, ;, {}, #include, cout, ::); L2 chỉ phạt mềm
 ├─ Guard: regex + tree-sitter + n-gram trùng với dòng thêm trong diff + grounding check
 └─ Vi phạm → sinh lại (danh sách seed cố định, N≤3) → template fallback ĐÚNG LEVEL
```

**Lưu ý khi triển khai:**
- **Che bớt thông tin chỉ làm giảm xác suất lộ**, không bảo đảm được không lộ. Renderer vẫn thấy code, nhãn và root_cause, mà LLM giỏi code thì tự suy ra cách sửa. Đo leakage bằng thực nghiệm; không dùng chữ "bảo đảm" trong báo cáo.
- **Cấm token phải quét toàn bộ vocab.** Với BPE, `;` nằm trong nhiều token (`";\n"`, `"};"`…).
- **Tác dụng phụ của cấm token:** câu gãy, model "nói code bằng lời", chặn mất trích dẫn hợp lệ ở L2. Vì vậy L2 chỉ phạt mềm.
- **SFT:**
  - Input: đề, code đánh số dòng, log, nhãn, control token `<L1>…<L4>`. Target: feedback giảng viên.
  - **Lọc trước** các mẫu gold L1/L2 vi phạm, nếu không model học cách vi phạm.
  - Upsample để cân bằng level.
- **Level-swap:** cùng code và chẩn đoán, sinh feedback ở cả 4 level (teacher LLM). Model học được level là biến điều khiển duy nhất, và đây cũng là nguồn cặp DPO.
- **DPO**, 4 loại rejected, khớp độ dài giữa chosen và rejected:
  1. chèn leak;
  2. mơ hồ (chống việc né bằng sự mơ hồ);
  3. hallucination (sai dòng, sai định danh);
  4. lệch level.
- **Thứ tự ưu tiên:** guard (W3) → SFT (W4–5) → DPO (W6, tuỳ chọn). Guard mới là thứ giữ compliance; DPO chỉ cải thiện chất lượng.
- **Giọng sư phạm:**
  - L1 hỏi kiểu Socratic, ví dụ "Với n = 0, vòng lặp ở hàm tính tổng chạy mấy lần?".
  - Độ dài theo median ± IQR của từng level, đo trên train.
  - Xưng hô theo tần suất của giảng viên.
- **Judge nội bộ:** model local cố định, có meta-eval trên train (đo tương quan với feedback giảng viên). Dùng để chọn model mà không tốn lượt nộp.
- **Tài liệu tham khảo** (đọc lại trước khi trích):
  - Phung et al. (hint lập trình, mô phỏng sinh viên);
  - Al-Hossami et al. 2023 (Socratic debugging);
  - Maurya et al. NAACL 2025 (taxonomy đánh giá tutor: mistake identification / location / revealing answer / actionability);
  - Sonkar et al. 2024 (DPO cho alignment sư phạm);
  - Hsieh et al. 2023 (Distilling step-by-step).

---

## 6. Kiến trúc: 4 phương án và khuyến nghị

| Phương án | Ý chính | Mạnh | Yếu |
|---|---|---|---|
| **A. Evidence-first Hybrid** | Tín hiệu thực thi + head LLM + decoder theo metric | T1 bắt được ca cực; ablation sạch; tái lập tốt; ra luôn bằng chứng cho bonus | Harness multi_problem tốn công; nguy cơ bị xem là "LLM chỉ là phụ" |
| B. Một LLM đa nhiệm | 1 LoRA học cả 3 task | Rẻ, ít hạ tầng | Co về giữa; nhãn hiếm chìm; không có policy engine; hộp đen |
| C. Teacher → student | LLM lớn sinh rationale, level-swap, cặp DPO, bug tổng hợp | T3 mạnh nhất | Phụ thuộc quyền gửi code ra API; tái lập phụ thuộc cache |
| D. Retrieval-heavy | kNN/RAG theo cùng đề | Rẻ | **Sụp nếu test có đề mới**; near-dup làm CV lạc quan |

**Khuyến nghị: "Evidence-first, Metric-aware Hybrid" (EMH).** Lấy A làm xương sống, ghép thêm:
- **Từ C:** level-swap và cặp DPO cho T3; bug tổng hợp cho nhãn hiếm. Chỉ làm sau khi MVP xong.
- **Từ D:** kNN chỉ làm feature và nguồn exemplar.
- **Từ B (ở tầng hạ tầng):** **1 backbone, 2 adapter.**
  - `A12`: head cho T1 và T2.
  - `A3`: renderer cho T3, **cách ly hẳn khỏi T1/T2** vì A3 đã thấy feedback.

```
OFFLINE (chỉ train): S1 ingest → S2 problem_card (policy) → S3 evidence (sandbox) → S6 index
                      → A12 (heads T1+T2), A3 (T3: SFT→DPO) → stacker/ngưỡng/rounder fit trên OOF
ONLINE (`make predict`, không người can thiệp):
  sample → evidence ─┬─ rules ─────────────────┐
         problem_card┼─ P-arm (Qwen3-8B, EV) ──┼─ stacker → decoder theo metric → validator → predictions.json
         kNN feats ──┼─ F-arm A12 (heads) ─────┤        (T1: policy→rescale/rounder→MAP 6 chiều;
                     └─ GBDT trên evidence ────┘         T2: calib→τ_l→nhất quán)
  T3: diagnose → render A3 → constrained decode → guard → retry → template
```

**Evidence card:** một biểu diễn JSON dùng chung cho cả 3 task và cho cả P-arm lẫn F-arm. Gồm compile, kết quả test (public, gen, verdict, `io_gap`), sanitizer, static, phần tách câu, cụm near-dup, và một bản text gọn.

**Sửa theo phản biện:**
- **Cross-fit K=3 rồi lấy trung bình 3 adapter cho test.** Trung bình làm phương sai dự đoán trên test nhỏ hơn trên OOF, nên ngưỡng QWK và ngưỡng nhãn bị lệch. Cách xử lý:
  - kiểm `std(test_avg)/std(oof)`;
  - hoặc dùng ngưỡng theo quantile;
  - hoặc **1 adapter + holdout 80/20**, rẻ hơn 3 lần.
- **Đừng đưa ma trận chấm điểm tự đánh giá giữa các phương án vào báo cáo** như bằng chứng.

**Ý tưởng novelty để làm điểm nhấn báo cáo** (P1 = nên làm chắc):

| # | Ý tưởng | Ưu tiên |
|---|---|---|
| N2 | **Decoding theo metric có cơ sở lý thuyết** (QWK = CCC nên de-shrink; τ\* = F\*/2; expected-F1 theo `zero_division`). Rẻ, đưa phần "phương pháp" từ "thử thấy tốt" lên lập luận toán học | P1 |
| N3 | **T3 chống lộ đáp án**: che thông tin, ràng buộc logit, guard nhiều tầng, fallback; kèm đường Pareto | P1 |
| N1 | **Policy engine** (bản cứng), sau đó bản gating mềm khả vi | P1 / P2 |
| N7 | **Trần nhiễu nhãn** từ cặp near-dup, để báo "đạt x% trần" | P1 (sau khi loại giả thuyết đạo văn) |
| N10 | Grounding checker cho T3 (dòng, định danh, giá trị test phải có thật) | P1 |
| N8 | Định tuyến theo độ bất định: bài khó mới đi nhánh đắt (CoT/SC). Cho ra đường chi phí–chất lượng cho mục triển khai | P2 |
| N4 | Giải mã chung T1+T2 có ràng buộc nhất quán (luật khai phá từ dữ liệu, chỉ áp khi độ tin ≥99%) | P2 |
| N5 | Peer-oracle differential testing (bài train điểm tối đa làm oracle) | P2 (chỉ khi đề trùng) |
| N11 | Đo độ bền trước prompt injection trong comment, đưa vào model card | P2 |
| N6 | LeakIG: đo leakage bằng information gain dưới một code LLM | P3 |

---

## 7. Săn điểm thưởng (+10%)

**Top-8 nên kiểm trước** (sắp theo khả năng đúng × giá trị × chi phí):

| # | Giả thuyết | Cách kiểm nhanh | Vì sao đáng |
|---|---|---|---|
| 1 | **Tam giác compilable ↔ compile_log ↔ nhãn CE** không nhất quán, hoặc log không tái lập được (toolchain Windows/MinGW) | `crosstab([compilable, has_error], CE)`; biên dịch lại bằng `-fsyntax-only` | Ràng buộc tất định nên bằng chứng không cãi được. Nếu CE dư thừa thì nhãn CE là "0.1 miễn phí" |
| 2 | **Gold L1/L2 chứa code** | Regex + đọc tay 30–50 mẫu train để đo precision | Đánh trúng điểm khác biệt nhất của đề: metric phạt cả giảng viên |
| 3 | **Nhãn suy biến + `zero_division` làm đổi hạng** | Prevalence theo split; mô phỏng 3 quy ước; bootstrap P(A > B) | Phải đi xa hơn "nhãn lệch": chứng minh thứ hạng T2 phần lớn là nhiễu |
| 4 | **Near-dup có nhãn lệch**, trong train và giữa các split | MinHash 5-gram (Jaccard ≥0.9), cùng đề; tính Δtotal | Ra nhiễu nhãn, trần hiệu năng và rò rỉ giữa split. **Trước khi claim, loại giả thuyết bài trùng bị cho 0 vì đạo văn** |
| 5 | **Gold vi phạm chính sách tiên quyết** | Câu tiên quyết fail theo tín hiệu tất định mà điểm câu sau vẫn được tính | Nghịch lý "làm đúng chính sách thì bị metric phạt" rất thuyết phục |
| 6 | **QWK bị chi phối bởi trung bình theo đề; tăng được nhờ tối ưu ngưỡng** | Baseline mean theo đề (LOO); round vs OptimizedRounder (QWK tăng mà MAE cũng tăng) | Vài dòng code là chứng minh được lỗ hổng của metric chính |
| 7 | **Compliance đạt tối đa bằng output rỗng hoặc template** | Chạy 4 hệ suy biến **chỉ trên máy** (không nộp lên public khi chưa được phép) | Ít nhóm nghĩ tới; đề xuất sửa (metric gộp) rõ ràng |
| 8 | **Vệ sinh schema:** tổng ≠ Σ6 chiều, điểm lẻ hoặc vượt thang, nhãn lẫn NFC/NFD, `sample_id` trùng, validator từ chối `[]` | Vài dòng pandas | Gần như miễn phí |

**Ứng viên dự bị:**
- Rò rỉ qua ID hoặc thứ tự (báo cáo nhưng không khai thác).
- Testcase công khai yếu theo từng đề (AUC(pass_rate → logic=4) < 0.7).
- Quan hệ cha–con giữa nhãn gán không nhất quán.
- Halo effect ở code_quality.
- Level là hàm của điểm (Cramér's V > 0.5).
- Feedback rập khuôn (> 20% là template).
- Lệch giữa hai môn.
- Encoding CP1258 làm `io_gap` giả.
- Ẩn danh còn sót (**chỉ đếm, không đọc**).

**Mẫu trình bày một phát hiện** (mỗi phát hiện 0.5–1 trang, chọn 2–4 phát hiện mạnh):
1. Claim một câu.
2. Cách vận hoá, kèm precision/recall của detector.
3. **Bằng chứng định lượng:** n, %, CI 95% (bootstrap, có seed), script `make audit-*`.
4. **Tác động lên metric/leaderboard** (đổi mấy hạng).
5. Nguyên nhân quan sát được (không đoán ý đồ người gán).
6. **Đề xuất sửa**, cho dữ liệu và cho metric.
7. **Thí nghiệm chứng minh bản sửa có ích.**
8. Giới hạn của phân tích.

**Thao tác:** gửi giảng viên **kèm timestamp** (khoảng W5). Trên diễn đàn chỉ bàn ý tưởng. Phát hiện rò rỉ thì báo cáo, **không khai thác**.

---

## 8. Model, hạ tầng, tái lập

### 8.1 Backbone

Các agent đã kiểm bằng WebSearch đến 10/2026. **Tự kiểm lại trước khi chốt.**

| Vai | Mặc định | Biến thể / ghi chú |
|---|---|---|
| P-arm và F-arm (cùng backbone) | `Qwen/Qwen3-8B` (Apache-2.0, text-only; TRL, PEFT, Unsloth, vLLM-LoRA và SequenceClassification đều ổn định) | `Qwen2.5-Coder-7B-Instruct` cho T1/T2. **Pilot ở W2** trên 2 model × 200 mẫu |
| T3 generator | Qwen3-8B (QLoRA) | `Qwen3.5-9B` (ra 03/2026) là multimodal, kiến trúc hybrid; Unsloth không khuyến nghị QLoRA cho nó → chỉ dùng làm biến thể khi có ≥24GB bf16. `gemma-4-E4B` cho ablation kích thước |
| Encoder rẻ | `answerdotai/ModernBERT-large` (8k ctx) | Chạy 5-fold OOF rẻ, làm baseline cho F-arm |
| Embedding | `Qwen3-Embedding-0.6B` + BM25 trên token đã chuẩn hoá | |
| Judge / teacher local | Model 27–32B bản 4-bit (ví dụ `Qwen3.6-27B`) | **Không** so P-arm 32B với F-arm 8B: lẫn hiệu ứng kích thước. Model lớn chỉ làm tham chiếu phụ |

### 8.2 Phần cứng

**Ước lượng ban đầu** (giả định khoảng 5k mẫu × 3k token, model 8B):
- 4090: khoảng 3.5 giờ/epoch (QLoRA khoảng 4.5 giờ).
- L4: khoảng 5.5 giờ/epoch.
- T4: không khả thi với 8B (không có bf16 và FA2).
- P100: tránh dùng.
- A100 80GB: khoảng 1.8 giờ/epoch.

**Theo red-team, tổng thực tế có thể lên 150–300 GPU-giờ cỡ 4090**, vì:
- cross-fit × seed;
- SFT, DPO;
- các dòng ablation;
- P-arm self-consistency trên toàn train.

**Cách xử lý:**
- Ablation fine-tune chạy trên 1 fold, 1 seed, model 1.5–4B.
- Chỉ cấu hình final mới chạy đủ.
- **Chốt nguồn GPU và ngân sách giờ ngay W1.**

### 8.3 Tái lập (cổng của 30% thứ hạng)

- **Seed ở mọi tầng:** split, fold, chọn exemplar, tie-break RAG, sampling.
- **Pin phiên bản:**
  - `revision` của model và checksum adapter;
  - lockfile (`uv.lock`);
  - Docker image theo digest;
  - phiên bản vLLM. Viết smoke test cho structured output, logprobs, `bad_words`, multi-LoRA, vì API vLLM đổi giữa các bản.
- **Cache mọi thứ không tất định:**
  - output API (khoá theo hash của prompt và tham số);
  - **evidence theo `code_sha`**: kết quả compile và chạy phụ thuộc máy.

  Lệnh `reproduce` mặc định chạy từ cache; thêm `reproduce FULL=1` để chạy lại từ đầu.
- **Bẫy cụ thể:**
  - `hash()` của Python bị salt theo process, nên cụm near-dup và `folds.csv` không tái lập. Dùng `hashlib.sha1`.
  - Wall-time và slope thời gian không tái lập. Ưu tiên thời gian trong `test_report` gốc.
  - Code có UB (undefined behavior) cộng ASLR cho output ngẫu nhiên. Chạy với `setarch -R`.
  - ASan không khởi động được dưới `ulimit -v`. Giới hạn bộ nhớ bằng cgroup (`--memory`).
- **Định nghĩa "tái lập":**
  - Inference: khớp argmax, hoặc metric lệch trong ±ε.
  - Train lại: nằm trong mean ± std của các seed.
  - Hỏi giảng viên dung sai cụ thể.
- **Sandbox cho code sinh viên:**
  - Docker `--network none`, cgroup giới hạn CPU và bộ nhớ, timeout, không mount secret.
  - Mỗi bài 1 container chạy hết các test (không mở 1 container cho mỗi lần chạy).
- **Chạy thử trên máy sạch ở W6 và W7.**

### 8.4 Bảo mật dữ liệu

- **Dữ liệu có thể rò ra ngoài qua:**
  - API thương mại;
  - GPU thuê (nhất là host cộng đồng);
  - Kaggle dataset, HF Hub (kể cả adapter, vì adapter có thể đã nhớ code);
  - W&B cloud;
  - **AI coding assistant đọc file trong repo**;
  - slide, báo cáo.
- **Cách phòng:**
  - Viết một văn bản ngắn "dữ liệu được phép đi đâu", gửi giảng viên duyệt ở W1.
  - Để dữ liệu **ngoài workspace** mà AI assistant đọc được.
  - Pre-commit chặn `*.cpp` và JSON dữ liệu; chạy gitleaks.
  - Báo cáo chỉ trích đoạn code đã che.

---

## 9. Kế hoạch 8 tuần

| Tuần | Mục tiêu | Gate (không qua thì dừng và xử lý) |
|---|---|---|
| W1 | Repo private, loader, validator, `make submit`, bản sao scorer có unit test, audit sanity, `folds.csv` v0, **baseline cả 3 task đã nộp**, gửi câu hỏi giảng viên | Cả 3 task có điểm public; baseline chạy được trên máy một thành viên khác |
| W2 | Sandbox và evidence, chỉ mục near-dup, LightGBM, P-arm v1, pilot backbone, **khoá validation protocol** | Mỗi task có ít nhất 1 hướng thắng baseline (CI của Δ không chứa 0); có ≥3 phát hiện bonus kèm số liệu |
| W3 | F-arm v1 (A12), T3 SFT v1, policy engine có unit test, ngưỡng theo nhãn, decoder QWK, **guard T3 + fallback theo level**, khung báo cáo | Inference chạy 2 lần cho diff = 0; có QWK multi_problem khi bật và tắt policy |
| W4 | Stacking v1, judge T3, review giữa kỳ, **chốt danh sách thành phần; sau W4 không mở hướng mới** | **MVP hoàn chỉnh** |
| W5 | DPO (nếu kịp), augmentation nhãn hiếm, gửi hồ sơ bonus, nháp EDA và phương pháp | Ablation đã có ≥80% số hàng |
| W6 | Train final (nhiều seed), ablation 100%, thẻ phân tích lỗi, **reproduce trên máy sạch lần 1**, đo throughput | Reproduce khớp trong ε |
| W7 | **Feature freeze**, báo cáo đầy đủ, model card, reproduce lần 2, slide v1 | Mỗi mục báo cáo được 2 người không viết mục đó review |
| W8 | Chạy private test từ tag `v1.0` bằng 1 lệnh, lưu hash, diễn tập 2 lần, luyện Q&A, bảng đóng góp | — |

**MVP (xong trước W4):**
- **Hạ tầng:** validator; `reproduce` chạy từ cache.
- **T1:** rule từ log, LightGBM, P-arm EV, policy engine cứng, rescale hoặc rounder.
- **T2:** rule CE, LR/GBDT, P-arm yes/no logprob, ngưỡng theo nhãn có shrinkage.
- **T3:** diagnose → render, regex guard, template theo từng level, SFT.
- **Fine-tune:** A12 train 1 lần, fit ngưỡng trên holdout.
- **Bonus:** 2 phát hiện có số liệu.

**Thứ tự cắt khi trễ** (cắt mục đầu trước):
1. N1 bản khả vi, LeakIG, mô phỏng sinh viên.
2. Harness `-Dmain` và compile riêng từng câu.
3. Peer-oracle và test sinh (giữ chạy lại test công khai và `io_gap`).
4. Augmentation cho nhãn hiếm.
5. DPO và level-swap (giữ SFT + guard).
6. Teacher API và distill.
7. Cross-fit nhiều seed (thay bằng 1 adapter + holdout).
8. Ablation backbone và ablation dạng output.
9. Stacker phức tạp và MAP 240 tổ hợp.
10. Meta-eval judge đầy đủ.

**Tuyệt đối không cắt:**
- Validator; tái lập 1 lệnh trên máy sạch.
- Tường lửa feedback kèm test tự động.
- ≥1 hướng prompting và ≥1 model fine-tune ≤10B chạy local, so sánh trên cùng split.
- Policy engine.
- Guard L1/L2 và fallback đúng level.
- Decoder theo metric.
- CV chia group đúng.
- ≥2 phát hiện bonus.
- Ablation các thành phần lõi và phân tích lỗi theo lát cắt.
- Đủ các mục báo cáo bắt buộc, khai báo LLM, log chi phí.

**Phân vai (6 người)**, mỗi người làm chủ 1 mục báo cáo và có 1 người dự phòng:

| Người | Phụ trách |
|---|---|
| 1 | Data, EDA, hồ sơ bonus |
| 2 | Sandbox và evidence, tách câu |
| 3 | T1: policy engine và decoder QWK |
| 4 | T2 và P-arm, vLLM |
| 5 | Fine-tune (A12/A3) và tái lập |
| 6 | T3: renderer, guard, judge |

- **Nhóm 5 người:** gộp vai 2 vào 1, và Eval vào 5.
- **Nhóm 7 người:** tách thêm một vai Tích hợp/Eval riêng.

**Nhịp làm việc:**
- Thứ Sáu là ngày tích hợp. Mọi model xuất OOF cùng định dạng `oof/{task}/{run_id}.parquet`.
- Chỉ một người chạy `make submit`.
- Peer assessment ẩn danh ở W4 và W8.

**Leaderboard:**
- Thực tế dùng được khoảng 35–50 lượt nộp.
- Chỉ nộp khi một trong ba điều kiện đúng:
  - (i) CV cải thiện ≥ max(ngưỡng thực tế, 1 SE);
  - (ii) cần hiệu chỉnh quan hệ CV↔LB (W1–W3);
  - (iii) cần đối chiếu scorer.
- Chọn bản nộp cuối theo **CV − 0.5·SE**, không theo đỉnh public.
- **Không task nào được rơi xuống đáy**, vì bảng tổng là trung bình thứ hạng.
- **Ưu tiên ban đầu:**
  - T3 guard: rẻ nhất, khoảng cách giữa các nhóm có lẽ lớn nhất.
  - T2 ngưỡng và rule.
  - T1 policy engine và de-shrink.

**Ablation và thống kê:**
- 3 seed cho fine-tune.
- Paired bootstrap với B = 2000; **lấy mẫu theo cụm đề** nếu có đề mới.
- Approximate randomization cho ΔQWK và Δmacro-F1. McNemar cho tỷ lệ vi phạm.
- Holm–Bonferroni khi có nhiều so sánh.
- Ngưỡng thực tế: ΔQWK hoặc ΔF1 ≥ 0.01.
- Làm cả hai chiều: cộng dần từ baseline (build-up) và bỏ từng thành phần khỏi hệ final (leave-one-out).

**Phân tích lỗi theo "kiểu bài nào và vì sao":**
1. Bảng lát cắt: type, môn, đề đã gặp/đề mới, nhóm điểm, độ dài/cắt context, biên dịch, số nhãn, level.
2. Xếp hạng lát theo phần đóng góp vào loss.
3. Hai người đọc 20–30 mẫu mỗi lát theo codebook, báo κ giữa hai người.
4. **Oracle-fix:** thay thành phần nghi ngờ bằng bản đúng rồi đo Δ.
5. Viết thẻ lỗi, mỗi thẻ khoảng ¼ trang.

**Thí nghiệm then chốt còn thiếu** (giá trị báo cáo cao):
- **Learning curve** 10/25/50/100% dữ liệu, so F-arm với P-arm, chạy trên model nhỏ cho rẻ.
- **Chuyển giao giữa 2 môn.**
- **Chế độ chỉ dùng tín hiệu được cấp** (không tự chạy code): vừa là một dòng ablation, vừa là fallback cho tái lập.
- **Ma trận bền vững theo quy ước scorer.**

---

## 10. Quyết định do dữ liệu chốt ở W1–W2

| D | Câu hỏi | Kiểm bằng | Nếu có | Nếu không |
|---|---|---|---|---|
| D0 | Dữ liệu đủ lớn? (`len(train)`, positive theo nhãn) | `Y.sum(0)` | Đi theo kế hoạch | Nếu < khoảng 1.5k bài hoặc có nhãn < 10 positive: bỏ head 8B cho T1/T2, dùng GBDT + P-arm, dồn fine-tune vào T3 |
| D1 | Tổng luôn bằng Σ6 chiều? | `(df[DIMS].sum(1)-df.total).abs().gt(1e-6).groupby(df.type).mean()` | MAP có ràng buộc tổng | Head riêng cho tổng |
| D2 | Dev/test có đề mới? | `(~dev.pid.isin(train.pid)).mean()` | Group theo đề; oracle do LLM sinh | Group theo near-dup; kNN và peer-oracle mạnh |
| D3 | Multi có điểm từng câu? | Đọc 5 JSON thô | Train head từng câu | Điểm câu là biến ẩn (N1) |
| D4/D6 | Chi tiết scorer (QWK, `zero_division`) | Hỏi + 1 lượt nộp dò | Theo đúng scorer | Chọn cấu hình bền với mọi quy ước |
| D5 | Tam giác compile khớp ≥99%? | crosstab | Override cứng | Chỉ dùng làm feature |
| D7 | Gold L1/L2 có code? | Regex + đọc tay | Lọc khỏi SFT, đưa vào hồ sơ bonus | Giữ nguyên |
| D8 | L2 có được trích code sinh viên không? | EDA + hỏi giảng viên | Nới cho L2 | L1/L2 không trích code |
| D9 | Được gửi code lên API? | Hỏi bằng văn bản | Teacher API (gói không train trên dữ liệu) | Teacher 27–32B local |
| D10 | p95 token của multi > 8k? | Tokenizer thật | Bắt buộc tách câu | Dùng cả file |
| D11 | Toolchain gốc là Windows/MinGW? | grep `mingw`, `C:\` trong log | Ưu tiên log gốc | g++ Linux |

**Ba ngày đầu:**
- **Ngày 1:**
  - Gửi câu hỏi (§3).
  - Tải dữ liệu, tạo `MANIFEST.sha256`.
  - Repo private: `.gitignore` chặn `data/`, `*.cpp`, `cache/`; gitleaks; lockfile.
  - Loader decode lần lượt utf-8 → cp1258 → latin-1, chuẩn hoá NFC, đếm số file phải fallback.
  - Chạy validator trên file giả (dummy) cho cả 3 task, gồm trường hợp T2 trả `[]`.
- **Ngày 2:**
  - Notebook audit sanity: schema, D1, D5, prevalence, chiều suy biến, phân bố tổng, token p50/p95/p99, D2, D7.
  - Khoá `folds.csv` v0.
  - `eval/metrics.py` kèm unit test.
- **Ngày 3:** baseline nộp được.
  - **T1:** rule từ log; mode của (đề, compilable); tổng = Σ.
  - **T2:** rule CE; TF-IDF char 3–5-gram + LR `balanced`; ngưỡng theo nhãn trên OOF.
  - **T3:** L1/L2 dùng template theo (nhãn × level); L3/L4 dùng LLM local zero-shot.

---

## 11. Rủi ro tuân thủ, xếp theo mức

| Mức | Rủi ro | Phòng ngừa |
|---|---|---|
| **S** (huỷ kết quả hoặc kỷ luật) | Dữ liệu ra khỏi phạm vi được phép | Văn bản "dữ liệu được đi đâu"; pre-commit; dữ liệu nằm ngoài tầm AI assistant |
| S | Feedback lọt vào T1/T2, trực tiếp hoặc gián tiếp | Loader theo whitelist; CI assert prompt T1/T2 không chứa chuỗi feedback dài ≥30 ký tự (trừ phần trùng với chính code sinh viên); bảng lineage |
| S | Chạm vào test | Input test chỉ được giải nén bên trong `make predict-private`; retry nằm sẵn trong code; log hash |
| S | Truy danh tính | PII scan chỉ đếm; không dùng mã SV làm feature |
| **A** (mất ~30% thứ hạng) | Không tái lập được | Cache evidence và output LLM; pin phiên bản; chạy trên máy sạch W6–W7 |
| A | Predictions private sai định dạng hoặc chạy không kịp | Chạy thử trên dev đúng lệnh đó; đo throughput; có sẵn cấu hình nhanh đã kiểm |
| **B** (trừ từng phần) | Thiếu một trong hai hướng hoặc so sánh không công bằng; thiếu báo cáo phần cứng, seed, phiên bản, log chi phí; báo cáo thiếu mục hoặc vượt trang; bảng đóng góp lệch commit | Checklist sản phẩm nộp chạy ở W7 |

---

## 12. Câu hỏi chưa giải quyết

- **Mọi câu ở §3.** Trong đó, các câu đổi kiến trúc là:
  - scorer (QWK, `zero_division`);
  - detector compliance và (b);
  - quyền gửi code lên API;
  - dev/test có đề mới;
  - multi có điểm từng câu;
  - phạm vi cấm feedback.
- **Kích thước và schema thật của dữ liệu.** Mọi ước lượng GPU-giờ, giờ công và ngưỡng ở đây đều là [GT]; phải đo lại ngay W1.
- **Năng lực thực tế của nhóm** (giờ/tuần/người, GPU sẵn có). Quyết định có làm cross-fit, DPO, distill hay không.
- **Cần kiểm chứng kỹ thuật:**
  - hỗ trợ head multi-label và `VLLM_BATCH_INVARIANT` trên phiên bản vLLM sẽ pin;
  - VRAM thực của teacher 27–32B bản 4-bit;
  - Docker cho toolchain MinGW;
  - cú pháp tmpfs của Docker.
- **Trích dẫn:** đọc lại các paper trước khi đưa vào báo cáo. Ví dụ red-team chỉ ra một tuyên bố về Hellas et al. (ICER 2023) có thể đã bị phóng đại.
