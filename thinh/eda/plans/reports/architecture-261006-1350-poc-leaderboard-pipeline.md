# Kiến trúc POC cho leaderboard: chấm rubric (T1), phân loại lỗi (T2), phản hồi theo level (T3)

Phạm vi: POC tối ưu độ chính xác trên leaderboard. Bỏ qua UX, giao diện, vận hành cho người dùng.
Số liệu bên dưới là **đo thật** trên 32 mẫu (`sample_dataset`) bằng các script trong `eda/poc/`. Mô hình LLM
trong thực nghiệm là Claude Opus 5.5 và Haiku 4.5, chạy **mù nhãn**: mỗi agent chỉ đọc file input đã bỏ hết
rubric/taxonomy/feedback gold. "Nhãn đã sửa" = released + `LABEL_FIXES` (S214, S211, S216, theo phản hồi của giảng viên).

Sơ đồ: [`eda/arch/grading-pipeline.html`](../../arch/grading-pipeline.html) (dataflow, mở bằng trình duyệt).

---

## 0. Tóm tắt

| Kết luận | Bằng chứng |
|---|---|
| Kiến trúc tốt nhất: **định tuyến theo loại đề**. EX02 lấy rubric từ tín hiệu tất định (gcc + test). EX01 lấy rubric từ LLM. Nhãn T2 lấy từ LLM, bầu chọn giữa nhiều lần chạy. | QWK 0.80 [0.59–0.94], MAE 1.19, macro-F1 0.40 (nhãn đã sửa) |
| Tín hiệu tất định quan trọng hơn chọn model | EX02: chỉ dùng luật đã đạt QWK 0.95, còn LLM tốt nhất đạt 0.87 |
| Prompt quan trọng hơn kích thước model | Đổi prompt v1 → v2 tăng QWK 0.69 → 0.76 (Opus). Haiku dùng v2 (0.72) đã hơn Opus dùng v1 (0.69) |
| EX01 (multi_problem) là điểm yếu chính | QWK 0.63. EX01 không có `test_report`, và gold của EX01 không theo luật "P1 sai thì bỏ P2–P4" ghi trong đề |
| T1 và T2 cần **tách** cấu hình | Giới hạn 1–2 nhãn giúp T1 nhưng làm giảm recall T2: macro-F1 của Haiku giảm 0.45 → 0.29 |
| Chưa nên dùng RL (SPPO/GRPO) cho POC | Chỉ có 32 nhãn, nhãn còn nhiễu, chưa đủ dữ liệu. Ưu tiên distill + SFT, và dùng DPO/KTO cho T3 |
| Self-host: Qwen3.8-27B làm model chính, Qwen3.5-9B làm dự phòng | Cùng prompt, cùng JSON schema, cùng gate, bầu chọn và eval harness với bản API |

Cảnh báo: n = 32 nên khoảng tin cậy 95% của QWK rộng khoảng ±0.2. Luật B1 và prompt v2 được rút ra
từ chính tập này, nên mọi con số đều **lạc quan**. Chênh lệch dưới khoảng 0.1 QWK chưa đủ để kết luận.

---

## 1. Dữ liệu quyết định thiết kế thế nào

| Đặc điểm dữ liệu (EDA) | Hệ quả thiết kế |
|---|---|
| Hai loại đề cố định (giảng viên xác nhận: chỉ single và multi) | Định tuyến cứng theo `exam_type`. Mỗi loại đề có prompt và luật riêng |
| `compile_log` dự đoán đúng 100% nhãn `compilable` (sau sửa nhãn S214), gcc 12.2 cho kết quả trùng khớp | **Cổng biên dịch** tất định, không hỏi LLM |
| EX02 build lỗi thì gold luôn là `[0,0,0,0,0,1]`. EX01 build lỗi thì gold toàn 0 | Hậu xử lý áp `COMPILE_FAIL_RUBRIC` |
| EX02 có `test_report` 8 test công khai. Tín hiệu "fail toàn bộ test" đạt F1 0.80 cho io_format (sau sửa nhãn) | Rubric EX02 tính chủ yếu từ test, LLM chỉ bổ sung |
| EX01 có `test_report = null` và `has_hidden_testcases: false`; gold không áp luật dây chuyền P1 | Không có tín hiệu thực thi. Gold EX01 khớp tốt với "tổng trọng số 1/4/2/3 của các câu có làm" (xem báo cáo phân tích `analysis-261006-1612-poc-results.md`) |
| EX02 có quy luật `edge = logic/2`, `code_quality ≡ 1`. EX01 có `complexity = code_quality = [logic > 0]` | Hậu xử lý theo quy luật rubric. Lưu ý: quy luật rút từ chính tập này |
| Gold có nhiễu (S211, S214, S216) | Đánh giá trên cả nhãn released và nhãn đã sửa, theo dõi chênh lệch |
| Nhãn gần nghĩa: logic, thuật toán, vòng lặp, hàm. Không có mẫu gold nào mang "Lỗi edge case" | Macro-F1 tối đa trên tập mẫu chỉ là 0.9. Cần định nghĩa nhãn (đang chờ giảng viên trả lời) |

---

## 2. Kiến trúc hệ thống

```
Bài nộp ─▶ Signal extractor (gcc, test verdict, regex) ─▶ Grader LLM (rubric EX01 + nhãn) ─▶ Gate + vote ─▶ predictions.jsonl
Đề+rubric ─▶ Few-shot retriever (cùng đề) ─────────────────┘                                      │
                                                                     Feedback LLM (T3, L1–L4) ◀─┘ ─▶ Level guard ─▶ predictions.jsonl
```

| Thành phần | Vai trò | API / self-host |
|---|---|---|
| Signal extractor (`eda/signals.py`) | Trạng thái build, test verdict (pass/prompt/wrong/crash/timeout), strict/lenient pass | Dùng chung, không gọi LLM |
| Router theo loại đề | EX02 tính rubric bằng luật B1. EX01 dùng LLM | Dùng chung |
| Few-shot retriever | k ví dụ cùng đề, chọn bằng kNN embedding code. Chỉ lấy ví dụ từ tập train | Dùng chung (embedding có thể chạy local) |
| Grader LLM | Một lần gọi trả JSON: rubric 6 chiều, taxonomy, evidence, confidence | Khác nhau ở backend, prompt giống nhau |
| Gate + vote (`poc/rubric_constraints.py`) | Ép cổng biên dịch, áp quy luật rubric, bầu nhãn ≥ 2/4 lần chạy | Dùng chung |
| Feedback LLM (T3) | Sinh phản hồi tiếng Việt từ nhãn T2 có sẵn trong input và level mục tiêu | Khác nhau ở backend |
| Level guard | Ở L1–L2 chặn code block; L1 chặn nêu vị trí/nguyên nhân; kiểm tiếng Việt; thử lại tối đa 2 lần | Dùng chung |
| Eval harness (`poc/score_predictions.py`) | QWK (trục 0..10), MAE, exact match từng chiều, macro/micro-F1, F1 từng nhãn; chấm trên nhãn released và nhãn đã sửa | Dùng chung |

**Các phần dùng chung cho cả LLM API và self-host:**
- Prompt (Jinja template).
- JSON schema (`eda/docs/contracts/`).
- Signal extractor, router, gate và vote.
- Level guard.
- Eval harness và bộ input mù nhãn.

Chỉ client là khác nhau. Một interface `complete(prompt, schema) -> dict` có hai cài đặt:
- Anthropic/OpenAI SDK cho bản API.
- vLLM OpenAI-compatible với `structured_outputs` cho bản self-host.

Nhờ vậy kết quả ablation so sánh được giữa hai track.

---

## 3. Thiết kế prompt

### 3.1 Prompt chấm (T1 + T2): v1 và v2

v1 có rubric chung chung, yêu cầu "theo grading_policy từng chữ", nhãn không có định nghĩa và không giới hạn số nhãn.
v2 là prompt đã đo được tốt hơn. Cấu trúc v2 (nguyên văn nằm trong script workflow, đây là tóm tắt):

1. **Vai trò và luật dữ liệu:** chỉ đọc input của bài này.
2. **Bước 1, cổng build:** tin `build_status` của harness, không được cãi.
3. **Bước 2:** tách đề thành từng yêu cầu. Mỗi yêu cầu xếp vào một trong năm mức: chạy đúng / gần đúng / một phần / sai / thiếu.
4. **Bước 3, mốc điểm (anchor)** cho từng chiều:
   - logic 4/3/2/1/0 tương ứng tất cả đúng / một lỗi nhỏ / khoảng một nửa / một phần nhỏ / không gì đúng.
   - io_format định nghĩa riêng cho bài một chương trình (khớp output chính xác, dòng nhắc thừa thì 0) và bài nhiều hàm (giữ đúng chữ ký và hợp đồng vào/ra).
   - **Cho điểm từng phần; một phần trước hỏng không kéo điểm các phần sau về 0.**
5. **Bước 4, taxonomy:** chỉ ghi 1–2 lỗi chính làm mất điểm, tối đa 3. Bài đạt điểm tối đa thì danh sách rỗng. Kèm định nghĩa 10 nhãn. Vấn đề nằm ngoài danh sách thì ép vào nhãn gần nhất (theo hướng dẫn của giảng viên).
6. **Bước 5, tự kiểm:** io_format = 0 thì phải có "Lỗi nhập/xuất"; điểm tối đa thì danh sách nhãn rỗng.
7. **Output bắt buộc theo JSON schema:**
   - rubric (số nguyên trong miền),
   - taxonomy (enum),
   - evidence (≤ 4 ghi chú, dẫn số dòng, không chép code),
   - confidence.

Nguyên tắc thiết kế:
- Tín hiệu tất định đặt trước code trong prompt.
- Lý do (evidence) đặt trước điểm trong schema khi chạy model không có thinking.
- Phần cố định (đề, rubric, định nghĩa nhãn, few-shot) đặt đầu prompt để dùng prompt cache.
- Phần thay đổi (code, log, test) đặt cuối.

### 3.2 Prompt phản hồi (T3)

- **Input:** đề, code, `taxonomy_error` (có sẵn trong input của T3), level mục tiêu, định nghĩa 4 level lấy từ `label_space.feedback_level`.
- **Prompt theo level:** mỗi level có một khối "ĐƯỢC / CẤM" riêng:
  - L1: chỉ báo có vấn đề hoặc khen, không chỉ ra vị trí.
  - L2: nêu vị trí và nguyên nhân, không nêu cách sửa, không code.
  - L3: thêm hướng sửa, không đưa lời giải hoàn chỉnh.
  - L4: được đưa code.
- **Bài không có lỗi:** khen ngắn gọn và xác nhận bài đạt. Không bịa lỗi để có gì đó mà góp ý.
- **Level guard (luật, không dùng LLM):**
  - L1–L2: không có code fence, không có dòng nào giống code (`;`, `{`, toán tử gán).
  - L1: không nhắc số dòng hoặc tên biến.
  - Mọi level: tỉ lệ ký tự tiếng Việt vượt ngưỡng.
  - Vi phạm thì gọi lại với thông điệp lỗi cụ thể.
- **Chấm T3:**
  - Mức tuân thủ level đo bằng guard (đo được tự động 100%).
  - Độ chính xác chẩn đoán đo bằng LLM-as-judge với rubric "có nêu đúng các nhãn trong input không", kèm kiểm tra tay một mẫu nhỏ.

---

## 4. Các hướng tiếp cận

| Hướng | Mô tả | Ưu | Nhược | Dùng cho |
|---|---|---|---|---|
| A. Chỉ luật | Cổng biên dịch, test verdict, prior theo đề | Rẻ, tất định, giải thích được | Không đọc được code: EX01 QWK ≈ 0, T2 F1 0.22 | Rubric EX02, cổng biên dịch |
| B. Chỉ LLM (zero-shot, đọc code) | Một prompt, model đọc code | Không cần luật | Dễ chấm ngược luật build, gán thừa nhãn | Baseline |
| **C. Lai: tín hiệu, rồi LLM, rồi luật** | A cho phần đo được, LLM cho phần cần hiểu code, luật sửa ràng buộc | Tốt nhất đo được | Luật rút từ tập nhỏ, có nguy cơ overfit | **Chọn cho POC** |
| D. Self-host + SFT bằng distill | Model API làm thầy gán nhãn cho vài trăm bài; lọc theo gold và luật; LoRA cho Qwen | Không phụ thuộc API, rẻ khi chạy số lượng lớn | Cần dữ liệu chưa nhãn, mất công setup | Pha 2 |
| E. Thực thi code thật cho EX01 | Tự viết test cho prob1–4 từ đề và file dữ liệu mẫu; chạy trong sandbox `--network none` | Có ích nếu gold EX01 chấm theo chạy thật | Phải viết harness cho từng đề; gold EX01 có vẻ không chấm theo chạy thật | Chỉ làm sau khi giảng viên xác nhận cách chấm EX01 |
| F. Đa agent hoặc LLM-judge kiểm chéo | Hai model chấm, model thứ ba phân xử | Có thể tăng ổn định | Tốn gấp 3 chi phí; chưa chứng minh hơn vote | Chỉ khi vote chưa đủ |

---

## 5. Bảng ablation (đo thật, 32 mẫu, nhãn đã sửa)

| # | Cấu hình | QWK | MAE | macro-F1 | micro-F1 |
|---|---|---|---|---|---|
| B0 | Prior theo đề (leave-one-out majority) | −0.07 | 3.53 | 0.02 | 0.10 |
| B1 | Cổng biên dịch + test verdict (không dùng LLM) | 0.53 | 1.97 | 0.23 | 0.49 |
| B1r | B1 + quy luật rubric | 0.59 | 1.66 | 0.23 | 0.49 |
| L1 | Opus, prompt v1, **chỉ code** | 0.59 | 2.12 | 0.39 | 0.50 |
| L2 | Opus, v1, + compile_log/test | 0.64 | 2.03 | 0.40 | 0.49 |
| L3 | L2 + cổng biên dịch | 0.69 | 1.72 | 0.39 | 0.50 |
| L4 | L2 + cổng + quy luật rubric | 0.71 | 1.66 | 0.39 | 0.50 |
| L5 | Haiku, v1, + tín hiệu + cổng | 0.63 | 2.06 | **0.45** | 0.52 |
| L6 | Opus, **prompt v2** + tín hiệu + cổng | 0.77 | 1.47 | 0.37 | 0.51 |
| L7 | Haiku, v2 + tín hiệu + cổng | 0.72 | 1.66 | 0.29 | 0.43 |
| H1 | Lai: EX01 = L3, EX02 = B1r, nhãn từ L3 | 0.76 | 1.22 | 0.39 | 0.50 |
| H2 | Lai: EX01 = L6, EX02 = B1r, nhãn từ L6 | 0.80 | 1.19 | 0.37 | 0.51 |
| H3 | H2 với Haiku thay Opus | 0.78 | 1.16 | 0.29 | 0.43 |
| H4 | H2 + bầu nhãn ≥ 1/4 lần chạy (hợp) | 0.80 | 1.19 | **0.43** | 0.49 |
| **H5** | **H2 + bầu nhãn ≥ 2/4** | **0.80** | 1.19 | 0.41 | 0.52 |
| H6 | H2 + bầu nhãn ≥ 3/4 | 0.80 | 1.19 | 0.36 | **0.57** |

Theo từng đề, cấu hình H5:

| Đề | QWK | MAE | macro-F1 |
|---|---|---|---|
| EX01 | 0.63 | 2.13 | 0.39 |
| EX02 | 0.95 | 0.35 | 0.30 |

Khoảng tin cậy bootstrap 95% (2000 lần lấy mẫu) và ảnh hưởng của việc sửa nhãn:

| Cấu hình | QWK (nhãn đã sửa) | QWK (nhãn released) |
|---|---|---|
| H5 | 0.80 [0.59, 0.94] | 0.70 [0.41, 0.91] |
| L3 | 0.69 [0.44, 0.87] | 0.61 [0.34, 0.82] |
| B1r | 0.59 [0.32, 0.83] | 0.52 [0.25, 0.77] |

Đọc bảng:
- **Thêm tín hiệu tất định (L1 → L3):** +0.10 QWK. Riêng cổng biên dịch đóng góp +0.05. LLM đoán đúng `compilable = 0` cho cả 4 bài EX02 build lỗi, nhưng không áp rubric build lỗi `[0,0,0,0,0,1]`: S212 được 7 điểm, S203 được 3, S214 và S217 được 0 (gold đều là 1). Luật của đề nên để code áp, không giao cho LLM.
- **Prompt v2 (L3 → L6):** +0.08 QWK, và MAE giảm 0.25. Đây là đóng góp lớn nhất của khâu prompt.
- **Định tuyến theo đề (L6 → H2):** +0.03 QWK. Riêng EX02, QWK tăng 0.87 → 0.95: test đáng tin hơn việc LLM đọc code.
- **Model:** dùng v2, Haiku kém Opus khoảng 0.05 QWK, nằm trong khoảng nhiễu. Với v1, Haiku gán nhiều nhãn hơn nên macro-F1 cao hơn.
- **T2 đánh đổi giữa precision và recall:**
  - Giới hạn số nhãn (v2) làm giảm recall các nhãn hiếm.
  - Bầu chọn có ngưỡng điều chỉnh được giữa hai chỉ số: ≥ 1/4 tối ưu macro-F1, ≥ 3/4 tối ưu micro-F1.
  - Nên chọn ngưỡng **theo từng nhãn** trên tập dev.

Còn thiếu (đã thiết kế, chưa chạy):
- k-shot retriever (k = 0/2/4).
- Self-consistency k = 3/5 cho rubric.
- Self-host Qwen.
- T3 compliance.
- Hiệu chỉnh isotonic, fit leave-one-out.

---

## 6. Phân tích lỗi: hệ thống sai ở bài nào, vì sao

Bảng đầy đủ từng bài sinh từ `final-scores.json` và các file dự đoán. Bên dưới là các bài sai lệch ≥ 3 điểm tổng ở cấu hình H5.

### 6.1 EX01 (multi_problem): 5/15 bài sai ≥ 3 điểm

| Bài | Gold | Dự đoán | Hệ thống thấy gì (evidence của LLM, diễn giải) | Vì sao lệch |
|---|---|---|---|---|
| S108 | 6 | 3 (v1: 1) | prob1 mở file nhưng đọc từ stdin; kích thước mảng hard-code | Gold chấm rộng tay cho prob2–4 dù prob1 hỏng. Cách gold cho điểm không suy ra được từ code |
| S110 | 8 | 4 (v1: 1) | n không được gán; vòng lặp lồng gán sai trường; parse ngày sai | Như trên: gold cao dù LLM thấy prob1 hỏng nặng. Gold khớp với tổng trọng số các câu có làm (8/8) |
| S112 | 9 | 1 | Mở file theo chuỗi literal thay vì tham số; con trỏ chưa khởi tạo; vòng lặp prob2 không chạy | Lệch lớn nhất. LLM (cả Opus lẫn Haiku, cả v1 lẫn v2) đều chấm thấp. **Ứng viên nhãn nhiễu** cần hỏi giảng viên |
| S114 | 1 | 5 | Parse đúng bản ghi đầu tiên, nhưng đóng file bên trong vòng lặp | LLM cho điểm từng phần; gold coi cả bài hỏng (khớp hành vi chạy thật: chỉ đọc được một bản ghi) |
| S107 | 2 | 5 | prob2 dựng danh sách đúng; prob3 lặp vô hạn | Prompt v2 "cho điểm từng phần" làm LLM cho dư. v1 trùng gold. **Lời nhắc cho điểm từng phần là con dao hai lưỡi** |

Nguyên nhân gốc ở EX01:
1. **Không có tín hiệu thực thi, và gold không chấm theo chạy thật.** EX01 có `has_hidden_testcases: false`. (Bản trước của báo cáo này viết là có hidden test; điều đó sai.) Gold EX01 khớp tốt với tổng trọng số 1/4/2/3 lấy từ đề, cộng cho các câu có thân hàm: 12/15 bài lệch ≤ 1 điểm. LLM thì chấm theo độ đúng của code, nên lệch khỏi gold.
2. **Luật trên giấy khác cách gold chấm.** Luật "P1 sai thì không chấm P2–P4" trong `grading_policy` không được gold tuân thủ. v1 làm theo luật thì chấm thấp hàng loạt (S108, S110, S112 bị 1 điểm). v2 bỏ luật này thì đỡ hơn, nhưng lại cho dư ở S107/S114.
3. **Lệch hệ thống:**
   - v1 chấm thấp hơn gold trung bình 1.9 điểm/bài EX01.
   - v1 gán trung bình 3.1 nhãn/bài, trong khi gold chỉ 1.2.
   - Nhãn thừa phổ biến: "Lỗi nhập/xuất" (do LLM coi parse file sai là lỗi I/O), "Lỗi hàm", "Lỗi edge case".
4. **Nhầm nhãn:** gold dùng "Lỗi thuật toán" ở S102/S104, LLM ghi "Lỗi logic". Ranh giới giữa hai nhãn chưa được định nghĩa.

### 6.2 EX02 (single_problem): rubric gần đúng, nhãn còn yếu

| Bài | Gold | Dự đoán | Vì sao |
|---|---|---|---|
| S209 | 7 | 3 | Fail cả 8 test (B1 suy ra io = 0, logic = 0), nhưng gold io = 1, logic = 2. Output sai không do định dạng. Gold có vẻ chấm theo đọc code chứ không theo test |
| S205 | 9 | 10 | Lỗi khoảng trắng thừa: harness strip whitespace nên pass 8/8, tín hiệu không thấy được. Cần so khớp stdout thô |
| S210 | 2 | 3 | complexity dư 1, do prior theo đề |

Về nhãn EX02:
- Rubric đúng tuyệt đối 14/17 bài, nhưng nhãn chỉ đạt macro-F1 0.30.
- Nhầm nhiều giữa vòng lặp, logic, thuật toán và hàm. Ví dụ S213, S215, S216: gold ghi "vòng lặp" hoặc "hàm", LLM ghi "logic" hoặc "thuật toán".
- Hai bài bị nhãn thừa "Lỗi style" (S204, S214) mà không tín hiệu nào bắt được.

### 6.3 F1 từng nhãn (H5)

| Nhãn | F1 |
|---|---|
| biên dịch | 1.00 |
| hard-code | 0.80 |
| logic | 0.67 |
| nhập/xuất | 0.55 |
| vòng lặp | 0.38 |
| hàm | 0.38 |
| mảng/chuỗi | 0.29 |
| edge case | 0 (không có mẫu gold nào) |
| thuật toán | 0 |
| style | 0 |

Macro-F1 đang bị kéo xuống bởi ba nhãn hiếm hoặc mơ hồ (thuật toán, style, edge case). Đó là vấn đề định nghĩa nhãn trước, model sau.

---

## 7. Chi phí và khả năng triển khai

Kích thước input đo trên bộ mẫu, ước lượng khoảng 3–3.5 ký tự mỗi token:

| Thành phần | EX01 | EX02 |
|---|---|---|
| Phần chung (đề, rubric, định nghĩa nhãn) | ≈ 1.1K token | ≈ 0.5K token |
| Mỗi bài, trung vị | ≈ 1.1K token (tối đa ≈ 1.9K) | ≈ 0.45K token |

Cộng thêm instruction ≈ 1K và few-shot k = 2 ≈ 2K, mỗi lần gọi khoảng **3–5K token input**. Output T1+T2 khoảng 0.4K token; T3 khoảng 0.5K token.

| Track | Đơn giá (USD/1M token, in/out) | Ước tính 1.000 bài (T1+T2+T3, 1 lần gọi mỗi task) | Ghi chú |
|---|---|---|---|
| Haiku 4.5 | 1 / 5 | ≈ $10 | Batch giảm 50%; cache phần chung giảm thêm |
| Sonnet 5.5 | 2 / 10 | ≈ $20 | |
| Opus 5.5 | 4 / 20 | ≈ $40 | Chỉ nên dùng làm "thầy" để distill |
| Bầu chọn 4 lần | gấp 4 lần các dòng trên | ≈ $40 (Haiku) | Chỉ cần cho T2. Có thể bầu bằng 2 model rẻ |
| Self-host Qwen3.8-27B (FP8/AWQ, 1 GPU 48 GB) | tiền thuê GPU khoảng $1–2/giờ | < $2 (vLLM batch, vài trăm bài/giờ) | Thêm công setup và một lần fine-tune |

Đơn giá lấy từ trang giá chính thức, ngày 2026-10-06. Ước tính chi phí của bảng là tính tay.

Khả năng triển khai thực tế (ngoài phạm vi POC, chỉ ghi lại rủi ro):
- **Thực thi code sinh viên:** chỉ chạy trong container `--network none` có giới hạn CPU/RAM/thời gian, như đã làm với gcc.
- **Dữ liệu sinh viên:** không được đưa ra ngoài lớp học. Track API phải dùng gói có cam kết không lưu và không train trên dữ liệu (hoặc chạy self-host). Đây là lý do thực tế để có track self-host.
- **Tất định:** đặt temperature 0 khi có thể và cố định seed, ghi log prompt hash, model id và kết quả cho từng bài để audit.

---

## 8. Self-host: hướng và model

| Vai trò | Model | Lý do |
|---|---|---|
| Model chính | **Qwen3.8-27B** (Apache-2.0, dense) | Lập luận code mạnh nhất cỡ này. Chạy FP8/AWQ trên 1 GPU 48 GB hoặc bf16 trên 1 GPU 80 GB. Model dense nên LoRA đơn giản |
| Dự phòng hoặc chạy nhanh | **Qwen3.5-9B** | Cùng template, prompt dùng lại nguyên vẹn. Chạy bf16 trên GPU 24 GB, QLoRA trên một GPU |
| Model khác họ để ensemble, hoặc cho T3 | Gemma 4 31B | Đa ngôn ngữ tốt, khác họ nên bỏ phiếu đa dạng hơn |

- Chưa có benchmark tiếng Việt công khai cho các model 2026 này. Cần tự đo T3 trên tập dev.
- Serving: vLLM hoặc SGLang, `structured_outputs` (xgrammar) để ép JSON schema.

**Lộ trình self-host:**
1. **Zero-shot:** chạy cùng prompt v2 và cùng harness.
2. **Distill + SFT (LoRA):**
   - Opus chấm vài trăm bài chưa nhãn (cùng hai đề, hoặc bài tự sinh bằng cách cấy lỗi vào lời giải đúng).
   - Chỉ giữ output khớp gold hoặc khớp luật tất định.
   - SFT Qwen trên tập đó.
3. **DPO/KTO cho T3:** cặp "đúng level" và "lộ code hoặc sai level" sinh tự động, không cần người gán nhãn.

---

## 9. RL ("SPA") cho fine-tuning service

- **"SPA" có thể là:**
  - SPA-RL (phân bổ credit theo từng bước cho agent): không hợp với bài toán này.
  - **SPPO** (Self-Play Preference Optimization, có công thức trong verl): chỉ có ý nghĩa cho chất lượng phản hồi T3.
  - Hoặc nói chung là PPO/GRPO.
- **Kết luận cho POC: chưa nên.**
  - Chỉ 32 nhãn, mà gold còn nhiễu (3/32 bài đã xác nhận sai).
  - GRPO cần nhiều prompt để các lần sinh trong một nhóm cho kết quả khác nhau.
  - Reward dạng rubric dễ bị model lách (reward hacking).
- **Khi nào nên dùng:**
  - T1 cần khoảng 300–500 bài có nhãn người cho mỗi loại đề, cộng một tập held-out.
  - T2 và T3 có thể dùng sớm hơn nếu reward kiểm được tự động: cấy lỗi đã biết vào code để có nhãn T2 chính xác; dùng guard luật làm reward level cho T3.
- **Thiết kế reward nếu làm:**
  - T1: r = 1 − ((ŷ − y)/10)², cùng dạng với trọng số bậc hai của QWK; QWK không tính được cho từng bài.
  - T2: F1 từng bài, trọng số theo nghịch đảo tần suất nhãn để bám macro-F1.
  - T3: guard leak code + kiểm tiếng Việt + nêu đúng nhãn + phạt KL.
- **Công cụ:** TRL + Unsloth là đủ cho POC. verl và OpenRLHF chỉ cần khi chạy nhiều GPU.

---

## 10. Khả năng diễn giải

| Kỹ thuật | Đã có trong POC | API | Self-host |
|---|---|---|---|
| Điểm tách theo 6 chiều, tổng tính bằng code | Có | Có | Có |
| Evidence dẫn số dòng, không chép code | Có (schema) | Có | Có |
| Kiểm evidence: số dòng phải tồn tại và khớp từ khóa trong code | Chưa | Có | Có |
| Quyết định tất định có lý do rõ (cổng build, test verdict) | Có, giải thích 100% | Có | Có |
| Mức đồng thuận giữa nhiều lần chạy làm độ tin cậy | Có (bầu nhãn) | Có (gọi lặp) | Có |
| Điểm kỳ vọng tính từ logprob của token điểm | Không | Không (Claude không trả logprob; GPT-5+ và Gemini 3 đã bỏ) | **Có** (vLLM trả top-20 logprob) |
| Hiệu chỉnh isotonic hoặc temperature | Chưa | Isotonic | Cả hai |

- **Lợi thế diễn giải của self-host:** có logprob, nên tính được điểm kỳ vọng và xác suất từng nhãn để chọn ngưỡng theo nhãn.
- **Track API:** chỉ có evidence và độ đồng thuận giữa các lần chạy.
- **Lưu ý:** thinking ẩn của reasoning model không phải là lời giải thích. Phải yêu cầu evidence thành một trường riêng trong output.

---

## 11. Hạn chế

1. **Mẫu quá nhỏ:** n = 32, khoảng tin cậy QWK rộng khoảng ±0.2. Prompt v2 và luật B1 tinh chỉnh trên chính tập này, nên số đo lạc quan và có rủi ro overfit cho leaderboard ẩn.
2. **Mỗi cấu hình chỉ chạy một lần:** chưa đo phương sai giữa các lần chạy của cùng cấu hình.
3. **Agent không giống hệt API production:**
   - Dự đoán sinh bằng agent Claude (Opus 5.5, Haiku 4.5), không qua API với temperature cố định.
   - Agent đọc nguyên file chứa cả đề (cả 15–17 bài trong một ngữ cảnh), nên có thể so sánh chéo giữa các bài. Pipeline thật chấm từng bài một.
4. **Luật không chắc đúng trên tập ẩn:** EX02 QWK 0.95 dựa trên quy luật rubric rút từ 17 bài; tập ẩn có thể không theo.
5. **Gold còn nhiễu và chưa định nghĩa nhãn:**
   - Ngoài 3 bài đã sửa, S112 (và có thể S108, S110) có dấu hiệu nhiễu.
   - Chưa có định nghĩa taxonomy chính thức (đã hỏi giảng viên).
6. **T3 chưa đo được:** chưa có thực nghiệm. Chỉ có thiết kế prompt và guard.

---

## 12. Bước tiếp theo, theo thứ tự lợi ích/chi phí

1. *(Đã sắp xếp lại, xem báo cáo phân tích `analysis-261006-1612-poc-results.md`.)* Hỏi giảng viên cách chấm EX01. Thêm tín hiệu "trọng số đề × câu có làm" và cho LLM chấm từng câu. Harness thực thi chỉ làm khi gold EX01 chấm theo chạy thật.
2. So khớp stdout thô cho EX02 (bắt được lỗi kiểu S205).
3. Ngưỡng bầu chọn theo từng nhãn và self-consistency k = 3 cho rubric EX01.
4. Few-shot retriever, đánh giá leave-one-out.
5. Self-host Qwen3.8-27B zero-shot trên cùng harness, rồi distill.
6. T3: prompt theo level + guard + LLM-judge.

## Cách chạy lại

```bash
cd eda
.venv/bin/python poc/blind_inputs.py --data-root <dataset> --out-dir <scratch>/blind
.venv/bin/python poc/rule_baseline.py --data-root <dataset> --out-dir <scratch>/preds
.venv/bin/python poc/rubric_constraints.py --data-root <dataset> [--gate-only] in.jsonl out.jsonl
.venv/bin/python poc/score_predictions.py --data-root <dataset> <scratch>/preds/*.jsonl --json scores.json
```

## Câu hỏi còn mở

1. S112 (gold 9, mọi cấu hình LLM chấm 1–2) có phải lỗi dữ liệu như S211/S214/S216 không?
2. Gold EX01 chấm thế nào? Metadata ghi không có hidden test. Gold khớp với tổng trọng số các câu có làm. Luật "P1 sai thì không chấm P2–P4" có còn áp dụng không?
3. Định nghĩa chính thức của 10 nhãn, nhất là ranh giới giữa logic, thuật toán và vòng lặp (đang chờ giảng viên).
4. Leaderboard có cho dùng dữ liệu tổng hợp hoặc distill để fine-tune không? Ngân sách GPU cho self-host là bao nhiêu?
5. Macro-F1 chính thức dùng `zero_division` = 0 hay 1? Điều này quyết định có nên dự đoán các nhãn hiếm hay không.
