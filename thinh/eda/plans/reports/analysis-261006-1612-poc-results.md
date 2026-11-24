# Phân tích kết quả POC: QWK 0.80 · MAE 1.19 · macro-F1 0.41

**Nguồn số liệu:**
- `eda/poc/analyze_results.py`, chạy trên các file dự đoán của POC (32 mẫu). Báo cáo trước: `architecture-261006-1350-poc-leaderboard-pipeline.md`.
- Mọi kết luận dưới đây đã qua phản biện đối nghịch: 3 agent độc lập (thống kê; rò rỉ/vòng lặp; dự báo leaderboard) và 1 agent rà xem còn thiếu gì.
- Các con số các agent đưa ra đều được tính lại bằng code trước khi đưa vào báo cáo.

## 0. Kết luận ngắn

| Con số POC | Nó thực sự đo gì |
|---|---|
| **QWK 0.80** | Phần lớn đến từ luật, không từ LLM: 7/32 bài do cổng biên dịch quyết định, 14/17 bài EX02 lấy từ test. Hệ **không dùng LLM** (cổng + hằng số 6 cho EX01 + luật EX02) đã đạt **0.76**. LLM chỉ thêm khoảng +0.05 QWK (CI [−0.14, 0.23], chưa chắc chắn) |
| **EX02 QWK 0.95** | Đúng trên tập này nhưng mong manh: tính trên nhãn released chỉ 0.73, vì riêng bài S214 đã kéo +0.22. Luật được rút ra từ chính 17 bài này |
| **EX01 QWK 0.63, điểm yếu chính** | Đúng là điểm yếu, nhưng **nguyên nhân khác với chẩn đoán cũ**. Gold EX01 không chấm theo độ đúng của code mà gần với "**tổng trọng số 1/4/2/3 của các câu có làm**". Predictor đó không dùng LLM mà đạt EX01 QWK **0.73** (MAE 1.53), hơn LLM (0.63, MAE 2.13) |
| **MAE 1.19** | Trung bình của hai đề rất khác nhau: EX01 2.13, EX02 0.35. Trùng chính xác điểm tổng 53%, lệch ≤ 1 điểm 72% |
| **Macro-F1 0.41** | Ba nhãn có F1 = 0 (thuật toán, style, edge case). Gán thừa nhãn (67 nhãn dự đoán so với 44 nhãn gold). Riêng nhãn biên dịch (do cổng tạo ra) đóng góp 0.10 trong 0.405 |
| **Dự báo leaderboard** | QWK khoảng **0.55–0.70** nếu nhãn ẩn giống bản released; khoảng 0.65–0.78 nếu nhãn ẩn sạch như bản đã sửa. Macro-F1 khoảng 0.30–0.45 |

---

## 1. QWK 0.80 đến từ đâu

### 1.1 Tách theo loại bài

| Tập | n | QWK | MAE |
|---|---|---|---|
| Cả 32 bài | 32 | 0.80 | 1.19 |
| 7 bài build lỗi (cổng biên dịch quyết định) | 7 | đúng 7/7 (trên nhãn đã sửa) | 0 |
| 25 bài build được | 25 | 0.65 | 1.52 |
| EX01 build được | 12 | 0.44 | 2.67 |
| EX02 build được | 13 | 0.88 | 0.46 |

Lưu ý: QWK trên tập con không so trực tiếp được với QWK toàn tập. QWK trên trục cố định 0..10 tương đương Lin's CCC (QWK = 1 − MSE / (var_y + var_p + (ȳ − p̄)²)), nên phụ thuộc độ trải điểm của tập. Ví dụ: cùng các lỗi ở EX02 cho 0.95 trên 17 bài nhưng 0.88 trên 13 bài. Vì vậy cần đọc kèm MAE.

### 1.2 So với baseline không dùng LLM

| Predictor | Đọc code? | QWK | MAE | Lệch ≤ 1 | Headline hơn baseline (bootstrap ghép cặp) |
|---|---|---|---|---|---|
| Cổng + điểm trung bình theo đề (EX01 = 6, EX02 = 7) | Không | 0.65 | 1.84 | 15/32 | +0.16, CI [−0.06, 0.40] |
| Cổng + EX01 = 6 + luật test EX02 | Không | **0.76** | 1.41 | 20/32 | **+0.05, CI [−0.14, 0.23]**, tỉ lệ dương 0.72 |
| **Cổng + trọng số đề cho câu có làm (EX01) + luật test EX02** | Chỉ regex tìm thân hàm, không dùng LLM | **0.82** | **0.91** | **28/32** | −0.01, CI [−0.26, 0.23] |
| Headline (hybrid, có LLM) | Có | 0.80 | 1.19 | 23/32 | — |

**Ý nghĩa:**
- Baseline B1 trong báo cáo trước là "người rơm": nó đoán hằng số 2 cho mọi bài EX01 build được. Với baseline công bằng hơn, phần đóng góp của LLM vào T1 nhỏ và chưa có ý nghĩa thống kê.
- Predictor trọng số đề **không dùng tham số nào rút từ gold**. Trọng số 1/4/2/3 lấy nguyên từ đề bài; "câu có làm" là hàm có thân, dò bằng regex trong `signals.ex01_functions`.
- Lưu ý: trường `problems[].max_score` trong `exams.json` ghi 2.5 cho mỗi câu, mâu thuẫn với 1/4/2/3 trong đề.

---

## 2. EX01: vì sao LLM yếu, và gold thực ra chấm theo gì

### 2.1 LLM nén thang điểm

Trên 12 bài EX01 build được:

| | Giá trị |
|---|---|
| Gold | {1, 1, 2, 2, 2, **6, 8, 8, 9, 9, 9, 10**} (**hai cụm**, không có giá trị nào từ 3 đến 5) |
| LLM | {1, 1, 2, 3, 4, 4, 5, 5, 6, 8, 8, 8} (nằm nhiều ở **giữa thang**) |
| Phương sai | gold 12.2, LLM 6.1. Hệ số góc của dự đoán theo gold 0.35; Pearson r 0.49 |

- **Bài tốt bị chấm thấp:** mọi bài có gold ≥ 6 đều bị ít nhất 4/5 cấu hình chấm thấp hơn (S101, S102, S104, S108, S110, S112, S115).
- **Bài kém bị chấm cao:** S106 và S107 (gold 2) bị ít nhất 4/5 cấu hình chấm cao hơn.
- **Lấy trung bình nhiều lần chạy không giúp** (0.60 so với 0.63): trung bình làm co phương sai, mà QWK/CCC phạt chính sự co này.
- **Đổi prompt hay model gần như không tác động** trên EX01 build được: QWK 0.431–0.439 cho cả v1, v2, Opus và Haiku. Phần tăng v1 → v2 (0.56 → 0.63) đến từ 3 bài build lỗi mà mọi cấu hình đều chấm đúng 0.

### 2.2 Gold EX01 gần với "tổng trọng số các câu có làm"

| Bài | Gold | Trọng số đề × câu có làm | LLM | Ghi chú |
|---|---|---|---|---|
| S101 | 8 | 8 | 6 | Thiếu P3 (2 điểm), khớp tuyệt đối |
| S108 | 6 | 5 | 3 | "Bài nghi nhãn sai" trước đây, được giải thích |
| S110 | 8 | 8 | 4 | "Bài nghi nhãn sai" trước đây, khớp tuyệt đối |
| S112 | 9 | 10 | 1 | "Bài nghi nhãn sai" trước đây, lệch 1 |
| S113, S114 | 1, 1 | 1, 1 | 2, 5 | Chỉ làm P1 |
| S102, S104, S115 | 9, 9, 10 | 10, 10, 10 | 8, 8, 8 | Làm đủ 4 câu |
| **S105, S106, S107** | 2, 2, 2 | **5, 10, 10** | 1, 4, 5 | **Có thân hàm nhưng hỏng nặng.** Ở đây LLM gần gold hơn |

**Tổng hợp:**
- Predictor trọng số: 12/15 bài lệch ≤ 1 điểm, EX01 QWK 0.73.
- LLM: 7/15 bài lệch ≤ 1, QWK 0.63.
- Lấy trung bình hai predictor (chỉ để tham khảo, vì chọn sau khi đã xem dữ liệu): QWK 0.78.

**Hệ quả:**
1. **S108, S110, S112 nhiều khả năng không phải nhãn sai.** Gold chấm "câu nào có làm", còn LLM chấm "câu nào chạy đúng". Báo cáo trước giải thích các bài này bằng "hidden test". Giải thích đó **sai**: EX01 có `has_hidden_testcases: false`, và báo cáo kiến trúc đã được sửa.
2. Bỏ 3 bài này thì QWK tăng lên 0.79/0.86, nhưng **không được coi là "điểm thật"**. Bỏ 3 bài có sai số lớn nhất của bất kỳ predictor nào cũng làm QWK tăng (bỏ S112, S110, S114 cho 0.88).
3. **Harness thực thi cho EX01 bị hạ ưu tiên.** Nếu gold chấm theo việc có làm câu nào, chạy code sẽ không sửa được S108, S110, S112, mà còn có thể làm lệch thêm.
4. **Ba bài S105, S106, S107 cho thấy gold không chỉ đếm câu có làm.** Bài có thân hàm nhưng sai hoàn toàn thì gold vẫn cho thấp. Thiết kế hợp lý là: **trọng số đề × trạng thái từng câu do LLM phân loại** (làm được / hỏng / không làm), thay cho một điểm tổng LLM tự cho.

---

## 3. EX02: QWK 0.95 vì sao cao, và vì sao mong manh

| Kiểm tra | QWK EX02 |
|---|---|
| Luật B1r, nhãn đã sửa | **0.95** (14/17 bài trùng chính xác, 16/17 lệch ≤ 1) |
| Cùng luật, nhãn released | 0.73 (riêng S214 kéo +0.217; S211 và S216 chỉ thêm +0.004) |
| Logic tính theo strict pass thay vì lenient | 0.83 (released: 0.58), **thấp hơn LLM Opus v2 (0.87)** |
| Logic tính theo tỉ lệ pass (`round(4·pass/8)`) | 0.95 |
| io = mọi test strict pass, thay vì có ≥ 1 test strict pass | 0.94 |

**Vì sao cao:**
- 4/17 bài build lỗi đều đúng nhờ cổng biên dịch.
- 13 bài còn lại có 8 test công khai, và gold EX02 gần như là hàm của kết quả test.

**Vì sao mong manh:**
- Có khoảng 6 hằng số rút từ chính 17 bài: rubric khi build lỗi là 1, code_quality = 1, edge = logic//2, complexity = 1, ngưỡng 4/2/0, và cách đếm lenient pass.
- Mới thử nhiễu 2 trong 6 lựa chọn này.
- Gold không thuần là hàm của test:
  - S206 và S210 cùng crash 8/8 nhưng gold là 3 và 2.
  - S209 timeout 8/8 mà gold là 7.
- Chỉ chuyển sang tập ẩn tốt nếu tập ẩn dùng **đúng đề EX02 này và cùng bộ test**.

---

## 4. Macro-F1 0.41: giới hạn ở đâu

| Nhãn | Số gold | Số dự đoán | P | R | F1 |
|---|---|---|---|---|---|
| Lỗi biên dịch | 7 | 7 | 1.00 | 1.00 | 1.00 |
| Lỗi hard-code | 3 | 2 | 1.00 | 0.67 | 0.80 |
| Lỗi logic | 12 | 18 | 0.56 | 0.83 | 0.67 |
| Lỗi nhập/xuất | 4 | 7 | 0.43 | 0.75 | 0.55 |
| Lỗi vòng lặp | 6 | 10 | 0.30 | 0.50 | 0.38 |
| Lỗi hàm | 5 | 11 | 0.27 | 0.60 | 0.38 |
| Lỗi mảng/chuỗi | 1 | 6 | 0.17 | 1.00 | 0.29 |
| Lỗi thuật toán | 4 | 2 | 0 | 0 | **0** |
| Lỗi style | 2 | 1 | 0 | 0 | **0** |
| Lỗi edge case | 0 | 3 | 0 | 0 | **0** |

- **Bỏ nhãn biên dịch ra,** 9 nhãn còn lại do LLM gán chỉ đạt trung bình khoảng 0.34.
- **Gán thừa nhãn** (67 dự đoán so với 44 gold) làm giảm precision và micro-F1. Nhưng macro-F1 bị giới hạn chủ yếu bởi **recall và nhầm lẫn ở các nhãn hiếm**: siết ngưỡng bầu chọn lại làm macro-F1 giảm (≥ 1/4: 0.43; ≥ 2/4: 0.41; ≥ 3/4: 0.36).
- **"Lỗi thuật toán" có 4 mẫu gold mà không trúng mẫu nào.** LLM ghi "Lỗi logic" thay vào. Đây là vấn đề định nghĩa nhãn.
- **Quy ước `zero_division`:**
  - Riêng đổi `zero_division=1` thì vẫn ra 0.405.
  - Lên được 0.505 chỉ khi **vừa** dùng `zero_division=1` **vừa** không bao giờ dự đoán "Lỗi edge case". Đó là chính sách dựa trên việc biết trước gold không có nhãn này; trên tập ẩn có nhãn này thì sẽ mất điểm.
- **Sửa nhãn cũng làm tăng T2 một cách cơ học:**
  - Gold đã sửa được thêm nhãn suy ra từ việc sửa (S214 thêm "Lỗi biên dịch"; S211, S216 thêm "Lỗi nhập/xuất").
  - Hậu xử lý cũng chèn đúng các nhãn đó vào dự đoán theo cùng quy tắc.
  - Macro-F1 trên nhãn released: 0.365; nhãn đã sửa: 0.405.

---

## 5. Có tin được không: các nguồn lạc quan

| Nguồn | Đo được? | Độ lớn |
|---|---|---|
| **Sửa nhãn theo một chiều** | Có | Cả 3 lần sửa đều kéo gold về đúng dự đoán của pipeline. Pipeline tự phát hiện chúng qua tín hiệu của mình. Các bất đồng theo chiều ngược lại (S209, S205) vẫn giữ nguyên, còn lỗi gold trùng với dự đoán thì không thể thấy. QWK toàn tập +0.10, macro-F1 +0.04. **Nên báo cáo số trên nhãn released làm số chính** |
| Chọn cấu hình tốt nhất trong 15 | Có, có đối chứng | Chọn trên in-bag 0.81 → out-of-bag 0.74. Nhưng headline cố định cũng chỉ đạt OOB 0.79 trên cùng các mẫu, nên **hiệu ứng chọn ≈ 0** (OOB của cấu hình được chọn − OOB của headline = −0.05). Khoảng chênh chủ yếu do QWK bị lệch khi chỉ có ~12 bài |
| **Rò rỉ thiết kế** (prompt v2, hằng số luật EX02, lenient pass, quy tắc "dòng nhắc thừa thì io = 0") | Một phần | Prompt v2 đóng góp +0.04 trong hybrid; lenient pass +0.12 trên EX02; sửa nhãn +0.10. Bootstrap không tính được phần này |
| LLM chấm cả đề trong một ngữ cảnh | Chưa | Agent thấy 15–17 bài cùng lúc nên có thể so chéo. Pipeline thật chấm từng bài một |
| Mỗi cấu hình chỉ chạy một lần | Chưa | Chưa có độ dao động giữa các lần chạy |
| Tỉ lệ bài build lỗi trên tập ẩn | Chưa | Ở đây là 7/32 (Wilson 95% khoảng 0.11–0.39). Riêng cổng đã cho 0.58. QWK toàn tập phụ thuộc mạnh vào tỉ lệ này |

---

## 6. Dự báo trên leaderboard ẩn (cùng hai loại đề)

| Kịch bản | QWK toàn tập | EX01 | EX02 | MAE | Macro-F1 |
|---|---|---|---|---|---|
| (a) Nhãn ẩn giống bản released (có nhiễu) | **0.55–0.70** | 0.45–0.65 | 0.65–0.80 | 1.6–1.9 | 0.30–0.37 |
| (b) Nhãn ẩn sạch như bản đã sửa | 0.65–0.78 | 0.55–0.70 | 0.80–0.95 | 1.2–1.5 | 0.35–0.45 |
| (c) = (a) + thêm tín hiệu trọng số đề cho EX01 (chưa kiểm chứng ngoài mẫu) | khoảng 0.65–0.78 | 0.60–0.75 | như (a) | thấp hơn | như (a) |

- Đây là **ước lượng theo phán đoán**, neo từ số trên nhãn released (0.70) và nhãn đã sửa (0.80), trừ đi rò rỉ thiết kế. Không phải khoảng tin cậy tính toán.
- **Một bài gold bỏ qua lỗi build** (như S214 bản released) làm mất khoảng 0.2 QWK trên EX02. Hỏi BTC xem nhãn ẩn có được sửa giống vậy không là câu hỏi có tác động lớn nhất.

---

## 7. Việc nên làm tiếp (đã xếp lại theo phân tích)

| # | Việc | Công sức | Vì sao (số liệu) |
|---|---|---|---|
| 1 | **Hỏi giảng viên/BTC:** (a) EX01 chấm thế nào (trọng số 1/4/2/3 hay 2.5; hàm chưa xong; vòng lặp vô hạn; còn `main`); (b) định nghĩa 10 nhãn; (c) macro-F1 tính thế nào và `zero_division`; (d) nhãn ẩn có được sửa như S214 không | Vài giờ | Trọng số đề giải thích 12/15 bài EX01; "thuật toán" F1 = 0; một bài gold lệch cổng = −0.2 EX02 |
| 2 | **Thêm tín hiệu "trọng số đề × câu có làm" cho EX01.** Prompt LLM chỉ cho **trạng thái từng câu** (làm được / hỏng / không làm); điểm = Σ trọng số × trạng thái, tính bằng code | 1 ngày | Predictor trọng số EX01 0.73 so với LLM 0.63; kết hợp 0.78 (tham khảo). **Đã làm**, xem `experiment-261006-1641-ex01-per-problem-grading.md`. Cách chính đã chốt (`runs`) thất bại (0.605). Cách khám phá `attempt+loop` đạt 0.815, nhưng gold tự mâu thuẫn ở S112 |
| 3 | **Nhờ chấm lại mù một tập ngẫu nhiên** (có cả bài đoán đúng lẫn bài bị chấm dư như S107, S114), thay vì chỉ xác nhận các bài mô hình bất đồng | Phụ thuộc giảng viên | Tránh sửa nhãn một chiều (§5) |
| 4 | **T2:** bảng nhầm lẫn giữa logic, thuật toán, vòng lặp và hàm; giới hạn tổng số nhãn; tách cấu hình T2 khỏi T1 | 0.5 ngày | 9 nhãn do LLM gán chỉ trung bình 0.34; dự đoán 67 so với 44 |
| 5 | Chạy lại EX01 **từng bài một mỗi lần gọi**, 3 lần, để đo phương sai và hiệu ứng ngữ cảnh | 0.5 ngày | Chưa đo (§5) |
| 6 | Báo cáo **nhãn released là số chính**, nhãn đã sửa là mức trần | Ngay | §5 |
| 7 | Harness thực thi EX01: **chỉ làm nếu (1) xác nhận gold chấm theo chạy thật.** Test phải viết từ đề và chốt trước khi chấm thử | 2–3 ngày | Bị hạ ưu tiên vì §2.2 |
| — | Self-host Qwen | Đưa ra khỏi lộ trình leaderboard | Model và prompt không làm thay đổi EX01 trên các bài build được (§2.1) |

---

## 8. Kết quả phản biện các nhận định (C1–C10)

| # | Nhận định ban đầu | Kết luận | Sửa lại |
|---|---|---|---|
| C1 | QWK 0.80 phóng đại khả năng hiểu code | **Có điều kiện** | Đúng chiều. Baseline công bằng là 0.76 (không phải 0.58 kiểu leave-one-out). 7/7 bài build lỗi chỉ đúng trên nhãn đã sửa |
| C2 | EX01 chỉ còn 0.44 trên bài build được | **Có điều kiện** | Riêng S112 chiếm 49% tổng bình phương sai số. Bằng chứng tốt hơn là MAE và baseline không đọc code |
| C3 | Lỗi EX01 có hệ thống | **Có điều kiện** | Đúng, và dạng của nó là **nén thang điểm** (không chỉ ở 3 bài nghi nhãn sai). Chấm thấp 3–8 điểm (không phải 2–8) |
| C4 | EX01 bị vài bài chi phối | **Xác nhận** | Nhưng 0.79/0.86 không phải điểm thật; bỏ 3 bài có sai số lớn nhất của bất kỳ predictor nào cũng tăng |
| C5 | EX02 0.95 mong manh | **Xác nhận** | Strict pass cho 0.83, thấp hơn LLM 0.87, tức lợi thế của luật bị đảo ngược |
| C6 | Sửa nhãn không vòng lặp | **Có điều kiện** | Không thiên vị luật so với LLM, nhưng **có hướng** (luôn kéo gold về đúng dự đoán) |
| C7 | Macro-F1: giới hạn do 3 nhãn F1 = 0 và gán thừa | **Có điều kiện** | 0.505 cần cả `zero_division=1` lẫn chính sách dựa trên gold; macro-F1 bị giới hạn bởi recall nhiều hơn gán thừa |
| C8 | Hybrid thắng B1 | **Có điều kiện** | B1 là baseline yếu. So với baseline công bằng, mức hơn khoảng +0.05 và chưa chắc chắn |
| C9 | Selection optimism | **Có điều kiện** | Có đối chứng thì hiệu ứng chọn ≈ 0. Rò rỉ thiết kế mới là phần lớn. Phải tách kịch bản nhãn released và nhãn sạch |
| C10 | Ưu tiên: harness thực thi EX01 trước | **Bác bỏ thứ tự** | Hỏi cách chấm EX01 trước. Thêm tín hiệu trọng số đề. Harness có điều kiện |

---

## Câu hỏi còn mở

1. EX01 chấm theo trọng số 1/4/2/3 cho câu có làm, hay theo chạy đúng? Vì sao `problems[].max_score` ghi 2.5?
2. Nhãn ẩn có được sửa như S214, S211, S216 không?
3. Macro-F1 chính thức: tính theo nhãn trên ma trận multilabel hay theo từng mẫu? `zero_division` bằng bao nhiêu?
4. Định nghĩa chính thức cho logic, thuật toán, vòng lặp, hàm và style.
5. Tập ẩn có dùng đúng đề EX01/EX02 này và cùng bộ test công khai không?
