# Thí nghiệm: LLM chấm từng câu, code nhân trọng số đề (EX01)

Bước 2 trong `analysis-261006-1612-poc-results.md` §7.
- **Code:** `poc/problem_scoring.py` (mới), `signals.statement_weights` (đọc trọng số từ đề), `poc/blind_inputs.py --per-sample`.
- **Bộ chạy:**
  - 48 agent chấm mù, mỗi agent chỉ thấy **1 bài**.
  - 5 agent phản biện dò tay lại trạng thái từng câu.

## 1. Thiết kế

- **LLM không cho điểm.** Với mỗi câu prob1..prob4, LLM chỉ trả:
  - `status` ∈ {correct, mostly, partial, broken, missing},
  - `infinite_loop`,
  - `error_label`,
  - một câu evidence.

  Mỗi câu được chấm độc lập, như unit test với input hợp lệ.
- **Code tính điểm:**
  - Tổng = Σ trọng số × điểm của câu. Trọng số đọc tự động từ đề (1/4/2/3), không viết cứng.
  - 6 chiều rubric được chia ra từ tổng theo thứ tự cố định.
  - Bài build lỗi đi qua cổng biên dịch, được 0.
- **Chốt trước khi chạy** (file `problem_scoring.py` ghi lúc 16:36, trước lần chạy đầu tiên), 4 cách quy đổi và cách chính:

  | Cách quy đổi | Câu được điểm khi |
  |---|---|
  | `attempt` | status ≠ missing |
  | **`runs` (cách chính đã chốt)** | correct / mostly / partial ("chỉ chấm chương trình chạy được") |
  | `graded` | correct = mostly = 1, partial = 0.5 |
  | `runs+prereq` | như `runs`, cộng luật ghi trong đề "P1 sai thì P2–P4 không chấm" |

- **Chạy trên 12 bài EX01 build được:**
  - Opus × 2 lần, để đo độ ổn định.
  - Haiku × 1 lần.
  - Đối chứng: prompt v2 cũ (LLM tự cho điểm), Opus, chạy **từng bài một**.

## 2. Kết quả (EX01: n = 15, gồm 3 bài build lỗi; "build được": n = 12)

| Cấu hình | EX01 QWK | EX01 MAE | Lệch ≤ 1 | QWK bài build được | QWK toàn tập (nhãn sửa) | QWK toàn tập (released) |
|---|---|---|---|---|---|---|
| Headline cũ: v2, cả đề trong một ngữ cảnh | 0.633 | 2.13 | 7/15 | 0.439 | 0.803 | 0.703 |
| Đối chứng: v2 chạy từng bài một | 0.587 | 2.27 | 6/15 | 0.405 | 0.782 | 0.686 |
| Trọng số đề × hàm có thân (regex, không LLM) | 0.726 | 1.53 | 12/15 | 0.557 | 0.817 | 0.724 |
| Theo câu, **`runs` (cách chính)**, Opus | 0.605 | 2.33 | 7/15 | 0.470 | 0.772 | 0.686 |
| Theo câu, `graded`, Opus r1 / r2 | 0.618 / 0.633 | 2.20 / 2.07 | 7–8 | 0.49 / 0.52 | 0.785 / 0.792 | 0.697 / 0.705 |
| Theo câu, `attempt` (Opus và Haiku cho kết quả **y hệt**) | 0.726 | 1.53 | 12/15 | 0.557 | 0.817 | 0.724 |
| Theo câu, `runs+prereq` (luật P1 đúng như đề) | **0.395** | 3.13 | 7/15 | 0.271 | 0.650 | 0.578 |
| *Khám phá:* `attempt+loop` (bỏ điểm câu có vòng lặp vô hạn) | **0.815** | **1.27** | 12/15 | — | **0.875** | **0.777** |
| *Khám phá:* `attempt+loop0` (có vòng lặp vô hạn thì cả bài 0 điểm) | 0.791 | 1.27 | 11/15 | — | 0.864 | 0.776 |

- **T2 tách khỏi T1:** dùng tổng điểm từ cách chấm theo câu, còn nhãn giữ nguyên bầu chọn ≥ 2/4. Macro-F1 vẫn 0.405 / 0.365.
- **Nhãn từ prompt chấm theo câu** (một lần chạy) kém hơn bầu chọn: macro-F1 toàn tập 0.32 so với 0.405. Nên giữ bầu chọn cho T2.
- **Bootstrap ghép cặp so với headline** (QWK toàn tập, 4000 lần):

  | Cấu hình | Nhãn sửa | Nhãn released |
  |---|---|---|
  | `attempt+loop` | +0.07, CI [−0.11, +0.30], tỉ lệ dương 0.76 | +0.07, CI [−0.10, +0.28], tỉ lệ dương 0.77 |
  | `attempt` | +0.02, tỉ lệ dương 0.55 | — |
  | `runs` | −0.03, tỉ lệ dương 0.26 | — |

- **Chính xác từng chiều EX01** (`attempt+loop` so với headline): io_format 1.00 so với 0.67; code_quality 0.80 so với 0.60; edge 0.67 so với 0.60; logic 0.53 bằng nhau; complexity 0.80 so với 0.87.

## 3. Phát hiện

1. **Giả thuyết chính đã chốt thất bại.** Quy đổi `runs` ("chỉ chương trình chạy được mới có điểm") cho 0.605: kém headline và kém cả regex. Gold EX01 **không** chấm theo độ đúng của code.
2. **Luật P1 ghi trong đề làm kết quả tệ hẳn (0.395).** Điều này xác nhận gold không áp luật đó.
3. **`attempt` trùng khớp tuyệt đối với regex "hàm có thân"** ở cả 3 lần chạy. Ở cách quy đổi này, LLM không thêm thông tin nào: regex đã làm được, rẻ hơn và tất định.
4. **LLM chỉ có ích khi phát hiện vòng lặp vô hạn**, theo luật trong đề "Nhận 0 điểm nếu … tạo vòng lặp vô hạn":
   - Opus r1, Opus r2 và Haiku cùng gắn cờ đúng 3 câu: P3 của S106, S107 và S112.
   - 5 agent phản biện dò tay **xác nhận cả 3 vòng lặp là thật**: biến chạy không bao giờ tiến lên.
   - Bỏ điểm câu bị lặp thì EX01 tăng từ 0.726 lên 0.815.
   - Đây là kết quả **tìm sau khi xem dữ liệu**, chỉ dựa trên 3 bài, nên phải coi là giả thuyết.
5. **Gold tự mâu thuẫn:**
   - S106 và S107 có vòng lặp vô hạn, gold = 2 (chỉ được compilable + io).
   - S112 cũng có vòng lặp vô hạn ở P3, và cả 4 câu đều được xác nhận là hỏng (mở file bằng chuỗi literal, vòng lặp P2 không bao giờ chạy, P4 ghi đè dữ liệu thay vì xoá), nhưng gold = 9.
   - Không cách quy đổi nhất quán nào khớp được cả ba bài. **S112 nhiều khả năng là lỗi dữ liệu**, nếu không thì luật vòng lặp vô hạn không được áp dụng đều.
6. **Rất ổn định:**
   - Opus r1 và r2 trùng 47/48 ô trạng thái. Tổng điểm trùng hệt với `attempt`, `runs` và `attempt+loop`.
   - Haiku cho cùng tổng điểm với `attempt` và `attempt+loop`. Tác vụ này **dùng model rẻ là đủ**.
7. **Hiệu ứng ngữ cảnh nhỏ:** v2 chạy từng bài một thấp hơn chạy cả đề khoảng 0.05 EX01 (0.587 so với 0.633). Headline cũ được lợi một chút nhờ so chéo giữa các bài; pipeline thật chấm từng bài sẽ mất khoảng chừng đó.

## 4. Đề xuất cho pipeline EX01 (multi_problem)

```
Cổng biên dịch ─▶ regex: hàm có thân (prob1..4) ─▶ LLM theo câu: status, infinite_loop, evidence (Haiku, từng bài một)
                                                     │
              tổng = Σ trọng số đề × [có làm ∧ không lặp vô hạn]  (tính bằng code) ─▶ chia ra 6 chiều
              status + evidence theo câu ─▶ giải thích kết quả, và làm input cho T3 (phản hồi theo câu)
```

- **T1:** dùng `attempt+loop` làm **ứng viên**. Chỉ chốt khi giảng viên xác nhận (a) cách chấm EX01 và (b) S112.
- **T2:** giữ bầu chọn nhãn ≥ 2/4. Nhãn từ prompt chấm theo câu (1 lần chạy) kém hơn.
- **Giữ status theo câu dù T1 chưa dùng hết:** nó ổn định (47/48), giải thích được, và là input tốt cho T3 (L2/L3: chỉ ra lỗi ở câu nào).
- **Chi phí:** 1 lần gọi Haiku mỗi bài, output ngắn (4 trạng thái). Rẻ hơn prompt v2.

## 5. Hạn chế

- Chỉ 12 bài. Lợi ích của luật vòng lặp đến từ 3 bài. Một trong ba (S112) lại đi ngược luật này.
- Cách quy đổi `attempt+loop` được chọn sau khi xem kết quả. Cần kiểm chứng trên dữ liệu mới, hoặc nhờ giảng viên xác nhận luật.
- Cách chia tổng ra 6 chiều là thứ tự cố định, chưa kiểm chứng riêng. Chiều logic vẫn chỉ đúng 53%.
- Agent chấm chưa phải API production với temperature cố định.

## Câu hỏi còn mở

1. S112: code hỏng cả 4 câu và có vòng lặp vô hạn, nhưng gold = 9. Đây có phải lỗi dữ liệu không?
2. Luật "vòng lặp vô hạn thì 0 điểm" áp cho cả bài hay chỉ cho câu đó? Gold S106 và S107 (= 2, chỉ có compilable + io) cho thấy mọi câu đều mất điểm logic.
3. Hàm có thân nhưng sai hoàn toàn (như P2 của S105) có được điểm không? Gold S105 = 2 cho thấy là không, trong khi S108 với P1 hỏng vẫn được tính.
