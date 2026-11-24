# EDA bộ mẫu 32 bài: chấm điểm, phân loại lỗi và phản hồi bài C++

- **Ngày:** 2026-10-03.
- **Dữ liệu:** `sample_dataset/`, gồm 32 bài: EX01 (multi_problem) 15 bài, EX02 (single_problem) 17 bài. README ghi rõ đây **không phải** train/test chính thức.
- **Cách làm:**
  - 5 agent phân tích song song, mỗi agent tự viết script và chạy trên dữ liệu thật.
  - Thêm **5 agent kiểm chứng độc lập**. Mỗi agent viết code mới để tính lại từng con số và tìm cách bác bỏ.
  - Kết quả: không phát hiện nào bị bác bỏ. **57/70 phát hiện được sửa lại câu chữ**, phần lớn vì diễn giải quá đà hoặc sai mẫu số.
  - Báo cáo này chỉ dùng **bản đã sửa**.
- **Chạy lại:** `./run-eda.sh <data-root> [--with-exec]` (xem §8).
- **Hạn chế:**
  - n = 32, nên mọi kết luận cho tập đầy đủ đều là **[GT]** (giả thuyết).
  - Biên dịch lại bằng Apple clang, không phải GNU g++ 13.
  - Nhãn level của T3 do một người (agent) gán tay, chưa đo độ đồng thuận.
- **Bản brainstorm trước:** [brainstorm-261003-2159-llm-grading-challenge.md](../../../plans/reports/brainstorm-261003-2159-llm-grading-challenge.md). §7 liệt kê những gì EDA này làm thay đổi.

---

## 0. Mười điều quan trọng nhất

1. **Dữ liệu rất nhỏ.**
   - README nhắc "4/75 bài EX02" và "5/46 bài EX01", nên tập đầy đủ có lẽ chỉ khoảng **121 bài** [GT]. Tập test vì vậy chỉ vài chục bài.
   - Metric sẽ rất nhiễu. Ví dụ: baseline luật biên dịch đạt QWK 0.51 với **CI95 ≈ [0.23; 0.76]**, và bỏ đi 1 mẫu là QWK đổi ±0.05.
   - Kế hoạch fine-tune head 8B cho T1/T2 trên khoảng 90 bài cần xem lại (cổng D0 ở §7).
2. **Nhãn T1 là rubric 6 chiều, không phải điểm thi chấm theo câu.**
   - total = Σ6 chiều ở 32/32 bài. Không có điểm từng câu.
   - Rubric có rất ít bậc tự do thật:
     - **EX02:** edge_case = logic/2 (17/17); code_quality luôn bằng 1; logic chỉ nhận {0, 2, 4}.
     - **EX01:** complexity = code_quality = [logic > 0] (15/15).
3. **Gold EX01 không theo chính sách tiên quyết đã ghi trong đề.**
   - Có 6 bài mà feedback nói P1 sai, nhưng total trải từ 1 đến 9; 3 bài được 6, 8 và 9 điểm.
   - Cùng một lỗi P1 ("assume n"): S105 được 2 điểm, S110 được 8.
   - → **Policy engine áp cứng theo đề sẽ làm tụt QWK so với gold.**
4. **EX02-S214: file bị comment toàn bộ (lỗi link, không có main) mà gold vẫn 10/10.** Bỏ comment ra chạy thì chính đoạn code đó chỉ pass 6/8 test. Nhãn T2 chỉ có `[Lỗi style]`.
5. **io_format ngược với harness test.**
   - Bài in prompt không bị trừ: S211 pass 0/8 vẫn được 10/10.
   - Bài in thừa một dấu cách thì bị trừ: S205 pass 8/8 mà `io_format=0`.
   - `test_report` đã strip khoảng trắng và không ghi returncode. `actual=''` thực chất là crash do hành vi không xác định (UB).
6. **T2: nhãn "Lỗi edge case" có 0/32 positive.**
   - Oracle đạt macro-F1 **0.90 hoặc 1.00 tuỳ `zero_division`**.
   - Ở `zd=0`, baseline "đoán cả 10 nhãn" (0.214) thắng một luật hợp lý (0.158).
   - Chỉ nhãn "Lỗi biên dịch" giải được bằng luật (6/6).
7. **Ràng buộc chéo T1↔T2 rất mạnh:**
   - `[]` ⇔ total = 10 ở 31/32 bài;
   - total < 10 ⇒ có ít nhất 1 nhãn (27/27).
8. **Input T3 chứa nguyên đáp án T2** (32/32, cả thứ tự) và **không có test_report**.
   - README ghi đây là thiết kế có chủ ý. Nó chỉ thành rò rỉ nếu test T2 và test T3 dùng chung bài.
9. **Gold T3 yếu.**
   - Level gần như do exam và điểm quyết định: EX01 toàn L2; ở EX02, L1 khi và chỉ khi total = 10. Không có L4.
   - Gold chỉ đúng định nghĩa level của chính label_space ở **36–61%** số bài, tuỳ cách đọc.
   - Theo quy tắc compliance hiện tại, output hằng số "Tốt." đạt **100%**, cao hơn cả gold.
   - Feedback EX01 là **ghi chú của người chấm** (gọi "SV" ở ngôi thứ ba, viết tắt, dán log), không phải lời viết cho sinh viên.
10. **compile_log do BTC sinh lại** (GNU g++ trên Linux, file `/tmp/tmp*/student.cpp`), khác môi trường của giảng viên.
    - S109: feedback ghi `student.h:91`, log ghi `.cpp:90`.
    - Có 1 log bị cắt ở đúng 2000 ký tự.
    - Clang local đồng ý trạng thái lỗi 32/32.
    - Không thấy PII sót, không thấy prompt injection.

---

## 1. Tổng quan dữ liệu

| Thành phần | Nội dung |
|---|---|
| `exams.json` | 2 đề, có đủ `statement` (đề bài nằm trong dữ liệu), metadata, `problems[]`, `grading_policy` |
| EX01 | multi_problem, môn CTDL&GT tuần 3. 4 hàm `prob1..prob4` trên struct General/Node (đọc file, linked list). Bài nộp **không có main**. `compile_log` sinh bằng `-fsyntax-only`, `test_report=null`. **Không kèm `generals.csv` hay test nào**, nên không chạy được |
| EX02 | single_problem, môn CSLT tuần 5, kiểm tra số Armstrong. stdin/stdout, 8 test công khai, `has_hidden_testcases=true`, "không được dùng pow" |
| `label_space.json` | Rubric 6 chiều; 10 nhãn; 4 level kèm "định nghĩa" và "cấm" |
| 3 file task | Cùng 32 sample_id, cùng thứ tự. Input T1 giống hệt T2. **T3 thêm `taxonomy_error` + `target_feedback_level` và bỏ `test_report`** |
| Định dạng file nộp | **Không được mô tả ở đâu** (0 lần nhắc) |

**Token** (ước lượng chars/3.5, với tiếng Việt và code đây là cận dưới):

| exam | đề | code p50/p95 | prompt T1/T2 p50/p95/max | feedback p50/p95 |
|---|---|---|---|---|
| EX01 | 635 | 889/1346 | 1582/1982/2309 | 39/59 |
| EX02 | 188 | 162/269 | 537/673/766 | 14/33 |

→ Prompt dưới 2.5k token. **Không cần tách câu vì lý do context.**

---

## 2. Task 1: rubric

### 2.1 Phân bố

| exam | vector (comp, io, logic, edge, cplx, qual) → total (n) |
|---|---|
| EX01 | 000000→0 (3, đều lỗi biên dịch) · 100000→1 (2) · 110000→2 (3) · 111111→6 (1) · 112211→8 (2) · 113211→9 (2) · 114111→9 (1) · 114211→10 (1) |
| EX02 | 000001→1 (3, đều lỗi biên dịch) · 100001→2 (1) · 100011→3 (1) · 104211→9 (1) · **112111→7 (7)** · 114211→10 (4) |

- Mọi giá trị đều nguyên, nằm trong miền, không thiếu. total = Σ6 chiều ở 32/32 bài.
- EX02 chỉ có 6 mức total: {1, 2, 3, 7, 9, 10}.
- 7/17 bài EX02 có cùng vector 112111, dù pass từ 0 đến 6/8 test và 7 feedback khác nhau. Điều này gợi ý **chấm theo khuôn 3 bậc** (đúng hết / đúng một phần / hỏng).

### 2.2 Cổng biên dịch xử lý khác nhau theo exam

- Bài không biên dịch được: **EX01 cho 000000** (3/3), **EX02 cho 000001** (code_quality vẫn bằng 1, 3/3).
- Điều này khớp với đề: chỉ statement EX01 có điều khoản "0 điểm". Vì vậy đây là **quy ước theo exam**, không phải lỗi.
- → Hậu xử lý T1 phải theo `exam_type`.

### 2.3 EX02: tín hiệu test

- **Tương quan với total** (Spearman, 13 bài có test):
  - so khớp chặt (strict): ρ = 0.57 (p = 0.04);
  - so khớp nới (bỏ prompt, không phân biệt hoa/thường): **ρ = 0.89**.
  - Tính cả 4 bài không có test (n = 17): 0.52 và 0.72.
- 3/13 bài pass 0/8 test vẫn được ≥7 điểm:
  - S211: in prompt.
  - S216: in prompt, nới ra thì chỉ đúng 3/8.
  - S209: TIMEOUT do biến chưa khởi tạo.
- **Phân loại 57/104 test fail:**

| Loại | Số test |
|---|---|
| Sai đáp án | 17 |
| Output rỗng (crash do UB) | 16 |
| Chỉ sai vì có prompt | 11 |
| Timeout | 8 |
| Prompt + sai đáp án | 5 |

- **Test công khai yếu:**
  - Bộ test có 6 True và 2 False, nên in hằng "True" đã pass 6/8.
  - S208 luôn in True (`if (is_armstrong_number)` là con trỏ hàm) mà vẫn được 7 điểm.
  - Input dài nhất 4 chữ số, trong khi ràng buộc là ≤10^9.
  - Tuy vậy, thêm 18 input lớn và biên **không đổi kết luận cho bài nào trên mẫu**. Rủi ro test che lỗi edge chỉ là [GT].

### 2.4 EX01: điểm theo câu so với rubric

![EX01 rubric so với policy](../../out/t1_/t1_ex01_rubric_vs_policy.png)

| Bài | P1–P4 (đọc code + feedback) | logic/edge | total gold | Theo policy 1/4/2/3 |
|---|---|---|---|---|
| S101 | đúng / đúng / không làm / sai | 2/2 | 8 | 5 |
| S102 | đúng / **sai (list rỗng, câu 4 điểm)** / đúng / đúng | **4**/1 | 9 | 6 |
| S108 | **P1 sai** (tiên quyết) | 1/1 | **6** | 0 |
| S110 | **P1 sai** (tiên quyết) | 2/2 | **8** | 0 |
| S112 | **P1 sai**, prob2 rỗng, prob4 không xoá, **prob3 lặp vô hạn** | 3/2 | **9** | 0 |
| S105, S113, S114 | P1 sai | 0/0 | 2, 1, 1 | 0 |
| S106, S107 | prob3 lặp vô hạn ("theo quy định là 0") | 0/0 | 2 | 0 |
| S115 | đúng cả 4 | 4/2 | 10 | 10 |

**Kết luận:**
- **Gold không áp chính sách cổng P1 một cách nhất quán.**
  - Bằng chứng không phụ thuộc trọng số: cùng lỗi "assume n", S105 được 2 còn S110 được 8.
  - Cùng nhóm "P1 sai, tiên quyết": S113/S114 được 1, còn S108/S110/S112 được 6–9.
- Luật "lặp vô hạn → 0" được áp **theo câu**: S106/S107 vẫn được 2 nhờ compilable + io. Riêng S112 cũng lặp vô hạn mà vẫn được logic = 3.
- **Trọng số mâu thuẫn:**
  - statement và README ghi 1/4/2/3;
  - `problems[].max_score` ghi 2.5 × 4.
  - Comment khung đề còn sót trong S102 ủng hộ bộ 1/4/2/3.
- **io_format của EX01 mang nghĩa khác:** io = 0 chỉ xảy ra khi P1 crash (stoi/stof ở S113, S114). Trong khi S108 đọc `cin` thay vì đọc file, S112 mở file bằng chuỗi cố định "filename", mà cả hai vẫn được io = 1.
- **Bài nộp hay thiếu hàm và hay còn main:**
  - Chỉ 6/15 bài có đủ 4 hàm. P2/P3/P4 thiếu lần lượt ở 4/8/6 bài.
  - Không bài nào đổi prototype.
  - 10/15 bài còn main dù đề cấm, và không bị trừ điểm (S115 được 10).

### 2.5 Metric QWK trên mẫu

| Baseline (không dùng gold làm input) | QWK pooled | EX01 | EX02 | MAE |
|---|---|---|---|---|
| Hằng số hoặc median | 0 | 0 | 0 | 3.2–3.5 |
| Median theo exam (chỉ dùng exam_id) | **0.219** | 0 | 0 | 2.94 |
| Luật biên dịch (lỗi → 0, ngược lại 10) | 0.512 (CI95 ≈ 0.23–0.76) | 0.413 | 0.636 | 2.91 |
| `rubric_prior` (EX02: 2 + 8·pass nới) | 0.487 | 0.413 | 0.691* | 2.47 |

\* Lạc quan: dạng công thức được chọn sau khi đã xem gold của chính 17 bài.

**Ba điều rút ra:**
1. **Gộp 2 exam làm QWK dương giả.** Baseline chỉ dùng exam_id đạt 0.219 pooled, nhưng bằng 0 trong từng exam. → Luôn báo QWK theo từng exam.
2. **Tham số `labels` của sklearn đổi kết quả theo cả hai chiều.**
   - Đổi 7 bài EX02 từ 7 điểm xuống 3: QWK 0.931 nếu `labels=None`, 0.738 nếu `labels=0..10`.
   - Ví dụ đồ chơi: 0.600 so với 0.976.
   - → Cố định `labels=list(range(11))`.
3. **Nhãn nhiễu kéo metric.** Bỏ 4 nhãn nghi sai (S214, S108, S110, S112) làm `rubric_prior` tăng 0.487 → 0.556. Bắt đúng lỗi link của S214 lại làm luật biên dịch *giảm* 0.512 → 0.429, vì gold của S214 là 10.

Ngoài ra, exact-match từng chiều bị thổi phồng: đoán mode cho EX02 đã đúng 100% ở code_quality và 82% ở compilable. → Luôn báo kèm baseline mode.

---

## 3. Task 2: taxonomy lỗi

### 3.1 Prevalence (n = 32)

| Nhãn | n | EX01/15 | EX02/17 |
|---|---|---|---|
| Lỗi logic | 12 | 5 | 7 |
| Lỗi biên dịch | 6 | 3 | 3 |
| Lỗi vòng lặp | 6 | 2 | 4 |
| Lỗi hàm | 5 | 2 | 3 |
| Lỗi thuật toán | 4 | 2 | 2 |
| Lỗi hard-code | 3 | 3 | 0 |
| Lỗi nhập/xuất | 2 | 0 | 2 |
| Lỗi style | 2 | 0 | 2 |
| Lỗi mảng/chuỗi | 1 | 1 | 0 |
| **Lỗi edge case** | **0** | 0 | 0 |

- Số nhãn/bài: trung bình 1.28, tối đa 3. Tập rỗng 4/32.
- Có 18 tổ hợp nhãn, 11 tổ hợp chỉ gặp 1 lần.
- Mỗi exam vắng 3/10 nhãn, nhưng Fisher p ≥ 0.09 cho mọi nhãn, nên "nhãn phụ thuộc đề" mới là [GT].
- Chuỗi nhãn sạch: 41/41 khớp chính xác label_space, đều NFC. Lưu ý cài đặt: cả 10 nhãn đều đổi khi chuyển sang NFD, nên parser phải chuẩn hoá NFC.
- **"Lỗi hard-code" trên mẫu** nghĩa là gán cứng nguồn dữ liệu (đường dẫn file, kích thước mảng, đọc `cin` thay file) ở P1 của EX01 (3/3). Không bài nào gán cứng output theo test.

### 3.2 Nhất quán với rubric và tín hiệu

| Cặp | Kết quả |
|---|---|
| Biên dịch ⇔ compilable = 0 ⇔ log có diagnostic error | **6/6, κ = 1.0**. Tính thêm lỗi link thì có 1 dương tính giả (S214), nên lỗi link chỉ nên là feature |
| `[]` ⇔ total = 10 | **31/32** (κ = 0.87; ngoại lệ S214). total < 10 ⇒ có nhãn: **27/27**. compilable và logic < 4 ⇒ có nhãn: 19/19 |
| `[]` ⇔ pass 8/8 | κ = 0.57: yếu hơn ràng buộc qua total |
| Nhập/xuất ~ io_format = 0 | Phủ 2/11 (6/11 là bài lỗi biên dịch). Ở EX01, io = 0 đi với "Lỗi hàm", không với nhập/xuất |
| Edge case ~ edge_case < 2 | 0 nhãn so với 22 bài. Nhưng 21/22 bài đó là trừ dây chuyền theo logic (Spearman 0.93); chỉ S102 bị trừ riêng edge_case |
| Style ~ code_quality = 0; thuật toán ~ complexity = 0 | **Không trùng bài nào.** code_quality = 0 và complexity = 0 chỉ xuất hiện khi điểm bị kéo về 0 dây chuyền |

**→ Không suy nhãn T2 từ chiều rubric**, trừ nhãn "Lỗi biên dịch" và ràng buộc liên quan tập rỗng.

### 3.3 Macro-F1

| Bộ dự đoán | zd = 0 | zd = 1 | zd = nan | micro-F1 |
|---|---|---|---|---|
| Luôn `[]` | 0.000 | 0.100 | 0.000 | 0 |
| **Đoán cả 10 nhãn** | **0.214** | 0.214 | 0.214 | 0.227 |
| Luật biên dịch, ngược lại {logic} | 0.158 | 0.258 | 0.175 | **0.466** |
| Oracle (= tra input T3) | **0.900** | 1.000 | 1.000 | 1.000 |

- `zero_division` mặc định của sklearn ("warn") cho kết quả như `zd=0`: mô hình hoàn hảo bị chặn ở 0.9.
- Thứ hạng chỉ đảo giữa mô hình **có** và **không** đoán nhãn vắng.
- Macro-F1 thưởng việc đoán tràn với nhãn hiếm. Đây là tính chất đã biết của metric: F1 khi đoán toàn positive là 2p/(1+p).
- **Xác suất nhãn hiếm vắng trong test,** với N ≈ 36 [GT, khoảng 30% của 121 bài]:

| Nhãn | P(không có positive nào) | P(≤2 positive) |
|---|---|---|
| Lỗi edge case | **0.58** | — |
| Lỗi mảng/chuỗi | 0.32 | — |
| Lỗi nhập/xuất, Lỗi style | — | 0.61 |
| Lỗi hard-code | — | 0.33 |

  → Gần như chắc chắn test có nhãn không có support, nên quy ước `zd` của scorer sẽ quyết định điểm.

---

## 4. Task 3: phản hồi

### 4.1 Level bị gắn với exam và điểm

| level | EX01 | EX02 |
|---|---|---|
| L1 | 0 | 4 (cả 4 bài đều total = 10) |
| L2 | 15 | 3 |
| L3 | 0 | 10 |
| L4 | **0** | **0** |

- EX02: L1 ⇔ total = 10 (Fisher p = 0.0004).
- Luật hậu nghiệm "exam + total" đoán đúng level 29/32 trong mẫu; chỉ dùng exam được 25/32.
- **Thiếu tổ hợp:**
  - không có L4;
  - L1 cho bài có lỗi chỉ có 1 bài (S214);
  - không exam nào có đủ các level.
- README đã thừa nhận đây là bias của mẫu nhỏ. Cần kiểm lại trên tập đầy đủ.

### 4.2 Gold có đúng định nghĩa level của label_space không? (28 bài có nhãn lỗi)

![target vs observed](../../out/t3/t3_target_vs_observed_level.png)

- **Tỷ lệ tuân thủ:** 10/28 (36%) nếu đọc chặt, 17/28 (61%) nếu đọc nới. Kết quả do một người gán, chưa đo độ đồng thuận.
- **Các phát hiện đúng theo cả hai cách đọc:**
  - **L2 đưa cách sửa:** S104 ("chỉ cần lấy abs(năm mất - năm sinh)") và S110 ("phải đếm số dòng để có n").
  - Ca ranh giới: S106/S107 nêu đúng câu lệnh còn thiếu (`cur = cur->next`).
  - **L3 thực chất là lời giải:** S207 ghi "bỏ chữ int ở dòng số 9 là được". Áp gợi ý đó vào code thì từ 4/8 lên **8/8 test** (đã chạy thật).
  - **L3 không có hướng sửa:** 5–8/10 bài; chỉ S206 và S207 có hướng sửa rõ.
  - **L2 của EX02 không nêu nguyên nhân:** 3/3 bài (ví dụ S210: "Chưa hoàn thiện.").

### 4.3 Metric (a) chỉ phạt vượt mức

| Output | Compliance theo quy tắc README |
|---|---|
| "Tốt." / "Chưa hoàn thiện." / "Em xem lại bài nhé." (hằng số) | **1.00** |
| Gold | 0.91 (đọc chặt: S104, S106, S107) · 0.97 (đọc nới) · 0.81 (áp mọi điều cấm của label_space) |

- Mô tả metric: "không được **vượt** mức, **VD** Level 1–2 mà đưa code sửa là vi phạm".
- Không có yêu cầu tối thiểu cho từng level. Các điều cấm khác trong label_space (L1 nêu nguyên nhân, L2 nêu cách sửa, L3 đưa lời giải) **chưa được định nghĩa vận hành**.
- Output hằng số sẽ bị (b) phạt, nên lỗ hổng chỉ nghiêm trọng nếu (a) được xếp hạng riêng.

### 4.4 Gold là tham chiếu yếu cho (b)

- Metric (b) được mô tả là "LLM-as-judge + chấm người trên mẫu con".
- Trong 28 bài có lỗi:
  - 4 bài không có chẩn đoán nào (S203, S208, S210, S217);
  - 4–5 bài chỉ nêu vị trí.
- Phủ nhãn: 80–90% (33–37/41). S102 và S210 không phủ nhãn nào.
- Có chẩn đoán sai so với code phát hành:
  - S109 lệch 1 dòng;
  - S209 nói "bỏ qua chữ số đầu tiên", trong khi code thực tế bỏ chữ số cuối.
- **Văn phong:**
  - EX01: 0/15 xưng "em", 9/15 gọi "SV", 7/15 nói về cách chấm, 3/15 dán log, khoảng 22 token viết tắt.
  - EX02: 9/17 xưng "em", 8/17 cụt và không xưng hô.
  - Động viên thật: 1/32. "Tốt." đúng nguyên văn: 3/32.
  - → **SFT thẳng trên gold sẽ học văn phong ghi chú của người chấm.**
- **Độ dài:** phụ thuộc cả exam lẫn level. Trong EX02, độ dài tăng đơn điệu theo level (trung vị 1 → 5 → 15.5 từ; ρ = 0.77). Nhưng L2 của EX01 (21 từ) lại dài hơn L3 của EX02.

### 4.5 Input T3 không có test_report

- Cả 32/32 input T3 đều không có `test_report`.
- Phần lớn gold chẩn đoán được bằng đọc code. Chỉ 2/17 bài EX02 nêu giá trị quan sát được khi chạy.
- Kiểm leak bằng so chuỗi con với code sinh viên sẽ **bỏ sót**: câu lệnh còn thiếu của S106 lại có ở dòng 95/112/117 trong cùng file. → Guard cần so theo ngữ cảnh (hàm hoặc vị trí).

---

## 5. Mã nguồn và tín hiệu phụ

- **Code ngắn.**
  - Số dòng (trung vị): EX01 140, EX02 41.
  - Không có `bits/stdc++`, `conio`, `system("pause")`, `void main`. Comment không có dấu tiếng Việt.
  - EX02 có code bị comment (S209 26 dòng, S214 toàn bộ).
  - 1 file có BOM (S107). 1 file có template Visual Studio (S110).
- **compile_log:**
  - 22/32 log rỗng; 7 có error (1 trong số đó là lỗi link); 3 chỉ có warning `-Wreturn-type`.
  - Định dạng GNU, file tạm `/tmp/tmp*/student.cpp`.
  - S111 bị **cắt ở đúng 2000 ký tự**.
  - Dòng code trích trong log khớp file phát hành 23/23.
- **README giải thích "compilable = 1 dù không biên dịch được" là do code viết theo MSVC.** Trên mẫu có **0 ca** như vậy.
  - Ca duy nhất là S214, do comment toàn bộ.
  - Xác suất ngẫu nhiên không gặp ca MSVC nào là 0.12 (EX01) và 0.35 (EX02), nên mẫu **không bác bỏ** README; chỉ cho thấy giải thích chưa đủ.
  - Trường `compiles_locally` mà README nhắc tới **không có** trong dữ liệu.
- **test_report không tái lập được bằng một toolchain cố định:**

| Toolchain | Số test tái lập / 104 |
|---|---|
| clang -O0 | 88 |
| clang -O2 | 80 |
| gcc 12 -O0 | 88 |
| gcc 12 -O2 | 96 |
| clang + trap khi hàm non-void thiếu return | **104** |

  - S206 và S210 có UB (hàm non-void thiếu return).
  - `actual=''` nghĩa là chương trình chết trước khi in. Nhưng report **không có returncode hay stderr**, nên không phân biệt được với trường hợp không in gì.
  - Ngay cả trên test mở rộng, 3/13 chương trình cho kết quả khác nhau giữa clang và gcc.
- **Trùng lặp:**
  - Không cặp nào có Jaccard ≥ 0.7 sau khi bỏ code mẫu của đề.
  - **S203–S206** có Jaccard 0.59–0.95 tuỳ cách chuẩn hoá, và chứa nguyên cùng biểu thức lỗi `res = res + (n % 10) * dem`. Đây là dấu hiệu cùng nguồn.
  - → Khi chia CV phải group theo cụm near-dup.
- **Ẩn danh:** 0 token `REDACTED` sót, 0 email, 0 từ khoá MSSV hay họ tên, 0 prompt injection. Hai chuỗi số dài đều là hằng `10^k` trong code.

---

## 6. Ứng viên cho điểm thưởng (đã kiểm chứng, xếp hạng)

| # | Vấn đề | Bằng chứng gọn | Mức | Cần kiểm trên tập đầy đủ |
|---|---|---|---|---|
| 1 | **Gold EX01 không theo chính sách tiên quyết, chấm không nhất quán** | Cùng lỗi P1: S105 = 2, S110 = 8. "P1 sai": 1–9 điểm. S112 lặp vô hạn mà logic = 3 | Cao | Tỷ lệ bài "P1 sai" có total > 2 |
| 2 | **io_format ngược với harness; test_report mất thông tin** | S211 0/8 → 10/10; S205 8/8 → io = 0; 104/104 actual bị strip; `''` là crash, không có returncode | Cao | Số bài in prompt hoặc thừa khoảng trắng; tỷ lệ `actual=''` |
| 3 | **Gold T3 vi phạm định nghĩa level của chính nó; metric (a) một chiều** | Tuân thủ 36–61%; output hằng số đạt 1.00, cao hơn gold (0.91–0.97); S207 L3 → 8/8 test | Cao | Chạy detector + gán tay khoảng 50 mẫu, đo κ giữa 2 người gán |
| 4 | **S214: file comment toàn bộ được 10/10** | Lỗi link; bản bỏ comment pass 6/8; nhãn chỉ có style | Cao (n = 1) | Đếm bài có 0 dòng code hiệu dụng hoặc lỗi link mà compilable = 1 |
| 5 | **Nhãn "Lỗi edge case" không có support + `zero_division`** | Oracle 0.90/1.00; "đoán 10 nhãn" thắng luật ở zd = 0; P0 ≈ 0.58 với N = 36 | Trung bình–cao (tính chất metric, thành thật khi test thiếu positive) | Support từng nhãn trong test |
| 6 | **Rubric suy biến: exact-match thổi phồng; edge_case không độc lập ở EX02** | edge = logic/2 (17/17); mode đúng 100% ở code_quality | Trung bình | Ràng buộc có còn đúng không |
| 7 | **QWK: tham số `labels`, gộp exam, CI rộng** | Chỉ dùng exam_id: pooled 0.219, trong từng exam 0; CI rộng khoảng 0.5 | Trung bình | Lặp lại với n thật |
| 8 | **Trọng số EX01 mâu thuẫn; quy định đề không khớp nhãn** | 1/4/2/3 so với 2.5×4; lệnh `-o id` so với yêu cầu bỏ main; 10/15 bài còn main | Trung bình–thấp | Metadata của tập đầy đủ |
| 9 | **Input T3 = gold T2** | 32/32 | Có điều kiện | Mức giao sample_id giữa test T2 và test T3 |
| 10 | **compile_log sinh lại, bị cắt 2000 ký tự, khác môi trường giảng viên** | S109 lệch dòng; S111 bị cắt | Thấp | Tỷ lệ log bị cắt |

**Thứ tự gợi ý khi viết hồ sơ bonus:** (1) → (2) → (3) → (5).

**Trước khi claim "nhãn nhiễu":** phải loại giả thuyết "chấm theo ý định" hoặc "chấm nương tay". Feedback S214 cho thấy giảng viên biết bài bị comment.

---

## 7. Cập nhật cho bản brainstorm

### 7.1 Các quyết định trả lời được từ mẫu

| D | Trên mẫu | Hệ quả |
|---|---|---|
| D0 cỡ dữ liệu | Khoảng 121 bài [GT, suy từ README] | **Cổng D0 bị kích hoạt:** không train head 8B cho T1/T2 trên khoảng 90 bài. Xem 7.2 |
| D1 total = Σ6 chiều | **Có** (32/32) | Dự đoán logic (và edge ở EX01), suy các chiều còn lại bằng ràng buộc theo exam, rồi cộng |
| D3 có điểm từng câu | **Không** | Trạng thái câu chỉ có trong feedback, không được dùng làm input |
| D5 tam giác compile | 31/32 (S214) | Luật CE dùng diagnostic error; lỗi link chỉ làm feature |
| D7 gold L1/L2 có code | **Có** (2–7/22 tuỳ định nghĩa) | Lọc hoặc viết lại trước khi SFT; đưa vào hồ sơ bonus |
| D10 token | p95 < 2k | Không cần tách câu vì context |
| D11 toolchain | GNU g++ Linux, do BTC sinh lại | Tự build bằng gcc 13 trong Docker. Cache evidence vì UB làm kết quả phụ thuộc toolchain |
| Đề có trong dữ liệu | **Có** (`statement`) | — |
| Định dạng file nộp | **Không biết** | Hỏi giảng viên |

### 7.2 Thay đổi khuyến nghị so với bản brainstorm

1. **Policy engine: chuyển từ "áp cứng" sang "feature + phát hiện bonus".**
   - Gold không theo chính sách đã viết, nên áp cứng sẽ làm tụt QWK.
   - Học ánh xạ thật từ dữ liệu. Báo cáo cả hai phiên bản: tối ưu theo gold, và đúng theo policy.
2. **T1.**
   - Mục tiêu chính là một biến thứ bậc: **logic** (EX02 có 3 bậc {0, 2, 4}). Các chiều khác suy bằng ràng buộc của từng exam.
   - Feature cho EX02:
     - tỷ lệ pass nới;
     - các detector trên **stdout thô**: prompt, khoảng trắng, hoa/thường;
     - cờ crash/UB.
   - Feature cho EX01:
     - có mặt và mức hoàn thành từng hàm;
     - phân tích tĩnh;
     - LLM đọc từng câu.

     EX01 không chạy được vì thiếu `generals.csv`. Nên xin file này.
3. **T2.**
   - Luật CE.
   - **Ràng buộc chung với T1:** total < 10 ⇒ có ≥1 nhãn; chỉ cho `[]` khi total = 10.
   - Ngưỡng riêng từng nhãn. Không đoán "Lỗi edge case" trừ khi có bằng chứng mạnh.
   - Prompt theo exam.
4. **T3.**
   - **Không SFT thẳng trên gold.** Cần:
     - viết lại ghi chú của người chấm thành lời nói với sinh viên;
     - gán lại level theo nội dung thực tế, hoặc sinh level-swap để lấp L1 trên bài có lỗi và L4.
   - Guard kiểm theo đúng điều cấm của label_space, so code mới theo ngữ cảnh.
   - **Kiểm leak bằng thực thi** cho EX02: áp gợi ý vào code rồi chạy test.
   - Hỏi giảng viên: T3 có được dùng `test_report` (ghép qua sample_id) không?
5. **Fine-tune bắt buộc.**
   - Với khoảng 121 bài, nên dồn fine-tune vào **T3** (SFT/DPO trên dữ liệu level-swap do teacher sinh, có ghi chi phí) hoặc vào model nhỏ.
   - Với T1/T2, head trên 8B dễ overfit; dùng LoRA trên model nhỏ hoặc classifier trên embedding.
   - **Learning curve** trở thành thí nghiệm then chốt cho báo cáo.
6. **Đánh giá.**
   - Báo metric **theo từng exam**, kèm CI bootstrap.
   - Cố định `labels=range(11)` cho QWK. Báo 3 quy ước `zero_division` cho macro-F1.
   - Group CV theo cụm near-dup.
   - Không chọn mô hình dựa trên chênh lệch nhỏ hơn khoảng 0.1 QWK.

---

## 8. Chạy lại trên tập đầy đủ

```bash
cd eda
./run-eda.sh /path/to/full_dataset              # tĩnh, an toàn, khoảng 7 giây trên mẫu
./run-eda.sh /path/to/full_dataset --with-exec  # thêm biên dịch lại và chạy EX02; chỉ chạy trong container dùng xong bỏ
```

- **Output:** `out/report/{t1-rubric,t2-taxonomy,t3-feedback,code-signals,meta-audit}.md`, hình trong `out/{t1_,t2,t3,code}/`, feature từng bài ở `out/code/code_features.csv`.
- **Script:**
  - Phân tích: `t1_rubric_eda.py`, `t2_label_eda.py`, `t3_feedback_eda.py`, `code_eda.py`, `meta_audit.py`.
  - Kiểm chứng độc lập: `verify_*.py`.
  - Loader dùng chung: `load_data.py`. Đã sửa để bắt cả lỗi link qua các cột `log_has_link_error` và `log_build_failed`.
- **Nhãn tay T3:** `t3_feedback_eda.py` đọc `t3_manual_annotations.json`, chỉ có cho 32 bài mẫu. Trên tập đầy đủ, script chỉ chạy detector regex.
- **Số cần xem đầu tiên trên tập đầy đủ:**
  - kích thước từng split;
  - mức giao sample_id giữa T2 và T3;
  - support từng nhãn (nhất là "Lỗi edge case");
  - ba ràng buộc rubric ở D1/§2.1;
  - tỷ lệ vi phạm cổng P1;
  - số bài in prompt hoặc thừa khoảng trắng;
  - số `actual=''`;
  - phân bố level × exam, và có L4 hay không;
  - số bài có 0 dòng code hiệu dụng.

---

## 9. Câu hỏi gửi giảng viên (rút từ mẫu)

1. **Định dạng file predictions** cho từng task: khoá, kiểu dữ liệu, total có phải số nguyên không?
2. **Scorer:** QWK có cố định `labels=0..10` không, tính gộp hay theo exam? Macro-F1 dùng `zero_division` nào, trên 10 nhãn cố định hay chỉ nhãn có support?
3. **EX01:** trọng số đúng là 1/4/2/3 hay 2.5×4? Rubric 6 chiều được suy từ kết quả từng câu như thế nào? Vì sao S108, S110, S112 có điểm dù P1 sai?
4. **io_format:** chuẩn chính thức là gì? In prompt có bị trừ không, thừa khoảng trắng có bị trừ không?
5. **S214 (file comment toàn bộ, 10/10):** chấm theo ý định hay lỗi nhãn? Có loại khỏi tập đánh giá không?
6. **Level ở tập test:** gán thế nào? Có L4 không, có L1 cho bài có lỗi không?
7. **Compliance:** có phạt "dưới mức" không? Trích code sinh viên ở L2 có tính là "đưa code" không? Điều cấm của L3 có được kiểm không?
8. **Metric (b):** judge dùng gold như reference cứng hay chỉ làm gợi ý? Gold không có chẩn đoán thì xử lý ra sao?
9. **Test T2 và test T3** có dùng chung sample_id không? T3 có được dùng `test_report` không?
10. **Bổ sung dữ liệu:** có phát `generals.csv` và test cho EX01 không? Có kết quả test ẩn của EX02 không? `test_report` có thể thêm returncode và stderr không?
11. **Toolchain:** `compiles_locally` có trong tập đầy đủ không, định nghĩa bằng lệnh nào? `compile_log` có luôn sinh bằng GCC 13 không, có bị cắt ở 2000 ký tự không?
12. **Kích thước tập đầy đủ** (khoảng 121 bài?) và cách chia train/dev/test (theo đề hay ngẫu nhiên)?

---

## 10. Hạn chế

- **n = 32 và có thể không ngẫu nhiên.** Mọi tỷ lệ chỉ là sự thật trên mẫu; p-value chỉ để tham khảo.
- **Toolchain khác:** biên dịch và chạy bằng Apple clang, thêm gcc 12 trong một image có sẵn. Không có GCC 13, nên giả thuyết "GCC 13 -O0 trap khi thiếu return" chưa xác nhận.
- **Nhãn tay T3** do một người gán, chưa đo κ. Con số tuân thủ dao động 36–61% tuỳ cách đọc.
- **Đọc feedback chỉ để EDA và lập hồ sơ bonus.** Không có feature nào cho T1/T2 lấy từ feedback.
- **An toàn khi chạy code:** chỉ chạy EX02, sau khi grep không có lời gọi nguy hiểm, trong thư mục tạm, timeout 2 giây. Không chạy EX01.
