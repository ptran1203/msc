# Task 1: chấm điểm theo rubric 6 chiều (metric chính: QWK trên tổng điểm)

Quy ước trong tài liệu này:
- **[GT-n]** là giả thuyết về dữ liệu. **[KT]** là cách kiểm tra.
- Không có số liệu nào về dữ liệu ở đây là số đo thật. Các con số đều đến từ biến đổi toán học hoặc là ước lượng có ghi rõ.
- Tên cột trong các snippet là giả định, cần sửa theo schema thật:

```python
dims = ["compilable","io_format","logic","edge_case","complexity","code_quality"]
```

---

## 1. Bản chất bài toán

### 1.1 Cấu trúc output
- Đây là bài toán **multi-output ordinal regression có ràng buộc**, không phải 6 bài classification độc lập.
  - Số mức của 6 chiều là 2·2·5·3·2·2, tức chỉ có **240 tổ hợp**.
  - Tổng điểm có 11 mức nếu là số nguyên.
- **[GT-1]** Với single_problem thì total = tổng 6 chiều. Với multi_problem thì total có thể bằng Σ w_q·score_q (trọng số theo câu, cộng luật tiên quyết), nên có thể là số lẻ hoặc khác tổng 6 chiều. [KT]:
  ```python
  df["sum6"] = df[dims].sum(1)
  df.assign(mis=(df.sum6 - df.total).abs() > 1e-6).groupby("type").mis.mean()
  df[dims + ["total"]].apply(lambda s: (s % 1 != 0).mean())   # tỷ lệ điểm lẻ
  ```
  - Nếu `mis` ≈ 0 thì tổng suy ra tất định từ 6 chiều.
  - Nếu `mis` > 0 thì có hai khả năng: do trọng số câu, hoặc nhãn không nhất quán. Trường hợp sau là ứng viên cho điểm thưởng.
- **[GT-2]** Các chiều có cấu trúc "cổng" (gating): compilable=0 kéo các chiều cần chạy được về 0, logic thấp kéo edge_case thấp. [KT]:
  - `df[df.compilable==0][dims].mean()`
  - `pd.crosstab(df.logic, df.edge_case)`
  - `df[dims].corr("spearman")`

### 1.2 QWK chính là Lin's CCC: hệ thức chi phối toàn bộ chiến lược
- Điều kiện: trọng số bình phương, các mức cách đều nhau, ma trận kỳ vọng lấy từ tích hai phân bố biên. Khi đó:

  **κ = 2·cov(y,ŷ) / (σ_y² + σ_ŷ² + (μ_y − μ_ŷ)²)**

- Chứng minh:
  - Σ w·O tỉ lệ với E[(y−ŷ)²] = σ_y² + σ_ŷ² + Δμ² − 2cov.
  - Σ w·E là cùng biểu thức đó nhưng với cov = 0.
- Hệ quả:
  - **Với tương quan ρ cố định, κ đạt cực đại khi σ_ŷ = σ_y và μ_ŷ = μ_y.** Khi đó κ = ρ.
  - Regression tối ưu MSE tự co về giữa (σ_ŷ = ρσ_y), cho κ = 2ρ²/(1+ρ²):
    - ρ = 0.8 cho 0.780 thay vì 0.800.
    - ρ = 0.6 cho 0.529 thay vì 0.600.
  - **Vậy không nên "bảo thủ về giữa". Phải làm ngược lại (de-shrink):** giãn dự đoán để khớp mean và variance của nhãn thật. Đây là lý do rescale và OptimizedRounder thắng trên Kaggle.
  - Lệch trung bình bị phạt theo hệ số 1/(1 + Δ²/(2σ²)) khi σ đã khớp. Ví dụ Δ = 0.5σ làm mất khoảng 11% κ. LLM zero-shot thường chấm rộng tay, nên **bắt buộc calibrate bias**.
  - MAE (metric phụ) lại ưa co về median, nên hai metric xung đột. Hãy tối ưu cho QWK và báo cáo MAE ở cả hai điểm vận hành trong bảng ablation.
  - Lỗi bị phạt theo bình phương: lệch 5 điểm nặng bằng 25 lần lệch 1 điểm. QWK bị chi phối bởi vài ca lệch xa, ví dụ:
    - bài không biên dịch, bài rỗng, bài chỉ nộp template mà lại đoán 7 điểm;
    - multi_problem có câu tiên quyết sai mà vẫn cộng điểm các câu sau.

    **Ưu tiên số 1 là bắt đúng các ca cực này bằng tín hiệu tất định.**

### 1.3 Vì sao tối ưu từng chiều không đồng nghĩa với tối ưu QWK tổng
- Exact-match từng chiều đạt tối ưu với **mode (argmax)** của chiều đó. QWK đạt tối ưu với **dự đoán liên tục (E-value) cộng bộ ngưỡng toàn cục**.
- Cộng argmax của từng chiều cho kết quả lệch hệ thống. Ví dụ edge_case có phân phối [0.4, 0.35, 0.25]: argmax = 0, còn E = 0.85.
- Lỗi giữa các chiều tương quan nên cộng dồn. Đúng 5/6 chiều nhưng sai logic 4→0 thì tổng vẫn lệch 4.
- Vì vậy nên tách 2 nhánh output:
  - 6 chiều: dùng MAP (argmax) có ràng buộc (§5.4).
  - total: dùng E-value rồi rescale hoặc áp ngưỡng.
- [KT] Script kiểm tra định dạng có bắt buộc total = tổng 6 chiều không? Câu trả lời quyết định có được tách 2 nhánh hay không.

### 1.4 Phân bố lệch và cách scorer tính QWK
- **[GT-3]** Phân bố tổng điểm đa đỉnh: một đỉnh ở 0 (không biên dịch hoặc bỏ trống), một đỉnh ở 8–10. Hệ quả:
  - κ tính gộp bị thổi phồng bởi phương sai giữa các đề và giữa nhóm compile được với nhóm không. Một model chỉ đọc compile_log cũng đã có κ đáng kể.
  - Trong từng nhóm, σ_y nhỏ nên κ rất nhạy.
  - [KT] Chạy hai baseline sau. Nếu baseline luật compile đã cao, đây là **vấn đề metric** đáng báo cáo. Đề xuất khắc phục: báo cáo thêm κ trên tập compilable=1 và κ trung bình theo đề.
  ```python
  from sklearn.metrics import cohen_kappa_score as ck
  q = lambda y,p: ck(y, p, weights="quadratic", labels=list(range(11)))
  loo = df.groupby("problem_id").total.transform(lambda s: (s.sum()-s)/(len(s)-1))
  q(df.total, loo.round().astype(int))                                   # baseline: mean theo đề
  q(df.total, np.where(df.compilable==0, 0, int(df[df.compilable==1].total.median())))  # baseline: luật compile
  ```
- **[GT-4]** Nếu scorer gọi `cohen_kappa_score(weights="quadratic")` mà không truyền `labels=range(11)`, sklearn tính trọng số theo **chỉ số** của các giá trị có mặt trong y ∪ ŷ, không theo giá trị thật. Hậu quả:
  - Thiếu mức 3 thì khoảng cách 2↔4 chỉ còn bằng 1.
  - Đoán thêm một mức chưa xuất hiện làm thay đổi trọng số của cả ma trận.
  - Điểm lẻ (7.5) trở thành một category riêng. Nộp float sẽ cho kết quả rất tệ.
  - [KT] Đọc script chấm; so sánh `ck(y,p,weights="quadratic")` với bản có `labels=list(range(11))` trên OOF. Nếu khác nhau thì đây là ứng viên điểm thưởng.
- **Luôn nộp total nằm trên lưới giá trị hợp lệ.**

---

## 2. Tín hiệu mạnh có sẵn hoặc tự tạo

Nguyên tắc: phần nào tất định được thì biến thành rule hoặc feature. LLM chỉ dùng cho phần ngữ nghĩa: logic của multi_problem, edge_case khi không có test, code_quality.

### 2.1 compile_log, dùng cho compilable
- **[GT-5]** compilable gần như tất định từ compile_log. [KT]:
  ```python
  pd.crosstab(df.compile_log.str.contains(r"\berror\b", case=False, na=False), df.compilable)
  ```
  Các ca lệch là nhiễu nhãn hoặc do khác compiler; cần phân tích riêng.
- **Bẫy môi trường biên dịch:** sinh viên Việt Nam hay dùng Dev-C++/MinGW, nên code thường có `conio.h`, `getch()`, `system("pause")`, `fflush(stdin)`. Nếu giảng viên biên dịch trên Windows còn nhóm biên dịch trên Linux, compilable sẽ lệch. [KT] Grep compile_log để tìm `C:\`, `mingw`, phiên bản g++, cờ `-std=`. Sau đó tái lập đúng toolchain đó và ghi phiên bản vào README.
- Feature từ log: số error, số warning theo loại khi build với `-Wall -Wextra` (`-Wuninitialized`, `-Wreturn-type`, `-Wsign-compare`), loại lỗi đầu tiên.

### 2.2 test_report (single_problem), dùng cho io_format và logic
- Feature: pass rate, phân bố verdict (WA/TLE/RE/MLE).
- **[GT-6]** io_format suy ra được từ khoảng cách giữa so sánh strict và so sánh normalized:
  - Tự chạy lại test công khai với 2 comparator: strict, và normalized (strip, gộp whitespace, lowercase, bỏ dòng prompt).
  - `io_gap = pass_norm − pass_strict`. Nếu io_gap > 0 thì io_format=0 rất có khả năng.
  - [KT] `pd.crosstab(io_gap>0, df.io_format)`.
- Feature tĩnh cho IO:
  - Có `cout << "Nhap n"` (prompt tiếng Việt) trước `cin`.
  - `freopen(...)` còn sót lại.
  - In thêm "Ket qua:".
  - Thiếu xuống dòng, format `printf` sai.
- **[GT-7]** test_report chỉ chạy trên test công khai, còn logic do giảng viên chấm theo test ẩn hoặc đọc code. Vì vậy pass rate tương quan với logic nhưng không quyết định hoàn toàn. [KT]:
  - Tính Spearman(pass_rate, logic) theo từng đề.
  - Soi các bài pass 100% mà logic < 4: có hard-code output kiểu `if(n==5) cout<<...` không?

### 2.3 Tự biên dịch và chạy (sandbox tất định)
- Chạy trong Docker `--network none`, đặt ulimit CPU/mem và timeout. Cố định phiên bản g++ và cờ `-std=c++17 -O2 -Wall -Wextra`.
- Build thêm một bản với `-fsanitize=address,undefined -g`. Đếm lỗi out-of-bounds, overflow, dùng biến chưa khởi tạo; đây là feature cho edge_case và logic.
- Harness cho multi_problem: biên dịch với `-Dmain=student_main` để link một driver riêng, gọi trực tiếp hàm của sinh viên theo chữ ký trong đề (trích bằng tree-sitter-cpp). Cách này cho phép test từng câu dạng hàm.

### 2.4 Sinh testcase, đặc biệt cho multi_problem và edge_case
- **Peer-oracle differential testing**, mạnh nhất nếu dev/test dùng chung đề với train:
  - Với mỗi mã đề, lấy các bài train có logic=4 và edge_case=2 làm oracle.
  - Sinh input bằng generator do LLM viết từ ràng buộc của đề, cộng input ngẫu nhiên và input biên (n=0, 1, max, số âm, phần tử trùng, mảng rỗng, mảng đã sắp xếp).
  - Expected output = kết quả đa số của các oracle. Bỏ những input mà oracle bất đồng.
  - Feature: pass rate trên test sinh, tách riêng nhóm test biên cho edge_case.
  - Cách này không cần lời giải mẫu và không đụng vào nhãn của tập test.
- Đề mới, không có oracle: cho LLM sinh k lời giải, chỉ giữ lời giải pass test công khai, lấy expected theo đa số. Cố định seed, ghi phiên bản model và cache kết quả.
- **Chống leakage:** khi tính feature cho một bài train, loại chính bài đó và các near-duplicate của nó khỏi tập oracle.

### 2.5 Độ phức tạp, dùng cho complexity
- Tĩnh:
  - độ sâu vòng lặp lồng nhau tối đa (dùng AST);
  - đệ quy không memo;
  - tự viết bubble/selection sort thay vì dùng `std::sort`;
  - gọi `erase` hoặc `find` trên vector bên trong vòng lặp.
- Động:
  - Chạy với n tăng gấp đôi theo ràng buộc của đề, fit slope của log t theo log n để ra bậc thực nghiệm.
  - Wall-time không tái lập được, nên lấy median nhiều lần chạy, hoặc đếm instruction bằng `perf stat -e instructions:u` (cần kiểm chứng trong Docker) để có kết quả tất định.
- **[GT-8]** complexity gần suy biến (đa số = 1). [KT]:
  ```python
  df[dims].apply(lambda s: s.value_counts(normalize=True).max())
  ```
  Nếu > 0.9 thì exact-match của chiều này gần như vô nghĩa. Nên báo cáo **mức cải thiện so với baseline majority**.

### 2.6 Chất lượng code, dùng cho code_quality
- Công cụ:
  - lizard: NLOC, CCN, độ dài hàm, số tham số.
  - cppcheck `--enable=style,performance,warning`.
  - clang-tidy với các check `readability-magic-numbers`, `readability-identifier-length`, `readability-function-size`, `bugprone-*`.
- Feature tự tính:
  - tỷ lệ dòng bị `clang-format` sửa (đo thụt lề không nhất quán);
  - tỷ lệ comment, số biến toàn cục, `goto`;
  - tỷ lệ tên biến 1 ký tự, khối code lặp lại.
- **[GT-9]** code_quality bị **halo effect**: bài đúng thì hay được cho code_quality=1. [KT] So AUC khi dự đoán code_quality từ (logic, edge_case, compilable) với AUC khi dự đoán từ feature style. Nếu cách đầu cao hơn rõ rệt thì nhãn không đo đúng thứ nó tuyên bố; đây là ứng viên điểm thưởng.

### 2.7 Bài gần rỗng và bài trùng lặp
- Bài gần rỗng: so diff với code khung (nếu đề có) hoặc với "code phổ biến nhất" của đề, đếm số dòng thực chất. Bài chỉ có `main` rỗng gần như chắc chắn được điểm thấp.
- **Near-duplicate** vừa là feature vừa dùng để đo nhiễu nhãn:
  - Chuẩn hoá code: bỏ comment và whitespace, thay identifier bằng loại token, rồi hash hoặc MinHash.
  - [KT]:
  ```python
  df.groupby(["problem_id","norm_hash"]).total.agg(["count","std","min","max"]).query("count>1")
  ```
  - κ giữa các cặp bài trùng xấp xỉ **trần inter-rater**. Nếu nhiều nhóm có std > 0 thì đó là nhiễu nhãn (điểm thưởng), đồng thời là trần thực tế của QWK.

### 2.8 Chiều nào gần tất định

| Chiều | Nguồn | Mức tất định [GT] |
|---|---|---|
| compilable | compile_log (đúng toolchain) | Gần tất định |
| io_format | test_report + io_gap + feature tĩnh IO (single); multi phải tự chạy | Cao với single, trung bình với multi |
| logic | pass rate trên test công khai, test sinh và test peer-oracle | Tương quan mạnh nhưng không tất định |
| edge_case | test biên, sanitizer | Trung bình |
| complexity | slope thực nghiệm, độ sâu vòng lặp | Có thể suy biến |
| code_quality | static analysis | Thấp, chủ quan |

Cách dùng: rule cho phần tất định, chỉ cho model ghi đè khi xác suất ngược lại rất cao. Ngưỡng ghi đè chọn trên OOF.

---

## 3. Xử lý multi_problem

### 3.1 Trích chính sách chấm từ đề bài (chạy 1 lần mỗi đề, có cache)
- Bước 1, regex:
  - Trọng số: `[Cc][aâ]u\s*(\d+)[^\n]*?\(\s*(\d+(?:[.,]\d+)?)\s*(điểm|đ)\)`, cộng dạng phần trăm.
  - Các cụm chỉ quan hệ tiên quyết: "tiên quyết", "chỉ (chấm|tính điểm) khi", "nếu câu … sai", "sử dụng (hàm|kết quả) (ở|của) câu".
- Bước 2, LLM extraction với structured output (JSON schema qua outlines, xgrammar hoặc JSON mode của API):
  ```json
  {"questions":[{"id":1,"weight":3,"io_kind":"function|stdin","signature":"int sum(int[],int)",
                 "depends_on":[],"gate":"zero|cap"}],"total_weight":10}
  ```
  - Self-consistency 3–5 lần, lấy kết quả đa số.
  - Validate: Σ weight = 10 (hoặc 100%), đồ thị phụ thuộc không có chu trình.
  - Nếu vẫn lỗi thì fallback về trọng số đều, không gating, và gắn cờ `policy_failed` làm feature.
- Chỉ audit tay trên đề của train để đo độ chính xác của extractor (và báo cáo con số đó). Đề dev/test chạy tự động hoàn toàn.
- Cần hỏi giảng viên: viết tay cấu hình cho từng đề (không phải từng bài) có vi phạm luật "không can thiệp" không.

### 3.2 Tách file nộp thành từng câu (từ rẻ đến đắt)
1. Comment đánh dấu, sau khi chạy `unidecode`: `(?im)^\s*(//|/\*)\s*(cau|bai|question|q)\s*\.?\s*(\d+)`.
2. Tên hàm: `\b(cau|bai|q|ex)_?(\d+)\s*\(`.
3. Menu trong `main`:
   - Hàm được gọi trong `case k:` của `switch` thuộc về câu k.
   - Dựng call graph bằng tree-sitter để gán các hàm phụ trợ; hàm dùng chung thì gán cho nhiều câu.
4. Khớp với `signature` lấy từ policy ở §3.1.
5. LLM segmentation:
   - Đưa code đã đánh số dòng, nhận về `{q: [[start,end],...]}`.
   - Validate: các đoạn không chồng lấn, ranh giới trùng ranh giới hàm trong AST, phủ đủ các hàm mà đề yêu cầu.
6. Fallback: dùng cả file cho mọi câu, kèm cờ `split_failed`.

[KT] Đo tỷ lệ bài được tách thành công ở từng bước. Đo mức đồng thuận giữa heuristic và LLM trên một mẫu train.

### 3.3 Chấm từng câu, rồi tổng hợp bằng code tất định
- Tín hiệu cho từng câu:
  - compile riêng nếu tách được;
  - pass rate trên test sinh hoặc test peer-oracle của câu đó;
  - nhận định của LLM {đúng, một phần, sai, không làm}, kèm E-value.
- **Luật tiên quyết áp bằng code, không để LLM tự nhớ:**
  ```python
  def aggregate(q_scores: dict[int, float], policy) -> float:   # q_scores ∈ [0,1]
      passed, total = {}, 0.0
      for q in topo_order(policy):
          s = q_scores[q]
          if any(not passed[p] for p in policy[q].depends_on):
              s = 0.0 if policy[q].gate == "zero" else min(s, policy[q].cap)
          passed[q] = s >= policy[q].pass_tau        # tau fit trên OOF
          total += policy[q].weight * s
      return total
  ```
- Gating cứng nhạy với ngưỡng, nên thêm bản "mềm": `s_q × Π P(pass_p)`. Đưa **cả hai bản** vào làm feature để stacker tự chọn.
- Nếu nhãn chỉ có ở cấp bài (không có điểm từng câu) thì q_scores là biến ẩn. Khi đó học tau bằng cách tối đa QWK của total tổng hợp trên OOF, hoặc dùng `aggregate` làm feature.
- **[GT-10]** Giảng viên không phải lúc nào cũng tuân theo chính sách tiên quyết. [KT]:
  - Lấy các bài có câu tiên quyết fail theo tín hiệu tất định (pass 0% test sinh hoặc lỗi compile ở phần đó).
  - Xem total của các bài này có bị chặn đúng như policy không.
  - Nếu vi phạm có hệ thống thì đó là bằng chứng cho điểm thưởng. Khi đó tối ưu theo nhãn cho leaderboard, và báo cáo cả hai phiên bản.
- **[GT-11]** Với multi_problem, 6 chiều có thể là tổng hợp có trọng số theo câu, hoặc là điểm chấm tổng thể. [KT]:
  - Điểm lẻ có xuất hiện ở các chiều của multi không?
  - Tương quan giữa logic và Σ w_q·pass_q.

---

## 4. Các hướng mô hình hoá

Chi phí API ước tính theo công thức N_bài × k_mẫu × (T_in + T_out) × giá/token (tra bảng giá hiện hành). Code 150–400 dòng vào khoảng 2–6k token (ước lượng, cần đo bằng tokenizer thật).

### (a) LLM zero/few-shot với rubric, CoT và self-consistency
- **Mô tả:**
  - Prompt gồm: đề bài; rubric có mô tả cho từng mức (viết từ đề, không lấy từ feedback); code đánh số dòng; compile_log rút gọn; test_report tóm tắt; kết quả tự chạy (pass rate, sanitizer).
  - Với mỗi chiều, model phải nêu bằng chứng (dòng code hoặc test) rồi mới cho điểm, xuất theo JSON schema.
  - Self-consistency: k = 5–10, T ≈ 0.7. Total lấy **trung bình** (không lấy mode) để có giá trị liên tục.
- **Ưu:**
  - Không cần train.
  - Xử lý được ngữ nghĩa và multi_problem không có test.
  - Có rationale để dùng trong phân tích lỗi.
  - Đáp ứng yêu cầu "có hướng prompting" của đề.
- **Nhược:**
  - Bias rộng tay và co về giữa, nên bắt buộc calibrate.
  - Không học được quy ước riêng của giảng viên.
  - API không tất định, nên phải cache response.
  - Dễ bị prompt injection qua comment trong code của sinh viên.
- **Chi phí:**
  - Cao nhất trong các hướng nếu k lớn và có CoT (CoT thêm vài trăm đến vài nghìn token output).
  - Chạy local 7B qua vLLM thì chỉ tốn GPU-hour.
- **Thắng khi:** đề mới; multi_problem không có test; ít dữ liệu train.

### (b) RAG bằng kNN trên các bài train cùng đề
- **Mô tả:**
  - Lấy k = 3–8 bài train cùng mã đề, gần nhất theo:
    - embedding code: `microsoft/unixcoder-base`, `Salesforce/codet5p-110m-embedding`, hoặc Qwen3-Embedding-0.6B (cần kiểm chứng chất lượng trên C++);
    - độ tương đồng token (Jaccard);
    - vector pattern pass/fail của testcase.
  - Chèn kèm nhãn 6 chiều làm few-shot.
  - Chọn mẫu đa dạng điểm: gần nhất, cộng 1 bài điểm cao và 1 bài điểm thấp, để neo thang điểm.
  - **Biến thể không dùng LLM:** kNN regression (trung bình có trọng số của total các láng giềng). Đây là baseline rẻ và là feature tốt cho stacker.
- **Ưu:** học được thang điểm cụ thể của từng đề và giảng viên, qua đó sửa bias calibration của (a); rẻ.
- **Nhược:**
  - Vô dụng với đề mới.
  - Rò rỉ nếu không loại chính bài đó và các near-duplicate của nó.
  - LLM có xu hướng chép điểm của láng giềng gần dù khác lỗi quan trọng; kiểm tra bằng ablation theo k và theo cách chọn đa dạng.
- **Chi phí:** embedding chạy 1 lần (CPU hoặc GPU nhỏ); prompt dài hơn khoảng k lần độ dài code.
- **Thắng khi:** dev/test cùng đề với train và mỗi đề có nhiều bài train.

### (c) Fine-tune LLM open-weight sinh JSON điểm (SFT)
- **Mô tả:**
  - Model chính: Qwen2.5-Coder-7B-Instruct. Bản 1.5B-Instruct dùng cho ablation nhanh.
  - Thay thế: Qwen3-8B (tắt thinking) hoặc Llama-3.1-8B-Instruct.
  - QLoRA r = 16–64, lr 1e-4 đến 2e-4, 2–3 epoch, chỉ tính loss trên token output, max_seq 4–8k. Dùng Unsloth hoặc TRL + PEFT.
  - Target là JSON 6 chiều.
  - Rationale ngắn (tuỳ chọn) chỉ distill từ teacher LLM. Dùng feedback làm target là vùng xám, cần hỏi giảng viên.
  - Data augmentation (đổi tên biến, bỏ comment) làm thay đổi code_quality, nên nếu augment thì phải **mask loss của chiều code_quality**.
- **Ưu:** học được quy ước của giảng viên; inference local và tất định (greedy + seed); đáp ứng yêu cầu fine-tune bắt buộc.
- **Nhược:**
  - Cross-entropy trên token không biết thứ tự: đoán "0" thay cho "4" bị phạt ngang đoán "3".
  - Argmax decoding làm mất thông tin, nên phải kết hợp với (g).
  - Mất cân bằng nhãn khiến model co về mode.
- **Chi phí:** QLoRA 7B với seq 8k trên 1 GPU 24GB là khả thi (ước lượng, cần đo); vài GPU-hour mỗi epoch, tuỳ N.
- **Thắng khi:** train có từ vài nghìn bài trở lên và đề lặp lại giữa các split.

### (d) Fine-tune với head hồi quy hoặc ordinal (CORAL/CORN)
- **Mô tả:**
  - Backbone Qwen2.5-Coder-1.5B hoặc 7B, thêm LoRA, pooling ở token cuối.
  - 6 head:
    - BCE cho 4 chiều nhị phân;
    - **CORN** cho logic (K=5) và edge_case (K=3), dùng thư viện `coral-pytorch`;
    - head phụ cho total với loss Huber.
  - Feature tất định đưa vào dạng text, ví dụ `[COMPILE:OK][PASS 7/10][IO_GAP 1]`.
- **Ưu:**
  - Hiểu thứ tự của nhãn; CORN đảm bảo các xác suất nhất quán theo thứ hạng.
  - Ra E-value liên tục, áp ngưỡng trực tiếp được.
  - Inference chỉ 1 forward pass, nhanh hơn sinh JSON.
- **Nhược:** phải tự viết code train; không có rationale.
- **Chi phí:** train tương đương (c), inference rẻ hơn.
- **Thắng khi:** mục tiêu thuần là QWK/MAE. Dự kiến thắng (c) dùng argmax.

### (e) Feature engineering + LightGBM
- **Mô tả:**
  - Feature:
    - toàn bộ feature ở §2;
    - target encoding OOF theo đề (mean và std của total);
    - SVD 64 chiều của embedding code;
    - feature kNN từ (b);
    - meta: type, môn, độ dài code.
  - Objective: regression trên total rồi áp ngưỡng, cộng một model multiclass cho mỗi chiều.
  - `monotone_constraints`: tăng theo pass_rate, giảm theo số error, để model bền vững hơn.
- **Ưu:** rẻ, chạy trong vài phút trên CPU; giải thích được bằng SHAP; mạnh khi tín hiệu thực thi chiếm ưu thế; là stacker tốt.
- **Nhược:** yếu về ngữ nghĩa khi không có test; đứng một mình thì không tính là hướng LLM.
- **Thắng khi:** single_problem có test_report giàu thông tin.

### (f) Hybrid: output có cấu trúc của LLM làm feature cho model nhỏ
- **Mô tả:**
  - LLM trả JSON gồm: nhận định từng câu, danh sách lỗi đã ánh xạ sang taxonomy, điểm 6 chiều, độ tự tin, E-value.
  - Đưa toàn bộ vào LightGBM cùng feature ở §2.
  - Dự đoán Task 2 (OOF) cũng dùng làm feature.
- **Leakage cần chặn:** nếu LLM có dùng RAG trên train, feature của mỗi bài train phải được tính với retrieval **loại chính bài đó**.
- **Ưu:** LLM chỉ cần chẩn đoán và xếp thứ tự đúng, model nhỏ lo phần calibration.
- **Nhược:** phải chạy LLM trên toàn bộ train; pipeline 2 tầng.
- **Chi phí:** 1 lần chạy LLM trên train + dev + test.
- **Thắng khi:** có ngân sách cho 1 lần chạy đó. Dự kiến đây là cấu hình mạnh nhất.

### (g) Expected-value decoding
- **Mô tả:**
  - Ép format để điểm mỗi chiều là đúng 1 token chữ số.
  - Lấy logprob của các token hợp lệ {0..K}, chuẩn hoá lại, tính E = Σ k·p_k.
  - Tổng = Σ E_dim. Linearity of expectation đúng bất kể các chiều phụ thuộc nhau.
  - Cách lấy logprob: vLLM `logprobs`, HF `output_scores`. Với API thì tuỳ provider có hỗ trợ logprobs hay không (cần kiểm chứng).
  - Temperature scaling trên OOF.
  - Khi có CoT đứng trước điểm: lấy trung bình E qua k mẫu CoT.
  - Cách làm tương tự trọng số theo xác suất trong G-Eval (Liu et al., 2023).
- **Ưu:**
  - Cải thiện gần như miễn phí.
  - Cho giá trị liên tục để áp ngưỡng.
  - Entropy là feature, và là tín hiệu để **chuyển bài khó sang model đắt hơn** (giảm chi phí).
- **Nhược:** phụ thuộc vào việc lấy được logprob; với CoT thì đây là phân phối có điều kiện trên một chuỗi CoT đã sample.
- **Thắng khi:** luôn nên bật cho (a) và (c).

### (h) Ensemble và stacking
- **Mô tả:**
  - Tầng 1: E-value OOF từ kNN (b), (c)+(g), (d), (e), (a)/(f).
  - Tầng 2: ridge không âm hoặc LightGBM nhỏ (thêm cờ type).
  - Áp ngưỡng **một lần, sau khi stack**.
- **Ưu:** ổn định hơn; trộn các nguồn lỗi khác nhau (LLM và thực thi).
- **Nhược:** bắt buộc nested CV; tái lập phức tạp hơn.
- **Thắng khi:** tương quan giữa residual OOF của các model thấp. Đo trước khi stack.

### (i) So sánh cặp với các bài anchor (tuỳ chọn)
- **Mô tả:**
  - Chọn các anchor cùng đề, đã biết điểm (ví dụ 2, 5, 8 điểm).
  - Hỏi LLM bài đang chấm tốt hơn hay kém hơn anchor; tìm vị trí bằng binary search hoặc Bradley–Terry.
  - Lý do: LLM so sánh tương đối đáng tin hơn chấm tuyệt đối, và theo §1.2 QWK chỉ cần thứ tự cộng thang đo.
- **Nhược:** cần khoảng log₂(số anchor) × N lượt gọi; có position bias, nên phải hoán đổi vị trí A/B.
- **Thắng khi:** (a) bị lệch thang điểm nặng.

### Bảng tóm tắt

| Hướng | Cần train cùng đề | Output liên tục | Chi phí | Vai trò dự kiến |
|---|---|---|---|---|
| (a) Prompting | Không | Có (SC, g) | Cao (API) | Bắt buộc theo đề; feature cho (f) |
| (b) kNN/RAG | Có | Có | Thấp | Baseline mạnh; few-shot |
| (c) SFT JSON | Có lợi | Chỉ khi dùng (g) | Trung bình (GPU) | Fine-tune bắt buộc theo đề |
| (d) Ordinal head | Có lợi | Có | Trung bình | Model QWK chính |
| (e) LightGBM | Không bắt buộc | Có | Rất thấp | Feature tất định; stacker |
| (f) Hybrid | Không | Có | Trung bình–cao | Ứng viên tốt nhất |
| (g) EV decoding | – | Có | ~0 | Bật cho mọi LLM |
| (h) Stack | – | Có | Thấp | Hệ thống cuối |

**Đề xuất hệ thống cuối:**
1. Rule cho compilable và io_format.
2. Feature thực thi (gồm cả peer-oracle).
3. kNN.
4. Qwen2.5-Coder-7B + LoRA + head CORN.
5. LLM prompting có RAG và EV decoding, dùng làm feature.
6. Stacker LightGBM hoặc ridge.
7. Chọn giữa rescale và OptimizedRounder bằng nested CV.
8. MAP có ràng buộc cho 6 chiều.

---

## 5. Tối ưu riêng cho QWK

### 5.1 Đổi dự đoán liên tục thành điểm trên lưới (so sánh cả 4 cách)
1. Round thẳng (baseline).
2. **Rescale tuyến tính** theo §1.2: ŷ' = μ_y + (ŷ − μ_ŷ)·σ_y/σ_ŷ, rồi round. Chỉ 2 tham số, ít overfit nhất.
3. **OptimizedRounder:**
   - 10 ngưỡng c₁ < … < c₁₀, khởi tạo tại k − 0.5.
   - Tối ưu bằng Nelder-Mead hoặc coordinate search để tối đa QWK trên OOF; sort lại để giữ thứ tự.
   - Cho ổn định: fit trên B mẫu bootstrap rồi lấy median của từng ngưỡng.
4. **Quantile matching:** gán điểm theo thứ hạng sao cho phân bố dự đoán bằng phân bố của train, làm riêng theo type. Mạnh nếu không có distribution shift, rủi ro nếu có.

Lưu ý chung:
- Fit ngưỡng riêng cho single và multi nếu mỗi loại có vài trăm mẫu trở lên.
- Fit **trong fold train** (nested), đánh giá trên fold validation.

### 5.2 Calibrate
- Nhãn dev bị ẩn, nên không calibrate trên dev.
- Public leaderboard (2 lượt/ngày) chỉ dùng để kiểm tra CV có khớp leaderboard không, không dùng để tune ngưỡng.
- Bias của LLM: sửa bằng linear hoặc isotonic calibration trên OOF train.

### 5.3 Hậu xử lý ràng buộc nhất quán (cần kiểm chứng [GT-2] trước)
```python
rules = {
  "nocompile_zero": lambda d: (d.compilable==0) & (d[["io_format","logic","edge_case","complexity"]].sum(1)>0),
  "logic0_edge0":   lambda d: (d.logic==0) & (d.edge_case>0),
  "logic_low_cplx": lambda d: (d.logic<=1) & (d.complexity==1),
}
{k: f(df).mean() for k, f in rules.items()}   # tỷ lệ vi phạm trên train
```
- Vi phạm < 0.5%: áp làm hard rule.
- 0.5–5%: chỉ dùng làm feature hoặc ràng buộc mềm.
- > 5%: bỏ.
- Không có rule nào cấm compilable=0 đi cùng code_quality=1, trừ khi dữ liệu cho thấy điều đó.

### 5.4 MAP có ràng buộc cho 6 chiều
- Liệt kê đủ **240 tổ hợp**, tính log P = Σ log p_dim (hoặc xác suất joint nếu model autoregressive).
- Loại các tổ hợp vi phạm hard rule.
- Nếu validator bắt total = tổng 6 chiều: chọn tổ hợp có xác suất cao nhất trong số các tổ hợp có tổng bằng total* (total* lấy từ nhánh QWK).
- Brute force 240 tổ hợp rất rẻ. Cách này giữ được QWK của total mà vẫn tối ưu exact-match của các chiều.

### 5.5 Dự đoán tổng trực tiếp hay cộng 6 chiều

| Biến thể | Cách tính | Dự kiến |
|---|---|---|
| A | Σ E[dim], rồi áp ngưỡng | Tốt với single (total = tổng 6 chiều) |
| B | Regress total trực tiếp, rồi áp ngưỡng | Tốt với multi (có trọng số, gating, điều chỉnh tổng thể) |
| C | Σ argmax(dim) | Baseline, dự kiến tệ nhất |
| D | Stack A + B + aggregate theo policy | Dự kiến tốt nhất |

So sánh bằng QWK trên OOF, kèm **paired bootstrap CI** cho chênh lệch giữa các biến thể.

---

## 6. Thiết kế validation

### 6.1 Xác định kịch bản từ dữ liệu
```python
pid = lambda s: s.str.extract(r"submissions/([^/]+)/")[0]
tr, dv = set(pid(train.code_file)), pid(dev.code_file)
print(len(set(dv) - tr), dv.nunique(), (~dv.isin(tr)).mean())   # số đề mới, tỷ lệ bài thuộc đề mới
```
Lặp lại với test nếu input của test đã có.

### 6.2 Chọn sơ đồ CV theo kịch bản
- **Kịch bản S, cùng đề (tỷ lệ bài thuộc đề mới ≈ 0):**
  - StratifiedGroupKFold: stratify theo (problem_id × nhóm total), group theo cụm near-duplicate để code trùng không lọt sang fold khác.
  - kNN, target encoding và peer-oracle đều hợp lệ.
- **Kịch bản N, đề mới:**
  - GroupKFold theo problem_id.
  - Target encoding và peer-oracle mất tác dụng; phải dùng lời giải tham chiếu do LLM sinh.
- **Kịch bản hỗn hợp:** báo cáo cả hai sơ đồ CV, lấy trung bình có trọng số theo tỷ lệ bài thuộc đề mới.

### 6.3 Các kiểm tra bổ sung
- **Adversarial validation:** dùng LightGBM phân biệt train với dev trên các feature không chứa nhãn (độ dài, type, môn, tín hiệu thực thi). AUC ≈ 0.5 nghĩa là hai tập cùng phân bố.
- Nếu mã sinh viên ẩn danh xuất hiện ở nhiều split: group theo mã đó khi đánh giá. Chỉ dùng như một mã bút danh, không truy ngược danh tính.

### 6.4 Độ ổn định
- Repeated CV 3 × 5 fold với seed cố định, báo mean ± std.
- Paired bootstrap cho mọi so sánh trong ablation.
- Bootstrap trên OOF với cỡ mẫu bằng |dev| để biết chênh lệch bao nhiêu trên leaderboard mới có ý nghĩa thống kê.
- Ghi tương quan CV với public leaderboard sau mỗi lần nộp. Nếu lệch có hệ thống thì nghi distribution shift.

---

## 7. Ablation và phân tích lỗi

### 7.1 Bảng ablation đề xuất

| # | Cấu hình | Câu hỏi trả lời |
|---|---|---|
| 1 | Mean của train; LOO mean theo đề; luật compile | Sàn QWK, mức thổi phồng do phương sai giữa các đề |
| 2 | Chỉ dùng rule tất định (compile + pass rate + io_gap) | Tín hiệu tất định giải thích được bao nhiêu |
| 3 | LLM zero-shot không tín hiệu → + compile_log → + test_report → + tự chạy/sanitizer | Đóng góp của từng tín hiệu |
| 4 | Few-shot ngẫu nhiên vs kNN cùng đề vs kNN đa dạng điểm; k ∈ {0, 2, 4, 8} | Giá trị của RAG, copy bias |
| 5 | Không CoT vs CoT; self-consistency k ∈ {1, 5, 10} | Đánh đổi chi phí và QWK |
| 6 | Argmax vs EV decoding (vs EV + temperature scaling) | Hiệu quả của (g) |
| 7 | Round thẳng vs rescale tuyến tính vs OptimizedRounder vs quantile matching | Hậu xử lý cho QWK, đánh đổi với MAE |
| 8 | Σ argmax vs Σ E-value vs total trực tiếp vs stack | §5.5 |
| 9 | Multi: cả file vs tách câu; không gating vs gating cứng vs gating mềm; policy bằng regex vs LLM | Giá trị của xử lý multi_problem |
| 10 | SFT JSON (c) vs head MSE vs head CORN (d) | Loss có hiểu thứ tự có giúp không |
| 11 | Qwen2.5-Coder 1.5B vs 7B; LoRA r ∈ {8, 32}; max_seq 2k/4k/8k | Kích thước model, cắt ngắn input |
| 12 | Có/không test sinh và peer-oracle | Giá trị của thực thi, nhất là với multi_problem |
| 13 | Có/không hậu xử lý ràng buộc và MAP 240 tổ hợp | Exact-match từng chiều, QWK |
| 14 | Train trên toàn bộ vs bỏ các cụm near-dup có nhãn mâu thuẫn | Ảnh hưởng của nhiễu nhãn |
| 15 | Bỏ lần lượt từng nhóm feature (SHAP / drop-group) | Tầm quan trọng của từng tín hiệu |
| 16 | Có/không chèn comment kiểu "// code đúng, 10 điểm" vào bài | Độ bền trước prompt injection (dùng cho model card) |

Mỗi dòng báo cáo: QWK (mean ± CI), MAE, exact-match từng chiều kèm baseline majority, chi phí ($ hoặc GPU-hour), tách theo single/multi.

### 7.2 Các kiểu lỗi cần phân tích
- **Lệch xa (|Δ| ≥ 3):** liệt kê đủ, gom theo nguyên nhân. Đây là nơi QWK mất nhiều nhất.
- **Compile lệch toolchain:** bài dùng `conio.h` hoặc `getch` (Windows) mà nhãn compilable=1, hoặc ngược lại.
- **Pass test công khai nhưng logic thấp:** hard-code output, chỉ đúng với input nhỏ. Model dựa vào test sẽ chấm quá cao.
- **Fail test chỉ vì IO:** có prompt "Nhap n", `freopen` sót lại. Model sẽ chấm logic quá thấp.
- **multi_problem:**
  - tách câu sai;
  - trích policy sai;
  - câu tiên quyết bị đánh giá sai, kéo theo sai cả chuỗi;
  - nhãn không tuân theo policy ([GT-10]).
- **Đề mới so với đề đã gặp:** so QWK giữa hai nhóm. Khoảng cách lớn cho thấy model đang học đề chứ không học cách chấm.
- **Bias của LLM:**
  - Length bias: tương quan giữa residual và số dòng code.
  - Thiên vị code có comment tiếng Việt, hoặc tiếng Việt không dấu.
  - Bias rộng tay, đo bằng mean(ŷ − y) trước khi calibrate.
- **Bài dài bị cắt ngắn:** residual theo nhóm độ dài token.
- **Chiều suy biến:** exact-match cao nhưng không cao hơn baseline majority.
- **Nhiễu nhãn:** κ giữa các cặp near-duplicate là trần. Tách lỗi của hệ thống khỏi phần không thể sửa vì nhiễu.
- **Giảng viên chấm lệch:** nếu có metadata lớp/đợt, so sánh residual theo nhóm đó.

---

## 8. Các điểm cần kiểm tra ngay trong tuần 1
1. Script chấm tính QWK thế nào: có truyền `labels` không, xử lý điểm lẻ ra sao, validator có bắt total = tổng 6 chiều không ([GT-4]).
2. total có bằng tổng 6 chiều không, chia theo type ([GT-1]).
3. Dev/test có đề mới không (§6.1). Câu trả lời quyết định kNN và peer-oracle có dùng được hay không.
4. Toolchain trong compile_log, và compile_log/test_report có đủ trong dev/test không:
   ```python
   dev[["compile_log","test_report"]].isna().mean()
   ```
5. JSON có điểm từng câu cho multi_problem không ([GT-11]).
6. Hỏi giảng viên:
   - Cấu hình chính sách chấm theo từng đề viết tay có được phép không.
   - Feedback có được dùng làm *target* rationale khi train không. Đề chỉ cấm dùng feedback làm input.

Status: DONE
Summary: Tài liệu đào sâu Task 1. Điểm then chốt: QWK đúng bằng Lin's CCC, nên cần giãn dự đoán (de-shrink) thay vì co về giữa; nếu scorer gọi sklearn không truyền `labels` thì trọng số tính theo chỉ số; có 9 hướng mô hình và 16 dòng ablation.
Concerns/Blockers: Chưa có dữ liệu, mọi [GT] cần kiểm chứng. Chưa xác minh bằng WebSearch: Qwen3-Embedding dùng cho C++, perf trong Docker, logprobs của từng API (đều đã ghi "(cần kiểm chứng)").