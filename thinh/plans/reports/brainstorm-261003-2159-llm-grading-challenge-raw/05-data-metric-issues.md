# Audit dữ liệu và metric: kế hoạch săn điểm thưởng (+10%)

## 0. Quy ước chung

**Tên trường dưới đây đều là giả định**, phải sửa lại sau khi tải dữ liệu: `sample_id, split, type, problem_id, code_file, compile_log, test_report, rubric.{6 chiều}, total, labels(list), feedback, level`. Toàn bộ số liệu đều là giả thuyết. Ngưỡng nêu trong tài liệu là ngưỡng đề xuất, cần nhóm thống nhất trước khi chạy để tránh p-hacking.

```python
import json, re, unicodedata as ud, numpy as np, pandas as pd, pathlib
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
ROOT = pathlib.Path('data')
S   = ['compilable','io_format','logic','edge_case','complexity','code_quality']
MAX = dict(compilable=1, io_format=1, logic=4, edge_case=2, complexity=1, code_quality=1)
def load(split):
    df = pd.json_normalize(json.load(open(ROOT/f'{split}.json')))   # tên file: giả định
    df.columns = [c.split('.')[-1] for c in df.columns]; df['split'] = split
    df['code'] = df.code_file.map(lambda p: (ROOT/p).read_bytes().decode('utf-8','replace'))
    return df
df = pd.concat([load(s) for s in ['train','dev']]); te = load('test')   # test: chỉ có input
def boot(metric, y, p, B=1000, seed=0):                                   # CI 95% dùng chung
    rng = np.random.default_rng(seed); n = len(y)
    s = [metric(y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(s, [2.5, 50, 97.5])
```

**Nguyên tắc cho mọi phát hiện**
- Luôn báo cáo n, tỷ lệ % và CI 95% (bootstrap, seed cố định).
- Lưu danh sách `sample_id` vi phạm ra file trong repo private.
- Được xem tay mẫu **train/dev** để phân loại nguyên nhân. **Không** gán nhãn test. Phân tích *input* của test (code, log) thì hợp lệ.

**Các fact về metric đã kiểm trên tài liệu và mã nguồn scikit-learn (bản stable hiện tại là 1.9.1)**
- `cohen_kappa_score` dựng ma trận trọng số từ `arange(n_classes)`, tức là theo **vị trí nhãn**, không theo giá trị.
- Khi mẫu số bằng 0, hàm trả `replace_undefined_by` (tham số thêm ở bản 1.9, mặc định `nan`).
- `f1_score(zero_division=...)` nhận `"warn"` (tính như 0), `0`, `1` hoặc `np.nan` (thêm ở bản 1.3). Với `np.nan`, nhãn đó bị loại khỏi trung bình macro.
- `confusion_matrix(labels=...)` **âm thầm bỏ** mẫu có giá trị không nằm trong `labels`. Điều này suy ra từ mã nguồn, nên tự kiểm lại trên máy.
- `np.round` / `round` / `pandas.round` làm tròn kiểu banker's: `np.round(2.5)=2`, `np.round(3.5)=4`.

---

## 1+2. Danh sách 35 giả thuyết

Mỗi giả thuyết gồm 4 phần: **kiểm chứng định lượng**, **ngưỡng kết luận**, **đề xuất sửa** (cho dữ liệu hoặc metric), **cách hệ thống khai thác hoặc phòng thủ**.

### (A) Nhất quán nội tại của nhãn

**A0. Vệ sinh schema: nhãn có nhiều biến thể chuỗi, tập rỗng bị validator từ chối**
- **Giả thuyết:**
  - Nhãn tiếng Việt lẫn NFC và NFD (dạng dựng sẵn và dạng tổ hợp dấu), lẫn hoa/thường, thừa khoảng trắng, làm số nhãn duy nhất vượt 10.
  - Level ghi lẫn `1` / `"Level 1"` / `"L1"`.
  - `sample_id` bị trùng.
  - Validator coi `[]` hoặc `null` là sai định dạng, trong khi đề nói tập rỗng là hợp lệ.
- **Kiểm:**
  ```python
  raw = df.labels.explode().dropna()
  print(raw.nunique(), raw.map(lambda s: ud.normalize('NFC', s).strip().casefold()).nunique())
  print(df.level.map(repr).value_counts(), df.sample_id.duplicated().sum(), df.isna().mean())
  ```
  Chạy **validator cục bộ** (không tốn quota) trên các file: nhãn NFD, `[]`, `null`, nhãn lạ.
- **Ngưỡng:** `raw.nunique() > 10`, hoặc số nhãn đổi sau khi chuẩn hoá, hoặc có ≥1 `sample_id` trùng → có lỗi. Validator từ chối `[]` → mâu thuẫn với đặc tả.
- **Sửa:** dùng vocabulary cố định dạng NFC. Scorer chuẩn hoá chuỗi trước khi so khớp, nếu không một đội xuất NFD sẽ bị 0 ở nhãn đó.
- **Hệ thống:** hậu xử lý để map mọi output về đúng chuỗi trong vocabulary (dùng constrained decoding hoặc fuzzy match), sau đó chạy validator trong CI.

**A1. Tổng ≠ tổng 6 chiều; điểm vượt thang; điểm không nằm trên lưới**
- **Kiểm:**
  ```python
  d = df[S].sum(1) - df.total
  print(df.assign(bad=d.abs()>1e-6).groupby('type').bad.agg(['mean','sum']))
  for c in S: print(c, ((df[c]<0)|(df[c]>MAX[c])).sum(), ((df[c]*2) % 1 != 0).sum(), df[c].isna().sum())
  ```
  Với multi_problem: nếu có điểm từng câu và trọng số, kiểm thêm `total ≈ Σ w_i·s_i`.
- **Ngưỡng:** đây là ràng buộc tất định, nên chỉ cần ≥1 mẫu vi phạm là đã là phát hiện. Phân biệt lệch do làm tròn (`|d| ≤ 0.25`) với lệch thật. Tỷ lệ >1% thì là lỗi có hệ thống.
- **Sửa:** công bố định nghĩa chính thức của `total`, kèm validator schema.
- **Hệ thống:**
  - Nếu `total ≡ Σ dims` ở ≥99% mẫu: dự đoán 6 chiều rồi cộng lại, đảm bảo nhất quán.
  - Nếu không: thêm một head riêng cho `total` và so hai cách trên dev.

**A2. Tam giác `compilable` ↔ `compile_log` ↔ nhãn "Lỗi biên dịch"**
- **Kiểm:**
  ```python
  err = df.compile_log.fillna('').str.contains(r'\berror\b|error C\d{4}|fatal error', case=False)
  ce  = df.labels.map(lambda L: 'Lỗi biên dịch' in L)
  print(pd.crosstab([df.compilable, err], ce, margins=True))
  print(df[df.compilable==0][['io_format','logic','edge_case']].describe())   # chính sách điểm khi không biên dịch được
  ```
- **Ngưỡng:**
  - Có vấn đề nếu `(compilable=1 & err)` hoặc `(compilable=0 & ¬err)` chiếm >2%.
  - Nếu `compilable==0 ⇔ ce` khớp ≥99% thì nhãn CE dư thừa: nó suy ra được tất định từ log nên chiếm "miễn phí" 1/10 macro-F1.
  - Nếu P(logic>0 | compilable=0) nằm trong [0.1, 0.9] thì chính sách "không biên dịch được" không thống nhất giữa các lần chấm.
- **Sửa:** chuẩn hoá bằng cách biên dịch lại (xem D1). Công bố chính sách cho các chiều còn lại khi `compilable=0`.
- **Hệ thống:** `compilable` và nhãn CE xử lý bằng rule hoặc một cây quyết định nhỏ trên đặc trưng của log, không cần LLM. Dành ngân sách LLM cho logic và edge_case.

**A3. 10/10 mà vẫn có nhãn lỗi; tập lỗi rỗng mà điểm thấp; chiều đạt max mà vẫn có nhãn lỗi tương ứng**
- **Kiểm:**
  ```python
  n = df.labels.str.len()
  print(((df.total==10)&(n>0)).mean(), ((n==0)&(df.total<=5)).mean())
  for c, lab in {'logic':'Lỗi logic', 'edge_case':'Lỗi biên'}.items():      # tên nhãn: giả định
      print(c, pd.crosstab(df[c]==MAX[c], df.labels.map(lambda L: lab in L), normalize='index'))
  ```
- **Ngưỡng:** có vấn đề nếu P(nhãn lỗi | chiều tương ứng đạt max) >5%, hoặc P(tập rỗng | total ≤ 5) >5%. Sau đó xem tay 20 mẫu train để tách hai khả năng: "lỗi nhẹ, không trừ điểm" (quy ước) hay mâu thuẫn thật.
- **Sửa:** công bố quan hệ giữa taxonomy và rubric (nhãn lỗi có bắt buộc kéo theo trừ điểm không).
- **Hệ thống:** giải mã chung T1+T2 với ràng buộc mềm, **chỉ khi** ràng buộc đúng ≥95% trên train. Có ablation có/không ràng buộc.

**A4. Nhãn của giảng viên tự vi phạm chính sách tiên quyết (multi_problem)**
- **Kiểm:**
  - Nếu có điểm từng câu và đồ thị tiên quyết (giả định trường `per_q`, `prereq`):
    ```python
    def viol(r): return any(r.per_q[q] > 0 and any(r.per_q[p] == 0 for p in r.prereq.get(q, [])) for q in r.per_q)
    mp = df[df.type=='multi_problem']; print(mp.apply(viol, axis=1).agg(['mean','sum']), mp.groupby('problem_id').apply(lambda g: g.apply(viol,axis=1).mean()))
    ```
  - Nếu chỉ có điểm cấp file: tách file thành từng câu, chạy test riêng từng câu để xác định câu tiên quyết sai, rồi so với logic gold.
  - Phương án phụ: regex trên feedback kiểu "câu 1 sai". Việc này chỉ dùng cho EDA, không làm input của model.
- **Ngưỡng:** ≥1 vi phạm đã xác thực là có phát hiện. Tỷ lệ >5%, hoặc vi phạm dồn vào vài đề, nghĩa là chính sách được áp dụng không nhất quán.
- **Sửa:** viết chính sách thành dữ liệu máy đọc được (`policy.json` theo từng đề), chấm lại các mẫu vi phạm.
- **Hệ thống:**
  - Thêm lớp rule thực thi chính sách sau LLM, làm ablation "LLM thuần" vs "LLM + policy layer".
  - Nếu gold vi phạm chính sách, rule có thể **giảm** QWK trên dev. Đây là nghịch lý "metric phạt hệ thống đúng chính sách", là bằng chứng mạnh cho điểm thưởng.

**A5. Level phản hồi mâu thuẫn nội dung: gold L1/L2 chứa code, gold L4 không có lời giải**
- **Kiểm:**
  ```python
  CODE = re.compile(r'```|#include|\b(cout|cin|printf|scanf)\b|\b(for|while|if)\s*\(.*\)|;\s*$', re.M)
  df['code_lines'] = df.feedback.map(lambda f: sum(bool(CODE.search(l)) for l in f.splitlines()))
  print(df.groupby('level').agg(has_code=('code_lines', lambda x: (x>0).mean()), len_=('feedback', lambda x: x.str.len().median())))
  ```
  Đo precision/recall của detector bằng cách xem tay 50 mẫu train.
- **Ngưỡng:** có vấn đề nếu:
  - L1/L2 chứa code >5% (sau khi hiệu chỉnh theo FP của detector): gold vi phạm chính quy tắc của metric, áp metric lên chính giảng viên thì không đạt 100%.
  - L4 không có code >20%.
  - Độ dài trung vị không tăng đơn điệu theo level.
- **Sửa:** gán lại level theo nội dung, hoặc loại các mẫu này khỏi tập đánh giá compliance. Công bố detector chính thức.
- **Hệ thống:** lọc mẫu L1/L2 vi phạm khỏi dữ liệu SFT, ablation lọc/không lọc và đo compliance. Thêm guard hậu kiểm: phát hiện code thì sinh lại.

**A6. Taxonomy chồng lấn; quan hệ cha–con được gán không nhất quán**
- **Kiểm:**
  ```python
  mlb = MultiLabelBinarizer(); Y = mlb.fit_transform(df.labels); C = Y.T @ Y
  P = pd.DataFrame(C / np.diag(C)[:, None], mlb.classes_, mlb.classes_)   # P.loc[a,b] = P(b | a)
  ```
  Chú ý các cặp ngữ nghĩa lồng nhau, ví dụ "Lỗi vòng lặp" → "Lỗi logic", "Lỗi hàm" → "Lỗi logic".
- **Ngưỡng:** P(cha | con) nằm trong [0.2, 0.8] → lúc gán cả cha, lúc chỉ gán con. Jaccard giữa hai nhãn >0.8 → hai nhãn gần trùng nhau.
- **Sửa:** thêm guideline gán nhãn cha–con, hoặc dùng hierarchical F1.
- **Hệ thống:** học quy ước từ dữ liệu (classifier chain hoặc prior đồng xuất hiện). Không tự suy ra nhãn cha bằng logic nếu gold không làm vậy.

### (B) Phân bố

**B1. Nhãn taxonomy suy biến (quá hiếm)**
- **Kiểm:** `pd.DataFrame(Y, columns=mlb.classes_).groupby(df.split.values).agg(['sum','mean']).T`. Số positive kỳ vọng trong test = prevalence_train × n_test. Bootstrap F1 từng nhãn trên dev.
- **Ngưỡng:** nhãn có <1% hoặc <10 positive ở dev → CI của F1 nhãn đó rộng hơn ±0.2, và một nhãn như vậy quyết định 10% macro-F1.
- **Sửa:** gộp nhãn hiếm; chỉ tính macro-F1 trên các nhãn có support ≥k, kèm CI. Split dùng iterative stratification (package `iterative-stratification`).
- **Hệ thống:** nhãn hiếm dùng rule, keyword hoặc few-shot chuyên biệt, kèm threshold riêng. Một TP duy nhất có thể đổi thứ hạng.

**B2. Chiều rubric gần như hằng số (toàn cục hoặc theo từng đề)**
- **Kiểm:** `df[S].apply(lambda c: c.value_counts(normalize=True).iloc[0])`; `(df.groupby('problem_id')[S].nunique()==1).mean()`.
- **Ngưỡng:** lớp đa số >95% (hoặc entropy <0.3 bit) → exact-match của chiều đó gần như không mang thông tin, vì baseline đa số đã đạt ≥95%.
- **Sửa:** báo cáo exact-match kèm baseline đa số, hoặc dùng Cohen κ thay accuracy.
- **Hệ thống:** chiều suy biến thì dự đoán theo đa số trong đề. Báo cáo κ để trung thực.

**B3. Điểm tổng dồn cục hoặc lưỡng đỉnh (0 và 10), có hạng điểm trống**
- **Kiểm:** `df.total.value_counts().sort_index()`; tách theo type và môn; `df.total.isin([0,10]).mean()`.
- **Ngưỡng:**
  - Có giá trị nguyên trong [0,10] với 0 mẫu ở dev → QWK của sklearn khi `labels=None` sẽ co khoảng cách (xem F1).
  - Đỉnh ở 10 >30% → MAE bị chi phối bởi các bài đúng hoàn toàn.
- **Sửa:** QWK với `labels` cố định; báo cáo metric theo từng tầng điểm.
- **Hệ thống:** dùng ordinal regression hoặc classification có calibrate. Tránh hồi quy MSE thuần vì nó kéo dự đoán về giữa.

**B4. Phương sai giữa các đề lấn át phương sai trong đề**
- **Kiểm:**
  ```python
  tr, dv = df[df.split=='train'], df[df.split=='dev']
  mu = tr.groupby('problem_id').total.mean()
  pred = dv.problem_id.map(mu).fillna(tr.total.mean()).round().astype(int)
  print(cohen_kappa_score(dv.total.round().astype(int), pred, weights='quadratic', labels=range(11)))
  ```
  Tính thêm η² = SS_between / SS_total của total theo problem_id.
- **Ngưỡng:** baseline "trung bình của đề" đạt QWK ≥0.4 → leaderboard thưởng việc nhận ra đề khó, chưa phải năng lực chấm từng bài.
- **Sửa:** QWK trung bình theo đề (macro qua các đề), hoặc tính QWK trên phần dư.
- **Hệ thống:** dùng prior theo đề như một feature (hợp lệ). Báo cáo thêm QWK theo từng đề để chứng minh hệ thống thật sự chấm bài.

**B5. Lệch phân phối giữa train, dev và test**
- **Kiểm:** test chỉ có input nên so trên input:
  - tỷ lệ theo type;
  - `set(te.problem_id) - set(tr.problem_id)`;
  - độ dài code, tỷ lệ log có error, pass rate trong test_report;
  - adversarial validation: LightGBM phân biệt train với test trên các đặc trưng này, đo AUC.
- **Ngưỡng:** AUC >0.7, hoặc ≥20% đề trong test không có ở train → dev không đại diện cho test, public LB gây hiểu lầm.
- **Sửa:** công bố thiết kế split.
- **Hệ thống:** nếu test có đề mới thì validation bằng GroupKFold theo problem_id, giảm phụ thuộc vào prior theo đề, RAG theo bài tương tự thay vì theo cùng đề.

**B6. Level lệch phân bố hoặc là hàm của điểm**
- **Kiểm:** `pd.crosstab(df.level, pd.cut(df.total, [-.1,3,6,8,10]), normalize='columns')` và Cramér's V; phân bố level theo đề.
- **Ngưỡng:** một level chiếm <5% → compliance của level đó nhiễu. V >0.5 → level gần như suy ra từ điểm; model fine-tune sẽ học "điểm → kiểu phản hồi" thay vì tuân theo level yêu cầu.
- **Sửa:** cân bằng level trong dev và test của Task 3; báo cáo compliance theo từng level.
- **Hệ thống:** augment bằng cách đổi level (cùng code và nhãn, dùng teacher LLM sinh phản hồi ở cả 4 level, báo cáo chi phí). Kiểm bằng thí nghiệm: đổi level ở input mà output không đổi thì model đang bỏ qua level.

### (C) Trùng lặp, rò rỉ, toàn vẹn

**C1. Code trùng hoặc gần trùng trong train mà nhãn khác nhau (nhiễu nhãn, từ đó ước lượng trần hiệu năng)**
- **Kiểm:** chuẩn hoá 2 mức. Mức 1: bỏ comment và khoảng trắng, đổi định danh thành `ID`, **giữ nguyên chuỗi** vì chuỗi ảnh hưởng io_format. Mức 2: bỏ luôn chuỗi.
  ```python
  KW = set('int long double char bool void if else for while do return include using namespace std cin cout endl vector string'.split())
  def norm(code, keep_str=True):
      code = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.S|re.M)
      if not keep_str: code = re.sub(r'"(\\.|[^"\\])*"', '"S"', code)
      return ' '.join(t if (t in KW or not re.match(r'[A-Za-z_]', t)) else 'ID' for t in re.findall(r'[A-Za-z_]\w*|\d+|\S', code))
  df['h'] = df.code.map(norm).map(hash)
  dup = df[df.duplicated(['problem_id','h'], keep=False)]
  print(dup.groupby(['problem_id','h']).total.agg(lambda s: s.max()-s.min()).describe())
  ```
  Gần trùng: MinHash LSH (thư viện `datasketch`) trên 5-gram token, Jaccard ≥0.9, cùng đề. Mỗi cặp tính |Δtotal|, tỷ lệ khác ở từng chiều, Jaccard của tập nhãn. **QWK giữa hai phần tử trong cặp** dùng làm proxy inter-rater, tức trần hiệu năng.
- **Ngưỡng:** có nhiễu nhãn đáng kể nếu trong các cặp trùng sau chuẩn hoá: P(Δtotal ≠ 0) >10%, hoặc QWK của cặp <0.85. Chiều có tỷ lệ khác cao nhất (dự đoán là code_quality) là chiều chủ quan nhất.
- **Sửa:** adjudication, lấy median hoặc majority; công bố trần hiệu năng.
- **Hệ thống:**
  - Khử trùng train bằng cách gộp nhãn, có ablation.
  - kNN-RAG: đưa bài gần trùng kèm nhãn train vào few-shot (hợp lệ).
  - Dùng trần để diễn giải kết quả, ví dụ "hệ thống đạt x% trần".

**C2. Trùng giữa các split (train ↔ dev/test)**
- **Kiểm:** dùng chung chỉ mục LSH với C1. Tính tỷ lệ mẫu dev/test có láng giềng trong train với Jaccard ≥0.9 (với test chỉ cần code). Với dev, kiểm nhãn có khớp nhãn của bản trùng không.
- **Ngưỡng:** >2% test có bản gần trùng trong train → điểm bị thổi phồng nhờ ghi nhớ. Nhãn lệch giữa các bản trùng >10% → trần trên dev.
- **Sửa:** group split theo cụm trùng.
- **Hệ thống:** tra kNN vẫn hợp lệ, nhưng báo cáo điểm tách hai tập con có/không có bản trùng để chứng minh khả năng khái quát.

**C3. Code khung do giảng viên phát và bài nộp gần rỗng**
- **Kiểm:** với mỗi đề, lấy code phổ biến nhất hoặc medoid làm khung; độ mới = 1 − Jaccard(code, khung); `df[novelty < 0.05].total.describe()`; đếm file <200 byte.
- **Ngưỡng:** bài gần như nguyên khung mà total ≥3 → nhãn đáng ngờ, hoặc chính sách cho điểm khung chưa được công bố. Nếu khung chiếm đa số token thì C1 sẽ báo gần trùng giả, phải trừ khung trước khi so.
- **Sửa:** trừ khung khi tính similarity; công bố chính sách cho bài nộp nguyên khung.
- **Hệ thống:** đưa phần diff so với khung vào LLM. Cách này giảm context và tập trung vào phần sinh viên tự viết.

**C4. Rò rỉ nhãn qua trường không phải feedback (thứ tự ID, test_report, đường dẫn)**
- **Kiểm:**
  - Với mỗi trường metadata: mutual information hoặc R² của model chỉ dùng trường đó để đoán total.
  - Spearman(thứ hạng `sample_id` trong đề, total).
  - Regex `điểm|score` trong test_report và compile_log.
- **Ngưỡng:** một trường không có lý do ngữ nghĩa mà giải thích >10% phương sai, hoặc |ρ| >0.2 → rò rỉ.
- **Sửa:** băm hoặc xáo lại ID, xoá trường đó.
- **Hệ thống:** **không khai thác** (mất điểm phương pháp, rủi ro liêm chính). Báo riêng cho giảng viên trước. Chạy thêm test "xáo trộn trường nghi rò rỉ" để chứng minh hệ thống của nhóm không phụ thuộc vào nó.

**C5. Ẩn danh còn sót, hoặc chính quá trình ẩn danh làm hỏng code**
- **Kiểm (chỉ đếm, không in nội dung):**
  - Regex trên comment và chuỗi: `\b\d{8,10}\b`, `MSSV|Họ\s*tên|Ho\s*ten|@\w+\.(edu|com)`. Báo cáo số file theo từng pattern.
  - Đếm token placeholder (ví dụ dạng `SV_xxxx`, giả định) nằm *trong định danh hoặc chuỗi in ra*.
  - So tỷ lệ biên dịch lại thất bại giữa file có và không có placeholder (Fisher exact test).
- **Ngưỡng:** ≥1 file còn sót → báo riêng cho giảng viên. Tỷ lệ hỏng ở file có placeholder cao hơn rõ (p <0.01) → code sau ẩn danh khác code đã được chấm, nên nhãn không còn khớp code.
- **Sửa:** chỉ ẩn danh trong comment, kiểm biên dịch lại sau khi ẩn danh.
- **Hệ thống:** bỏ comment trước khi đưa vào LLM (giảm rủi ro PII, giảm token). Không ghi raw comment vào log.
- **Đạo đức:** tuyệt đối không thử khôi phục danh tính; trong báo cáo chỉ có số đếm.

**C6. Toàn vẹn file**
- **Kiểm:**
  - File không tồn tại hoặc rỗng.
  - Mã đề trong path khác `problem_id`.
  - Một `code_file` được nhiều `sample_id` dùng chung.
  - Encoding: `charset_normalizer.from_bytes(b).best().encoding`, BOM, CRLF, số ký tự `\ufffd` sau khi decode.
- **Ngưỡng:** ≥1 trường hợp là đủ thành phát hiện cụ thể.
- **Sửa:** manifest kèm checksum.
- **Hệ thống:** decode lần lượt utf-8 → cp1258 → latin-1, log số file phải dùng fallback.

### (D) Tín hiệu phụ

**D1. Không tái lập được `compile_log`**
- **Kiểm:**
  - Biên dịch lại trong sandbox: Docker `--network none`, có timeout, **chỉ compile, không chạy**: `g++ -std=c++17 -fsyntax-only`. Thử thêm c++11 và clang.
  - So kết quả với `compilable` và lỗi trong log gốc.
  - Phân loại nguyên nhân lệch bằng regex: `bits/stdc++.h`, `conio.h`, VLA, `gets`, `system("pause")`.
- **Ngưỡng:** lệch >3% → log được sinh bằng compiler hoặc cờ khác. Lệch dồn vào một nhóm header → môi trường chấm là Windows/MSVC.
- **Sửa:** công bố compiler và cờ biên dịch, sinh lại log.
- **Hệ thống:** dùng cả log gốc (khớp với nhãn) lẫn log tái lập (để tổng quát). Khi hai log mâu thuẫn, ưu tiên log gốc vì nhãn được tạo từ đó.

**D2. `test_report` không khớp `logic` / `io_format`**
- **Kiểm:**
  - Tính pass_rate; Spearman(pass_rate, logic).
  - Bảng chéo: `pass_rate==1 & logic<4` và `pass_rate==0 & logic>=3`.
  - Chạy lại test trong sandbox, so khớp sau khi bỏ khoảng trắng: tỷ lệ test chuyển từ fail sang pass là "lỗi chỉ do định dạng", đem so với io_format gold.
- **Ngưỡng:** ρ <0.5, hoặc >15% bài pass hết test mà logic <4 → testcase yếu hoặc giảng viên dùng hidden test. io_format=1 trong khi có fail chỉ do định dạng ở >5% mẫu → io_format không bám theo test_report.
- **Sửa:** công bố hidden test; định nghĩa io_format theo checker.
- **Hệ thống:** dùng pass_rate và cờ "fail chỉ do định dạng" làm feature cho io_format; đưa diff output cho LLM đọc.

**D3. `test_report` bị thiếu không ngẫu nhiên (MNAR) hoặc không đồng bộ**
- **Kiểm:**
  - `df.test_report.isna().groupby([df.split, df.type]).mean()`.
  - test_report có mặt dù compile lỗi.
  - Số test khác nhau giữa các bài cùng đề (dấu hiệu nhiều phiên bản testset).
  - So phân bố total giữa nhóm thiếu và nhóm có (KS test).
- **Ngưỡng:** tỷ lệ thiếu giữa train và test lệch >5 điểm %, hoặc KS p <0.01 → việc thiếu tự nó mang thông tin.
- **Sửa:** sinh lại test_report đồng bộ cho mọi mẫu.
- **Hệ thống:** thêm cờ "thiếu"; khi train, ngẫu nhiên bỏ trường phụ để model chịu được trường hợp test thiếu.

**D4. Testcase công khai yếu, đo theo từng đề**
- **Kiểm:** theo từng đề, AUC(pass_rate → logic==4), số lượng test, tỷ lệ bài pass hết mà logic <4.
- **Ngưỡng:** AUC <0.7 → bộ test của đề đó gần như không phân biệt được bài đúng/sai.
- **Sửa:** bổ sung test biên.
- **Hệ thống:**
  - Dùng LLM sinh test biên, chạy differential testing so với một lời giải tham chiếu.
  - Lời giải tham chiếu có thể lấy từ feedback Level 4 trong train. **Đây là vùng xám** so với quy định cấm dùng feedback cho Task 1/2, phải hỏi giảng viên trước khi dùng.

**D5. multi_problem: một câu lỗi biên dịch kéo theo cả file**
- **Kiểm:** P(logic>0 | compilable=0), so giữa multi_problem và single_problem.
- **Ngưỡng:** chênh >20 điểm % → chiều `compilable` mang nghĩa khác nhau tùy loại đề.
- **Sửa:** chấm `compilable` theo từng câu.
- **Hệ thống:** tách file theo hàm hoặc marker comment, biên dịch từng phần, đưa kết quả cho từng câu vào prompt.

### (E) Người chấm

**E1. Độ lệch giữa người chấm (leniency)**
- **Kiểm:**
  - Nếu có mã người chấm hoặc mã lớp ẩn danh: `statsmodels` `mixedlm('total ~ pass_rate + C(problem_id)', groups=grader)`, tính ICC và random intercept.
  - Nếu không có: dùng proxy là cụm phong cách feedback (TF-IDF char n-gram + KMeans; chỉ để phân tích), rồi so độ lệch điểm giữa các cụm sau khi kiểm soát pass_rate.
- **Ngưỡng:** chênh leniency ≥0.5 điểm, hoặc variance component >10% phương sai phần dư.
- **Sửa:** hiệu chỉnh bằng many-facet Rasch; chấm đôi khoảng 10% bài.
- **Hệ thống:** nếu mã lớp có trong input test thì dùng làm feature sẽ tăng điểm leaderboard. Khi đó vẫn phải báo cáo thêm một bản không dùng thông tin người chấm cho mục đích triển khai, và thảo luận về công bằng.

**E2. Drift theo thời gian hoặc thứ tự chấm**
- **Kiểm:** nếu có timestamp hoặc thứ tự: phần dư = total − f(pass_rate, đề); Spearman(thứ tự, phần dư) trong từng đề; độ dài feedback theo thứ tự (dấu hiệu mệt mỏi).
- **Ngưỡng:** |ρ| >0.15 với p <0.01, hoặc nửa đầu và nửa sau lệch ≥0.5 điểm.
- **Sửa:** hiệu chỉnh, hoặc chấm lại một mẫu.
- **Hệ thống:** không dùng thứ tự làm feature.

**E3. Hai môn hiểu "rubric chung" khác nhau**
- **Kiểm:** `df.groupby('course')[S].agg(['mean','std'])` (trường môn học là giả định, có thể suy từ mã đề); liệt kê nhãn chỉ xuất hiện ở một môn.
- **Ngưỡng:** ví dụ complexity gần hằng ở Cơ sở lập trình nhưng biến thiên ở Cấu trúc dữ liệu & Giải thuật → cùng tên chiều nhưng ngữ nghĩa khác.
- **Sửa:** mô tả rubric riêng cho từng môn.
- **Hệ thống:** đưa môn học vào prompt; few-shot lấy từ cùng môn.

**E4. Feedback rập khuôn**
- **Kiểm:** tỷ lệ feedback trùng hệt sau chuẩn hoá; TF-IDF cosine >0.9; tỷ lệ feedback <20 từ; tách theo level.
- **Ngưỡng:** >20% feedback là template → "độ chính xác chẩn đoán" so với các tham chiếu này gần như không mang thông tin.
- **Sửa:** loại template khỏi tập tham chiếu Task 3; dùng nhiều tham chiếu.
- **Hệ thống:** giảm trọng số hoặc lọc template khỏi dữ liệu SFT để model không học trả lời chung chung.

### (F) Metric và leaderboard

**F1. QWK trên điểm tổng có phần lẻ: làm tròn và trọng số theo vị trí nhãn**
- **Vấn đề:**
  - Trọng số quadratic tính theo **vị trí nhãn** trong tập nhãn xuất hiện. Với `labels=None`, nếu không ai được 3 điểm thì khoảng cách 2→4 bị tính như 1 bước.
  - Điểm 7.5 thành một hạng riêng.
  - Truyền `labels=range(11)` thì các mẫu có điểm 7.5 **bị loại âm thầm**.
  - Làm tròn mặc định là banker's rounding.
- **Kiểm:** trên cùng một bộ dự đoán dev, so 4 phiên bản:
  1. `labels=None`;
  2. `labels=np.arange(0, 10.5, 0.5)`;
  3. làm tròn half-up về số nguyên rồi `labels=range(11)`;
  4. `np.round` (banker's) rồi `labels=range(11)`.
- **Ngưỡng:** chênh lệch >0.01 QWK, tức cỡ khoảng cách giữa các đội trên leaderboard → đặc tả metric mơ hồ làm thay đổi xếp hạng.
- **Sửa:** công bố lưới điểm, `labels` cố định và quy tắc làm tròn.
- **Hệ thống:** tự viết lại scorer, nộp 1 lần lên public để đối chiếu với điểm leaderboard; chỉ xuất giá trị nằm trên lưới.

**F2. QWK có thể tăng chỉ bằng cách khớp phân phối biên, và mâu thuẫn với MAE**
- **Kiểm:** từ dự đoán liên tục, so QWK và MAE của 3 cách hậu xử lý:
  - (a) làm tròn thường;
  - (b) tối ưu ngưỡng cắt (OptimizedRounder, Nelder-Mead, fit bằng CV trên train);
  - (c) quantile-matching về phân phối điểm của train.
- **Ngưỡng:** (b) hoặc (c) tăng QWK ≥0.02 trong khi MAE tăng → metric chính tối ưu được bằng hậu xử lý mà không chấm từng bài tốt hơn.
- **Sửa:** xếp hạng bằng chỉ số gộp, hoặc dùng MAE để phá hoà; báo cáo kèm exact-match từng chiều.
- **Hệ thống:** dùng (b) là hợp lệ, nhưng phải báo cáo minh bạch kèm ablation.

**F3. QWK suy biến, và QWK tổng che sai số từng chiều**
- **Vấn đề:** dự đoán hằng số cho κ = 0 đúng (không báo lỗi). Cả hai phía đều hằng thì trả `nan`, nên QWK theo từng đề nhỏ rất dễ ra `nan`.
- **Kiểm:** tỷ lệ mẫu dev có total đúng nhưng ≥1 chiều sai (lỗi bù trừ); số đề có QWK `nan`.
- **Ngưỡng:** >20% số mẫu total đúng có chiều sai → QWK tổng che giấu sai cấu trúc.
- **Sửa:** thêm metric "all-dim exact" và QWK riêng cho chiều logic.
- **Hệ thống:** báo cáo các metric này trong phần phân tích lỗi.

**F4. Macro-F1 với nhãn ít hoặc không có positive, và quy ước `zero_division`**
- **Vấn đề:** nếu một nhãn có 0 positive trong test:
  - đội nào đoán nhãn đó ít nhất 1 lần → F1 = 0;
  - đội không đoán → 0, 1 hoặc bị loại, tùy quy ước.
  - Với `zero_division=1`, chiến lược "không bao giờ đoán" được thưởng. Với `=0`, trần macro-F1 tụt xuống 0.9 cho mọi đội.
- **Kiểm:**
  - Tính macro-F1 dưới 3 quy ước cho ≥2 hệ thống (của nhóm và baseline), xem thứ hạng có đổi không.
  - Bootstrap 1000 lần trên dev, tính P(A > B).
  - Đếm nhãn có support bằng 0 hoặc <5 trong dev.
- **Ngưỡng:** thứ hạng đổi theo quy ước, hoặc CI 95% rộng hơn khoảng cách giữa các đội top → xếp hạng Task 2 phần lớn là nhiễu.
- **Sửa:** công bố quy ước; chỉ tính nhãn có support ≥k; dùng split phân tầng đa nhãn.
- **Hệ thống:** threshold riêng cho từng nhãn. Hỏi giảng viên quy ước chính thức trước khi chọn chiến lược cho nhãn cực hiếm.

**F5. Level compliance đạt tối đa bằng output rỗng hoặc chung chung; detector bị né được**
- **Kiểm:** chạy scorer (nếu được phát) hoặc bản tự viết, **cục bộ trên dev**, cho 4 hệ suy biến:
  1. chuỗi rỗng;
  2. template cố định "Em hãy kiểm tra lại logic";
  3. chỉ liệt kê lại nhãn lỗi đã cho sẵn;
  4. luôn đưa lời giải đầy đủ.

  Thêm các phép tấn công né detector: mô tả cách sửa bằng lời ("đổi điều kiện `i<=n` thành `i<n`"), pseudo-code, code không có dấu `;`. Kiểm thêm: L4 mà không có lời giải có bị phạt không (nghi metric bất đối xứng).
- **Ngưỡng:**
  - (1) hoặc (2) đạt compliance 100% trong khi điểm chẩn đoán không gần 0.
  - (3) đạt điểm chẩn đoán gần hệ tốt nhất.
  - Các phép né giữ được compliance → detector chỉ bắt hình thức bề mặt.
- **Sửa:**
  - Chỉ tính compliance khi phản hồi vượt ngưỡng chẩn đoán, hoặc gộp hai thành một (tích compliance × chẩn đoán).
  - Định nghĩa "vi phạm" theo ngữ nghĩa: tiết lộ thay đổi cụ thể ở mức token.
  - Dùng LLM-judge được xác thực bằng một tập nhãn tay trên dev.
- **Hệ thống:** tối ưu đồng thời cả compliance và chẩn đoán. Chỉ minh hoạ các hệ suy biến cục bộ, không đưa chúng vào bài nộp chính thức.

**F6. "Độ chính xác chẩn đoán" chỉ so với một tham chiếu**
- **Kiểm:** với các cặp code gần trùng (từ C1) có 2 feedback: chấm feedback A với B làm tham chiếu, ra trần của người. Nếu metric dùng LLM-judge hoặc embedding, đo thêm độ nhạy với độ dài và với paraphrase cùng ý.
- **Ngưỡng:** trần của người <0.7, hoặc thấp hơn hệ thống tốt nhất → metric đang đo "giống phong cách người chấm này", chưa phải đo chẩn đoán đúng.
- **Sửa:** dùng nhiều tham chiếu; chấm dựa trên nhãn lỗi và vị trí dòng lỗi.
- **Hệ thống:** phản hồi phải bao phủ mọi nhãn đã cho và chỉ ra dòng code cụ thể; few-shot theo văn phong giảng viên.

**F7. Bảng tổng xếp theo trung bình thứ hạng; overfit public leaderboard**
- **Kiểm:**
  - Bootstrap tập đánh giá để ra phân phối thứ hạng của từng đội.
  - Mô phỏng vi phạm IIA (independence of irrelevant alternatives): thêm hoặc bớt một đội làm đảo thứ tự của hai đội khác.
  - Overfit: 2 lượt/ngày × ~56 ngày ≈ 112 lượt nộp/đội. So SE của QWK trên dev (bootstrap) với khoảng cách giữa các đội.
- **Ngưỡng:** CI thứ hạng của các đội liền kề chồng lấn nhau, hoặc khoảng cách top <2×SE.
- **Sửa:** xếp theo bậc có ý nghĩa thống kê; dùng cơ chế Ladder (Blum & Hardt, ICML 2015) cho public LB; quy tắc xử lý đồng hạng rõ ràng.
- **Hệ thống:** chọn bài nộp cuối theo CV kết hợp dev, không theo đỉnh public. Lên kế hoạch dùng quota.

---

## 3. Top-8 nên kiểm trước

Các điểm 1–3 là ước lượng chủ quan, cần cập nhật sau ngày đầu chạy EDA.

| # | Giả thuyết | Khả năng đúng | Giá trị bonus | Chi phí | Lý do |
|---|---|---|---|---|---|
| 1 | **A2 (+D1)** Tam giác compilable–log–nhãn CE, biên dịch lại | 3 | 3 | Thấp, sandbox thì trung bình | Ràng buộc tất định nên bằng chứng không cãi được. Chạm cả T1 và T2; nếu CE dư thừa thì còn ảnh hưởng macro-F1 |
| 2 | **A5** Gold L1/L2 chứa code | 3 | 3 | Thấp | Đánh trúng điểm khác biệt nhất của đề (Task 3): metric phạt cả giảng viên |
| 3 | **B1+F4** Nhãn suy biến và quy ước zero_division làm đổi hạng | 3 (đề đã gợi ý) | 3 nếu có thí nghiệm đổi hạng | Thấp | Muốn được thưởng phải đi xa hơn "nhãn lệch": chứng minh thứ hạng là nhiễu |
| 4 | **C1+C2** Gần trùng với nhãn lệch, trong train và giữa các split | 2–3 | 3 | Trung bình (một chỉ mục LSH) | Ra cùng lúc nhiễu nhãn, trần hiệu năng và rò rỉ giữa split. Tái dùng cho F6 |
| 5 | **A4** Gold vi phạm chính sách tiên quyết | 2 | 3 | Thấp đến cao, tùy có điểm từng câu không | Đề nhấn mạnh chính sách này. Nghịch lý "đúng chính sách thì bị phạt" rất thuyết phục |
| 6 | **B4+F2** QWK bị chi phối bởi trung bình đề và tăng được bằng tối ưu ngưỡng | 3 | 2–3 | Thấp | Một baseline vài dòng code là chứng minh được metric chính có lỗ hổng |
| 7 | **F5** Compliance đạt tối đa bằng output rỗng hoặc template | 3 | 3 | Thấp nếu có scorer | Ít đội nghĩ tới. Đề xuất sửa (metric gộp) rõ ràng |
| 8 | **A0+A1** Schema, Unicode, tổng ≠ Σ chiều, điểm vượt thang | 2 | 1–2 | Rất thấp (dưới 1 giờ) | Gần như miễn phí. Tiếng Việt dễ lẫn NFC/NFD; lỗi này ảnh hưởng thẳng tới scorer |

Ứng viên dự bị: C4 (rò rỉ qua ID, giá trị cao nếu đúng), D2 (testcase yếu), A6 (quan hệ cha–con), E1 (người chấm).

**Thứ tự thực thi gợi ý**
- **Ngày 1–2:** một notebook `audit/00_sanity` chạy A0, A1, A2 (bảng chéo), B1–B3, E4, C6.
- **Tuần 1:** chỉ mục trùng lặp (C1–C3), detector code trong feedback (A5), các baseline cho B4 và F2.
- **Tuần 2:** sandbox để biên dịch lại và chạy lại test (D1, D2), A4, các thí nghiệm metric F1/F4/F5.
- **Tuần 3 trở đi:** E1, rồi viết báo cáo.

---

## 4. Mẫu trình bày một phát hiện để tối đa điểm thưởng

Chọn 2–4 phát hiện mạnh, mỗi phát hiện đi đủ cả chuỗi dưới đây, thay vì liệt kê 20 phát hiện rời rạc. Mỗi phát hiện chiếm khoảng nửa trang đến một trang, có một hình và một bảng.

**F-x. [Claim một câu, kiểm chứng được]**
Ví dụ: "Gold ở Level 1/2 tự vi phạm quy tắc compliance của Task 3."

1. **Định nghĩa và giả thuyết:** quy tắc trong đề, cách vận hành hoá (detector regex kèm precision/recall đo trên [n] mẫu train gán tay).
2. **Bằng chứng định lượng:**
   - "[k]/[N] mẫu L1/L2 (= [x]%, CI 95% [a, b], bootstrap B=1000, seed=0) chứa code"; tách theo level và theo đề.
   - Có ví dụ đã che (chỉ trích dòng code, không trích comment).
   - Kèm script tái lập: `make audit-A5`.
3. **Tác động lên metric và leaderboard:**
   - Áp metric compliance lên chính gold được [y]%.
   - Model SFT trên gold có compliance [z₁]%, sau khi lọc là [z₂]%.
   - Chênh lệch đó so với khoảng cách giữa các đội trên public LB (đổi được mấy hạng).
4. **Nguyên nhân khả dĩ:** chỉ nêu những gì quan sát được (vd. vi phạm tập trung ở đề X hoặc cụm phong cách Y), không đoán ý đồ người gán.
5. **Đề xuất sửa:** cho dữ liệu (gán lại level, hoặc loại khỏi tập đánh giá), cho metric (detector công bố kèm định nghĩa vi phạm theo ngữ nghĩa), có thể kèm patch cho validator hoặc scorer.
6. **Thực nghiệm chứng minh sửa có ích:**
   - Ablation trước/sau khi sửa, cùng seed và cùng phiên bản model.
   - Metric chính kèm CI; kiểm định bootstrap ghép cặp (paired bootstrap) P(sau > trước).
   - Nếu được: chỉ ra thứ hạng các hệ thống ổn định hơn khi dùng metric đã sửa.
7. **Giới hạn:** độ phủ của detector, phần chưa kiểm được (vd. test chưa có nhãn).

**Checklist ăn điểm**
- Mỗi số liệu đi kèm n, CI và script tái lập.
- Ghi rõ đâu là phân tích trên train/dev, đâu là chỉ dùng input của test.
- Mỗi phát hiện về metric phải kèm một ví dụ đổi hạng: mô phỏng hai hệ thống, đổi quy ước metric thì thứ tự đảo.
- Mỗi phát hiện về dữ liệu phải kèm một thí nghiệm "sửa rồi hệ thống tốt lên hoặc đánh giá công bằng hơn".
- Gửi riêng cho giảng viên trước các phát hiện nhạy cảm: rò rỉ (C4), PII (C5). Giữ timestamp để chứng minh phát hiện trước. Trên diễn đàn chỉ thảo luận ý tưởng.
- Phát hiện kiểu rò rỉ thì báo cáo nhưng **không khai thác**. Kỹ thuật hợp lệ như tối ưu ngưỡng QWK thì được dùng, nhưng phải nêu minh bạch.
- Phần privacy chỉ có số đếm, không trích nội dung, không thử truy ngược danh tính.

---

## Câu hỏi cần hỏi giảng viên hoặc diễn đàn (chưa có câu trả lời)
1. Scorer QWK dùng lưới điểm, `labels` và quy tắc làm tròn nào? Điểm tổng có phần lẻ không?
2. Macro-F1 dùng quy ước `zero_division` nào? Có loại nhãn có support 0 trong test không?
3. Detector level compliance dùng regex hay LLM-judge? "Độ chính xác chẩn đoán" được tính bằng gì?
4. Có được dùng feedback Level 4 của train làm lời giải tham chiếu cho testing (feature Task 1/2) không? Đây là vùng xám so với quy định cấm dùng feedback.
5. Chính sách tiên quyết có sẵn ở dạng máy đọc được theo từng đề không? Có điểm từng câu không?
6. Có trường người chấm, mã lớp hoặc timestamp không?
7. Test có chứa đề không xuất hiện trong train không?