# TASK 2: Phân loại lỗi đa nhãn (10 nhãn, macro-F1)

> Quy ước: **[GT]** là giả thuyết về dữ liệu, chưa kiểm chứng; đi kèm luôn là phép kiểm tra. Tên field (`labels`, `rubric.compilable`, `total`, `problem_type`, `problem_id`) là tên giả định, phải đổi theo schema thật. Đề bài chỉ nêu 4/10 nhãn (Lỗi biên dịch, Lỗi logic, Lỗi vòng lặp, Lỗi hàm). 6 nhãn còn lại nhắc tới bên dưới (biên, I/O, mảng/con trỏ, tràn số, độ phức tạp, code quality) đều là **[GT]**. Việc đầu tiên cần làm:
```python
from collections import Counter
cnt = Counter(l for s in df.labels for l in s); print(cnt, len(df))
df['card'] = df.labels.str.len(); print(df.card.value_counts().sort_index())   # tỉ lệ tập rỗng, cardinality
print(df.labels.apply(lambda s: tuple(sorted(s))).value_counts().head(30))      # số label-set khác nhau
```

---

## 1. Toán học của macro-F1 đa nhãn

### 1.1 Định nghĩa và hệ quả trực tiếp
- `F1_l = 2TP_l / (2TP_l + FP_l + FN_l)`, `macroF1 = (1/10)·Σ_l F1_l`. **TN không xuất hiện**, nên bài sạch đoán đúng là ∅ không được cộng điểm nào. Bài sạch chỉ có thể làm mất điểm, qua FP.
- Có **hai biến thể** macro-F1 (Opitz & Burst 2019, "Macro F1 and Macro F1"): (a) trung bình các F1_l, như sklearn `average='macro'`; (b) harmonic mean của macro-P và macro-R. Hai biến thể xử lý nhãn rỗng khác nhau. Cần hỏi BTC scorer dùng biến thể nào, vì đề chỉ phát script kiểm tra định dạng.
- **Mỗi nhãn chiếm 0.1 macro-F1 bất kể support.** Minh họa số học, không phải dữ liệu:
  - Nhãn hiếm có 5 positive, đoán 5 bài, đúng 3: F1 = 6/10 = 0.6. Đúng thêm 1 bài: 8/10 = 0.8, tức **+0.02 macro**.
  - Nhãn phổ biến có 300 positive, TP=240, FP=FN=60: F1 = 0.800. Đúng thêm 1 bài: 0.8033, tức **+0.0003 macro**.
  - Vậy **một mẫu nhãn hiếm đáng giá khoảng 60 lần** một mẫu nhãn phổ biến. Thời gian, augmentation và tuning nên dồn cho nhãn hiếm.
- Hệ quả về **phương sai**: F1 nhãn hiếm trên dev/test nhảy ±0.2 chỉ vì 1 mẫu. Public LB (dev, 2 lượt/ngày) nhiễu, nên **không tune theo LB**. Hãy ước lượng std của macro-F1 bằng bootstrap OOF với kích thước bằng tập dev và bỏ qua các chênh lệch trên LB nhỏ hơn khoảng 1 std.
- **Baseline hằng số của từng nhãn** (prevalence π): đoán toàn 1 cho `F1 = 2π/(1+π)`. Ví dụ π=0.02 → 0.039; π=0.1 → 0.18; π=0.5 → 0.67; π=0.9 → 0.95. Mọi model phải thắng mốc này **trên từng nhãn**. Nếu OOF F1 của model thấp hơn mốc thì fallback về hằng số.

### 1.2 Nhãn "gần suy biến": ba kiểu
| Kiểu | Ảnh hưởng tới macro-F1 | Cách kiểm chứng |
|---|---|---|
| **Gần luôn 0** (π rất nhỏ) | F1 cực nhiễu. Có xác suất test **không có positive nào**: `P = (1−π)^N` (π=0.5%, N=400 → ≈0.13; π=1% → ≈0.02). Khi đó kết quả phụ thuộc hoàn toàn vào zero_division (mục 1.3) | `cnt` theo từng split; mô phỏng `(1-π)**N_test` |
| **Gần luôn 1** (vd [GT] "Lỗi logic" có mặt ở hầu hết bài sai) | Đoán hằng 1 đã đạt khoảng 0.9 nên model gần như không thêm gì. Rủi ro là threshold tune lệch làm F1 **tụt dưới** mức hằng số | So OOF F1 của model với `2π/(1+π)` |
| **Trùng tất định với nhãn hoặc chiều rubric khác** (vd [GT] Lỗi biên dịch ⇔ compilable=0 ⇔ compile_log có `error:`; hoặc nhãn con ⇒ nhãn cha) | Ai có luật là được "free" 0.1. Leaderboard thực chất chỉ phân hạng trên các nhãn còn lại; metric bị thổi phồng. Đây là bằng chứng tốt cho **điểm thưởng** (dư thừa nhãn, rò rỉ giữa Task 1 và Task 2) | Bảng 2×2, φ/Cohen κ, xác suất có điều kiện (đoạn code dưới) |

```python
df['ce'] = df.labels.apply(lambda s: 'Lỗi biên dịch' in s)
print(pd.crosstab(df.ce, df.compilable, margins=True))
from sklearn.metrics import cohen_kappa_score; print(cohen_kappa_score(df.ce, df.compilable==0))
# quan hệ cha-con giữa nhãn: P(B | A) cho mọi cặp
Y = pd.DataFrame(MultiLabelBinarizer(classes=LABELS).fit_transform(df.labels), columns=LABELS)
cond = (Y.T @ Y) / Y.sum().values[:, None]     # hàng A, cột B: P(B|A); giá trị ≈1 nghĩa là A ⇒ B
```

### 1.3 Nhãn không có positive trong dev/test (zero_division)
- Khi TP=FN=0: nếu **không đoán bài nào** thì TP+FP+FN=0 và F1 = giá trị `zero_division`. Nếu **đoán ≥1 bài** thì F1 = 0/(FP) = 0, bất kể zero_division (sklearn hiện tính F1 = 2TP/(2TP+FP+FN)).
- Hệ quả chiến lược:
  - `zero_division=0`/`"warn"` (mặc định sklearn): nhãn đó đóng góp 0 với mọi đội, trần macro chỉ còn 0.9. Dự đoán trên nhãn đó không ảnh hưởng gì.
  - `zero_division=1`: không đoán gì thì được trọn 0.1, **chỉ 1 FP là mất cả 0.1**. Với nhãn có π rất nhỏ, ngưỡng nên bảo thủ hơn hẳn mức F1-optimal.
  - `zero_division=np.nan` (sklearn ≥1.3): nhãn bị loại khỏi trung bình, các nhãn còn lại nặng 1/9.
- **Bẫy cài đặt scorer**: nếu scorer dựng `MultiLabelBinarizer` từ `y_true ∪ y_pred` thay vì cố định 10 lớp, thì nhãn vắng ở cả hai phía sẽ biến khỏi mẫu số. Khi đó đoán nhầm 1 lần sẽ *thêm* một cột F1=0, macro giảm từ Σ/L' xuống Σ/(L'+1). Cần hỏi BTC, rồi viết unit test mô phỏng cả 3 trường hợp.
- **Bẫy Unicode**: nhãn tiếng Việt có thể ở dạng NFC hoặc NFD. Lệch dạng chuẩn hóa thì so khớp chuỗi thất bại và mọi dự đoán thành FP+FN. Kiểm tra bằng `unicodedata.is_normalized('NFC', s)` trên gold, và chuẩn hóa output về đúng dạng của gold.

### 1.4 Tập rỗng
- Với macro/micro-F1, bài có y=∅ và ŷ=∅ chỉ sinh TN, nên đóng góp bằng 0; ŷ≠∅ thì sinh FP. Tập rỗng chỉ "có giá trị" qua việc **giảm FP**, và không cần tối ưu exact-set match.
- Cách làm:
  1. Output phải là `[]`, không phải `null` hay thiếu field. Chạy validator trước mỗi lượt nộp vì file lỗi vẫn bị tính quota.
  2. Thêm một "clean gate" (bài qua hết test, không có warning nặng, predicted total ≈ 10) để chặn các nhãn có p sát ngưỡng. Chỉ bật khi đã kiểm chứng precision của gate trên train: `df[gate].card.eq(0).mean()`.
  3. LLM có bias "tìm lỗi bằng được" khi prompt yêu cầu tìm lỗi. Cần nói rõ "có thể không có lỗi nào" và đưa few-shot có bài sạch.

---

## 2. Tối ưu trực tiếp macro-F1

### 2.1 Phân rã theo nhãn
Macro-F1 cộng tách được theo nhãn, nên **mỗi nhãn được chọn riêng** threshold, model nguồn, loss weight và luật. Đây là đòn bẩy lớn nhất của Task 2.

### 2.2 Lý thuyết ngưỡng
- Bộ phân loại tối ưu cho F1 (ở mức population) là **threshold trên P(y=1|x)** (Koyejo et al., NeurIPS 2015, *Consistent Multilabel Classification*; Narasimhan et al. 2014).
- Lipton, Elkan & Narayanaswamy (2014): với xác suất đã calibrate, **ngưỡng F1-optimal = F\*/2**, trong đó F\* là F1 tối ưu đạt được. Hệ quả:
  - Ngưỡng luôn ≤ 0.5.
  - Nhãn mà model yếu (F\* ≈ 0.3) thì ngưỡng khoảng 0.15, tức đoán positive nhiều hơn prior.
  - Có thể dùng làm sanity check: nếu t_l tune được lệch xa F̂_l/2 thì nghi miscalibration hoặc overfit.
- Kết quả này chỉ đúng khi xác suất **calibrated**. pos_weight, focal và ASL đều làm lệch calibration, nên cần recalibrate (temperature hoặc Platt theo nhãn trên OOF) hoặc tune ngưỡng thực nghiệm.

### 2.3 Quy trình tune ngưỡng cross-validated
- **OOF**: `MultilabelStratifiedKFold` (gói `iterative-stratification`) 5-fold × 3 repeats. Nếu dev/test chứa **đề không có trong train**, phải chia theo nhóm `problem_id`. Kiểm tra: `set(dev.problem_id) - set(train.problem_id)`; mã đề có ngay trong đường dẫn `submissions/<mã đề>/`.
- Với mỗi nhãn, quét ngưỡng trên các giá trị OOF duy nhất và lấy argmax F1 trên OOF gộp.
- **Chống overfit nhãn hiếm** (n_pos ≲ 30, ngưỡng cắt này là [GT]):
  1. *Bootstrap threshold*: B=200 lần resample OOF, lấy median của argmax, hoặc maximize F1 trung bình qua các lần (đường cong mượt hơn).
  2. *Shrinkage*: `t_l = w·t_l^opt + (1−w)·t_global`, với `w = n_l/(n_l+κ)`; chọn κ bằng nested CV.
  3. *Prior/quantile matching*: đoán positive cho top-q bài test, q = c·π_l (c>1 với nhãn yếu, vì F1-optimal đoán dư). Cách này bền với lệch calibration giữa đề train và đề test, và chạy tự động hoàn toàn.
  4. *Expected-F1 trên test (decision-theoretic, Nan et al. ICML 2012)*: với p_i đã calibrate, chọn k để maximize `E[F1] ≈ 2·Σ_{i≤k} p_(i) / (k + Σ_i p_i)`. Cách này tự thích nghi với số positive "có vẻ" có trong test; nếu Σp rất nhỏ thì k*=0, hợp với zero_division=1.
  5. *Nested CV* để đo optimism: F1 khi tune ở inner fold so với F1 ở outer fold. Ghi khoảng chênh này vào báo cáo.
- Micro-F1 (metric phụ) ưu tiên ngưỡng khác. Báo cáo cả hai, nhưng chọn ngưỡng theo macro.

### 2.4 Loss (cùng backbone, đưa vào ablation)
- **BCE + `pos_weight_l = (N−n_l)/n_l`**, thường lấy căn hoặc chặn trên (cap) để ổn định. Logit bị dịch khoảng log w; muốn calibrate lại thì trừ log w. Khởi tạo bias head `b_l = log(π_l/(1−π_l))` để train ổn định từ đầu.
- **Focal** (Lin et al. 2017, γ=1–2): giảm ảnh hưởng của negative dễ, nhưng làm model underconfident nên bắt buộc tune ngưỡng.
- **ASL** (Ridnik et al., ICCV 2021): γ+=0, γ−=4, margin m=0.05 (`p_m = max(p−m, 0)` cho negative). Margin loại negative quá dễ và **giảm phạt các negative có thể bị gán thiếu**. Điều này liên quan trực tiếp nếu **[GT] giảng viên chỉ gắn nhãn chính, bỏ sót nhãn phụ**. Kiểm chứng: đọc thủ công top-50 FP có độ tin cao của model (chỉ trên **train**). Có thể đối chiếu bằng từ khóa trong field feedback; dùng feedback cho EDA không vi phạm luật vì không đưa vào input.
- **Distribution-Balanced Loss** (Wu et al., ECCV 2020): thiết kế riêng cho multi-label long-tail có đồng xuất hiện nhãn. Class-balanced loss (Cui et al., CVPR 2019) là phương án đơn giản hơn.
- **Soft-F1 / sigmoidF1** (Bénédict et al., TMLR 2022): `sTP=Σp·y, sFP=Σp(1−y), sFN=Σ(1−p)y`, loss = `1 − mean_l 2sTP/(2sTP+sFP+sFN+ε)`.
  - Cạm bẫy: nếu batch không có positive của nhãn l thì sTP≡0 và **gradient bằng 0**, tức nhãn hiếm không nhận tín hiệu. Gradient accumulation không cứu được vì loss không phân rã theo mẫu.
  - Khắc phục: dùng sampler đảm bảo mỗi batch có ≥1 positive của mỗi nhãn hiếm, hoặc dùng memory bank. Nên dùng dạng `BCE + λ·(1−softF1)` với λ khoảng 0.3–1.
- **Oversampling** các mẫu chứa nhãn hiếm: lưu ý cách này kéo theo oversample cả các nhãn phổ biến đồng xuất hiện.

### 2.5 Ensembling xác suất
- Lấy trung bình xác suất đã calibrate (hoặc logit) qua seed × fold × họ model. **Tune ngưỡng lại trên OOF của ensemble**; ngưỡng của từng thành viên không dùng lại được.
- Khi thang điểm khác nhau (logprob LLM so với sigmoid encoder), rank-average được, nhưng rank-average phá calibration nên phải đi với ngưỡng kiểu quantile (2.3, ý 3).
- **Stacking theo nhãn**: logistic regression L2 trên `[p_encoder, p_LLM, p_kNN, rule_flags, rubric dự đoán]`, chỉ dùng OOF. Nhãn hiếm dùng LR thay vì GBDT.
- **Chọn model theo nhãn**: tối đa 3 ứng viên, kiểm chứng bằng nested CV, vì chọn trên 10 nhãn × M model cũng là một dạng overfit.

---

## 3. Các họ phương pháp

| Phương pháp | Ưu | Nhược | Gợi ý dùng |
|---|---|---|---|
| **Binary Relevance** | Đơn giản; mỗi nhãn có ngưỡng và model riêng; khớp với việc macro-F1 phân rã theo nhãn | Bỏ qua tương quan nhãn, có thể ra tổ hợp vô lý | Backbone mặc định |
| **Classifier Chains / ECC** (Read et al. 2011) | Khai thác phụ thuộc (vd [GT] khi bài có Lỗi biên dịch, giảng viên ít gắn nhãn khác) | Lỗi lan truyền theo chuỗi, phụ thuộc thứ tự | Đặt nhãn tất định (biên dịch) đầu chuỗi; dùng ECC với GBDT trên feature, chi phí thấp |
| **Label Powerset / RAkEL** | Bắt trọn tổ hợp nhãn | 2^10 lớp, không đoán được tổ hợp chưa thấy; nhãn hiếm nằm trong các tổ hợp hiếm nên **hại macro-F1** | Chỉ dùng nếu top-20 label-set phủ ≥95% dữ liệu ([GT], kiểm bằng `value_counts` ở đầu bài) |
| **LLM sinh JSON có ràng buộc** (enum 10 nhãn; vLLM ≥0.12 dùng `StructuredOutputsParams(json=...)`, `guided_json` đã deprecated) | Zero/few-shot, suy luận được trên code, giải thích được | **Không có xác suất** nên không tune ngưỡng theo nhãn được; thiên về nhãn chung chung ("Lỗi logic"); bias theo thứ tự liệt kê nhãn | Self-consistency: sample n=5–10 lần, **tỉ lệ phiếu theo nhãn dùng làm xác suất giả** rồi tune ngưỡng |
| **LLM hỏi yes/no từng nhãn, đọc logprob** | Ra điểm liên tục cho từng nhãn nên threshold được; 10 câu hỏi dùng chung prefix (đề+code+log), prefix caching của vLLM làm chi phí chỉ khoảng 1.x lần một lượt | Gấp 10 số lượt gọi trên API; cần logprob | Xem các lưu ý bên dưới |
| **Encoder code** (CodeBERT 512 tok, UniXcoder khoảng 1k tok *(cần kiểm chứng)*, CodeT5+ 110M-embedding; ModernBERT 8k ctx cho code dài) | Rẻ, train nhiều seed, head sigmoid cho xác suất tử tế | Context ngắn; yếu ở ngữ nghĩa thực thi (lỗi logic cần "chạy trong đầu"); không quen đề tiếng Việt | Chunk theo hàm (tree-sitter) rồi **max-pool logit theo nhãn**. Đây là đúng ngữ nghĩa multi-instance: một hàm có lỗi thì cả file mang nhãn |
| **Decoder LLM + classification head** (Qwen2.5-Coder-7B-Instruct hoặc Qwen3-8B, LoRA/QLoRA; `AutoModelForSequenceClassification`, `problem_type="multi_label_classification"`) | Context dài, hiểu cả C++ lẫn tiếng Việt, ra xác suất. Đáp ứng yêu cầu fine-tune open-weight ≤10B | Tốn compute; dễ overfit dữ liệu nhỏ; phải đặt `pad_token` và padding side đúng khi pool last-token | **Ứng viên chính** cho hướng fine-tune |
| **Decoder SFT sinh text nhãn** | Gần với instruction tuning, dễ ghép với Task 3 | Không có xác suất trực tiếp | Mẹo **slot template**: train target theo thứ tự cố định `bien_dich:N vong_lap:Y ...`; lúc suy luận teacher-force template và đọc P(Y) ở từng slot. Ra xác suất 10 nhãn trong một pass, kiểu classifier chain bên trong LLM |
| **kNN/RAG theo cùng đề** | Bài cùng đề hay mắc cùng lỗi. Ngoài embedding (UniXcoder) còn hai khóa: shingle token đã chuẩn hóa định danh (kiểu winnowing/MOSS), và **vector test fail (Jaccard)** cho single_problem | Rò rỉ trong CV qua bài gần trùng (đạo bài); vô dụng nếu test là đề mới | Đưa p_kNN thành feature cho stacker. Few-shot RAG cho LLM chỉ đưa **code + nhãn** của demo, **không đưa feedback** vì feedback của demo cũng là input bị cấm. Đưa thêm **diff với bài đúng gần nhất cùng đề** để LLM định vị dòng lỗi |

**Lưu ý cho cách hỏi yes/no và đọc logprob:**
- Dùng token trả lời một mảnh ("Yes"/"No" hoặc "A"/"B"). "Có" có thể bị tách thành nhiều token, kiểm tra bằng tokenizer. Cộng xác suất các biến thể (`" Yes"`, `"Yes"`), rồi `p = e^{l_Y}/(e^{l_Y}+e^{l_N})`.
- Khử bias bằng contextual calibration (Zhao et al. 2021, *Calibrate Before Use*).
- CoT trước khi trả lời làm tăng accuracy nhưng đẩy xác suất về 0/1, threshold kém tác dụng. Phương án thay thế là sample n chuỗi CoT rồi lấy tỉ lệ phiếu.
- **API Anthropic không trả logprobs** (đã kiểm chứng). OpenAI có `logprobs/top_logprobs` cho một số model non-reasoning *(cần kiểm chứng theo model cụ thể)*. Model local qua vLLM hoặc transformers thì truy cập đầy đủ.
- **Verbalized confidence** ("0–100") thô, dồn cụm và quá tự tin (Xiong et al., ICLR 2024). Chỉ nên dùng như tín hiệu xếp hạng sau khi isotonic-calibrate.

**Stack đề xuất theo tiến độ:**
- Tuần 1–2: rule + GBDT trên feature tool.
- Tuần 2–4: LLM API làm từng nhãn với self-consistency hoặc logprob (đáp ứng yêu cầu prompting).
- Tuần 3–6: Qwen-Coder-7B LoRA + head đa nhiệm (đáp ứng yêu cầu fine-tune).
- Tuần 6–7: stacking theo nhãn và ngưỡng OOF.

---

## 4. Tín hiệu và luật tất định

### 4.1 Tự tái tạo tín hiệu, không chỉ tin log có sẵn
Biên dịch lại toàn bộ bằng toolchain cố định (Docker, ghi version g++ để tái lập) và so với `compile_log` được cấp.
- **[GT] dataset issue**: bài dùng `conio.h`, `windows.h`, `void main()`, `scanf_s`, hoặc chỉ build được trên Dev-C++/MSVC thì `compilable` của giảng viên có thể ≠ kết quả trên Linux.
- Kiểm tra: `pd.crosstab(our_compile_ok, df.compilable)` và grep các header đó.

### 4.2 compile_log → feature
Regex trên các dòng `error:`, mỗi pattern là một feature:

| Pattern trong log | Ý nghĩa |
|---|---|
| `expected ';'` | Cú pháp |
| `was not declared in this scope` | Định danh chưa khai báo |
| `no matching function for call`, `too few arguments to function` | Ứng viên **Lỗi hàm** |
| `undefined reference to` | Linker: khai báo prototype mà không định nghĩa |
| `return-statement with a value, in function returning 'void'` | Lỗi kiểu trả về |
| `invalid conversion` | Lỗi chuyển kiểu |

[GT] giảng viên gắn *cả* Lỗi biên dịch lẫn Lỗi hàm khi lỗi biên dịch nằm ở lời gọi hàm. Kiểm tra: `P(Lỗi hàm | pattern hàm trong log)`.

### 4.3 test_report → feature (single_problem)
- Feature cơ bản: pass_rate, any_TLE, any_RE, any_WA, chỉ fail ở test nhỏ, và vector test fail.
- Feature theo **kiểu diff output**:
  - Chỉ khác whitespace hoặc xuống dòng → io_format / [GT] Lỗi I/O.
  - Có **dòng thừa kiểu `Nhap n:`** (prompt nhập liệu, rất phổ biến ở lớp nhập môn) → I/O.
  - Số lệch đúng ±1 → off-by-one vòng lặp hoặc biên.
  - Output rỗng hoặc bị cắt → RE hoặc đọc input sai.
- Map **[GT]**: TLE → Lỗi vòng lặp (vòng vô hạn) hoặc độ phức tạp; RE → mảng/con trỏ hoặc đệ quy sâu; chỉ fail ở test biên → Lỗi biên. Mọi map đều phải kiểm bằng precision/recall trên train.

### 4.4 Static analysis → map nhãn [GT]
- `g++ -O2 -Wall -Wextra -Wshadow` (cần -O2 để kích hoạt maybe-uninitialized):

| Warning | Nhãn ứng viên |
|---|---|
| `-Wreturn-type` (thiếu return) | Lỗi hàm |
| `-Wmaybe-uninitialized` / `-Wuninitialized` | Logic |
| `-Wparentheses` (`if (a = b)`) | Logic / điều kiện |
| `-Wempty-body` (`for(...);`) | Vòng lặp |
| `-Wsign-compare` (int so với `size()`) | Biên / vòng lặp |
| `-Waggressive-loop-optimizations`, `-Warray-bounds` | Tràn mảng |
| `-Wformat` | I/O |
| `-Wshadow`, `-Wunused-*` | Code quality / logic chết |

- **clang-tidy**: `bugprone-infinite-loop`, `bugprone-integer-division`, `bugprone-assignment-in-if-condition`, `bugprone-branch-clone`, `bugprone-narrowing-conversions`, `misc-redundant-expression`, `clang-analyzer-core.*` (null deref, chia 0), `readability-function-cognitive-complexity`.
- **cppcheck**: `uninitvar`, `arrayIndexOutOfBounds`, `nullPointer`, `zerodiv`, `missingReturn`, `knownConditionTrueFalse`, `memleak`, `unreadVariable`.
- **AST** (tree-sitter-cpp): số vòng lặp, độ lồng, có đệ quy, số hàm, có biến toàn cục, pattern `i <= n` khi truy cập `a[i]`, cộng dồn bằng `int`.

### 4.5 Chạy động
- Chạy lại public test với `-fsanitize=address,undefined -D_GLIBCXX_DEBUG`. `_GLIBCXX_DEBUG` bắt `vector[]` vượt biên mà ASan đôi khi bỏ sót. Các lỗi bắt được: heap/stack-buffer-overflow, signed-integer-overflow ([GT] nhãn tràn số), stack overflow do đệ quy.
- Sandbox bằng `docker --network none` cộng ulimit (hoặc nsjail/firejail), vì đây là code chưa tin cậy.
- TLE dễ flaky: chạy 3 lần lấy đa số, ghi rõ timeout để tái lập.
- **Differential testing** để có thêm test:
  - Mỗi đề, LLM viết một bộ sinh input ngẫu nhiên (seeded, cache lại, chạy tự động nên không tính là can thiệp từng bài).
  - Oracle là output đa số của các bài train đạt 10/10 cùng đề. Với đề chỉ có ở test thì dùng lời giải do LLM viết, đã kiểm trên public test.
  - Sinh feature: fail ở input nhỏ nhất, TLE ở input lớn, fail khi có phần tử trùng.
  - Không áp dụng được cho multi_problem (không có test_report); nhánh này dựa vào LLM đọc code.

### 4.6 Kết hợp luật với model
- **Override cứng** chỉ khi luật đạt precision và recall ≥99% trên train, và ổn định theo môn và loại đề. Ứng viên duy nhất có khả năng là Lỗi biên dịch.
- **Luật làm feature** trong stacker (LR/GBDT), và làm khối "Tool evidence" trong prompt LLM, kèm chỉ dẫn rằng đây là bằng chứng, không phải đáp án.
- **Logit offset học được**: `z' = z + β_l·rule_l`, β học bằng LR trên OOF. Đây là dạng mềm, nằm giữa feature và override.
- **Routing**: bài không biên dịch được thì không có feature động, nên dùng model/ngưỡng riêng. Thêm cờ missing-indicator cho multi_problem.
- Báo cáo một dòng **rule-only** trong bảng ablation, kèm bảng precision/recall của từng luật theo nhãn (nội dung này cũng dùng được cho EDA).

---

## 5. Tăng dữ liệu cho nhãn hiếm

### 5.1 Mutation có kiểm soát
| Operator | Nhãn [GT] |
|---|---|
| `<`↔`<=`, khởi tạo 0→1, `i++`→`i--`, xóa bước tăng (vòng vô hạn), xóa `break` | Lỗi vòng lặp |
| Xóa nhánh `if (n==0)`, `>=`→`>` ở so sánh biên | Lỗi biên |
| `int& a`→`int a` (mất cập nhật), xóa `return`, đảo thứ tự đối số khi gọi | Lỗi hàm |
| `&&`↔`||`, phủ định điều kiện, `+`↔`-` | Lỗi logic |
| `long long`→`int` | Tràn số |
| Thêm `cout<<"Nhap n: "`, `"\n"`→`" "` | I/O |
| Xóa `;`, đổi tên định danh, xóa `#include` | Lỗi biên dịch (vốn đã dễ, không cần augment) |

- Công cụ: tự viết mutator dựa trên tree-sitter (dễ kiểm soát nhãn nhất), hoặc Mull (dựa trên LLVM), Universal Mutator, dextool.
- Nguồn: bài full điểm, tập nhãn ∅, **cùng đề** trong train.
- **Kill check**: mutant phải đổi hành vi trên test. Operator vòng vô hạn phải gây TLE; operator biên dịch phải fail compile. Loại equivalent mutant.
- **Sinh lại compile_log/test_report/static feature cho từng mutant**, nếu không feature sẽ mâu thuẫn với nhãn.
- **Học mapping operator → nhãn từ dữ liệu thật**, đừng tự gán: với các bài thật chỉ có 1 nhãn, diff với bài đúng gần nhất cùng đề, phân loại kiểu edit, rồi tính P(nhãn | kiểu edit) thực nghiệm.
- **Partial-label loss**: chỉ supervise nhãn được chèn là positive và các nhãn chắc chắn negative. Mask các nhãn có thể bị kéo theo (vd nhãn cha "Lỗi logic", hoặc biên kéo theo vòng lặp).

### 5.2 LLM sinh bug
- Prompt gồm định nghĩa nhãn, 2–3 bài thật mang nhãn đó cùng đề để LLM bắt chước phong cách, và yêu cầu "sửa tối thiểu, kiểu sinh viên năm nhất, không comment".
- Sau khi sinh: xóa comment (LLM hay để lại `// bug`), xác minh bằng test, và lọc theo phân bố kích thước diff (số dòng đổi) sao cho khớp với diff thật giữa bài sai và bài đúng gần nhất.
- **Augmentation giữ nhãn** cho positive hiếm thật, ít lệch phân bố hơn: đổi tên biến, `for`↔`while`, đổi format, đảo thứ tự các hàm độc lập. Xác minh vector verdict test không đổi.
- Chi phí API cho phần sinh dữ liệu phải ghi vào báo cáo.

### 5.3 Cảnh báo distribution shift
- **Artifact**: mutant là "code gần đúng + đúng 1 edit", còn lỗi thật thường nhiều, mang tính hệ thống (hiểu sai đề) và style lộn xộn. Model học được artifact này sẽ over-predict.
- **Lệch ngữ nghĩa nhãn**: cái mình gọi là "Lỗi hàm" chưa chắc trùng với cách giảng viên dùng nhãn đó.
- **Prevalence bị thổi phồng** làm lệch calibration.
- **Biện pháp:**
  - Synthetic **chỉ nằm trong fold train**. Mutant phải **cùng fold với bài gốc** để tránh rò rỉ theo nhóm.
  - Ngưỡng **chỉ tune trên OOF thật**.
  - Sample weight cho synthetic khoảng 0.3–0.5; ablation tỉ lệ 0.5/1/2× so với positive thật.
  - Huấn luyện **domain classifier** phân biệt synthetic và thật. AUC≈1 nghĩa là artifact mạnh.
  - Chỉ giữ augmentation cho nhãn mà F1 OOF *thật* tăng.

---

## 6. Nhất quán giữa Task 1 và Task 2

### 6.1 Ràng buộc giả thuyết [GT]
- C1: Lỗi biên dịch ⇔ compilable=0.
- C2: total=10 ⇒ tập nhãn = ∅. Chiều ngược lại có thể sai, vì có thể trừ code_quality mà không gắn nhãn nào.
- C3: compilable=0 ⇒ logic=io=edge=0. Chính sách này ảnh hưởng gián tiếp tới các nhãn khác.
- C4: Lỗi vòng lặp hoặc TLE ⇒ complexity=0.
- C5–C7: io_format=0 ⇔ nhãn I/O; edge_case<2 ⇔ nhãn biên; logic<4 ⇒ ít nhất một nhãn thuộc nhóm logic.
- Với multi_problem: [GT] nhãn gắn theo **file** hay theo **câu**? Nếu câu tiên quyết sai, lỗi ở các câu sau có được gắn nhãn không? Cần hỏi BTC và kiểm tra trên dữ liệu.

### 6.2 Kiểm chứng trước khi áp dụng
```python
def rule(df, A, B):   # A ⇒ B
    s = df[A]; return dict(support=s.sum(), conf=df.loc[s, B].mean(), violations=(s & ~df[B]).sum())
df['empty'] = df.card.eq(0); df['full'] = df.total.eq(10)
print(rule(df, 'full', 'empty'), rule(df, 'empty', 'full'))
print(df.groupby(['subject','problem_type']).apply(lambda g: rule(g, 'full', 'empty')))  # ổn định theo nhóm?
```
- Khai phá tự động: nhị phân hóa rubric (chiều < max) và nhãn, chạy `mlxtend.apriori` + `association_rules` (conf ≥0.95) để ra danh sách ứng viên ràng buộc.
- Ngưỡng áp dụng: **hard** chỉ khi conf ≥99% và các vi phạm giải thích được; còn lại chỉ dùng làm feature. Trước khi áp, mô phỏng trên OOF xem macro-F1 và QWK thay đổi thế nào.

### 6.3 Khai thác
- **Multi-task chung backbone** (Qwen-LoRA hoặc encoder):
  - Head 10 sigmoid cho nhãn lỗi.
  - Head rubric: compilable/io/complexity/quality dạng binary; logic 0–4 và edge 0–2 dạng ordinal (CORAL hoặc cumulative link).
  - Rubric có giá trị ở **mọi** mẫu, nên là giám sát dày giúp regularize biểu diễn cho nhãn hiếm.
  - Trọng số loss: tune λ, hoặc dùng uncertainty weighting (Kendall et al. 2018).
- **Cross-task stacking**: rubric *dự đoán* (OOF) làm feature cho stacker Task 2 và ngược lại. Rubric gold không có lúc test nên không dùng được.
- **Joint decoding**: với các nhãn có p ∈ [t−δ, t+δ], liệt kê các tổ hợp cùng argmax rubric, chọn cấu hình có tổng log-prob lớn nhất thỏa ràng buộc đã kiểm chứng. Đảm bảo nộp Task 1 và Task 2 không mâu thuẫn nhau.
- **Điểm thưởng**: vi phạm của chính các ràng buộc "hiển nhiên" là bằng chứng chất lượng nhãn.
  - Ví dụ: total=10 mà có nhãn; có Lỗi biên dịch mà compilable=1; compile_log thành công mà compilable=0.
  - Bài **gần trùng** (hash token sau khi chuẩn hóa định danh) mà khác nhãn cho phép ước lượng tỉ lệ nhiễu nhãn và **trần F1 đạt được**.
  - Từ khóa trong feedback ("vòng lặp", "biên") mà thiếu nhãn tương ứng cho thấy gán thiếu nhãn. Đề xuất khắc phục: kiểm tra nhất quán trong công cụ chấm, và adjudication cho các ca lệch.

---

## 7. Ablation và phân tích lỗi

### 7.1 Ablation (cùng split, 3 seed, paired bootstrap CI)
1. **Tín hiệu đầu vào** (cộng dần): code → +compile_log → +test_report → +static → +sanitizer/differential. Báo cáo Δ F1 theo từng nhãn.
2. **Quy tắc quyết định**: 0.5 cố định / một ngưỡng chung / ngưỡng theo nhãn / theo nhãn có shrinkage / quantile-matching / expected-F1 top-k. Kèm khoảng optimism đo bằng nested CV.
3. **Loss**: BCE / BCE+pos_weight / focal / ASL / BCE+softF1, cùng backbone.
4. **Họ model**: rule-only / GBDT feature / UniXcoder / long-ctx encoder / Qwen-Coder LoRA+head / LLM zero-shot JSON / LLM logprob theo nhãn / few-shot RAG.
5. **Output của LLM fine-tune**: classification head / SFT sinh text / slot-template teacher-forced, cùng base model.
6. **Nguồn xác suất của LLM API**: verbalized / tỉ lệ phiếu self-consistency (n=1, 5, 10) / logprob. Đo ECE và macro-F1 sau khi tune ngưỡng.
7. **Tích hợp luật**: không dùng / feature / override cứng / logit offset.
8. **Augmentation**: không / mutation / LLM-bug / cả hai × tỉ lệ 0.5/1/2× × có/không mask loss; kèm AUC của domain classifier.
9. **Multi-task**: có/không head rubric; có/không chiếu nhất quán.
10. **Retrieval**: k ∈ {0, 3, 8}; cùng đề hay khác đề; khóa embedding hay vector test fail; có/không diff với bài đúng gần nhất.
11. **Mô tả nhãn trong prompt**: chỉ tên / định nghĩa / định nghĩa + phản ví dụ giữa các cặp dễ nhầm. Thêm hoán vị thứ tự nhãn để đo position bias.
12. **Code dài**: cắt đầu / chunk theo hàm + max-pool.
13. **Giao thức chia**: random split hay group theo đề. Ablation này cho thấy mức lạc quan của random split và ghi điểm ở phần phương pháp luận.

### 7.2 Phân tích lỗi
- **Theo nhãn**: đường PR, F1 kèm support và CI 95% bootstrap, reliability diagram.
- **Nhầm lẫn theo cặp nhãn**: đếm (FN ở A ∧ FP ở B) trên cùng một bài, tức "thay thế nhãn". Dùng MLCM (Heydarian et al., IEEE Access 2022; repo `mrh110/mlcm`) có cột NTL/NPL cho tập rỗng. Các cặp nghi nhầm [GT]: vòng lặp↔logic, biên↔logic, hàm↔biên dịch.
- **Cắt theo lát**: multi_problem vs single_problem; CTDL&GT vs CSLT; từng `problem_id`; bucket điểm (0–3/4–6/7–9/10); decile độ dài token (đặc biệt là bài vượt context); cardinality nhãn thật (0/1/2/3+); trạng thái biên dịch; có/không có test_report.
- **FP trên bài ∅**: tỉ lệ theo model; LLM có "bịa lỗi" không.
- **Nhiễu nhãn**: top FP/FN có độ tin cao **trên train/dev** (tuyệt đối không động vào test), đọc thủ công để ước lượng tỉ lệ gán sai theo nhãn. Kết quả nuôi phần điểm thưởng.
- **Độ ổn định của LB**: std bootstrap trên tập kích thước dev so với các bước cải thiện thực tế giữa các lượt nộp.

---

## Câu hỏi cần làm rõ với BTC / trên diễn đàn
1. Scorer macro-F1 dùng biến thể nào (trung bình F1 hay F1 của macro-P/R)? `zero_division` bằng bao nhiêu? Không gian nhãn cố định 10 nhãn hay suy ra từ dữ liệu? Chuẩn Unicode của chuỗi nhãn là gì?
2. Dev và test có chứa đề **không có trong train** không? Dev có nhãn công khai không?
3. Có được dùng feedback làm **target phụ lúc train** (không phải input lúc suy luận) cho Task 2 không?
4. Với multi_problem, nhãn gắn theo file hay theo câu? Lỗi ở câu sau khi câu tiên quyết sai có được gắn nhãn không?

Nguồn đã kiểm chứng:
- [sigmoidF1 (arXiv 2108.10566, TMLR 2022)](https://arxiv.org/pdf/2108.10566)
- [sklearn f1_score: zero_division gồm 0/1/np.nan](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [sklearn discussion: F1 khi TP=FP=FN=0](https://github.com/scikit-learn/scikit-learn/discussions/28675)
- [Anthropic OpenAI-compat: logprobs bị bỏ qua](https://platform.claude.com/docs/en/api/openai-sdk)
- [MLCM (IEEE Access 2022)](https://github.com/mrh110/mlcm)
- [vLLM Structured Outputs (guided_* deprecated từ v0.12)](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [Qwen3-Coder repo](https://github.com/QwenLM/Qwen3-Coder): các bản Coder gần đây là MoE 30B-A3B trở lên, vượt khuyến nghị ≤10B. Vì vậy đề xuất Qwen2.5-Coder-7B hoặc Qwen3-8B; tên bản Qwen3-Coder 7B *(cần kiểm chứng)*.