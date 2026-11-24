# Task 3: sinh phản hồi có kiểm soát mức Level 1–4

Ghi chú chung: chưa có dữ liệu trong tay. Mọi nhận định về dữ liệu dưới đây là **GIẢ THUYẾT (GT)**, và mỗi GT đi kèm một phép kiểm chứng. Tên cột trong các đoạn code (`fb`, `level`, `labels`, `code`…) là tên giả định.

---

## 1. Giải phẫu metric

### 1(a) Level compliance: BTC có thể đo bằng cách nào

| Cách đo | Bắt được | Bỏ lọt | Khả năng BTC dùng |
|---|---|---|---|
| M1. Regex/heuristic: ```` ``` ````, backtick, `#include`, dòng kết thúc bằng `;`, `{}`, `cout<<`, `for(` | Code fence, đoạn code nguyên văn | Pseudo-code, lệnh sửa viết bằng lời | GT: cao, vì rẻ, tất định và chạy tự động được cho leaderboard |
| M2. Classifier học từ feedback giảng viên theo level | Phong cách "lời giải" | Dạng code mới chưa thấy khi train | Trung bình |
| M3. LLM-judge với rubric "có code sửa/lời giải không" | Pseudo-code, "đổi `<=` thành `<`", công thức truy hồi đầy đủ | Phụ thuộc prompt, có nhiễu | Cao, vì đề nhấn mạnh tính sư phạm |
| M4. Độ tương đồng với code đúng hoặc lời giải mẫu | Phản hồi chứa đúng thay đổi cần sửa | Diễn đạt lại | Thấp đến trung bình |

**Hệ quả thiết kế: tối ưu cho trường hợp xấu nhất (minimax).** Phản hồi phải qua được M1 ở bề mặt và M3 ở ngữ nghĩa. Ở L1/L2 cần tránh cả những "dạng code trá hình" sau:
- Pseudo-code hoặc liệt kê từng bước thuật toán. Với đề DSA, mô tả đủ các bước thì đã là lời giải.
- Lệnh sửa viết bằng lời: "đổi dấu nhỏ hơn hoặc bằng thành nhỏ hơn", "thêm `return` sau dòng 12".
- Biểu thức hoặc định danh của bản đã sửa: `dp[i] = max(dp[i-1], …)`, `i < n`, `sort(a, a+n)`.
- Công thức hoặc điều kiện biên viết đủ để chép lại máy móc ("khởi tạo `max` bằng `a[0]` thay vì 0").
- Một dòng code đơn lẻ (`return 0;`, `long long`) kèm vị trí chèn.

**Ranh giới mơ hồ ở L2.** Quy tắc làm việc đề xuất (sẽ kiểm chứng ở §2):
- L2 được nói **lỗi là gì (WHAT)** và **ở đâu (WHERE)**: vị trí, loại lỗi, nguyên nhân.
- L2 không được nói **sửa thế nào (HOW)**, tức là thay đổi cụ thể cần làm.
- Phép thử "áp dụng máy móc": nếu sinh viên làm theo phản hồi mà không cần suy nghĩ vẫn ra code đúng thì phản hồi đó là lộ lời giải.
- Trích lại biểu thức sai của chính sinh viên (ví dụ "điều kiện `i <= n` ở dòng 8") có thể vẫn hợp lệ ở L2. Đây là GT, kiểm chứng bằng cách đo tỷ lệ feedback L2 của giảng viên có trích code của sinh viên.
- Arxiv 2026 "Auditable Release Control for Pedagogical Leakage in LLM Tutors" (chưa peer-review) coi cả việc thu hẹp không gian tìm kiếm quá mức là leakage. Theo định nghĩa này, L2 quá cụ thể vẫn có thể bị phạt.

**L3 và L4 có thể là gì (GT):**
- L3: hướng sửa cụ thể kèm một phần lời giải, như pseudo-code, hoặc code cho một bước hay một hàm con.
- L4: code đúng đầy đủ kèm giải thích.
- Có thể metric đối xứng: L4 mà không có lời giải cũng bị tính "không tuân thủ". Cần hỏi BTC; trong lúc chờ, render L4 với code đầy đủ đã qua test.

### 1(b) Độ chính xác chẩn đoán: các cách đo khả dĩ
- **LLM-judge so với feedback giảng viên** (dùng feedback GV làm reference): "có xác định cùng lỗi chính, cùng vị trí, không khẳng định sai không". Khả năng BTC dùng cao nhất.
- **So khớp nhãn:** trích loại lỗi từ phản hồi bằng classifier rồi so với nhãn. Vì nhãn đã được cho sẵn ở input, cách đo này dễ bị game bằng cách nhắc lại tên nhãn.
- **So khớp vị trí:** so dòng, hàm hoặc câu được nhắc tới với feedback GV.
- **BERTScore/ROUGE với feedback GV:** thiên về văn phong và độ dài. ROUGE tiếng Việt cần tách từ.

### Rủi ro bị game và trade-off
- **Phản hồi mơ hồ** ("Em kiểm tra lại logic vòng lặp nhé"): compliance gần như 100% nhưng chẩn đoán thấp.
- **Nhắc lại nhãn:** thổi phồng điểm nếu metric dựa trên nhãn.
- **Liệt kê dàn trải nhiều lỗi khả dĩ:** tăng recall với judge chỉ hỏi "có nhắc lỗi của GV không", nhưng sai về sư phạm.
- **Trade-off cốt lõi:** càng cụ thể thì chẩn đoán càng tăng nhưng leakage cũng tăng, nhất là ở L2.
  - Taxonomy của Maurya et al. (NAACL 2025) tách đúng các chiều này: *mistake identification, mistake location, revealing of the answer, providing guidance, actionability*. Mâu thuẫn giữa *actionability* và *revealing answer* chính là trade-off của Task 3.
  - Đề xuất: vẽ đường **Pareto** (compliance, chẩn đoán) và chọn điểm vận hành bằng ngưỡng của bộ phát hiện leakage.
- **Ứng viên điểm thưởng (metric):** chứng minh một baseline mơ hồ cố định (template theo nhãn) đạt compliance gần tối đa và chẩn đoán không tầm thường trên public LB. Đó là bằng chứng định lượng rằng metric gộp có thể bị game.

---

## 2. Suy ra định nghĩa Level từ dữ liệu train (EDA văn bản)

### Đặc trưng đo được trên mỗi feedback
- **Hình thức code:** `has_fence`, số backtick, `n_cpp_lines` (dòng khớp regex C++), tỷ lệ token là toán tử C++ (`<=`, `++`, `[]`, `->`, `::`).
- **Mức cụ thể:**
  - số lần nhắc "dòng N";
  - số định danh trùng với định danh trong code sinh viên (trích bằng tree-sitter-cpp);
  - số giá trị testcase được nhắc (khớp với test_report).
- **Lệnh sửa:** regex `(đổi|sửa|thay|thêm|bỏ) … (thành|bằng)`, động từ mệnh lệnh.
- **Độ lộ so với lời giải** (khi có code sửa: L4 của GV hoặc repair đã qua test): tỷ lệ token của phần *dòng thêm vào* trong diff xuất hiện trong feedback. Đây là tín hiệu leakage mạnh nhất.
- **Giọng sư phạm:** số câu hỏi (`?`), đại từ ("em"/"bạn"), cụm khen ("tốt", "đúng hướng").
- **Độ dài:** ký tự, số từ sau tách từ (underthesea/pyvi), số câu.
- **Luôn chuẩn hoá Unicode NFC trước khi đo.** Văn bản tiếng Việt lẫn NFC/NFD làm hỏng regex và so khớp.

```python
import re, unicodedata as ud, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
CPP  = re.compile(r'(;\s*$)|^\s*(for|while|if|return|int|long|#include)\b|cout\s*<<|cin\s*>>|[{}]', re.M)
EDIT = re.compile(r'\b(đổi|sửa|thay|thêm|bỏ)\b.{0,40}\b(thành|bằng)\b', re.I)
LREF = re.compile(r'dòng\s*\d+', re.I)
def feats(fb, idents):
    fb = ud.normalize('NFC', fb); toks = re.findall(r'[A-Za-z_]\w*', fb)
    return dict(n_char=len(fb), fence='```' in fb, tick=fb.count('`'),
                cpp=sum(1 for _ in CPP.finditer(fb)), edit=len(EDIT.findall(fb)),
                lref=len(LREF.findall(fb)), q=fb.count('?'),
                ident=sum(t in idents for t in toks))
F = pd.DataFrame([feats(r.fb, r.idents) for r in df.itertuples()]); F['level'] = df.level.values
print(F.groupby('level').describe().T)                    # phân bố từng đặc trưng theo level
print(F.drop(columns='level').corrwith(F.level, method='spearman'))  # tính đơn điệu theo level
pred = cross_val_predict(LogisticRegression(max_iter=1000), F.drop(columns='level'), F.level, cv=5)
print(confusion_matrix(F.level, pred))                    # L1/L2 có tách được không?
viol = F[(F.level <= 2) & (F.fence | (F.cpp >= 2) | (F.edit > 0))]
print(len(viol) / (F.level <= 2).sum())                   # tỷ lệ feedback GV L1/L2 "tự vi phạm"
```

### Các GT cần kiểm chứng và ý nghĩa
1. **GT: các đặc trưng tăng đơn điệu theo level.** Nếu đúng, quy tắc của §1 đứng vững. Nếu ma trận nhầm lẫn cho thấy L1 và L2 (hoặc L2 và L3) chồng lấn nặng thì nhãn level bị nhiễu. Đây là ứng viên điểm thưởng về dữ liệu.
2. **GT: một phần feedback GV ở L1/L2 chứa code fence hoặc lệnh sửa.** Nếu tỷ lệ `viol` > 0, đọc tay khoảng 30 mẫu để ước lượng precision của regex.
   - Hệ quả 1: metric sẽ phạt cả bản thân dữ liệu gold.
   - Hệ quả 2: SFT trên dữ liệu này dạy model vi phạm, nên phải lọc trước khi train.
3. **GT: level tương quan với điểm hoặc nhãn lỗi** (điểm thấp → L3/L4). Kiểm tra bằng `pd.crosstab(df.level, pd.cut(df.total, [0,3,6,10]))`, tách theo môn (DSA và CSLT) và theo loại đề.
4. **GT: feedback GV có template.** Đo tỷ lệ trùng chính xác và near-duplicate (MinHash, `datasketch`). Nếu cao thì baseline kNN/retrieval sẽ mạnh, và metric chẩn đoán có thể thưởng cho việc chép template.
5. **GT: có feedback không nhắc tới nhãn được cho.** Dùng LLM hoặc keyword để kiểm tra feedback có đề cập từng nhãn không. Tỷ lệ không nhất quán cao là ứng viên điểm thưởng.
6. **multi_problem:** khi câu tiên quyết sai, GV chỉ nhận xét câu đó hay nhận xét cả các câu sau? Kiểm tra số "Câu k" được nhắc so với cấu trúc đề.
7. **Nhãn rỗng (bài đúng):** GV viết gì? Khen, gợi ý code_quality, hay để trống? Điều này quyết định cách render cho trường hợp không có lỗi.

---

## 3. Kiến trúc sinh 2 bước: Diagnose rồi Render

### Bước 0: thu bằng chứng (tất định, không dùng LLM)
- **Biên dịch:** g++ cố định phiên bản, `-std=c++17`, chạy sandbox có timeout. Parse log lỗi ra (dòng, thông điệp).
- **Chạy test:** với single_problem, chạy testcase công khai và lấy input gây sai, output mong đợi, output thực tế. Có thể sinh thêm input biên như n=0, n=1, n max, số âm, tràn `int`.
- **Phân tích tĩnh:** tree-sitter-cpp để lấy định danh và map dòng ↔ hàm/câu. Có thể thêm cppcheck.
- **multi_problem:** tách file theo từng câu (theo hàm hoặc comment đánh dấu) và áp dụng chính sách tiên quyết.
- **Repair nội bộ:** LLM sửa code; chỉ giữ bản sửa qua được test. Diff thu được cho vị trí chính xác.
  - Phung et al. (LAK 2024) cho thấy đưa failing test và bản sửa dạng symbolic vào prompt làm hint tốt lên rõ rệt.
  - Bản sửa này **chỉ dùng nội bộ** cho chẩn đoán, cho detector leakage (§4) và làm lời giải ở L4.

### Bước 1: chẩn đoán có cấu trúc
Output là JSON, ép đúng schema bằng guided decoding (outlines / xgrammar / guided JSON của vLLM):

```json
{"issues":[{"label":"<∈ nhãn cho sẵn>","question":"Câu 2|null","line_span":[8,10],
  "identifiers":["i","n"],"symptom":"...","root_cause":"...",
  "failing_input":"...","expected":"...","actual":"...",
  "fix_sketch":"<CHỈ NỘI BỘ>","priority":1,"confidence":0.8}]}
```

**Kiểm tra cứng sau khi sinh:**
- dòng nằm trong phạm vi file;
- `identifiers` thuộc tập định danh do tree-sitter trích;
- `label` thuộc tập nhãn cho sẵn;
- mỗi nhãn cho sẵn được phủ bởi ít nhất 1 issue (vì nhãn là ground truth);
- khẳng định "không biên dịch được" phải khớp compile_log.

**Self-consistency:** lấy k mẫu, gom theo (label, line_span), giữ các issue được đa số đồng ý.

### Bước 2: render theo level với chính sách tiết lộ (information masking)

| Trường | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| label (khái niệm), symptom trừu tượng | có | có | có | có |
| question / line_span / identifiers | không (chỉ hàm hoặc câu) | có | có | có |
| root_cause, failing_input | gợi ý dạng câu hỏi | có | có | có |
| fix_sketch | **không đưa vào prompt** | **không đưa vào prompt** | có (bằng lời / pseudo-code một phần) | có |
| repaired code | không | không | một phần | đầy đủ, đã qua test |

**Lợi ích:**
- **Tách độ đúng khỏi mức tiết lộ.** Chẩn đoán được tối ưu và đánh giá riêng; renderer chỉ lo phần sư phạm.
- **Đảm bảo theo nghĩa thông tin:** renderer L1/L2 không thể lộ cái nó không được nhìn thấy.
- **Dùng lại cho Task 1/2:** chẩn đoán cũng là đặc trưng cho phân loại lỗi và chấm điểm. Riêng Task 1/2 phải chạy chẩn đoán **không có** feedback và **không có** nhãn.
- **Có bằng chứng trong tài liệu:** Daheim et al. (EMNLP 2024) cho thấy "verify-then-generate" (phát hiện lỗi trước, sinh phản hồi sau) làm phản hồi bám lỗi tốt hơn.

**Sắp thứ tự ưu tiên lỗi:** lỗi biên dịch, rồi câu tiên quyết, rồi lỗi logic chính, rồi edge case, cuối cùng là code_quality. Số issue được render theo level lấy từ EDA (GT: L1 khoảng 1, L2 khoảng 1–2).

---

## 4. Đảm bảo compliance nhiều lớp (từ rẻ đến đắt)

**L-a. Prompt ràng buộc theo level.**
- Hợp đồng level ghi rõ được nói gì và cấm nói gì; few-shot lấy cùng level, cùng nhãn từ train (đã lọc vi phạm); ví dụ âm tính.
- Cấm backtick; định danh viết trong ngoặc kép.
- Chỉ dựa vào prompt là không đủ: Hellas et al. (ICER 2023) ghi nhận model họ GPT gần như luôn chèn code, kể cả khi không được yêu cầu.

**L-b. Information masking** như ở §3.

**L-c. Constrained decoding cho L1/L2.**
- Cài bằng custom LogitsProcessor. Với HF dùng `bad_words_ids` hoặc `sequence_bias`; với vLLM dùng `bad_words` hoặc logits processor (API đổi giữa V0 và V1, cần kiểm chứng).
- **Phải quét toàn bộ vocab.** Với BPE, `;` nằm trong nhiều token như `";\n"`, `"};"`, `" ;"`. Lấy mọi token mà chuỗi giải mã chứa chuỗi bị cấm:
  `[i for t,i in tok.get_vocab().items() if any(s in tok.convert_tokens_to_string([t]) for s in BAN)]`
- Danh sách đề xuất:
  - **cấm cứng:** ```` ``` ````, `` ` ``, `#include`, `{`, `}`, `;`, `cout`, `cin`, `printf`, `::`, `->`;
  - **phạt mềm** (bias khoảng −3 đến −5): `<=`, `>=`, `==`, `!=`, `++`, `[`, `]`.
- **Tác dụng phụ:**
  1. Model né bằng cách viết code bằng lời ("cho i chạy từ 0 tới n trừ 1"). Lớp này chỉ chặn được bề mặt, M3 vẫn bắt được, nên vẫn cần L-d.
  2. Khi token bị cấm đang có xác suất cao, phân bố bị lệch: câu gãy, lặp, đôi khi chuyển sang tiếng Anh.
  3. Chặn cả trích dẫn hợp lệ biểu thức sai của sinh viên ở L2 (`a[i]`, `i <= n`), nên chẩn đoán có thể giảm. Vì vậy L2 chỉ nên phạt mềm, không cấm cứng.
  4. Không ảnh hưởng tính tái lập: vẫn tất định khi cố định seed.

**L-d. Bộ phát hiện leakage sau khi sinh.** Sinh xong thì chấm; nếu bị reject thì sinh lại.
- **Regex:** các đặc trưng ở §2, cộng mẫu lệnh sửa tiếng Việt.
- **tree-sitter-cpp:** parse từng câu hoặc đoạn. Đoạn có ít nhất k token, parse thành statement/expression và tỷ lệ node `ERROR` thấp thì coi là code.
- **Trùng với phần sửa:** n-gram overlap giữa feedback và *dòng được thêm* trong diff repair (hoặc lời giải mẫu nếu có). Đây là tín hiệu mạnh nhất.
- **Information gain (đề xuất mới):**
  `LeakIG = [log p(diff | code, fb) − log p(diff | code)] / |diff|`
  - tính dưới một code LLM cố định, ví dụ Qwen2.5-Coder-7B;
  - đo xem feedback làm bản sửa dễ đoán hơn bao nhiêu;
  - là thước đo liên tục, dùng được cả làm margin cho DPO.
- **Mô phỏng học sinh:** theo hướng "student model" của Phung et al. (2024), một model nhỏ nhận code + feedback và sửa trong 1 lần. Ở L1/L2, nếu sửa đúng *và* diff trùng gần nguyên văn với bản sửa thì phản hồi quá lộ.
- **Quyết định:**
  - reject thì sinh lại, dùng danh sách seed cố định (để tái lập) và prompt chặt hơn, tối đa N lần;
  - hết N lần thì fallback sang **template tất định** dựng từ JSON chẩn đoán. Template này luôn compliant; chẩn đoán kém hơn nhưng không bao giờ vi phạm.

**L-e. Self-critique.** Một LLM chấm theo rubric ("Có câu nào cho phép sửa mà không cần suy nghĩ? Có nêu cách sửa không?"), rồi viết lại phần vi phạm. CodeHelp (Liffiton et al., Koli Calling 2023) dùng pipeline prompting có guardrail với cùng mục tiêu không lộ lời giải.

**L-f. Classifier compliance học từ train.**
- Dương tính: feedback L3/L4 của GV. Âm tính: L1/L2 đã lọc sạch.
- Thêm hard negatives tổng hợp: lấy L1/L2 rồi chèn snippet hoặc lệnh sửa lấy từ diff.
- Model: PhoBERT (cần tách từ) hoặc XLM-R / multilingual-e5.
- Dùng làm reranker cho best-of-N:
  `score = diag_score − λ·P(leak)`, với λ tinh chỉnh trên dev để chọn điểm Pareto.

---

## 5. Fine-tune

### Model và phần cứng
- **Base:** Qwen2.5-Coder-7B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B, Llama-3.1-8B-Instruct.
  - Ưu tiên họ Qwen vì vừa mạnh code vừa ổn tiếng Việt. Kiểm chứng license và ghi rõ revision hash.
- **Huấn luyện:** QLoRA bằng PEFT + TRL hoặc Unsloth, vừa với 1 GPU 24GB ở seq khoảng 4k (cần đo thực tế).
- **Bằng chứng khả thi:**
  - Solano, Koutcheme et al. (2025) SFT Qwen3-4B, Llama-3.1-8B và Qwen3-32B trên 40k cặp giải thích lỗi biên dịch C; họ ghi nhận model frontier hay "over-assist".
  - Kotalwar et al. (NeurIPS 2024 D&B) fine-tune Llama3-8B và Phi3 trên dữ liệu tổng hợp từ GPT-4, vượt GPT-3.5.

### SFT
- **Input:**
  - đề;
  - code đánh số dòng (`12| ...`), để model trích dòng chính xác;
  - compile_log / test_report;
  - nhãn;
  - control token level, ví dụ `<L1>`…`<L4>` (special token mới) hoặc tiền tố "[MỨC 2]".
- **Output:** feedback GV.
- Mask loss trên phần prompt.
- **Làm sạch dữ liệu:**
  - bỏ L1/L2 vi phạm (theo §2);
  - dedupe template;
  - cân bằng level bằng upsample.

### Level-swap augmentation (phản thực tế)
- Với cùng (code, chẩn đoán), dùng LLM thương mại sinh feedback ở cả 4 level. Model học rằng level là biến điều khiển duy nhất, nên khả năng kiểm soát tốt hơn.
- Đây cũng là nguồn cặp ưu tiên tự nhiên cho bước sau.

### Học theo ưu tiên (DPO / ORPO / KTO)
Cặp được sinh tự động. Chosen là feedback GV đã sạch, hoặc mẫu tự sinh qua được detector và có `diag_score` cao. Rejected gồm 4 loại:
1. **Chèn leak:** chosen cộng một snippet hoặc lệnh sửa lấy từ diff. Hai bên khác nhau chỉ ở thuộc tính leakage.
2. **Mơ hồ:** compliant nhưng chẩn đoán kém. Loại này chống việc model học cách "né" bằng sự mơ hồ.
3. **Hallucination:** sai dòng, hoặc định danh không tồn tại trong code.
4. **Lệch level:** dùng output L3 làm rejected cho prompt L2.

Lưu ý khi dùng:
- **Khớp độ dài** giữa chosen và rejected để DPO không học thiên kiến độ dài.
- **KTO** hợp khi chỉ có nhãn tốt/xấu không ghép cặp (từ detector). **ORPO** gộp SFT và preference trong một pha, đỡ compute.
- Vị trí module trong TRL có thể đổi theo phiên bản (cần kiểm chứng).
- **Bằng chứng:**
  - Sonkar et al. (EMNLP Findings 2024): DPO/KTO tốt hơn SFT rõ rệt ở xu hướng gợi mở thay vì đưa lời giải.
  - Scarlatos et al. (AIED 2025): DPO theo rubric sư phạm đưa model 8B lên ngang GPT-4o trên rubric.
- **Mở rộng (rủi ro cao):** GRPO với reward `diag − λ·LeakIG`. Dễ bị reward hacking, nên chỉ làm khi còn thời gian.

### Distillation từ LLM thương mại
Dùng để:
- sinh chẩn đoán JSON cho train, được kiểm chứng bằng compile/test;
- sinh dữ liệu level-swap;
- sinh rejected loại 1 và 2.

Quản lý:
- Log token × đơn giá cho từng call và cache kết quả, để báo cáo chi phí.
- Cần hỏi BTC về chính sách gửi code sinh viên ra API (mục 9 của đề) và ưu tiên chế độ không lưu dữ liệu (zero-retention).

---

## 6. Tiếng Việt, độ dài, giọng sư phạm, chống hallucination

**Giọng theo level** (lấy theo phân bố của GV trong train):
- L1: câu hỏi Socratic, ví dụ "Với n = 0, vòng lặp ở hàm tính tổng chạy bao nhiêu lần?". TreeInstruct (Kargupta et al., EMNLP Findings 2024) và Socratic Debugging Benchmark (Al-Hossami et al., BEA 2023) là mẫu tham khảo cho phong cách hỏi dẫn dắt.
- L2: chỉ vùng lỗi + hiện tượng + nguyên nhân.
- L3: chiến lược sửa + một bước mẫu.
- L4: lời giải và giải thích.

**Văn phong:**
- Đại từ ("em"/"bạn"), cấu trúc khen–chỉ–hỏi, và thuật ngữ ("vòng lặp" / "loop", "mảng") theo tần suất của GV, vì judge so với feedback GV nhạy cả văn phong.
- Lập glossary thuật ngữ từ train.

**Độ dài:**
- Khớp trung vị ± IQR theo level đo ở §2.
- Kiểm soát bằng chỉ dẫn "≤ N câu" và `max_new_tokens`.
- Kiểm tra judge có thiên vị độ dài không (xem §7).

**Chất lượng chữ:**
- Chuẩn hoá NFC.
- Đo tỷ lệ từ thiếu dấu.
- Cấm lẫn tiếng Anh dài.

**Kiểm tra tham chiếu (chống hallucination), sai thì sinh lại:**
- Mọi "dòng N" được nhắc phải tồn tại và nằm trong `line_span` ± 1 của một issue.
- Mọi token ASCII giống định danh phải thuộc tập định danh trong code. Ngoại lệ là từ khoá C++ ở L3/L4.
- Mọi giá trị testcase được nhắc phải có trong test_report.
- "Câu k" phải tồn tại trong đề multi_problem.

**Số dòng:**
- Số dòng GV dùng tham chiếu file gốc. Cần chuẩn hoá CRLF và dòng trống *giống cách GV đếm*.
- GT: kiểm chứng bằng cách so "dòng N" trong feedback với vị trí của định danh được nhắc cùng câu.

---

## 7. Đánh giá nội bộ (không tốn quota leaderboard)

**Chia dữ liệu:**
- Kiểm tra mã đề trùng giữa train và dev: `len(set(train.pid) & set(dev.pid)) / dev.pid.nunique()`.
- Nếu dev chứa đề chưa thấy thì dùng GroupKFold theo mã đề.

**Compliance proxy:** ensemble gồm regex, tree-sitter, classifier, LeakIG và LLM-judge. Báo cáo từng thành phần và mức đồng thuận giữa chúng.

**Judge chẩn đoán:**
- Open-weight cố định, ví dụ Qwen2.5-32B-Instruct-AWQ hoặc Qwen3-14B/32B chạy trên vLLM.
- `temperature=0`, seed cố định, ghi revision hash.
- Rubric lấy từ các chiều của Maurya et al. (2025), mỗi chiều 0–2, output JSON:
  1. cùng lỗi chính với GV;
  2. cùng vị trí;
  3. không có khẳng định sai;
  4. mức hướng dẫn phù hợp level.

**Meta-evaluation trên dev:**
1. Feedback GV của chính bài đó phải được điểm cao (kiểm tra trần).
2. Feedback GV của bài khác cùng đề nhưng khác nhãn phải được điểm thấp (đối chứng âm).
3. Chèn câu thừa vô nghĩa không được làm tăng điểm (kiểm tra thiên kiến độ dài).
4. Đảo vị trí trong so sánh cặp (kiểm tra thiên kiến vị trí).
5. Nhóm tự chú thích khoảng 100 mẫu **train/dev** (không đụng tới test), tính Spearman và Cohen's κ giữa judge và người chấm.
6. Đối chiếu điểm nội bộ với kết quả public LB qua các lượt nộp, để hiệu chỉnh judge.

Tài liệu tham khảo cho LLM-as-judge với feedback lập trình: Koutcheme et al. (ITiCSE 2024) và "Evaluating LMs for Generating and Judging Programming Feedback" (SIGCSE TS 2025).

**Thống kê và quota:**
- Bootstrap CI 95% trên dev; chỉ coi là cải thiện khi CI không chồng lấn hoặc khi kiểm định cặp có ý nghĩa.
- Quota tối đa khoảng 2 lượt/ngày × 56 ngày, chia chung cho cả 3 task (cần hỏi BTC). Chỉ nộp các điểm trên đường Pareto nội bộ.

---

## 8. Ý tưởng ablation cho Task 3

Tất cả đo bằng cặp (compliance, chẩn đoán) và vẽ trên đường Pareto.

1. **1 bước vs 2 bước** (chẩn đoán rồi render).
2. **Information masking:** renderer L1/L2 thấy hay không thấy `fix_sketch` và repair.
3. **Tín hiệu bằng chứng:** không có / compile_log / + test_report / + chạy testcase biên / + diff repair.
4. **Constrained decoding:** không có / chỉ cấm fence / cấm cứng toàn bộ / phạt mềm; đo kèm độ trôi chảy và chẩn đoán.
5. **Detector + sinh lại:** N = 0, 1, 3, 5; có hoặc không có fallback template; số đo phụ là tỷ lệ phải fallback.
6. **Hàm tín hiệu leakage:** regex / tree-sitter / classifier / LeakIG / LLM-judge. Đo precision và recall so với khoảng 100 mẫu nhóm tự gán (train/dev).
7. **SFT vs SFT+DPO vs ORPO vs KTO**, và tác động riêng của từng loại rejected (leak / mơ hồ / hallucination / lệch level).
8. **Cách điều khiển level:** special token vs chỉ dẫn bằng chữ vs LoRA riêng từng level; bật/tắt level-swap augmentation.
9. **Lọc dữ liệu gold:** train có và không có các mẫu L1/L2 tự vi phạm (định lượng hệ quả của vấn đề dữ liệu, gắn với điểm thưởng).
10. **Retrieval few-shot:** k = 0/2/4; ngẫu nhiên / cùng nhãn / cùng đề / cùng level.
11. **Self-consistency chẩn đoán:** k = 1 vs 5.
12. **Kiểm tra tham chiếu** (dòng/định danh) bật/tắt: đo tỷ lệ hallucination và mức thay đổi điểm chẩn đoán.
13. **Base model và lượng tử hoá:** Qwen2.5-Coder-7B vs Qwen3-8B vs mô hình 3–4B; 4-bit vs bf16; kèm chi phí và độ trễ.
14. **Baseline mơ hồ theo template:** chứng minh metric có thể bị game (ứng viên điểm thưởng về metric).

**Phân tích lỗi:** chia kết quả theo level × loại đề (multi/single) × nhãn × môn. GT là lỗi tập trung ở L2 với lỗi logic DSA, nơi khó tách "chỉ chỗ sai" khỏi "nói cách sửa".

---

## Câu hỏi mở cho BTC
1. Compliance được đo bằng regex, classifier hay LLM-judge? Có phạt L3/L4 khi tiết lộ thiếu không?
2. Chẩn đoán được đo bằng judge so với feedback GV, so nhãn hay so vị trí? Hai metric (a) và (b) gộp thế nào để xếp hạng?
3. Ở L2 có được trích nguyên văn biểu thức sai của sinh viên không?
4. Quota 2 lượt/ngày tính chung cho 3 task hay riêng từng task?
5. Có được gửi code sinh viên lên API thương mại để sinh dữ liệu train không, và với điều kiện gì?

## Nguồn
- Phung et al. 2024 (LAK): GPT-4 tutor sinh hint + GPT-3.5 student kiểm chứng hint. https://arxiv.org/pdf/2310.03780
- Phung et al. 2023: benchmark ChatGPT/GPT-4 so với tutor người, có kịch bản hint generation. https://arxiv.org/pdf/2306.17156
- Al-Hossami et al. 2023 (BEA): Socratic Debugging Benchmark. https://aclanthology.org/2023.bea-1.57/
- Kargupta et al. 2024 (EMNLP Findings): TreeInstruct. https://arxiv.org/abs/2406.11709
- Liffiton et al. 2023 (Koli Calling): CodeHelp, guardrail không lộ lời giải. https://arxiv.org/abs/2308.06921
- Hellas et al. 2023 (ICER): LLM gần như luôn chèn code. https://research.aalto.fi/en/publications/exploring-the-responses-of-large-language-models-to-beginner-prog/
- Daheim et al. 2024 (EMNLP): verify-then-generate. https://aclanthology.org/2024.emnlp-main.478/
- Maurya et al. 2025 (NAACL): taxonomy 8 chiều sư phạm, có "revealing of the answer". https://aclanthology.org/2025.naacl-long.57/
- BEA 2025 Shared Task (đánh giá năng lực sư phạm của AI tutor). https://sig-edu.org/sharedtask/2025
- Sonkar et al. 2024 (EMNLP Findings): Pedagogical Alignment, DPO/KTO so với SFT. https://aclanthology.org/2024.findings-emnlp.797/
- Scarlatos et al. 2025 (AIED): DPO cho tutorbot. https://arxiv.org/pdf/2503.06424
- Koutcheme et al. 2024 (ITiCSE): model mở sinh feedback + GPT-4-as-judge. https://juholeinonen.com/assets/pdf/koutcheme2024open.pdf
- Koutcheme et al. (SIGCSE TS 2025): đánh giá LM khi sinh và khi chấm feedback. https://arxiv.org/pdf/2407.04873
- Kotalwar et al. 2024 (NeurIPS D&B): Hints-In-Browser. https://neurips.cc/virtual/2024/poster/97759
- Solano, Koutcheme et al. 2025: SFT model mở cho giải thích lỗi biên dịch. https://arxiv.org/abs/2507.05305
- Auditable Release Control for Pedagogical Leakage (arXiv 2026, chưa peer-review). https://arxiv.org/pdf/2608.00515

Status: DONE
Summary: Đã viết đủ 8 mục cho Task 3; các trích dẫn chính đã kiểm chứng bằng WebSearch. Mọi nhận định về dữ liệu ghi rõ là GIẢ THUYẾT kèm phép kiểm chứng.
Concerns: API logits processor của vLLM và vị trí trainer ORPO/KTO trong TRL thay đổi theo phiên bản (đã ghi "cần kiểm chứng"); paper arXiv 2026 về pedagogical leakage chưa peer-review.