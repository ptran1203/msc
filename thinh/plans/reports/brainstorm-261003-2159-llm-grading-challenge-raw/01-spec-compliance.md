# Phân tích đặc tả: đồ án LLM chấm điểm và phản hồi bài lập trình

## 0. Quy ước

- **[GT]** là giả thuyết về dữ liệu, chưa kiểm chứng. Mỗi [GT] đi kèm một phép kiểm.
- Các snippet giả định `df` là tập train đã load với các cột `sample_id, type, code_file, statement, compile_log, test_report, labels (list), feedback, level`, 6 cột rubric và `total`. Tên cột thật phải đối chiếu khi có dữ liệu.
  - `DIMS = ['compilable','io_format','logic','edge_case','complexity','code_quality']`
  - `df['pid'] = df.code_file.str.split('/').str[1]` lấy mã đề từ đường dẫn.
- Ký hiệu thành phần điểm:

| Ký hiệu | Thành phần | Trọng số |
|---|---|---|
| **RANK** | Thứ hạng private leaderboard | 30% |
| **METH** | Phương pháp và thiết kế thí nghiệm | 25% |
| **ABL** | Ablation và phân tích lỗi | 15% |
| **REP** | Báo cáo, model card, trình bày | 20% |
| **REPRO** | Tái lập và chất lượng mã nguồn | 10% |
| **BONUS** | Vấn đề thật trong dữ liệu hoặc metric | +10% |
| **GATE** | Vi phạm có thể mất nhiều hơn trọng số danh nghĩa (kết quả "không được chấm", hoặc vi phạm liêm chính) | — |

---

## 1. Checklist yêu cầu tường minh

| # | Yêu cầu | Mức | Thành phần | Hậu quả nếu bỏ sót |
|---|---|---|---|---|
| R1 | Nộp đủ 3 task. Bảng tổng xếp theo trung bình thứ hạng 3 bảng | MUST | RANK | Thiếu 1 task thì gần như đứng cuối bảng đó, kéo cả hạng tổng xuống |
| R2 | T1 trả về 6 điểm thành phần và tổng, đúng miền (0–1/0–1/0–4/0–2/0–1/0–1, tổng 0–10) | MUST | RANK | Sai miền hoặc thiếu trường thì file bị từ chối, vẫn mất quota |
| R3 | T1 multi_problem tuân chính sách chấm của đề (câu tiên quyết, trọng số câu) | MUST, đề ghi "không phải chi tiết phụ" | RANK, METH | QWK sụt ở multi_problem; bị coi là bỏ qua yêu cầu cốt lõi |
| R4 | T2 trả về tập nhãn nằm trong 10 nhãn taxonomy; tập rỗng là hợp lệ | MUST | RANK | Nhãn sai chính tả bị tính là FP và FN |
| R5 | T2 tối ưu macro-F1, tức phải xử lý lệch nhãn | MUST (metric chính) | RANK, METH | Đề nói thẳng: chỉ đoán nhãn phổ biến sẽ bị phạt nặng |
| R6 | T3 sinh phản hồi tiếng Việt đúng level; L1/L2 không có code sửa hay lời giải | MUST | RANK | Vi phạm compliance (a) thì điểm thấp dù chẩn đoán đúng |
| R7 | T3 chẩn đoán khớp phản hồi của giảng viên | MUST | RANK | Điểm (b) thấp |
| R8 | Có ít nhất 2 hướng tiếp cận và so sánh chúng | MUST | METH, ABL | Mất phần lớn 25% |
| R9 | Có ít nhất 1 hướng prompting (zero/few-shot, CoT, self-consistency, RAG trên train, hoặc kết hợp) | MUST | METH | Không đạt yêu cầu tối thiểu |
| R10 | Có ít nhất 1 model open-weight được fine-tune (khuyến nghị ≤10B) | MUST; phần ≤10B là SHOULD | METH | Không đạt yêu cầu cốt lõi của môn |
| R11 | Chạy model fine-tune cục bộ và báo cáo cấu hình phần cứng | MUST | REPRO, REP | Mất điểm tái lập; Q&A dễ bị hỏi khó |
| R12 | Báo cáo chi phí nếu dùng API thương mại | MUST nếu có dùng | REP, METH | Thiếu mục "chi phí và triển khai" mà báo cáo bắt buộc phải có |
| R13 | Suy luận tự động 100%, không người can thiệp từng bài | MUST | GATE (RANK) | Có thể bị coi là gian lận, huỷ kết quả |
| R14 | Cố định seed và ghi phiên bản model | MUST | REPRO, GATE | "Không tái lập được sẽ không được chấm": có thể mất cả 30% RANK chứ không chỉ 10% |
| R15 | Không dùng trường feedback làm input cho T1/T2 | MUST | GATE | Huỷ kết quả T1/T2 |
| R16 | predictions.json gồm các phần tử `{sample_id, output}` và qua được script kiểm tra | MUST | RANK | Bị từ chối mà vẫn mất quota |
| R17 | Tối đa 2 lượt nộp/ngày/nhóm trên public LB (tập dev) | MUST | RANK | Lượt đã lỡ không bù được |
| R18 | Xếp hạng chính thức lấy theo private test | SHOULD (chiến lược) | RANK | Overfit dev thì tụt hạng ở private |
| R19 | Repo có mã nguồn, hướng dẫn cài, và một lệnh tái lập kết quả tốt nhất | MUST | REPRO | Mất 10% và kéo theo rủi ro gate R14 |
| R20 | Báo cáo 10–15 trang gồm: dữ liệu và EDA, các hướng tiếp cận, bảng ablation, phân tích lỗi (sai ở kiểu bài nào và vì sao), chi phí và triển khai thực tế, hạn chế | MUST | REP, ABL, METH | Thiếu mục nào bị trừ trực tiếp mục đó |
| R21 | Model card gồm dữ liệu train, giới hạn, rủi ro khi chấm thật | MUST | REP | Bị trừ điểm báo cáo |
| R22 | Slide, 15 phút trình bày, 10 phút hỏi đáp | MUST | REP | Bị trừ điểm trình bày |
| R23 | predictions.json cho private test | MUST | RANK | Không có hạng, mất 30/30 |
| R24 | Bảng đóng góp có xác nhận của cả nhóm | MUST | Điểm cá nhân | Điểm cá nhân bị lệch khỏi điểm nhóm |
| R25 | Không phát tán dữ liệu, repo không public, không dùng cho mục đích khác | MUST | GATE (liêm chính) | Bị kỷ luật, có thể 0 điểm |
| R26 | Không gán nhãn thủ công tập test, dù chỉ một phần | MUST | GATE | Như trên |
| R27 | Không truy ngược danh tính sinh viên | MUST | GATE | Như trên |
| R28 | Không chia sẻ mã nguồn hay dự đoán giữa các nhóm | MUST | GATE | Như trên |
| R29 | Cuối báo cáo ghi rõ đã dùng công cụ LLM nào cho việc gì | MUST | REP, liêm chính | Bị trừ điểm, có thể thành vấn đề liêm chính |
| R30 | Phát hiện và xử lý vấn đề dữ liệu: mất cân bằng, lệch phân bố điểm, nhãn suy biến | SHOULD (đề gọi là "một phần thách thức") | METH, ABL, REP (EDA) | Phương pháp bị đánh giá là ngây thơ |
| R31 | Báo cáo đủ metric phụ: MAE, exact-match từng chiều (T1), micro-F1 (T2) | SHOULD | METH, REP | Bảng kết quả thiếu |
| R32 | Chỉ ra vấn đề thật của dataset hoặc metric, kèm bằng chứng định lượng và đề xuất khắc phục | SHOULD | BONUS | Bỏ lỡ khoản +10% rẻ nhất |

---

## 2. Yêu cầu ngầm và bẫy

### 2.1 Phạm vi lệnh cấm feedback (R15)

| Cách dùng | Đánh giá | Khuyến nghị |
|---|---|---|
| Đưa feedback của chính bài vào prompt hoặc feature T1/T2 lúc suy luận | Cấm rõ ràng | Không dùng |
| Đưa feedback của bài train vào exemplar few-shot hoặc index RAG cho T1/T2 | Theo câu chữ vẫn là "trường feedback làm input" | Bỏ feedback khỏi exemplar và index của T1/T2 |
| Feature dẫn xuất từ feedback (`level`, độ dài, nhãn trích ra, embedding) | Coi như chính feedback | Cấm cho T1/T2 |
| Feedback làm target phụ khi fine-tune (multi-task, sinh rationale rồi mới chấm) | Không phải input lúc suy luận, nhưng thuộc vùng xám | Chỉ dùng khi công bố rõ, giữ một ablation "không dùng feedback ở bất kỳ đâu", và hỏi giảng viên |
| LLM teacher đọc feedback để sinh rationale distill cho T1/T2 | Thông tin feedback chui vào trọng số model; xám đậm | Mặc định teacher chỉ thấy code và nhãn rubric/taxonomy |
| Dùng feedback để làm sạch hoặc kiểm tra nhãn train | Xám nhẹ, chỉ tác động lên train | Chỉ dùng cho EDA và hồ sơ BONUS, không đưa vào model |

- **[GT] `level` tương quan mạnh với điểm**, nên nó là kênh rò rỉ nếu lọt vào T1.
  - Kiểm: `df.groupby('level').total.agg(['mean','std','size'])` và `scipy.stats.spearmanr(df.level, df.total)`.

### 2.2 Rò rỉ chéo giữa các task

- Input của T3 chứa **nhãn lỗi gold** và level.
  - Nếu test T2 và test T3 dùng chung bài, nhãn T3 chính là đáp án T2. `level` cũng có thể thành feature cho T1.
  - Kiểm ngay khi có file: `len(set(t2_test.code_file) & set(t3_test.code_file))`, và làm tương tự với T1.
  - Nếu có giao nhau: không khai thác. Báo giảng viên bằng văn bản; đây là ứng viên BONUS vì là lỗ hổng thiết kế benchmark.
- Không dùng nhãn taxonomy gold làm feature cho T1 khi train, vì lúc test không có.
  - Nếu cần nối T2 vào T1, dùng nhãn **dự đoán out-of-fold** (stacking).

### 2.3 Input tập test nằm trong tay nhóm

Vì nhóm tự nộp predictions.json, nhóm sẽ có input của private test.

- Không ai đọc bài test để chỉnh prompt, rule hay threshold. Làm vậy là can thiệp thủ công gián tiếp và overfit.
- Quy trình: tag git, chạy một lệnh, qua validator, nộp. Lưu commit hash và hash file kết quả.
- Không sửa tay output hỏng. Lỗi parse phải có fallback tự động: retry, rồi constrained decoding (JSON schema), rồi baseline rule.
- "Không người can thiệp" còn bao gồm: không tự chọn giữa các ứng viên cho từng bài, không chạy lại có chọn lọc những bài "trông sai".

### 2.4 Chính sách chấm multi_problem (R3)

- Chính sách có thể nằm trong **văn bản đề** (ngôn ngữ tự nhiên) chứ không phải metadata. Cần một bước trích chính sách thành JSON `{câu, trọng số, tiên quyết}`, kiểm tra bằng rule và cache theo `pid`.
- Kiến trúc khuyến nghị:
  - Model đánh giá **từng câu** trước.
  - Một **policy engine tất định** viết bằng Python áp quy tắc tiên quyết và trọng số rồi gộp ra 6 chiều và tổng.
  - Lợi ích: chắc chắn tuân thủ; ablation sạch (bật/tắt engine); dễ giải thích trong báo cáo.
- **[GT] Nhãn chỉ có ở mức file**, không có điểm từng câu. Khi đó không supervise trực tiếp từng câu được, mà phải đánh giá từng câu bằng zero/few-shot hoặc học gián tiếp.
  - Kiểm: đọc 5 JSON thô của multi_problem và liệt kê `columns`.
  - Tìm từ khoá trong đề: `df.statement.str.contains('tiên quyết|nếu câu|không được tính|không chấm', case=False).mean()`.
- **[GT] Hiệu ứng tiên quyết hiện ra trong nhãn** dưới dạng các bài biên dịch được, code dài, nhưng `logic` gần 0.
  - Kiểm: so phân phối `logic` giữa multi_problem và single_problem trong nhóm `compilable==1`.

### 2.5 Metric và định dạng

- **QWK cần lớp rời rạc.**
  - Nếu tổng có số lẻ (do trọng số câu) hoặc model dự đoán số thực, cách scorer làm tròn quyết định kết quả.
  - Python và numpy dùng banker's rounding: `round(2.5)=2`, `round(3.5)=4`.
  - Kiểm: `(df.total % 1 != 0).mean()` và `df.total.value_counts().sort_index()`.
- **`sklearn.metrics.cohen_kappa_score` tính trọng số bậc hai theo chỉ số lớp**, không theo giá trị điểm, khi không truyền `labels=`.
  - Nếu một mức điểm vắng mặt, hoặc có mức 7.5, khoảng cách giữa các mức bị nén hoặc méo.
  - Nếu scorer được phát, kiểm xem có truyền `labels=range(11)` không. Nếu không, đây là ứng viên BONUS.
- **Tối ưu QWK**: dự đoán giá trị liên tục rồi tối ưu ngưỡng cắt trên dự đoán out-of-fold, thay vì làm tròn trực tiếp.
- **[GT] `total` bằng tổng 6 chiều.** Kiểm: `(df[DIMS].sum(1) - df.total).abs().max()`.
  - Nếu đúng, ép output nhất quán (`total = Σ`).
  - Có hai cách: tối ưu tổng rồi phân rã ra từng chiều, hoặc dự đoán từng chiều rồi hiệu chỉnh. Lựa chọn phụ thuộc scorer đọc `total` hay tự cộng (xem §4).
- **Ràng buộc cứng giữa các chiều.**
  - **[GT]** `compilable=0` kéo `logic` và `edge_case` về 0.
  - Kiểm: `df[df.compilable==0][DIMS].describe()`.
  - Nếu đúng, áp làm hậu xử lý. Mọi output phải là số nguyên trong miền (clip rồi ép int).
- **Macro-F1 với nhãn cực hiếm.**
  - Nhãn không có mẫu dương trong test cho F1 phụ thuộc `zero_division`; sklearn mặc định là 0 kèm cảnh báo.
  - Với 10 nhãn, một nhãn bằng 10% macro-F1. Có nên đoán nhãn hiếm hay không phụ thuộc thiết lập này.
  - Kiểm tần suất: `Y = MultiLabelBinarizer().fit_transform(df.labels); pd.Series(Y.mean(0))`.
  - Dùng ngưỡng riêng cho từng nhãn, tối ưu trên out-of-fold.
- **Đoán đúng tập rỗng không được thưởng.**
  - F1 tính theo nhãn chỉ dựa vào TP/FP/FN. Đoán đúng "không lỗi" chỉ tạo ra TN, nên metric không đo khả năng nhận ra bài sạch mà chỉ phạt FP.
  - Đây là ứng viên BONUS. Đề xuất: F1 theo mẫu với quy ước rỗng khớp rỗng được 1 điểm, hoặc thêm nhãn giả "Không lỗi".
- **Chuỗi nhãn tiếng Việt.**
  - Chênh lệch NFC/NFD, khoảng trắng hay hoa/thường làm so khớp chuỗi thất bại, F1 nhãn đó về 0.
  - Lấy chuỗi đúng từ dữ liệu, chuẩn hoá NFC, và `assert set(pred) <= TAXONOMY` trước khi ghi file.
  - Kiểm dữ liệu: `{l for L in df.labels for l in L if unicodedata.normalize('NFC', l) != l}`. Nếu chính dữ liệu lẫn NFC/NFD thì là ứng viên BONUS.
- **Bảng tổng là trung bình thứ hạng**, nên task yếu nhất có sức nặng ngang task mạnh nhất.
  - [GT về đối thủ] T3 có độ chênh giữa các nhóm lớn nhất vì nhiều nhóm sẽ để lọt code ở L1/L2. Một guardrail tốt có thể mua được nhiều bậc hạng.
- **Quota.** 2 lượt/ngày cho tối đa khoảng 110 lượt trong 8 tuần, có thể tính chung cho cả 3 task.
  - Validator phải nằm cứng trong lệnh `make submit`; không bao giờ nộp tay.
  - Public LB chỉ để xác nhận. Chọn model dựa trên CV cục bộ kèm bootstrap CI.

### 2.6 Dữ liệu

- **Lệch phân bố dev/test. [GT] Dev/test có đề không xuất hiện trong train.**
  - Kiểm: `set(dev.pid) - set(train.pid)`, và tỷ lệ 2 môn, 2 loại đề theo từng split.
  - Nếu đúng: CV phải dùng `GroupKFold(groups=pid)`; không dựa vào kNN hay RAG kiểu "cùng đề".
- **Trùng code. [GT] Có bài gần như trùng nhau xuyên split, và có bài trùng code mà khác điểm.**
  - Kiểm: chuẩn hoá code (bỏ comment và khoảng trắng, thay identifier bằng token), băm, rồi `df.groupby('h').total.nunique().gt(1).sum()`.
  - Hệ quả 1: validation lạc quan quá mức.
  - Hệ quả 2: nhiễu nhãn tạo trần cho QWK, và là bằng chứng cho chuyện "hai giảng viên chấm lệch" mà đề nhắc. Ứng viên BONUS.
- **Nhãn suy biến. [GT] `complexity` và `code_quality` gần như hằng số; một số nhãn taxonomy dưới 1%.**
  - Kiểm: `df[d].value_counts(normalize=True)` cho từng chiều.
  - Hệ quả: exact-match trên chiều đó cao giả tạo. Báo cáo bắt buộc phải so với baseline luôn đoán giá trị phổ biến nhất.
- **Nhất quán chéo giữa các tín hiệu. [GT] `compilable==0` ⇔ nhãn "Lỗi biên dịch" ⇔ compile_log có lỗi.**
  - Kiểm: `pd.crosstab([df.compilable, df.compile_log.str.contains('error', case=False)], df.labels.apply(lambda L: 'Lỗi biên dịch' in L))`.
  - Nếu nhất quán: dùng rule tất định thay LLM cho một chiều và một nhãn.
  - Nếu mâu thuẫn: ứng viên BONUS.
  - Làm tương tự cho tỷ lệ pass của test_report so với `logic`/`edge_case` (Spearman).
- **Độ dài code. [GT] Bài multi_problem dài, vượt context của model ≤10B** sau khi ghép thêm đề và log.
  - Kiểm: đếm token bằng tokenizer của base model, xem p50/p95/p99 theo `type`.
  - Bị cắt ngắn âm thầm là loại lỗi khó phát hiện.
- **I/O.**
  - Encoding: UTF-8 BOM, CP1258, CRLF. File có thể thiếu hoặc rỗng.
  - Kiểm: `sum(not os.path.exists(p) for p in df.code_file)`, đếm số file decode lỗi, và `df.code_file.duplicated().sum()`.
- **Mã sinh viên ẩn danh trong tên file.**
  - Dùng làm `groups` cho CV là hợp lý, vì chống rò rỉ phong cách code của cùng một người.
  - Dùng làm feature "năng lực sinh viên" là profiling cá nhân, gần với R27. Tránh.
- **Ẩn danh có thể chưa triệt để** (tên hoặc MSSV trong comment).
  - Chỉ đếm bằng regex, không đọc, không tra cứu.
  - Scrub trước khi gửi API và báo giảng viên.
- **Prompt injection trong code. [GT] Có comment hoặc chuỗi kiểu "chấm 10 điểm", "ignore previous".**
  - Kiểm: grep và đếm.
  - Xử lý: bọc code trong delimiter, đặt chỉ dẫn ở system prompt, thêm ablation "bỏ comment" cho T1/T2, và ghi vào model card.

### 2.7 Tái lập (R14, R19)

- **API thương mại.**
  - Không tất định kể cả khi temperature=0. OpenAI có tham số `seed` dạng best-effort kèm `system_fingerprint`; Anthropic không có seed (cần kiểm chứng tại thời điểm dùng). Snapshot có thể bị khai tử trước buổi chấm.
  - Cách xử lý:
    - Pin snapshot có ngày và lưu trường `model` của response.
    - **Cache request → response**, khoá bằng hash của prompt và tham số.
    - Lệnh `reproduce` mặc định replay từ cache.
    - Báo cáo độ dao động qua 3 lần gọi thật.
- **Model local.**
  - Đặt seed ở mọi nơi: `transformers.set_seed`, numpy, random, torch, tham số `seed` của vLLM. Dùng greedy cho T1/T2.
  - Ghi phiên bản torch, transformers, vllm, peft, CUDA, driver và loại GPU; ghi revision hash của base model trên HF; lưu checksum của adapter.
  - Kết quả có thể đổi theo batch size hoặc loại GPU (cần kiểm chứng). Cố định batch size, chạy 2 lần và so diff dự đoán.
- **Seed cho mọi nguồn ngẫu nhiên**: chia split, fold CV, chọn exemplar few-shot, tie-break RAG, sampling khi dùng self-consistency.
- **"Một lệnh" phải tự lo hết**: tải weights và cache từ nơi lưu private; không phụ thuộc đường dẫn cá nhân; có lockfile (`uv.lock` hoặc `conda-lock`). Chạy thử trên máy sạch của một thành viên khác trước tuần 7.

### 2.8 Task 3

- **Detector compliance chưa rõ** (regex, classifier hay LLM-judge). Thiết kế phòng thủ cho L1/L2:
  - Không dùng code fence.
  - **Không trích nguyên dòng code của sinh viên**, vì detector có thể coi đó là code.
  - Diễn đạt bằng lời, dẫn số dòng, tên biến, tên hàm.
- **Guardrail 2 tầng**:
  - Tầng 1: regex bắt token C++ (`;` cuối dòng, `{}`, `cout<<`, `for(`, `#include`).
  - Tầng 2: verifier (classifier nhỏ hoặc LLM).
  - Nếu vi phạm: sinh lại tối đa k lần, sau đó rơi về template an toàn. Đo tỷ lệ vi phạm trên val.
- **[GT] Chính phản hồi gold ở L1/L2 có chứa code.**
  - Kiểm: `df[df.level<=2].feedback.str.contains(r"```|#include|cout\s*<<|for\s*\(", regex=True).mean()`.
  - Nếu tỷ lệ > 0:
    - Lọc hoặc viết lại các mẫu đó trước khi SFT, nếu không model sẽ học cách vi phạm.
    - Ứng viên BONUS: metric sẽ phạt cả giảng viên.
- **Nhãn cho sẵn.**
  - Có thể rỗng: khi đó viết nhận xét về chất lượng hoặc tối ưu.
  - Có thể nhiều nhãn: sắp theo nhân quả, biên dịch trước, rồi I/O, logic, trường hợp biên.
- **L4 (lời giải mẫu).** Với single_problem, có thể tự biên dịch và chạy public testcase trong sandbox để xác minh, chỉ trả về code đã pass.
- **[GT] Metric (a) chỉ phạt lộ đáp án ở L1/L2, không phạt thiếu ở L3/L4.**
  - Nếu đúng, một hệ thống luôn kín đáo vẫn đạt (a) tối đa, nên metric không đo đủ. Ứng viên BONUS: đề xuất đo cả hai chiều.
- **Tiếng Việt.** Model ≤10B phải viết tiếng Việt tự nhiên. Kiểm sớm bằng 30 mẫu val, chấm mù.
  - Ứng viên: Qwen2.5-7B-Instruct, Qwen2.5-Coder-7B-Instruct, Qwen3-8B. Chất lượng tiếng Việt từng model cần kiểm chứng.
- **Nếu metric (b) dùng LLM-judge**, bản thân điểm leaderboard cũng không tất định. Ứng viên BONUS về độ tin cậy của metric.

### 2.9 Quy trình

- **"Cục bộ"**: tối thiểu là suy luận trên máy nhóm kiểm soát. Ghi GPU, VRAM, RAM, CPU, thời gian train, throughput suy luận (bài/giây) và peak VRAM. Đây cũng là số liệu cho mục chi phí và triển khai.
- **Log chi phí API tự động**: token vào/ra nhân đơn giá, kèm ngày áp giá, cho mọi lần gọi kể cả thí nghiệm hỏng.
- **Gán nhãn thủ công trên train không bị cấm** (đề chỉ cấm test).
  - Viết tay khoảng 50 exemplar L1/L2 chuẩn hoặc rationale cho few-shot là đòn bẩy hợp lệ; ghi rõ trong báo cáo.
  - Không gán tay dev, vì dev chạy public LB nên thuộc vùng xám.
- **Dữ liệu hoặc model ngoài** (CodeNet, dataset feedback công khai): đề không cấm. Công bố và hỏi lại.
- **Điểm cá nhân**: commit history, bảng đóng góp và khả năng trả lời Q&A phải khớp nhau.
- **Khai báo LLM**: tách bạch công cụ hỗ trợ viết code/báo cáo với LLM là thành phần của hệ thống.
- **Thời gian chạy private test**: đo throughput sớm để chắc chạy kịp trong cửa sổ cho phép (chưa biết là bao lâu).
- **Phân tích lỗi**: xác định khoá phân tầng từ đầu (`type`, môn, `pid`, nhóm điểm, độ dài code, trạng thái biên dịch, nhãn). R20 đòi chỉ ra "kiểu bài nào".

---

## 3. Mâu thuẫn và căng thẳng giữa các điều khoản

| # | Điều khoản A | Điều khoản B | Căng thẳng | Xử lý an toàn |
|---|---|---|---|---|
| 1 | Được dùng LLM thương mại qua API | Không phát tán dữ liệu ra ngoài lớp, không dùng cho mục đích khác | Gửi code sinh viên cho bên thứ ba, có thể bị lưu log hoặc dùng để train | (i) Hỏi giảng viên bằng văn bản ngay tuần 1. (ii) Nếu dùng: chọn gói hoặc endpoint cam kết không train trên dữ liệu và lưu tối thiểu (kiểm chứng ToS lúc dùng); scrub comment và chuỗi; chỉ gửi train/dev. (iii) Phương án B: teacher open-weight lớn chạy trên GPU của nhóm hoặc trường, ví dụ Qwen2.5-Coder-32B-Instruct bản lượng tử hoá. Ghi vào model card |
| 2 | API dùng "để huấn luyện" | Yêu cầu có hướng prompting | Câu chữ chỉ cho phép dùng API để huấn luyện, không nói gì về lúc suy luận | Mặc định hệ thống nộp private chạy bằng model local. API chỉ để distill, sinh dữ liệu và làm baseline so sánh (có cache). Hỏi giảng viên |
| 3 | Cố định seed; không tái lập thì không chấm | API thương mại, self-consistency có sampling | API không tất định | Cache-replay là nguồn sự thật; báo cáo phương sai. Local sampling dùng seed cố định |
| 4 | Không gán nhãn thủ công test | Pseudo-label, self-training, transductive bằng LLM trên test | Câu chữ chỉ cấm "thủ công"; tinh thần là không khai thác test | Mặc định không train trên input test. Self-training dùng input dev hoặc train. Chỉ dùng test khi giảng viên đồng ý, chạy hoàn toàn trong pipeline một lệnh, có công bố và ablation |
| 5 | Cấm feedback làm input T1/T2 | Input T3 test có nhãn gold và level | Rò rỉ chéo nếu cùng bài | Kiểm giao nhau (§2.2); không khai thác; báo giảng viên |
| 6 | Compliance (a) | Chẩn đoán chính xác (b) | Càng cụ thể (dòng, biến, bản chất cách sửa) càng gần lời giải | Coi (a) là ràng buộc cứng, (b) là mục tiêu. Định nghĩa thang level rõ ràng (§4 #8). Vẽ đường Pareto giữa (a) và (b) theo độ cụ thể trên val và đưa vào ablation |
| 7 | Fine-tune ≤10B chạy cục bộ | Thứ hạng leaderboard | Hệ thống tốt nhất có thể là model lớn hoặc API | Hệ thống cuối là model fine-tune distill từ teacher. Bảng so sánh báo cả hai. Nếu final dùng API thì phải gánh rủi ro của dòng 1 và 3 |
| 8 | Repo và tái lập một lệnh | Không phát tán dữ liệu | Adapter có thể ghi nhớ code sinh viên: công khai weights gần như phát tán dữ liệu. Cache API chứa code sinh viên | Weights và cache để ở storage private, cấp quyền cho giảng viên. Không đẩy lên HF Hub public. Repo chỉ chứa code, config và hash |
| 9 | Khuyến khích thảo luận công khai trên diễn đàn | Bonus mang tính cạnh tranh; không chia sẻ code | Đăng phát hiện về dữ liệu sớm thì nhóm khác cùng hưởng | Gửi phát hiện cho giảng viên trước (có mốc thời gian), sau đó chỉ thảo luận ở mức ý tưởng. Không đăng code, dự đoán hay mẫu code sinh viên |
| 10 | Báo cáo 10–15 trang | Nhiều mục bắt buộc, ablation, phân tích lỗi | Thiếu chỗ | Hỏi phụ lục có tính trang không. Dùng bảng dày và hình tự sinh |
| 11 | Phải so sánh ít nhất 2 hướng | Quota 2 lượt/ngày | Không đủ lượt để so sánh trên dev | So sánh chính bằng CV cục bộ (cùng split, cùng seed, bootstrap CI). Mỗi hướng chính nộp dev một lần để có điểm đối chiếu chung |
| 12 | Một rubric 6 chiều dùng chung | Đề nhiều câu có trọng số | Không rõ 6 chiều ở mức file được gộp thế nào | Xem §4 #1 |
| 13 | Không người can thiệp từng bài | Phân tích lỗi phải đọc bài | Không thật sự mâu thuẫn nếu tách bạch | Phân tích chỉ trên train/val. Suy luận test không chạm tay |

---

## 4. Điểm mơ hồ

| # | Điểm mơ hồ | Ảnh hưởng | Giả định mặc định | Câu hỏi gửi giảng viên |
|---|---|---|---|---|
| 1 | Tổng của multi_problem là tổng 6 chiều, hay tổng có trọng số theo câu? | Kiến trúc T1, QWK | Tổng 6 chiều; 6 chiều ở mức file đã gộp trọng số (kiểm bằng snippet §2.5) | "Với multi_problem, 6 chiều tính ở mức file hay theo từng câu rồi gộp? `total` có luôn bằng tổng 6 chiều?" |
| 2 | Điểm có số lẻ không? Scorer làm tròn thế nào trước QWK? | QWK | Số nguyên; nhóm tự làm tròn và xuất int | "Scorer làm tròn `total` ra sao? QWK có dùng tập nhãn cố định 0..10 không?" |
| 3 | Scorer đọc `total` hay tự cộng 6 chiều? Nếu hai số lệch nhau thì sao? | Đánh đổi giữa tối ưu QWK và exact-match | Xuất nhất quán, `total = Σ` | "QWK tính trên trường `total` nộp lên hay trên tổng các chiều?" |
| 4 | Chính sách tiên quyết nằm ở đâu? "Sai" nghĩa là 0 điểm hay dưới ngưỡng? Áp lên những chiều nào? | Policy engine | Nằm trong văn bản đề. "Sai" là câu tiên quyết không đạt `logic`. Câu phụ thuộc bị 0 ở mọi chiều theo câu | "Xin 2–3 ví dụ đã chấm có câu tiên quyết sai, và định nghĩa 'sai'." |
| 5 | Exact-match báo từng chiều hay trung bình? Có dùng để xếp hạng không? | Thứ tự ưu tiên tối ưu | Chỉ là metric phụ; LB xếp theo QWK | "Bảng T1 xếp theo QWK thuần?" |
| 6 | Danh sách đủ 10 nhãn và chuỗi chính xác (đề chỉ ghi "…") | Định dạng T2 | Lấy từ train, chuẩn hoá NFC | "Có file taxonomy chính thức kèm định nghĩa từng nhãn không?" |
| 7 | Macro-F1 dùng `zero_division` bao nhiêu? Nhãn vắng trong test tính ra sao? | Chiến lược với nhãn hiếm | Mặc định sklearn, bằng 0 | "Scorer T2 có công bố không? `zero_division`=?" |
| 8 | Định nghĩa Level 1–4; ranh giới L2/L3 | Toàn bộ T3 | L1: loại lỗi hoặc khái niệm cần xem lại, không chỉ vị trí. L2: vị trí và bản chất lỗi, không cách sửa. L3: hướng sửa bằng lời hoặc pseudocode. L4: lời giải hoàn chỉnh | "Có guideline viết phản hồi theo level không? Pseudocode hoặc trích dòng code của sinh viên có tính là 'code sửa' không?" |
| 9 | Compliance (a) đo bằng gì? Có phạt L3/L4 quá kín không? | Thiết kế guardrail | Chỉ phạt L1/L2; detector tự động | "Detector là regex, classifier hay LLM-judge? Có được dùng thử không?" |
| 10 | "Độ chính xác chẩn đoán" (b) đo bằng gì: LLM-judge (model, prompt nào), embedding, so khớp nhãn trích ra, hay người chấm? | Hình thức output T3 | LLM-judge so từng điểm chẩn đoán, nên viết chẩn đoán tường minh và ngắn | "Judge có cố định seed và phiên bản không? Có công bố không?" |
| 11 | Định dạng output T3: chuỗi thuần hay markdown? Giới hạn độ dài? Code fence ở L3/L4? | Validator, (a) | Chuỗi UTF-8, markdown nhẹ, L1/L2 khoảng ≤200 từ | Hỏi trực tiếp |
| 12 | Nhãn lỗi cho sẵn ở T3 test là gold hay dự đoán? Có thể rỗng không? | T3 | Gold, có thể rỗng | Hỏi trực tiếp |
| 13 | Quota 2 lượt/ngày tính chung cả nhóm hay theo từng task? Reset theo múi giờ nào? Một file hay ba file? | Kế hoạch nộp | Chung cho cả nhóm, reset 0h GMT+7 | Hỏi trực tiếp |
| 14 | Bản nộp nào tính cho private: lần cuối, lần tốt nhất, hay file nộp kèm sản phẩm? Khi nào phát input test? | Chiến lược bản final | File nộp kèm sản phẩm (mục 8.5) | Hỏi trực tiếp |
| 15 | Lệnh cấm feedback có bao gồm target phụ khi train, exemplar RAG, `level`, rationale distill từ feedback không? | GATE | Chỉ cấm ở input suy luận và exemplar; target phụ cho phép nếu công bố | Liệt kê từng trường hợp ở §2.1 và hỏi |
| 16 | API thương mại: chỉ để huấn luyện hay cả suy luận? Báo chi phí đơn vị gì, có tính thí nghiệm hỏng không? | Privacy, hệ thống cuối | Chỉ train và baseline; báo tổng USD và VND, gồm mọi thí nghiệm | Hỏi trực tiếp |
| 17 | "Chạy cục bộ" áp cho cả train hay chỉ suy luận? Colab/Kaggle/VM thuê có tính không? | Hạ tầng | Suy luận trên máy nhóm; train được dùng cloud nếu báo cáo | Hỏi trực tiếp |
| 18 | "≤10B" là tổng tham số hay tham số active (MoE)? | Chọn model | Tổng tham số | Hỏi trực tiếp |
| 19 | Ngưỡng "tái lập được": dự đoán trùng khít hay metric lệch ≤ε? | REPRO | Local greedy trùng khít; API lệch ≤0.005 | Hỏi trực tiếp |
| 20 | Dùng input test (không nhãn) để pseudo-label hay self-training có hợp lệ? Input dev thì sao? | Phương pháp | Không dùng test; dev được | Hỏi trực tiếp |
| 21 | Dev có công bố nhãn không? Sau đó có được train trên dev? | Lượng dữ liệu | Dev không có nhãn | Hỏi trực tiếp |
| 22 | Có được dùng dữ liệu hoặc model ngoài? | Phương pháp | Được, kèm công bố | Hỏi trực tiếp |
| 23 | Có được tự biên dịch, chạy code sinh viên (sandbox) và sinh testcase mới lúc suy luận? | Tín hiệu cho multi_problem | Được, nếu tự động | Hỏi trực tiếp |
| 24 | compile_log và test_report sinh bằng compiler/flag nào? Testcase công khai có phải toàn bộ test giảng viên dùng? | Độ tin cậy tín hiệu | g++ chuẩn; test công khai là tập con | Hỏi trực tiếp |
| 25 | Đề bài có nằm trong dữ liệu không? Có lời giải tham khảo không? | RAG, L4 | Có đề, không có lời giải | Hỏi trực tiếp |
| 26 | 10–15 trang có tính phụ lục và tài liệu tham khảo không? Template, ngôn ngữ? | REP | Phụ lục không tính; viết tiếng Việt | Hỏi trực tiếp |
| 27 | **Mục 7 bị thiếu** (đề nhảy từ 6 sang 8): có nội dung bị lược không, như mốc thời gian, checkpoint giữa kỳ, hỗ trợ GPU, baseline chính thức? | Có thể sót yêu cầu bắt buộc | Không có | "Mục 7 có nội dung gì bị thiếu?" |
| 28 | Thứ hạng quy ra điểm (30%) thế nào: tuyến tính? theo bảng tổng hay từng bảng? đồng hạng? | Phân bổ công sức | Theo bảng tổng, tuyến tính | Hỏi trực tiếp |
| 29 | Tiêu chí "vấn đề thật" cho bonus? Nhiều nhóm cùng phát hiện thì sao? | Chiến lược bonus | Chấm theo chất lượng bằng chứng | Hỏi trực tiếp |

---

## 5. Sản phẩm nộp và tiêu chí "xong"

1. **Repository (private)**
   - Clone sạch trên một máy khác, chạy `make reproduce`, thu được predictions cho cả 3 task và metric val, khớp số trong báo cáo (≤ε).
   - README nêu: yêu cầu phần cứng; cài đặt bằng lockfile; vị trí đặt dữ liệu (repo không chứa dữ liệu); cách tải weights và cache API từ storage private.
   - `configs/` chứa seed, model id, revision hash và snapshot API.
   - Validator chạy trong pre-submit hoặc CI.
   - Không có secret; chỉ có `.env.example`.
   - Có log chi phí API và manifest phần cứng, phiên bản.
   - Repo không public; giảng viên có quyền đọc. `git log` không chứa dữ liệu sinh viên.
2. **Báo cáo 10–15 trang**
   - Đủ các mục: dữ liệu và EDA (mất cân bằng, nhãn suy biến, lệch phân bố); các hướng (ít nhất một prompting và một fine-tune); bảng ablation; phân tích lỗi phân tầng theo `type`, môn, đề, nhóm điểm, nhãn, kèm nguyên nhân; chi phí và triển khai (USD trên 1000 bài, latency, phần cứng); hạn chế; khai báo công cụ LLM.
   - Mọi con số truy được về run-ID và config.
   - Có baseline (luôn đoán giá trị phổ biến nhất, và baseline rule từ compile_log/test_report).
   - Metric chính có bootstrap CI. Số trang đúng giới hạn.
3. **Model card (1–2 trang)**
   - Dữ liệu train: nguồn, kích thước, phân bố, những gì đã lọc.
   - Mục đích sử dụng và ngoài phạm vi.
   - Metric theo nhóm (2 môn, 2 loại đề, theo level).
   - Giới hạn: đề mới, code dài, chất lượng tiếng Việt.
   - Rủi ro khi chấm thật: thiên lệch theo phong cách code, prompt injection (có số đo trên val), lộ lời giải, sai ở bài điểm sát ngưỡng, người dùng quá tin máy, quyền riêng tư.
   - Khuyến nghị có người kiểm tra (human-in-the-loop). Phiên bản, phần cứng, chi phí.
4. **Slide, 15 phút trình bày, 10 phút hỏi đáp**
   - Diễn tập ≤14 phút.
   - Mỗi thành viên trả lời được về toàn hệ thống.
   - Có backup slide: ablation chi tiết, vấn đề dữ liệu, chi phí.
   - Có video demo lệnh tái lập.
5. **predictions.json cho private test**
   - Sinh từ commit đã tag, bằng đúng một lệnh.
   - Qua validator; đủ sample_id và mỗi sample_id xuất hiện đúng một lần.
   - Lưu hash file và commit hash.
6. **Bảng đóng góp**: phần việc cụ thể và tỷ lệ % của từng người, cả nhóm xác nhận, khớp với commit history.
7. **(Ngầm) Hồ sơ BONUS**: mỗi vấn đề gồm mô tả, bằng chứng định lượng (script tái lập được), tác động lên metric hoặc model, đề xuất khắc phục. Gửi giảng viên kèm mốc thời gian.
8. **(Ngầm) Manifest tái lập**: phiên bản thư viện, seed, phần cứng, log chi phí. Có thể nằm trong repo.

---

## 6. Phân tích trọng số: đầu tư ở đâu lợi nhất trên mỗi giờ công

| Thành phần | % | Nhóm kiểm soát được bao nhiêu | Đòn bẩy chính | ROI/giờ |
|---|---|---|---|---|
| RANK | 30 | Thấp–trung bình (tương đối so với lớp) | Hậu xử lý theo metric, policy engine, guardrail T3 | Cao ở giai đoạn đầu, giảm dần nhanh |
| METH | 25 | Cao | Split đúng (GroupKFold), baseline, so sánh có kiểm soát, xử lý dữ liệu | Cao |
| ABL | 15 | Rất cao | Ablation dạng config flag ngay từ đầu; phân tích lỗi phân tầng | Rất cao |
| REP | 20 | Rất cao | Khung báo cáo từ tuần 3; hình và bảng tự sinh từ log | Rất cao |
| REPRO | 10 (và là GATE) | Cao | Cache, seed, lockfile, chạy thử trên máy sạch | Rất cao, vì bảo vệ cả 30% RANK |
| BONUS | +10 | Trung bình–cao | EDA sớm, kiểm định các [GT] | Cao |

**Nhận định chính**

- 70% (25+15+20+10) cộng 10% bonus không phụ thuộc thứ hạng. Tuy vậy REPRO và liêm chính là điều kiện nhân: vi phạm có thể xoá luôn 30% RANK.
- Nếu thứ hạng quy ra điểm tuyến tính với N nhóm (ví dụ 10, chưa biết chính xác), mỗi bậc hạng khoảng 3.3% tổng điểm. Bảng tổng tính trung bình thứ hạng nên **tránh thảm hoạ ở bất kỳ task nào** quan trọng hơn tối ưu một task.

**Thứ tự đầu tư theo ROI**

1. **Tuần 1: hạ tầng tái lập, validator, baseline đủ 3 task và nộp sớm.** Việc này bảo vệ gate, tạo điểm sàn trên LB và cho METH một baseline. Kèm theo: gửi câu hỏi cho giảng viên.
2. **EDA và kiểm định các [GT] ở §2.6.** Một người làm 1–2 tuần, cùng lúc phục vụ ba thứ: EDA (REP), xử lý dữ liệu (METH) và hồ sơ BONUS. Ứng viên bonus mạnh nhất:
   - Code trùng mà điểm khác (nhiễu giữa người chấm).
   - Phản hồi gold L1/L2 chứa code.
   - Mâu thuẫn giữa rubric, taxonomy và compile_log.
   - Metric F1 không thưởng việc đoán đúng tập rỗng.
   - Compliance chỉ đo một chiều.
   - QWK tính theo chỉ số lớp.
   - Rò rỉ chéo giữa T2 và T3.
   - Phương sai lớn của macro-F1 do nhãn hiếm (chứng minh bằng bootstrap).
3. **Hậu xử lý theo metric (rẻ, tăng RANK nhiều)**: tối ưu ngưỡng cho QWK, ngưỡng riêng từng nhãn cho macro-F1, ràng buộc nhất quán (rule biên dịch), policy engine cho multi_problem, guardrail cho T3. Hai thứ đề nhấn mạnh là câu tiên quyết và compliance; nhiều nhóm sẽ làm hời hợt cả hai.
4. **Ablation dạng flag** (compile_log/test_report bật/tắt, CoT, số exemplar RAG k, engine bật/tắt, guardrail bật/tắt, cân bằng dữ liệu, rank LoRA) cộng bảng tự sinh. ABL gần như miễn phí.
5. **Fine-tune ở mức "đủ tốt"**: QLoRA 7–8B, một cấu hình chính và 1–2 biến thể. Không sa vào sweep siêu tham số.
6. **ROI thấp**: mài prompt để thêm khoảng +0.01 QWK; ensemble nhiều API đắt tiền.

**Phân công gợi ý cho 6 người** (mỗi người sở hữu một mục báo cáo)

- Data/EDA/BONUS.
- T1: policy engine, hồi quy và tối ưu ngưỡng.
- T2: classifier và ngưỡng từng nhãn.
- T3: sinh phản hồi và guardrail.
- Fine-tune và hạ tầng tái lập.
- Bộ đánh giá, báo cáo và slide.

**Mốc thời gian**

| Tuần | Việc |
|---|---|
| W1 | Hạ tầng, baseline, câu hỏi gửi giảng viên |
| W2 | EDA, bản nháp các vấn đề dữ liệu |
| W3–5 | Các hướng tiếp cận, khung báo cáo |
| W6 | Fine-tune bản cuối, ablation |
| W7 | Freeze, chạy tái lập trên máy sạch, viết báo cáo |
| W8 | Slide, chạy private test |

**Câu hỏi gửi giảng viên ngay tuần 1** (xếp theo tác động)

1. Phạm vi cấm feedback (bao gồm `level`) và việc test T2/T3 có dùng chung bài không.
2. Chi tiết scorer: cách làm tròn trước QWK, scorer đọc `total` hay tự cộng các chiều, `zero_division` của macro-F1.
3. Định nghĩa từng level, cách phát hiện vi phạm compliance, judge của metric (b).
4. Gửi code sinh viên lên API có được phép không, và API có được dùng lúc suy luận không.
5. Quota tính chung hay theo từng task.
6. Chính sách tiên quyết được mã hoá ở đâu.
7. Có được dùng input test (không nhãn) không.
8. Nội dung của mục 7 bị thiếu.