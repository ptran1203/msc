# SRS: hệ thống chấm rubric, phân loại lỗi và sinh phản hồi cho bài C++ (đồ án LLM)

| Mục | Giá trị |
|---|---|
| Phiên bản | 0.1, bản nháp để nhóm review |
| Ngày | 2026-10-04 |
| Vai trò soạn | Requirements Engineer |
| Căn cứ | Đề đồ án (4 trang, §1–§9, không có §7); bộ mẫu 32 bài `sample_dataset/`; [báo cáo EDA](../plans/reports/eda-261003-2335-sample-dataset.md) (ưu tiên khi mâu thuẫn); [brainstorm](../../plans/reports/brainstorm-261003-2159-llm-grading-challenge.md); quy ước Python PY-00..PY-13 |
| ID định nghĩa ở đây | C-xx, BR-xx, FR-<MOD>-xx, NFR-<CAT>-xx, Q-xx. STK/G/KPI/UC/US/RSK/A thuộc tài liệu BA. SC/TC thuộc tài liệu scenario/test |

**Ký hiệu nguồn:**
- Đề §n: mục n của đề.
- RM: README của bộ mẫu. LS: `label_space.json`. EXM: `exams.json`.
- EDA §n: báo cáo EDA. BS §n: brainstorm. PY-xx: quy ước Python.

**Độ chắc chắn:**
- **Chắc:** đề, README hoặc LS ghi rõ.
- **[GT]:** quan sát trên mẫu n = 32. README ghi mẫu không phải train/test chính thức, nên phải kiểm lại trên train chính thức (FR-EVL-10) trước khi áp cứng.

**MoSCoW:** M = Must, S = Should, C = Could, W = Won't (không làm ở bản này).

**Quy mô:**

| Loại | Số lượng |
|---|---|
| Ràng buộc C | 21 |
| Quy tắc nghiệp vụ BR | 25 |
| Yêu cầu chức năng FR | 88, thuộc 12 module (M 70, S 15, C 2, W 1) |
| Yêu cầu phi chức năng NFR | 22 |
| Câu hỏi mở Q | 33 |

---

## 1. Mục đích, phạm vi, thuật ngữ

### 1.1 Mục đích

- Đặc tả yêu cầu kiểm chứng được cho hệ thống của nhóm. Đây là căn cứ cho plan, thiết kế, test (TC-*) và báo cáo.
- Mỗi yêu cầu truy ngược được về điều khoản đề, quy tắc nghiệp vụ hoặc phát hiện EDA (§7).

### 1.2 Phạm vi: hai "hệ thống"

| | (1) Hệ thống challenge (đối tượng của SRS) | (2) Bối cảnh thật: trợ lý chấm bài cho giảng viên |
|---|---|---|
| Mục tiêu | Tối đa metric private (QWK, macro-F1, compliance + chẩn đoán) trong ràng buộc đề; tái lập bằng 1 lệnh | Giảm thời gian chấm, chấm nhất quán hơn, phản hồi có tính sư phạm |
| Người dùng | Nhóm 5–7 học viên; giảng viên chấm đồ án | Giảng viên, trợ giảng; sinh viên nhận phản hồi |
| Chuẩn "đúng" | Nhãn gold, kể cả khi gold lệch chính sách đã viết | Chính sách chấm đã công bố và phán quyết cuối của giảng viên |
| Mức tự động | 100% lúc suy luận (C-05) | Chỉ gợi ý; giảng viên duyệt trước khi công bố |
| Đặc tả ở | Toàn bộ §2–§6 | FR-RPT-05 (model card) và bảng §1.3 |

**Ngoài phạm vi:**
- giao diện người dùng, triển khai production;
- ngôn ngữ khác C++;
- gán nhãn mới cho dữ liệu;
- mọi thao tác trên input test ngoài lệnh `make predict-private`.

### 1.3 Thay đổi yêu cầu khi chuyển từ (1) sang (2) (đưa vào model card)

| Khía cạnh | (1) Challenge | (2) Triển khai thật |
|---|---|---|
| Chính sách tiên quyết | Dùng làm feature, tối ưu theo gold (BR-08) | Áp đúng chính sách đã viết (phiên bản (b) của FR-POL-03) |
| Bài nghi nhãn nhiễu (EX02-S214) | Giữ nguyên nhãn train | Gắn cờ để giảng viên xem |
| Phản hồi T3 | Qua guard; vi phạm thì dùng template fallback | Giảng viên duyệt; không tự gửi L4 khi bài tập còn mở |
| Hậu quả khi sai | Mất điểm leaderboard | Chấm sai ảnh hưởng sinh viên; thiên lệch theo văn phong; prompt injection trong comment |

### 1.4 Kiến trúc tham chiếu (chỉ để định vị các module)

```
ING → EVD (sandbox, cache theo code_sha) → POL (problem card, aggregate → feature)
        └─ evidence card ─┬─ T1: P-arm, F-arm, GBDT → stack → decoder QWK → ràng buộc theo exam → vector 6 chiều
                          ├─ T2: luật CE, P-arm, classifier → ngưỡng theo nhãn → ràng buộc chéo T1↔T2
                          └─ T3: diagnose → render theo level → GRD (guard → sinh lại → template)
EVL (scorer, CV, CI, judge) · VAL (validator, nộp bài) · REP (cache, seed, run_id)
SEC (feedback firewall, sandbox, dữ liệu ra ngoài) · RPT (ablation, phân tích lỗi, model card, bonus)
```

### 1.5 Thuật ngữ

| Thuật ngữ | Định nghĩa dùng trong SRS |
|---|---|
| QWK | Quadratic Weighted Kappa giữa total gold và total dự đoán. Luôn tính với `labels=list(range(11))` cố định (EDA §2.5) |
| MAE | Trung bình trị tuyệt đối của (total gold − total dự đoán) |
| Exact-match từng chiều | Tỷ lệ bài đoán đúng giá trị của một chiều rubric. Luôn báo kèm baseline mode (EDA §2.5) |
| Rubric vector | Bộ 6 số (compilable, io_format, logic, edge_case, complexity, code_quality), viết gọn, ví dụ `112111` |
| macro-F1 / micro-F1 | F1 lấy trung bình đều trên 10 nhãn / F1 tính trên tổng TP, FP, FN của mọi nhãn |
| zero_division (zd) | Giá trị F1 gán cho nhãn có TP + FP + FN = 0 (theo sklearn): 0, 1, hoặc nan (bỏ nhãn khỏi trung bình) |
| 0-support | Nhãn không có positive nào trong tập đánh giá. Ví dụ "Lỗi edge case" có 0/32 (EDA §3.1) |
| Level compliance | Tỷ lệ phản hồi không vi phạm điều cấm của level yêu cầu. Đây là metric (a) của T3 |
| Độ chính xác chẩn đoán | Mức phản hồi nêu đúng lỗi, so với phản hồi giảng viên; đo bằng LLM-as-judge + chấm người trên mẫu con. Đây là metric (b) |
| Vượt mức / dưới mức | Phản hồi lộ thông tin của level cao hơn / thiếu nội dung tối thiểu của level yêu cầu |
| Leak (lộ lời giải) | Phản hồi đủ để sinh viên sửa thành code đúng mà không cần suy nghĩ. Với EX02 kiểm bằng thực thi (FR-GRD-05) |
| Evidence card | JSON tất định cho mỗi bài, khoá (`code_sha`, `toolchain_id`): compile, test, phân tích tĩnh, near-dup, trạng thái câu, bản text gọn. Dùng chung cho mọi task và mọi hướng |
| Problem card | JSON cho mỗi exam, trích từ EXM: các câu, trọng số, đồ thị tiên quyết, điều khoản 0 điểm, ràng buộc cấm |
| Policy engine | Module POL: problem card + hàm `aggregate()` tất định. Đầu ra là **feature**, không ghi đè dự đoán (BR-08) |
| Guard | Module GRD: hậu kiểm phản hồi T3 theo điều cấm của level. Vi phạm thì sinh lại, hết lượt thì dùng template fallback đúng level |
| Template fallback | Phản hồi dựng sẵn theo (level × nhãn), đã qua guard |
| P-arm / F-arm | Hướng prompting (không cập nhật trọng số) / hướng fine-tune: cập nhật trọng số model open-weight ≤10B (LoRA/QLoRA/full, có thể kèm head). Classifier trên embedding đóng băng KHÔNG tính là F-arm, chỉ là baseline |
| OOF | Out-of-fold: dự đoán cho mỗi bài train từ model không thấy fold chứa bài đó. Dùng để fit ngưỡng, decoder, stacker |
| Near-dup | Cặp bài cùng exam rất giống nhau sau khi bỏ code mẫu của đề. Ví dụ EX02-S203..S206 (EDA §5) |
| Group CV | Chia fold sao cho mọi bài cùng cụm near-dup (hoặc cùng exam, nếu test có đề mới) nằm trong cùng fold |
| Pass strict / pass nới | So `actual` với `expected` nguyên văn / sau khi bỏ prompt, chuẩn hoá khoảng trắng, bỏ phân biệt hoa thường (EDA §2.3) |
| io_gap | Pass nới − pass strict |
| CE | Compile error: `compile_log` có diagnostic `error:`, **không** tính lỗi link |
| UB | Undefined behavior, ví dụ hàm non-void thiếu return. Làm `actual=''` và làm kết quả phụ thuộc toolchain (EDA §5) |
| De-shrink | Giãn dự đoán total về đúng μ và σ của gold (rescale hoặc OptimizedRounder) để tối đa QWK (BS §4.1) |
| Level-swap | Sinh phản hồi cho cùng một chẩn đoán ở cả 4 level, để model học level là biến điều khiển (EDA §7.2.4) |
| Feedback firewall | Tập cơ chế bảo đảm trường feedback và mọi dẫn xuất của nó không đi vào T1/T2 (BR-21) |
| Cache-replay | Chạy lại từ cache (khoá theo hash nội dung) thay cho gọi lại LLM hay sandbox |
| run_id | Định danh duy nhất của một lần chạy; gắn vào log, OOF, predictions, manifest |
| [GT] | Giả thuyết: mới quan sát trên mẫu, chưa kiểm trên tập đầy đủ |

---

## 2. Ràng buộc (C-xx)

| ID | Ràng buộc | Nguồn | Yêu cầu liên quan |
|---|---|---|---|
| C-01 | Ba task có input/output cố định. T1 ra 6 chiều + tổng 0–10. T2 ra tập con của 10 nhãn (rỗng hợp lệ). T3 ra đoạn phản hồi tiếng Việt theo level | Đề §3 | FR-T1-01, FR-T2-01, FR-T3-01 |
| C-02 | Triển khai và so sánh ít nhất 2 hướng: ≥1 hướng prompting và ≥1 model open-weight được fine-tune | Đề §4 | FR-T3-04 (fine-tune bắt buộc), FR-T1-04, FR-T2-03, FR-EVL-08; FR-T1-05, FR-T2-04 khi là LoRA ≤10B |
| C-03 | Model fine-tune khuyến nghị ≤10B, chạy được cục bộ, báo cấu hình phần cứng | Đề §4 | FR-RPT-03, NFR-PERF-04, Q-10 |
| C-04 | Được dùng LLM thương mại qua API để huấn luyện, nhưng phải báo chi phí | Đề §4 | FR-SEC-04, FR-RPT-03, NFR-COST-01 |
| C-05 | Toàn bộ pipeline suy luận chạy tự động, không có người can thiệp từng bài | Đề §4 | FR-POL-01, FR-GRD-06, FR-SEC-05 |
| C-06 | Cố định seed và ghi rõ phiên bản model. Không tái lập được thì không được chấm | Đề §4 | FR-REP-*, NFR-REPRO-* |
| C-07 | Cấm dùng trường feedback làm input cho T1 và T2 | Đề §4, RM | BR-21, FR-ING-05, FR-SEC-01 |
| C-08 | Public LB chạy trên dev, 2 lượt nộp/ngày/nhóm. Private LB chạy trên test, xếp hạng chính thức theo private. Bảng tổng xếp theo trung bình thứ hạng 3 task | Đề §5 | FR-VAL-03, FR-VAL-05 |
| C-09 | Nộp một file `predictions.json`, mỗi phần tử gồm `sample_id` và trường `output` của task. File sai định dạng bị từ chối và vẫn tính vào quota | Đề §5 | FR-VAL-01..04, Q-01 |
| C-10 | Sản phẩm nộp: repo có hướng dẫn cài và script tái lập kết quả tốt nhất bằng 1 lệnh; báo cáo 10–15 trang (EDA, các hướng, ablation, phân tích lỗi, chi phí và triển khai, hạn chế); model card; slide 15 phút + hỏi đáp 10 phút; predictions cho test riêng tư | Đề §8 | FR-REP-01, FR-RPT-*, FR-RPT-08 |
| C-11 | Không phát tán dữ liệu ra ngoài lớp, không đưa lên repo công khai, không dùng cho mục đích khác | Đề §9 | FR-SEC-03/04, NFR-PRIV-* |
| C-12 | Không gán nhãn thủ công cho tập test, dù chỉ một phần | Đề §9 | FR-SEC-05 |
| C-13 | Không truy ngược danh tính sinh viên | Đề §9 | BR-25, FR-SEC-06 |
| C-14 | Không chia sẻ mã nguồn hay dự đoán giữa các nhóm (được thảo luận ý tưởng) | Đề §9 | FR-SEC-03 |
| C-15 | Được dùng LLM hỗ trợ viết mã và báo cáo, nhưng phải ghi rõ công cụ nào cho việc gì ở cuối báo cáo | Đề §9 | FR-RPT-07 |
| C-16 | Với đề nhiều câu, hệ thống phải tôn trọng chính sách chấm của đề (ví dụ câu tiên quyết sai thì câu sau không tính). Đề nhấn mạnh đây không phải chi tiết phụ | Đề §3 T1 | BR-07, BR-08, FR-POL-* |
| C-17 | T3: sinh ở Level 1/2 mà đưa code sửa hoặc lời giải là vi phạm. Đo cả tính sư phạm | Đề §3 T3 | BR-18, BR-19, FR-GRD-* |
| C-18 | Code nằm ở `submissions/<mã đề>/<mã đề>-<mã SV>.cpp`; JSON chỉ tham chiếu qua `code_file` | Đề §2 | FR-ING-02 |
| C-19 | Trọng số chấm đồ án: private rank 30%, phương pháp 25%, ablation + phân tích lỗi 15%, báo cáo + trình bày 20%, tái lập + mã nguồn 10%. Bonus tối đa +10% cho vấn đề thật của dữ liệu hoặc metric, kèm bằng chứng định lượng và đề xuất khắc phục | Đề §6 | FR-RPT-01/02/04, FR-EVL-* |
| C-20 | Bảng đóng góp từng thành viên có xác nhận của cả nhóm | Đề §6 | FR-RPT-07 |
| C-21 | Thời lượng 8 tuần, nhóm 5–7 người | Đề (đầu trang) | Ưu tiên MoSCoW; thứ tự cắt giảm theo BS §9 |

---

## 3. Quy tắc nghiệp vụ (BR-xx)

| ID | Phát biểu | Nguồn | Độ chắc chắn |
|---|---|---|---|
| BR-01 | total_score = Σ 6 chiều; mọi giá trị là số nguyên | Đề §2 (thang tổng 0–10 bằng Σ giá trị tối đa); EDA §2.1 (32/32) | [GT], mạnh (32/32) |
| BR-02 | Miền giá trị: compilable, io_format, complexity, code_quality ∈ {0, 1}; logic ∈ {0..4}; edge_case ∈ {0..2}; total ∈ {0..10} | Đề §2, LS | Chắc |
| BR-03 | Cổng biên dịch của EX01: bài CE nhận vector `000000` (total 0) | EXM (statement EX01 có điều khoản 0 điểm khi không biên dịch được); EDA §2.2 (3/3) | [GT] |
| BR-04 | Cổng biên dịch của EX02: bài CE nhận `000001` (total 1, code_quality vẫn bằng 1). Cổng là **quy ước theo exam**, không phải lỗi nhãn | EDA §2.2 (3/3) | [GT] |
| BR-05 | EX02: logic ∈ {0, 2, 4}; edge_case = logic/2; code_quality luôn bằng 1 | EDA §0.2, §2.1 (17/17) | [GT] |
| BR-06 | EX01: complexity = code_quality = [logic > 0] | EDA §0.2, §2.1 (15/15) | [GT] |
| BR-07 | Chính sách **như đề viết** (EX01): P1 (đọc file) sai thì P2–P4 không được chấm; lặp vô hạn thì 0 điểm. Trọng số P1..P4 = 1/4/2/3 theo statement, nhưng `problems[].max_score` ghi 2.5 × 4 (mâu thuẫn) | EXM `grading_policy`, statement; EDA §2.4 | Chắc (văn bản); trọng số đúng → Q-08 |
| BR-08 | Chính sách **như gold thực tế**: gold không áp cổng P1 nhất quán. Cùng lỗi P1, EX01-S105 được 2 còn EX01-S110 được 8; nhóm "P1 sai" trải từ 1 đến 9. Hệ thống (1) tối ưu theo gold, policy chỉ là feature, và báo song song một phiên bản đúng policy | EDA §0.3, §2.4, §7.2.1 | [GT], mạnh |
| BR-09 | Gold áp luật "lặp vô hạn → 0" theo **câu**, không theo file: EX01-S106/S107 vẫn được 2 nhờ compilable + io; EX01-S112 lặp vô hạn mà logic vẫn bằng 3 | EDA §2.4 | [GT] |
| BR-10 | io_format chưa có chuẩn chính thức. Gold EX02 không trừ việc in prompt nhưng trừ việc thừa khoảng trắng; ở EX01, io = 0 chỉ xảy ra khi P1 crash. Không suy io_format trực tiếp từ pass strict | EDA §0.5, §2.3, §2.4 | [GT]; chuẩn chính thức → Q-12 |
| BR-11 | Ground truth là nhãn giảng viên. `compile_log` và `test_report` chỉ là tín hiệu phụ và có thể lệch: log do BTC sinh lại, có log bị cắt ở 2000 ký tự, `test_report` đã strip khoảng trắng và không có returncode | RM; EDA §0.10, §5 | Chắc |
| BR-12 | Nhãn T2 là đúng 10 chuỗi trong LS, so khớp sau chuẩn hoá Unicode NFC. Không có nhãn ngoài vocab, không trùng nhãn | Đề §3; LS; EDA §3.1 | Chắc |
| BR-13 | Tập rỗng `[]` là output T2 hợp lệ | Đề §3 T2 | Chắc |
| BR-14 | Ràng buộc chéo T1↔T2: `[]` ⇔ total = 10 (31/32, ngoại lệ EX02-S214); total < 10 ⇒ có ít nhất 1 nhãn (27/27) | EDA §0.7, §3.2 | [GT] |
| BR-15 | "Lỗi biên dịch" ⇔ compilable = 0 ⇔ log có diagnostic `error:` (6/6, κ = 1.0). Lỗi link **không** thuộc quy tắc này (sinh 1 dương tính giả: EX02-S214), chỉ dùng làm feature | EDA §3.2, §7.1 D5 | [GT], mạnh |
| BR-16 | Không suy nhãn T2 từ chiều rubric: nhập/xuất không đi với io_format, style không đi với code_quality, thuật toán không đi với complexity, edge case không đi với edge_case (không trùng bài nào, hoặc chỉ là trừ dây chuyền theo logic) | EDA §3.2 | [GT] |
| BR-17 | "Lỗi hard-code" trên mẫu nghĩa là gán cứng nguồn dữ liệu (đường dẫn file, kích thước mảng, đọc `cin` thay vì đọc file) ở P1 của EX01 (3/3). Không có bài gán cứng output theo test | EDA §3.1 | [GT]; định nghĩa chính thức → Q-18 |
| BR-18 | Bốn level và điều cấm. **L1 Gợi ý nhẹ:** chỉ báo có vấn đề hoặc khen, không chỉ vị trí lỗi; cấm nêu nguyên nhân cụ thể, cấm đưa code sửa. **L2 Chẩn đoán:** chỉ ra lỗi ở đâu và nguyên nhân; cấm đưa cách sửa cụ thể, cấm đưa code. **L3 Gợi ý sửa:** chẩn đoán + hướng sửa; cấm đưa lời giải hoàn chỉnh. **L4 Lời giải mẫu:** được đưa code sửa hoặc lời giải | LS `feedback_level` | Chắc |
| BR-19 | Metric (a) theo mô tả chỉ phạt **vượt mức** (ví dụ L1–2 đưa code sửa); các điều cấm khác chưa được định nghĩa vận hành. Hệ thống áp **toàn bộ** điều cấm của BR-18 (chặt hơn metric) và tự đo dưới mức | Đề §3 T3; trường metric của task 3; EDA §4.3 | Chắc (văn bản); cách vận hành → Q-04 |
| BR-20 | Trên mẫu, level gắn với exam và điểm: EX01 toàn L2; ở EX02, L1 ⇔ total = 10; không có L4. Hệ thống **không** được giả định phân bố này mà phải xử lý đủ 4 level cho mọi exam | EDA §4.1; RM (thừa nhận bias của mẫu nhỏ) | [GT] |
| BR-21 | Feedback firewall: trường `feedback` và mọi dẫn xuất (level suy từ feedback, độ dài, embedding, rationale, model hay artefact đã huấn luyện trên feedback) không được làm input, feature, exemplar few-shot, index RAG hay target phụ của T1/T2. Feedback chỉ dùng cho T3, EDA và hồ sơ bonus | Đề §4; BS §2.2 | Chắc (phần lõi); phạm vi mở rộng → Q-05 |
| BR-22 | Input T3 chứa nguyên đáp án T2 (`taxonomy_error`, 32/32). Không dùng input T3 để dự đoán T2. Nếu test T2 và test T3 có chung sample_id thì báo giảng viên, không khai thác | EDA §0.8; BS §2.2 | Chắc (quan sát) + quyết định nhóm |
| BR-23 | Input T3 không có `test_report`. Chỉ ghép `test_report` của T1 sang T3 (qua sample_id) khi được phép | EDA §4.5 | Chắc; quyền ghép → Q-14 |
| BR-24 | Phản hồi T3 viết bằng tiếng Việt, nói với sinh viên (xưng hô ngôi thứ hai), không phải ghi chú của người chấm. Gold EX01 có 9/15 bài gọi "SV" ở ngôi thứ ba và 3/15 bài dán log | Đề §1, §3; EDA §4.4 | Chắc (tiếng Việt); văn phong là quyết định nhóm |
| BR-25 | Dữ liệu đã ẩn danh (`[NAME_REDACTED]`, `[CLASS_REDACTED]`, mã S1xx/S2xx). Mã SV chỉ dùng làm khoá hoặc group, không làm feature (tránh profiling). Tài liệu không trích code sinh viên, chỉ dẫn sample_id | Đề §9; RM; BS §2.2 | Chắc |

---

## 4. Yêu cầu chức năng (FR-<MOD>-xx)

**Hai hướng tiếp cận theo task** (C-02). So sánh có kiểm soát theo FR-EVL-08.

| Task | Hướng prompting (P-arm) | Hướng fine-tune (F-arm, ≤10B, local) | Ghi chú |
|---|---|---|---|
| T1 | FR-T1-04: forced-choice logprob → kỳ vọng | FR-T1-05: LoRA model nhỏ + head thứ bậc (classifier trên embedding chỉ là baseline) | ≈ 121 bài [GT]: head 8B dễ overfit (EDA §7.2.5) |
| T2 | FR-T2-03: yes/no từng nhãn, đọc logprob | FR-T2-04: LoRA ≤10B cho xác suất 10 nhãn; GBDT/LR/classifier trên embedding là nguồn học được, không tính F-arm | Ngưỡng riêng từng nhãn (FR-T2-05) |
| T3 | FR-T3-04: zero/few-shot, Diagnose → Render | FR-T3-04/06: QLoRA SFT (tuỳ chọn DPO) trên dữ liệu đã làm sạch và level-swap → adapter A3 | **Fine-tune bắt buộc** (C-02) đặt ở đây |

### 4.1 ING: nạp dữ liệu

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-ING-01 | Hệ thống PHẢI nạp `exams.json`, `label_space.json` và 3 file task, validate theo schema §6.1 (trường bắt buộc, kiểu, miền, enum). Sai schema thì dừng ngay; thông báo nêu file, sample_id và trường | M | Đề §2; PY-05 (fail sớm) | Bộ mẫu: 0 lỗi, mỗi task 32 sample. 5 file hỏng cố ý (thiếu trường, sai kiểu, sai enum, điểm ngoài miền, sample_id trùng) → exit ≠ 0 và nêu đúng vị trí |
| FR-ING-02 | PHẢI đọc `code_file` theo đường dẫn tương đối với data-root; decode lần lượt utf-8 → cp1258 → latin-1; bỏ BOM; chuẩn hoá NFC cho bản text; giữ bytes gốc để tính `code_sha` (sha1) | M | C-18; EDA §5 (EX01-S107 có BOM); BS §10 | 32/32 đọc được; text của EX01-S107 không còn BOM; log số file phải fallback decode |
| FR-ING-03 | PHẢI tạo và kiểm `MANIFEST.sha256` của data-root; bỏ qua file rác (`.DS_Store`); cảnh báo file `.cpp` không được JSON nào tham chiếu. `code_file` không tồn tại: chế độ train/audit → `DataIntegrityError`, exit 2; chế độ predict/predict-private → **không dừng**: bài đó chạy `given_signals_only` với code rỗng, ra fallback hợp lệ, gắn cờ `missing_code` và ghi vào `private_audit` (C-09: không để 1 file thiếu làm mất toàn bộ predictions) | M | Bộ mẫu có 2 file `.DS_Store`; BS §10; C-09 | Đổi 1 byte bất kỳ → báo sai checksum. Xoá 1 file `.cpp`: train → exit 2, lỗi nêu đúng sample_id; predict → exit 0, output đủ id, bài đó có cờ `missing_code` |
| FR-ING-04 | PHẢI join 3 task theo sample_id và kiểm: sample_id duy nhất; input T1 ≡ input T2; input T3 = input T1 − `test_report` + `taxonomy_error` + `target_feedback_level`. Báo mức giao sample_id giữa T2 và T3: với train/dev tính bình thường; với split test chỉ được tính **bên trong `PrivateRun`**, chỉ ghi số đếm (không ghi id hay nội dung) vào `runs/private_audit.jsonl`; giao > 0 thì exit cảnh báo và báo giảng viên | M | EDA §1, §0.8; BR-22; FR-SEC-05 | Mẫu: 32/32 khớp; xuất bảng giao sample_id cho train/dev. Archive test tổng hợp giao 2 id → audit ghi `overlap=2`, 0 id được in |
| FR-ING-05 | PHẢI cấp input cho từng task qua whitelist trường. Loader của T1/T2 không có đường nào trả về `feedback`, `target_feedback_level` hay trường riêng của T3 | M | C-07; BR-21 | Unit test: object input T1/T2 không có key bị cấm; cố truy cập → lỗi |
| FR-ING-06 | PHẢI phân tích `compile_log` thành: `log_has_error` (diagnostic `error:` theo regex GNU), `log_has_link_error`, `log_build_failed` (OR của hai cờ trên), `log_warning_only`, `log_truncated` (độ dài bằng 2000), số lỗi, các dòng lỗi | M | EDA §5, §3.2; `load_data.py` | Mẫu: 22 log rỗng; 6 diagnostic error + 1 lỗi link; 3 chỉ có warning; 1 log bị cắt (EX01-S111). `log_has_error` ⇔ compilable = 0: 6/6 |

### 4.2 EVD: evidence và sandbox biên dịch-chạy

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-EVD-01 | PHẢI dựng evidence card JSON tất định cho mỗi bài, khoá (`code_sha`, `toolchain_id`). Card gồm: compile (FR-ING-06, cộng kết quả rebuild nếu có); test (strict, nới, phân loại fail, cờ crash/timeout); phân tích tĩnh (FR-EVD-05/06); near-dup (FR-EVD-07); trạng thái câu (FR-POL-04); bản text gọn ≤ 1.5k token. Card dùng chung cho mọi task và mọi hướng | M | BS §6; EDA §7.2.2 | Card validate theo JSON Schema; 100% bài có card; card không chứa trường feedback (kiểm bởi FR-SEC-01) |
| FR-EVD-02 | PHẢI tính pass strict, pass nới (bỏ prompt hoặc tiền tố, chuẩn hoá khoảng trắng, bỏ phân biệt hoa thường) và `io_gap` từ `test_report` | M | EDA §2.3 | Trên 13 bài EX02 có test: Spearman(pass nới, total) = 0.89 ± 0.01; bản strict = 0.57 ± 0.01 |
| FR-EVD-03 | PHẢI phân loại mỗi test fail vào: sai đáp án, output rỗng, chỉ sai vì prompt, timeout, prompt + sai đáp án. `actual=''` gắn cờ nghi crash (UB), không coi là "không in gì" | M | EDA §2.3, §5 | Mẫu: 57/104 test fail được phân đúng 17/16/11/8/5 |
| FR-EVD-04 | NÊN biên dịch lại và chạy lại EX02 trong sandbox (FR-SEC-02) với toolchain pin (GNU g++ 13, `-std=c++11 -O0`, trap khi hàm non-void thiếu return). Ghi stdout thô, returncode, stderr, thời gian. Từ stdout thô phát hiện: in prompt, thừa hoặc thiếu khoảng trắng, sai hoa thường. Cache theo (`code_sha`, `toolchain_id`, test) | S | EDA §5 (104/104 khi có trap), §7.1 D11, §7.2.2 | Tái lập ≥ 96/104 giá trị `actual` của mẫu; cờ prompt bật cho EX02-S211 và EX02-S216; cờ khoảng trắng bật cho EX02-S205; chạy 2 lần cho card giống hệt |
| FR-EVD-05 | PHẢI phân tích tĩnh bài multi_problem: có mặt từng hàm theo prototype trong problem card; hàm rỗng hoặc stub; còn `main`; thư viện ngoài danh sách đề cho phép; vòng lặp nghi vô hạn (heuristic); hard-code nguồn dữ liệu (đường dẫn cố định, `cin` thay vì đọc file, kích thước mảng cố định) | M | EDA §2.4, §3.1; BS §5.1 | Mẫu: 6/15 bài đủ 4 hàm; thiếu P2/P3/P4 lần lượt 4/8/6 bài; 10/15 bài còn main; 3/3 bài có nhãn "Lỗi hard-code" bật cờ hard-code |
| FR-EVD-06 | PHẢI phát hiện file có 0 dòng code hiệu dụng (toàn comment) và khối code bị comment dài | M | EDA §0.4, §5 | EX02-S214 bật cờ `all_commented`; EX02-S209 bật cờ khối comment ≥ 20 dòng |
| FR-EVD-07 | PHẢI tính cụm near-dup trong cùng exam sau khi bỏ code mẫu của đề (Jaccard/MinHash trên token chuẩn hoá). `cluster_id` phải ổn định (dùng hashlib, không dùng `hash()`) | M | EDA §5; BS §8.3 | EX02-S203..S206 cùng một cụm; chạy lại trên máy khác cho cùng `cluster_id` |
| FR-EVD-08 | Harness chạy EX01 (`-Dmain`, `generals.csv`, test từng câu): KHÔNG làm cho tới khi BTC phát dữ liệu chạy được | W | EDA §1 (không có `generals.csv`); BS §9 (thứ tự cắt); Q-16 | Mở lại khi Q-16 có trả lời |

### 4.3 POL: chính sách multi_problem

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-POL-01 | PHẢI tự động trích problem card cho mỗi exam từ EXM (regex trước, sau đó LLM structured output + self-consistency): các câu; trọng số từ **cả** statement lẫn `problems[].max_score` (bật cờ `weight_conflict` nếu lệch); đồ thị tiên quyết (`is_prerequisite_for`); điều khoản 0 điểm; hàm và thư viện bị cấm; `io_mode`. Validate: đồ thị không có chu trình, Σ trọng số hợp lệ. Thất bại thì dùng trọng số đều + cờ `policy_failed`, pipeline không dừng. Không viết tay card cho exam chỉ xuất hiện ở dev/test | M | C-05, C-16; BS §2.2, §5.1; EDA §2.4 | EX01: P1 → {P2, P3, P4}, `weight_conflict = true` (1/4/2/3 so với 2.5 × 4). EX02: cấm `pow`. Exam giả không có `grading_policy` → `policy_failed = true`, pipeline chạy hết |
| FR-POL-02 | PHẢI cài `aggregate()` tất định, có unit test, với 2 biến thể: cứng (câu tiên quyết sai thì câu phụ thuộc bằng 0) và mềm (s_q × Π P(pass_p)). Đầu ra là **feature** của T1, không ghi đè dự đoán | M | BR-07, BR-08; EDA §7.2.1 | ≥ 10 unit test (chuỗi tiên quyết, câu không làm, câu lặp vô hạn); feature có trong evidence card |
| FR-POL-03 | PHẢI xuất 2 phiên bản T1 cho multi_problem: (a) tối ưu theo gold, policy là feature; (b) đúng policy đã viết, áp cứng (`prereq_failed` lấy từ FR-POL-04). Báo QWK và MAE theo exam cho cả hai. Điểm vận hành **nộp** chọn theo FR-EVL-05 trên OOF của train chính thức, giữa 3 phương án: (a), (b), và lai (chỉ áp cổng khi P(P1 sai) ≥ τ, τ fit trên OOF). Quyết định ghi vào `runs/decisions.jsonl`; báo cáo có đoạn lập luận cho C-16. (b) luôn dùng cho model card và hồ sơ bonus | M | C-16; EDA §7.2.1; bằng chứng gold lệch policy mới n = 15 [GT] | Bảng 3 phương án × exam, kèm Δ so với (a) và CI95; `decisions.jsonl` có dòng chọn điểm vận hành |
| FR-POL-04 | PHẢI ước lượng trạng thái từng câu {đúng, một phần, sai, không làm} cho multi_problem, chỉ từ evidence và LLM đọc code (không từ feedback). Là đầu vào của `prereq_failed` cho FR-POL-03 (b) | M | EDA §7.1 D3 (không có điểm từng câu); BS §5.1; FR-POL-03 | 100% bài EX01 có trạng thái cho 4 câu; test lineage xác nhận không đọc feedback |
| FR-POL-05 | CÓ THỂ học gating mềm khả vi (τ, T) chỉ với giám sát ở mức file, để đo mức giảng viên thực sự áp chính sách | C | BS §5.1 (N1); EDA §6 #1 | Báo τ kèm CI95 bootstrap |

### 4.4 T1: chấm rubric

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-T1-01 | PHẢI xuất cho mỗi sample: 6 chiều là số nguyên trong miền, và `total_score` = Σ 6 chiều | M | C-01; BR-01, BR-02 | Validator: 0 lỗi; 100% total = Σ |
| FR-T1-02 | PHẢI mô hình hoá biến thứ bậc chính là `logic` (EX02 ∈ {0, 2, 4}) và thêm `edge_case` ở EX01; suy các chiều còn lại bằng ràng buộc theo exam (BR-03..06), chỉ khi ràng buộc đúng ≥ 99% trên train (FR-EVL-10); ràng buộc không đạt thì hạ thành feature. Vector cuối phải thuộc tập vector hợp lệ của exam (MAP có ràng buộc) | M | EDA §7.1 D1, §7.2.2; BS §4.1 | Báo % train thoả từng ràng buộc; 100% vector output hợp lệ; ablation có và không có ràng buộc |
| FR-T1-03 | PHẢI áp cổng biên dịch theo exam: `log_has_error` ⇒ vector cổng của exam, học từ train (mẫu: EX01 `000000`, EX02 `000001`). Lỗi link không kích hoạt cổng | M | BR-03, BR-04, BR-15; EDA §2.2 | Mẫu: 6/6 bài CE ra đúng vector cổng; EX02-S214 không bị cổng |
| FR-T1-04 | PHẢI có hướng prompting (P-arm): LLM local đọc đề + code + evidence card, xuất JSON theo enum. Lấy phân phối forced-choice (logprob của từng giá trị) để tính kỳ vọng; không đọc logprob dưới grammar mask | M | C-02; BS §5.1 | P-arm chạy hết pipeline; tỷ lệ parse-fail sau retry ≤ 1%; manifest ghi model revision và seed |
| FR-T1-05 | PHẢI có hướng fine-tune open-weight ≤10B chạy local cho T1, **hoặc** ghi trong báo cáo lý do dồn fine-tune sang T3 hay model nhỏ (dữ liệu ≈ 121 bài [GT]). Dạng F-arm hợp lệ: LoRA trên model nhỏ + head thứ bậc (CORN). Classifier trên embedding đóng băng chỉ là baseline. C-02 được bảo đảm bởi FR-T3-04, không bởi FR này | M | C-02, C-03; EDA §7.1 D0, §7.2.5 | Có OOF T1 của F-arm, hoặc có mục báo cáo kèm learning curve (FR-RPT-06) |
| FR-T1-06 | PHẢI có các baseline: median theo exam; luật biên dịch dựa trên `compile_log` (không dùng gold `compilable`); `rubric_prior` cho EX02 | M | EDA §2.5; BS §4.1 | Tái tạo trên mẫu: luật biên dịch QWK pooled 0.512 (EX01 0.413, EX02 0.636), MAE 2.91; median theo exam: pooled 0.219, trong từng exam bằng 0 |
| FR-T1-07 | PHẢI có decoder theo metric: de-shrink (rescale μ/σ hoặc OptimizedRounder) fit trên OOF, chọn phương án bằng nested CV. Tổng ra số nguyên trên lưới hợp lệ; không dùng banker's rounding | M | BS §4.1 | std(pred)/std(gold) trên OOF nằm trong [0.8; 1.2]; 0 total lẻ; ablation có và không có decoder |
| FR-T1-08 | NÊN dùng các nhóm feature sau. EX02: pass nới, `io_gap`, cờ prompt/khoảng trắng, crash/UB, timeout. EX01: có mặt và mức hoàn thành từng hàm, trạng thái câu, phân tích tĩnh, feature policy | S | EDA §7.2.2 | Ablation theo từng nhóm feature, báo ΔQWK + CI |
| FR-T1-09 | NÊN stacking (Ridge/LR cho exam ít mẫu, GBDT nhỏ cho exam có test) fit trên OOF theo group CV. Stack không thắng thành phần tốt nhất (vượt ngoài CI) thì loại | S | BS §5.1 | OOF của stack ≥ thành phần tốt nhất, hoặc có ghi quyết định loại |

### 4.5 T2: phân loại lỗi đa nhãn

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-T2-01 | PHẢI xuất danh sách nhãn ⊆ 10 nhãn của LS: chuỗi NFC đúng nguyên văn, không trùng, thứ tự theo LS; `[]` hợp lệ. Output LLM được map về vocab (chuẩn hoá NFC, khớp mờ có ngưỡng; ngoài vocab thì bỏ và ghi log) | M | C-01; BR-12, BR-13 | Validator 0 lỗi; input dạng NFD hoặc lệch dấu → map đúng 10/10 nhãn |
| FR-T2-02 | PHẢI dùng luật "Lỗi biên dịch" ⇔ `log_has_error`. Chỉ override cứng khi bộ ba (compilable, log, nhãn) khớp ≥ 99% trên train; nếu không, dùng làm feature mạnh | M | BR-15; EDA §3.2 | Mẫu 6/6; báo tỷ lệ khớp trên train |
| FR-T2-03 | PHẢI có P-arm: LLM local hỏi yes/no cho từng nhãn (kèm định nghĩa, phản ví dụ và câu "có thể không có lỗi nào"), prompt theo exam, đọc logprob ra xác suất | M | C-02; BS §5.2; EDA §7.2.3 | Có xác suất 10 nhãn cho 100% bài; manifest ghi seed và revision |
| FR-T2-04 | PHẢI có ≥1 bộ phân loại học được (GBDT/LR trên evidence, classifier trên embedding, hoặc LoRA ≤10B) cho ra xác suất OOF của 10 nhãn. Chiều rubric dự đoán chỉ là feature mềm, không dùng để suy nhãn trực tiếp. Chỉ tính vào C-02 khi bộ phân loại là LoRA ≤10B | M | BR-16; C-02 chỉ khi là LoRA ≤10B | File OOF đủ 10 cột; có ablation bỏ feature rubric |
| FR-T2-05 | PHẢI chọn ngưỡng riêng cho từng nhãn trên OOF (argmax F1, lấy median qua bootstrap; nhãn hiếm co về ngưỡng chung). Nhãn nào không thắng mốc hằng số 2π/(1+π) trên OOF thì dùng chiến lược hằng số | M | BS §4.2; EDA §7.2.3 | Bảng ngưỡng + F1 OOF từng nhãn so với mốc; 0 nhãn thua mốc mà vẫn dùng model |
| FR-T2-06 | PHẢI áp ràng buộc chéo BR-14, bật/tắt bằng cấu hình: total dự đoán < 10 mà tập nhãn rỗng thì thêm nhãn có xác suất cao nhất; chỉ cho `[]` khi total dự đoán = 10 | M | BR-14; EDA §7.2.3 | Khi bật: 100% output thoả ràng buộc; báo Δmacro-F1 bật so với tắt, kèm CI |
| FR-T2-07 | PHẢI có chính sách cho nhãn 0-support hoặc cực hiếm (đặc biệt "Lỗi edge case"): chỉ dự đoán khi xác suất ≥ τ_strong (cấu hình) **và** có bằng chứng thực thi (test biên fail). Chiến lược tham số hoá theo `zero_division` | M | EDA §3.3, §7.2.3; Q-03 | Mẫu: 0 dự đoán "Lỗi edge case"; báo macro-F1 ở zd ∈ {0, 1, nan} |
| FR-T2-08 | CÓ THỂ tăng dữ liệu cho nhãn hiếm bằng mutation có kiểm soát (off-by-one, sai điều kiện biên, sai chữ ký hàm). Mẫu sinh ra nằm cùng fold với bài gốc | C | BS §5.2 | Báo AUC của domain classifier (thật so với sinh); ΔF1 nhãn hiếm + CI |

### 4.6 T3: sinh phản hồi theo level

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-T3-01 | PHẢI sinh phản hồi tiếng Việt không rỗng cho mỗi sample, từ (đề, code, `compile_log`, `taxonomy_error`, `target_feedback_level`), nói với sinh viên | M | C-01; BR-24 | 100% sample có output; detector ngôn ngữ nhận tiếng Việt ≥ 99% |
| FR-T3-02 | PHẢI theo luồng Diagnose → Render. Diagnoser xuất JSON {nhãn, hàm/câu, dòng, định danh, hiện tượng, nguyên nhân, `fix_sketch` (chỉ dùng nội bộ), confidence} và qua kiểm grounding: dòng tồn tại, định danh có trong code, khẳng định khớp log | M | BS §5.3 | Grounding pass ≥ 95% trên tập đánh giá nội bộ; chạy grounding trên gold thì EX01-S109 bị gắn lệch dòng |
| FR-T3-03 | PHẢI render theo level, có che thông tin: L1/L2 không nhận `fix_sketch` hay diff; L3 nhận hướng sửa bằng lời; L4 nhận code sửa đã pass test công khai (single_problem). Khi Q-19 hoặc Q-14 không cho phép thực thi (hoặc không có `public_tests`): L4 đưa lời giải gắn `unverified`, báo tỷ lệ | M | BR-18; BS §5.3; Q-14, Q-19 | Unit test nội dung prompt của từng level; L4 của EX02 khi được thực thi: code kèm theo pass 8/8 test công khai; khi không: 100% L4 có cờ `unverified`, tỷ lệ có trong summary |
| FR-T3-04 | PHẢI có ≥1 hướng prompting (zero/few-shot) VÀ PHẢI có hướng fine-tune T3: LoRA/QLoRA trên model open-weight ≤10B, chạy local (mặc định Qwen3-8B → adapter A3). Đây là FR bảo đảm C-02 | M | C-02, C-03; EDA §7.2.5 | sha256 adapter A3 có trong manifest; số tham số trainable > 0; có bảng so sánh 2 hướng theo FR-EVL-08 |
| FR-T3-05 | PHẢI KHÔNG SFT thẳng trên gold. Trước khi huấn luyện: lọc hoặc viết lại mẫu vi phạm level (theo GRD), chuyển ghi chú người chấm thành lời nói với sinh viên, gán lại level theo nội dung thực tế. Lưu lineage từ mẫu gốc sang mẫu huấn luyện | M | EDA §4.2, §4.4, §7.2.4 | 0 mẫu SFT vi phạm guard; báo số mẫu bị lọc hoặc viết lại theo exam × level |
| FR-T3-06 | NÊN sinh dữ liệu level-swap (cùng chẩn đoán, đủ 4 level; lấp L1 cho bài có lỗi và lấp L4) bằng teacher local hoặc API đã được phép, có cache và log chi phí. CÓ THỂ dựng cặp DPO với 4 loại rejected: leak, mơ hồ, bịa, lệch level | S | EDA §7.2.4; BS §5.3 | Tỷ lệ mỗi level trong dữ liệu huấn luyện lệch ≤ ±10 điểm % so với phân bố đều; chi phí ghi vào FR-RPT-03 |
| FR-T3-07 | NÊN kiểm soát độ dài và văn phong theo (exam × level): độ dài trong median ± IQR đo trên train đã làm sạch; xưng "em"; không viết tắt kiểu ghi chú | S | EDA §4.4 | ≥ 90% output nằm trong khoảng độ dài; 0 output gọi "SV" ở ngôi thứ ba |
| FR-T3-08 | NÊN xử lý `taxonomy_error = []` bằng nhận xét tích cực hoặc gợi ý nhẹ, không bịa lỗi. Với nhãn nghi nhiễu, cho phép chẩn đoán confidence thấp, không ép bịa lỗi cho khớp nhãn | S | BS §5.3; EDA §0.4 (EX02-S214) | Bài có `[]`: 0 phản hồi nêu lỗi cụ thể (theo judge nội bộ) |

### 4.7 GRD: guard compliance cho T3

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-GRD-01 | PHẢI hậu kiểm mọi output T3 theo **toàn bộ** điều cấm BR-18 của level yêu cầu. L1: cấm vị trí cụ thể, nguyên nhân, code. L2: cấm cách sửa cụ thể, code. L3: cấm lời giải hoàn chỉnh | M | BR-18, BR-19; EDA §7.2.4 | Trên bộ kiểm guard (≥ 50 phản hồi từ train hoặc tổng hợp, 2 người gán, báo κ): recall vi phạm ≥ 0.90, precision ≥ 0.80 |
| FR-GRD-02 | PHẢI có tầng phát hiện từ vựng và cú pháp: code fence, backtick, `;`, `{}`, `#include`, `cout`/`cin`, biểu thức gán hoặc so sánh kiểu C++. Đoạn nghi ngờ được parse bằng tree-sitter để xác nhận là code | M | BS §4.3, §5.3 | ≥ 30 unit test dương và âm; 0 bỏ sót với code nguyên văn |
| FR-GRD-03 | PHẢI so code trong phản hồi với code sinh viên **theo ngữ cảnh** (cùng hàm, cùng vị trí) và với diff sửa nội bộ, không chỉ so chuỗi con trên toàn file. Việc trích nguyên văn code sinh viên ở L2 được phép hay không do cấu hình (Q-04) | M | EDA §4.5 (EX01-S106: câu lệnh còn thiếu lại có ở dòng khác trong cùng file) | Ca EX01-S106 bị gắn "đề xuất sửa" dù chuỗi đó có trong file |
| FR-GRD-04 | PHẢI phát hiện lệnh sửa viết bằng lời ở L1/L2 ("đổi … thành …", "thêm …", "bỏ …", "chỉ cần …") bằng pattern + LLM-judge local cố định | M | EDA §4.2 | Gold EX01-S104 và EX01-S110 bị gắn vi phạm L2; gold EX02-S207 bị gắn "L3 gần như lời giải" |
| FR-GRD-05 | NÊN kiểm leak bằng thực thi cho single_problem: cho LLM áp đúng nội dung phản hồi vào code sinh viên, rồi chạy test công khai trong sandbox (tắt khi không có `public_tests` hợp lệ, Q-14). Nếu từ fail chuyển thành pass toàn bộ ở L1–L3 thì bật cờ leak | S | EDA §4.2 (EX02-S207: 4/8 → 8/8), §7.2.4 | Gold EX02-S207 bị cờ leak; báo tỷ lệ leak theo level |
| FR-GRD-06 | PHẢI xử lý vi phạm tự động: sinh lại với danh sách seed cố định (N ≤ 3); hết lượt thì dùng template fallback **đúng level** (theo level × nhãn) đã qua guard | M | C-05; BS §5.3 | 100% output cuối qua guard; log tỷ lệ retry và fallback mỗi run |
| FR-GRD-07 | NÊN đo dưới mức (L2 thiếu vị trí hoặc nguyên nhân, L3 thiếu hướng sửa, L4 thiếu lời giải) và đưa vào báo cáo, không chặn output | S | EDA §4.3; Q-04 | Báo tỷ lệ dưới mức; output hằng số ("Tốt.") bị gắn dưới mức ở L2–L4 |

### 4.8 EVL: metric và đánh giá nội bộ

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-EVL-01 | PHẢI có bản sao scorer kèm unit test: QWK quadratic với `labels=list(range(11))`; MAE; exact-match từng chiều; macro/micro-F1 trên 10 nhãn cố định với zd ∈ {0, 1, nan}; compliance (dùng GRD) | M | EDA §2.5, §3.3 | Tái tạo hai ví dụ `labels` của EDA §2.5: 0.931 (`labels=None`) so với 0.738 (`labels=0..10`), và ví dụ đồ chơi 0.600 so với 0.976; oracle T2 trên mẫu = 0.900/1.000/1.000; "đoán cả 10 nhãn" = 0.214 |
| FR-EVL-02 | PHẢI báo mọi metric theo từng exam và pooled, kèm CI95 bootstrap (B = 2000, seed cố định, lấy mẫu theo cụm near-dup) | M | EDA §2.5, §7.2.6 | Mọi bảng kết quả có cột exam và CI |
| FR-EVL-03 | PHẢI dùng group CV theo cụm near-dup (theo exam nếu dev/test có đề mới, Q-07), stratify theo exam × khoảng total; `folds.csv` có version. Mọi model xuất OOF cùng định dạng `oof/{task}/{run_id}.parquet` | M | EDA §5, §7.2.6; BS §9 | 0 cụm bị chia qua 2 fold; sha của `folds.csv` ghi trong manifest |
| FR-EVL-04 | PHẢI báo kèm baseline mode cho exact-match từng chiều, và mốc hằng số cho F1 từng nhãn | M | EDA §2.5 (đoán mode của EX02 đã đúng 100% ở code_quality) | Mọi bảng exact-match có cột "mode" |
| FR-EVL-05 | PHẢI có quy tắc chọn mô hình: không chọn dựa trên chênh lệch < Δ_min(task) hoặc chênh lệch nằm trong CI; chọn bản cuối theo CV − 0.5·SE, không theo đỉnh public LB. Δ_min = `evl.select_min_delta`: T1 QWK 0.1 (EDA §7.2.6); T2 macro-F1 và T3 điểm judge chốt ở W2 (Q-33), mặc định bằng nửa độ rộng CI95 bootstrap của baseline cùng task | M | EDA §7.2.6; BS §9; Q-33 | Nhật ký quyết định ghi task, Δ, Δ_min, CI và lý do |
| FR-EVL-06 | PHẢI có judge T3 nội bộ: compliance (GRD) + chẩn đoán bằng LLM-as-judge local cố định (pin model, revision, prompt). Judge thuộc họ model khác generator T3, hoặc bắt buộc kiểm self-preference (hoán vị nguồn, báo Δ). Meta-eval trên train: đối chiếu với chấm người trên mẫu con, 2 người chấm | M | Đề §3 T3; EDA §4.4 | κ (hoặc Spearman) judge–người ≥ 0.6 và không thấp hơn κ người–người quá 0.1 [GT]. Không đạt thì judge không được dùng cho FR-EVL-05, chỉ được báo cáo |
| FR-EVL-07 | NÊN đo các hệ suy biến của T3 (hằng số, rỗng, template) **chỉ trên máy nhóm** và vẽ đường Pareto compliance × chẩn đoán | S | EDA §4.3; BS §7 #7 | Có hình Pareto trong báo cáo; tái tạo được compliance 1.00 của output hằng số |
| FR-EVL-08 | PHẢI so sánh có kiểm soát giữa 2 hướng: cùng split và folds, cùng evidence card, cùng backbone hoặc cùng họ + kích thước; dùng paired bootstrap hoặc approximate randomization, hiệu chỉnh Holm–Bonferroni | M | C-02; BS §0, §9 | Mỗi task có bảng so sánh với Δ, CI, p đã hiệu chỉnh |
| FR-EVL-09 | NÊN có ma trận bền vững theo quy ước scorer (`labels`, zd, gộp hay tách exam), và phân tích độ nhạy khi loại các nhãn nghi nhiễu (EX02-S214, EX01-S108, EX01-S110, EX01-S112). Không sửa nhãn train. Danh sách nhãn nghi nhiễu rút từ việc đọc feedback, nên kết quả chỉ để báo cáo hoặc bonus; không là input của FR-EVL-05 hay FR-T1-07 | S | EDA §2.5, §3.3; BR-21 | Mẫu: tái tạo được `rubric_prior` 0.487 → 0.556 khi loại 4 nhãn nghi nhiễu |
| FR-EVL-10 | PHẢI chạy audit trên train chính thức trước khi áp cứng bất kỳ BR [GT] nào, với 2 đầu ra tách biệt. (a) `constraints.json`: BR-01, BR-03..06, BR-14, BR-15, support từng nhãn, cổng D0, số bài in prompt hoặc thừa khoảng trắng, số `actual=''`, số bài 0 dòng code hiệu dụng, mức giao sample_id T2/T3 của train/dev; nguồn chỉ là nhãn T1/T2, `compile_log`, evidence; tỷ lệ cổng P1 (nếu dùng cho thiết kế T1) chỉ tính từ trạng thái câu FR-POL-04; lineage `touched_feedback=false`; là input của train T1/T2. (b) `audit_feedback.md`: cổng P1 theo feedback, BR-20 và level; lineage `touched_feedback=true`; chỉ cho bonus và báo cáo | M | EDA §8, §7.1 D3; BR-21 | `make audit` sinh 2 file; `constraints.json` không có khoá `p1_gate*`, `level*`, `feedback*`; ràng buộc đúng < 99% bị hạ thành feature (FR-T1-02) |

### 4.9 VAL: validator và nộp bài

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-VAL-01 | PHẢI validate offline `predictions.json` của từng task trước khi nộp: UTF-8, JSON hợp lệ, mảng các phần tử {`sample_id`, `output`}, đủ và đúng tập sample_id của split, không trùng, `output` đúng schema task (§6.3) | M | C-09 | 12 file hỏng cố ý (thiếu id, thừa id, trùng id, sai kiểu, float, nhãn NFD, total ≠ Σ, feedback rỗng…) → bắt 12/12; file đúng → pass |
| FR-VAL-02 | PHẢI chạy script kiểm định dạng chính thức (khi BTC phát) như một cổng bắt buộc. Validator nội bộ và script chính thức kết luận khác nhau thì chặn nộp | M | Đề §5 | Không file nào được nộp mà chưa qua cả hai |
| FR-VAL-03 | PHẢI có `make submit TASK=…` chỉ tạo và kiểm file, không tự upload; ghi nhật ký lượt nộp (ngày, task, run_id, sha file, lý do, điểm public). Đơn vị quota là **một lần upload** (bất kể số task trong file), theo trả lời Q-01/Q-11; nhật ký đếm theo upload; lượt vượt quota bị **từ chối** tại máy (không chỉ cảnh báo) | M | C-08; Q-01, Q-11 | Lượt thứ 3 trong cùng ngày bị từ chối ngay tại máy; 3 file task trong cùng 1 upload được tính là 1 lượt |
| FR-VAL-04 | PHẢI chuẩn hoá trước khi ghi file: số là `int` (không ghi `7.0`), chuỗi NFC, `ensure_ascii=False`, không có ký tự điều khiển, thứ tự phần tử theo input | M | BR-12; Q-01 | Đọc lại file cho đúng đối tượng gốc; 0 giá trị float |
| FR-VAL-05 | NÊN chỉ nộp khi: (i) CV cải thiện ≥ max(Δ_min(task), 1 SE), Δ_min như FR-EVL-05; (ii) cần hiệu chỉnh quan hệ CV↔LB; hoặc (iii) cần đối chiếu scorer/định dạng. Một người giữ quyền nộp | S | BS §9; EDA §7.2.6 (thay ngưỡng 0.01 của BS) | Mọi dòng nhật ký có trường lý do ∈ {i, ii, iii}; lý do (i) ghi Δ và Δ_min |
| FR-VAL-06 | PHẢI dry-run lệnh `make predict-private` trên dev (cùng lệnh, cùng cấu hình) trước buổi chạy private | M | BS §11 (rủi ro mức A) | Dry-run ở W6 và W7 qua validator; thời gian chạy ghi vào NFR-PERF-01 |

### 4.10 REP: tái lập và hạ tầng

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-REP-01 | PHẢI có một lệnh `make reproduce` sinh lại predictions tốt nhất cho cả 3 task. Mặc định chạy từ cache; `FULL=1` chạy lại từ đầu | M | C-06, C-10 | Trên máy sạch: clone + data-root + 1 lệnh → sha predictions khớp bản nhóm đã nộp |
| FR-REP-02 | PHẢI đặt seed ở mọi tầng (split, fold, exemplar, tie-break, sampling, bootstrap) từ một `Settings`; pin model revision (commit HF), checksum adapter, lockfile, digest Docker image, phiên bản vLLM | M | C-06; PY-05; BS §8.3 | Manifest liệt kê đủ các mục; đổi 1 mục pin thì manifest đổi theo |
| FR-REP-03 | PHẢI cache theo hash nội dung: LLM/API theo sha1(prompt, tham số, model revision); evidence theo (`code_sha`, `toolchain_id`). Cache-replay là nguồn sự thật cho mọi bước không tất định | M | BS §8.3; EDA §7.1 D11 | Ngắt mạng, không có GPU: replay từ cache vẫn ra predictions giống hệt |
| FR-REP-04 | PHẢI sinh `run_id` và run manifest mỗi lần chạy: config, git sha, sha của data manifest, sha của folds, model revisions, seed, phần cứng, thời gian, chi phí | M | C-06; NFR-OBS-01 | 100% artefact (OOF, predictions, log) mang run_id |
| FR-REP-05 | NÊN có smoke test môi trường: structured output, logprobs/prompt_logprobs, multi-LoRA, `bad_words` trên phiên bản vLLM đã pin | S | BS §8.3 | `make smoke` pass trên máy sạch |
| FR-REP-06 | NÊN có chế độ "chỉ dùng tín hiệu được cấp" (không tự biên dịch hay chạy code), làm dòng ablation và làm fallback tái lập | S | BS §9; Q-19 | Chạy được độc lập; báo Δmetric so với chế độ đầy đủ |

### 4.11 SEC: bảo mật, riêng tư, tuân thủ

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-SEC-01 | PHẢI kiểm feedback firewall tự động trong CI: (a) whitelist của loader (FR-ING-05); (b) assert mọi prompt và feature của T1/T2 không chứa chuỗi con ≥ 30 ký tự của bất kỳ feedback nào (trừ phần trùng với code sinh viên); (c) bảng lineage các artefact đã chạm feedback, cấm dùng làm input T1/T2 | M | C-07; BR-21; BS §11 | Test đỏ khi cố ý chèn 1 feedback vào prompt T1; bảng lineage có trong báo cáo |
| FR-SEC-02 | PHẢI chỉ chạy code sinh viên trong sandbox: container `--network none`; cgroup giới hạn CPU, bộ nhớ, số tiến trình; timeout mỗi test; rootfs chỉ đọc + tmpfs; user không phải root; không mount secret; mỗi bài một container | M | BS §8.3; EDA §10 | Các bài thử (mở socket, fork bomb, ghi ra ngoài tmp, vòng lặp vô hạn) đều bị chặn và có verdict |
| FR-SEC-03 | PHẢI giữ repo private; `.gitignore` và pre-commit chặn `data/`, `*.cpp`, JSON dữ liệu, `cache/`, adapter và weights; chạy gitleaks | M | C-11, C-14 | Commit thử file `*.cpp` hoặc JSON dữ liệu bị chặn |
| FR-SEC-04 | PHẢI kiểm soát dữ liệu ra ngoài: chỉ gửi tới endpoint trong allowlist đã được giảng viên duyệt bằng văn bản; mặc định chạy offline; mỗi call ra ngoài ghi đích, số token, chi phí | M | C-04, C-11; Q-06 | Gọi tới host ngoài allowlist → exception; nhật ký chi phí khớp hoá đơn trong ±5% |
| FR-SEC-05 | PHẢI cô lập test: input test chỉ được giải nén và đọc bên trong `make predict-private`; log hash của input và output; không công cụ nào hiển thị input test cho người; không sửa tay output, không chạy lại có chọn lọc | M | C-05, C-12 | Audit log cho thấy mọi lần chạy trên test đều được ghi, và output nộp là của đúng một lần chạy |
| FR-SEC-06 | PHẢI không dùng mã SV làm feature; PII scan chỉ đếm, không hiển thị; tài liệu, báo cáo, slide chỉ dẫn sample_id, không trích code sinh viên chưa che | M | C-13; BR-25 | Danh sách feature không chứa mã SV; linter tài liệu (n-gram ≥ 8 token so với code SV và feedback) báo 0 hit trên báo cáo và slide (TC-SEC-08) |
| FR-SEC-07 | NÊN bền vững trước prompt injection trong comment code: comment được đánh dấu là dữ liệu trong prompt; có bộ test injection tổng hợp | S | BS §6 (N11); EDA §0.10 (mẫu có 0 ca injection) | Tỷ lệ output bị đổi do injection được báo trong model card |

### 4.12 RPT: báo cáo, ablation, artefact

| ID | Phát biểu | Ưu tiên | Nguồn / lý do | Tiêu chí chấp nhận |
|---|---|---|---|---|
| FR-RPT-01 | PHẢI có bảng ablation hai chiều (build-up và leave-one-out) cho các thành phần lõi: cổng CE, ràng buộc rubric, policy làm feature so với áp cứng, decoder, ngưỡng theo nhãn, ràng buộc chéo, guard, làm sạch gold, level-swap. Fine-tune chạy 3 seed | M | C-19; Đề §8 | Mỗi thành phần lõi có ≥ 1 dòng, kèm CI |
| FR-RPT-02 | PHẢI phân tích lỗi theo lát cắt (exam, khoảng total, CE, số nhãn, level, cụm near-dup, có hay không có test), viết thẻ lỗi "kiểu bài nào, vì sao", và chạy oracle-fix | M | Đề §8; BS §9 | ≥ 5 thẻ lỗi; mỗi lát cắt có phần đóng góp vào loss |
| FR-RPT-03 | PHẢI báo phần cứng và chi phí: GPU, VRAM, GPU-giờ theo run, chi phí API (model, token, $), thời gian suy luận private | M | C-03, C-04; Đề §8 | Bảng chi phí khớp nhật ký FR-SEC-04 và manifest |
| FR-RPT-04 | PHẢI lập hồ sơ bonus. Mỗi phát hiện theo mẫu 8 bước (claim, cách vận hoá, bằng chứng n/%/CI, tác động lên metric, nguyên nhân quan sát được, đề xuất sửa, thí nghiệm chứng minh, giới hạn) và có lệnh `make audit-*`. Thứ tự ưu tiên theo EDA §6: (1) → (2) → (3) → (5). Trước khi claim nhãn nhiễu, phải loại giả thuyết "chấm theo ý định". Chỉ chạy trên train chính thức (+ dev nếu có nhãn); không chạy `run-eda.sh` hay `make audit-*` trên input test | M | C-12, C-19; EDA §6; BS §7 | ≥ 2 phát hiện đủ 8 bước, đã chạy lại trên train chính thức |
| FR-RPT-05 | PHẢI có model card cho hệ thống (2): dữ liệu huấn luyện (2 exam, ≈ 121 bài [GT]); giới hạn (nhãn nhiễu, level gắn với exam, phụ thuộc toolchain); rủi ro khi chấm thật; yêu cầu người duyệt; các khác biệt ở bảng §1.3; điều kiện duyệt sâu (BA §9.3) như đề xuất triển khai | M | Đề §8 | Model card ≤ 2 trang, đủ các mục; có bảng điều kiện duyệt sâu BA §9.3 |
| FR-RPT-06 | NÊN có learning curve (10/25/50/100% dữ liệu) so sánh P-arm với F-arm | S | EDA §7.2.5; BS §9 | Có hình learning curve kèm CI |
| FR-RPT-07 | PHẢI khai báo công cụ LLM hỗ trợ (công cụ nào, dùng cho việc gì) ở cuối báo cáo, gồm cả việc dùng AI agent trên dữ liệu (ngày, phạm vi; ví dụ EDA bộ mẫu ngày 2026-10-03 và soạn bộ tài liệu này), và có bảng đóng góp được cả nhóm xác nhận | M | C-15, C-20; Q-06 | Có mục khai báo, có dòng dùng agent trên dữ liệu; bảng đóng góp đối chiếu được với lịch sử commit |
| FR-RPT-08 | PHẢI có báo cáo 10–15 trang (phụ lục tính riêng, chờ Q-20) đủ 6 mục của đề §8(2): dữ liệu & EDA, các hướng, ablation, phân tích lỗi, chi phí & khả năng triển khai, hạn chế; slide cho 15 phút trình bày + 10 phút hỏi đáp; ≥ 2 lần diễn tập; mỗi mục được 2 người không viết mục đó review | M | C-10, C-19; Đề §8(2), §8(4) | Checklist đủ 6 heading; số trang ∈ [10; 15]; biên bản 2 lần diễn tập; bảng người review từng mục |

---

## 5. Yêu cầu phi chức năng (NFR-<CAT>-xx)

| ID | Yêu cầu | Cách đo / kiểm | Nguồn |
|---|---|---|---|
| NFR-REPRO-01 | Chạy lại `make reproduce` trên cùng máy, cùng cache phải cho `predictions.json` giống từng byte với lần trước (diff = 0) ở cả 3 task | So sha256 hai lần chạy; gate W3 | C-06; BS §9 (gate W3) |
| NFR-REPRO-02 | Chạy `FULL=1` trên máy khác cùng image digest: lệch metric trong ε (QWK ≤ 0.01, macro-F1 ≤ 0.01, compliance ≤ 0.01); argmax T1/T2 trùng ≥ 99% số bài. ε là đề xuất, chờ Q-09 | Chạy trên máy sạch ở W6 và W7 | C-06; BS §8.3 |
| NFR-REPRO-03 | Huấn luyện lại fine-tune với 3 seed, báo mean ± std; model final phải nằm trong mean ± 2 std | Bảng seed trong báo cáo | C-06; BS §9 |
| NFR-REPRO-04 | Không dùng `hash()` của Python hay thứ tự set chưa sắp xếp cho artefact cần ổn định. LLM dùng greedy hoặc sampling có seed cố định, batch-invariant nếu phiên bản hỗ trợ. Code có UB chạy với ASLR tắt (`setarch -R`) | Lint rule + test chạy 2 lần | BS §8.3; EDA §5 (UB) |
| NFR-PERF-01 | Suy luận private cho 3 task (N ≈ 36 [GT]) mất ≤ 60 phút trên 1 GPU 24 GB, không tính thời gian build image. Có sẵn cấu hình nhanh đã kiểm, ≤ 15 phút | Đo ở dry-run FR-VAL-06 | BS §11 (rủi ro A); Q-11 |
| NFR-PERF-02 | Báo throughput (bài/phút) của từng task ở cấu hình final và cấu hình nhanh. Context ≤ 8k token là đủ (p95 prompt < 2.5k) | Log thời gian theo stage | EDA §1; Đề §8 (khả năng triển khai) |
| NFR-PERF-03 | Sandbox: timeout 2 giây/test, ≤ 30 giây/bài; cache hit thì không chạy lại | Log verdict + thời gian | EDA §10; BS §8.3 |
| NFR-PERF-04 | NÊN: mọi fine-tune chạy được trên 1 GPU ≤ 24 GB (LoRA/QLoRA); ablation fine-tune chạy 1 fold, 1 seed, model nhỏ; chỉ cấu hình final chạy đủ | Manifest ghi GPU và VRAM đỉnh | C-03; BS §8.2 |
| NFR-SEC-01 | Sandbox: không mạng, ≤ 1 CPU, ≤ 512 MB RAM qua cgroup (không dùng `ulimit -v`), giới hạn ≤ 64 tiến trình, rootfs chỉ đọc, user không phải root, seccomp mặc định | Bộ test thoát sandbox (FR-SEC-02) | BS §8.3 |
| NFR-SEC-02 | Không có secret trong repo hay log. API key đọc qua `Settings` từ biến môi trường | gitleaks báo 0 phát hiện ở CI | PY-05; FR-SEC-03 |
| NFR-PRIV-01 | Lịch sử git có 0 file dữ liệu, code sinh viên, adapter hay weights; repo private | Quét lịch sử git trước mỗi mốc nộp | C-11 |
| NFR-PRIV-02 | Không đưa dữ liệu ra ngoài khi chưa có phép bằng văn bản. Phạm vi gồm: API thương mại, GPU thuê, Colab/Kaggle, HF Hub, W&B cloud, AI coding assistant đọc repo. Tracking chạy offline hoặc self-host. Việc AI agent đã xử lý bộ mẫu (EDA, soạn tài liệu) trước khi có trả lời Q-06 được khai báo theo FR-RPT-07 | Văn bản "dữ liệu được đi đâu" được duyệt ở W1; allowlist FR-SEC-04 | C-11; BS §8.4; Q-06 |
| NFR-PRIV-03 | Log mức INFO không chứa code sinh viên hay feedback; chỉ ghi sample_id và hash | Test trên log mẫu | C-11, C-13 |
| NFR-COST-01 | Trần chi phí API đặt trong `Settings` (giá trị chốt ở W1, Q-23). Vượt trần thì dừng gọi và báo lỗi. Ghi chi phí mỗi call (model, token vào/ra, $) | Tổng nhật ký ≤ trần; khớp hoá đơn ±5% | C-04 |
| NFR-COST-02 | Ngân sách GPU-giờ chốt ở W1; ghi GPU-giờ theo run | Tổng GPU-giờ trong báo cáo ≤ ngân sách | BS §8.2; Q-23 |
| NFR-MAINT-01 | Mã nguồn tuân quy ước PY-01..PY-13 (bảng §5.1) | ruff + mypy pass ở CI; checklist review | PY-xx |
| NFR-MAINT-02 | Chốt sync hay async một lần, ghi vào README trước khi init repo | Có quyết định trong README | PY-00; Q-22 |
| NFR-MAINT-03 | Có unit test cho các module lõi: scorer, `aggregate()`, validator, guard, loader, firewall. Coverage của các module này ≥ 80%; CI chạy ruff, mypy, pytest | Báo cáo coverage ở CI | C-19 (tái lập + mã nguồn 10%) |
| NFR-OBS-01 | Log có cấu trúc (JSON) gồm run_id, sample_id, task, stage, duration, cache hit/miss, cờ retry/fallback. Lỗi chỉ log một lần, tại nơi xử lý | Kiểm schema log | PY-07; FR-REP-04 |
| NFR-OBS-02 | Mỗi run sinh báo cáo tóm tắt: metric theo exam kèm CI, tỷ lệ parse-fail, vi phạm guard, fallback, cache hit, chi phí | File `runs/<run_id>/summary.md` tồn tại | FR-EVL-02 |
| NFR-USAB-01 | Mỗi luồng chạy bằng một lệnh: `make setup`, `make audit`, `make train`, `make predict SPLIT=…`, `make reproduce`, `make submit TASK=…`. Có README cài đặt. Thành viên khác chạy được baseline trên máy mình | Gate W1 (BS §9) | C-10 |
| NFR-USAB-02 | Thông báo lỗi nêu điều đã thấy (file, sample_id, trường, status, host), không đoán nguyên nhân | Review thông điệp lỗi | PY-08 |

### 5.1 Áp quy ước Python vào hệ thống

| Mã | Áp dụng cụ thể |
|---|---|
| PY-00 | Sync hay async: chưa có mặc định, phải hỏi và chốt (Q-22) |
| PY-01 | Tên, log, exception, comment viết tiếng Anh. Tiếng Việt chỉ có ở phản hồi T3, template fallback và chuỗi nhãn LS (dữ liệu) |
| PY-02 | Tên grep được, ví dụ `ProblemCard.from_exam(...)`, `build_evidence_card(...)`; không có hàm cấp module tên mơ hồ |
| PY-03, PY-04 | `core/exceptions.py` (`AppError` → `DataSchemaError`, `SandboxError`, `UpstreamError`, `GuardViolation`…); decorator `@handle_errors` ở biên CLI |
| PY-05, PY-06 | Một `Settings` (pydantic-settings) ở `core/config.py`: seed, data-root, ε, trần chi phí, allowlist, τ_strong, timeout. Hằng nghiệp vụ (10 nhãn, 4 level, miền rubric) đặt ở `constants.py`, nạp từ LS và đối chiếu khi khởi động |
| PY-07, PY-08 | Log một lần bằng `log.exception`; giữ status và body gốc của API (`raise … from e`) |
| PY-09 | `shared/unicode.py` (NFC), `shared/jsonio.py`, `shared/hashing.py` (sha1 ổn định); không có `utils.py` thập cẩm |
| PY-10 | Không load model, không gọi mạng hay đọc dữ liệu lúc import |
| PY-11 | Comment chỉ giải thích "vì sao", 1–3 dòng; quyết định dài viết vào docs/ADR |
| PY-12 | Bộ mẫu 32 bài chỉ dùng làm fixture trong `tests/`; không có đường fallback về dữ liệu mẫu khi chạy thật |
| PY-13 | Type hint cho hàm public; ruff + mypy cấu hình trong `pyproject.toml`, chạy ở CI |

---

## 6. Yêu cầu dữ liệu

### 6.1 Schema input quan sát được (bộ mẫu)

| Trường | T1 | T2 | T3 | Kiểu | Null / rỗng | Ghi chú |
|---|---|---|---|---|---|---|
| `sample_id` | ✓ | ✓ | ✓ | str, dạng `EXnn-Snnn` | Không | Khoá join; duy nhất; 3 file có cùng 32 id, cùng thứ tự |
| `exam_id` | ✓ | ✓ | ✓ | str | Không | Khoá tới EXM |
| `exam_type` | ✓ | ✓ | ✓ | enum {`multi_problem`, `single_problem`} | Không | |
| `language` | ✓ | ✓ | ✓ | str (`cpp11`) | Không | |
| `code_file` | ✓ | ✓ | ✓ | str, đường dẫn tương đối | Không | `submissions/<exam>/<exam>-<SV>.cpp` |
| `compile_log` | ✓ | ✓ | ✓ | str | `""` khi build không có diagnostic | GNU g++ do BTC sinh lại; EX01 dùng `-fsyntax-only`; có log bị cắt ở 2000 ký tự |
| `test_report` | ✓ | ✓ | — | `null` (multi) hoặc list[obj] (single); `[]` khi single không build được | Có thể null hoặc rỗng | obj = {`input`: str, `expected`: str, `actual`: str, `passed`: bool}; đã strip khoảng trắng; không có returncode/stderr; `actual=''` có thể là crash |
| `taxonomy_error` | — | — | ✓ | list[str] ⊆ LS | Có thể `[]` | Trùng gold T2 ở 32/32 bài (BR-22) |
| `target_feedback_level` | — | — | ✓ | enum 4 chuỗi "Level k - <tên>" | Không | Mẫu không có L4 (BR-20) |

**Output gold (trường `output` trong file task):**

| Task | Schema | Ràng buộc |
|---|---|---|
| T1 | `{"rubric": {compilable, io_format, logic, edge_case, complexity, code_quality: int}, "total_score": int}` | BR-01, BR-02 |
| T2 | `{"taxonomy_error": list[str]}` | BR-12, BR-13 |
| T3 | `{"feedback": str}` | BR-18, BR-24 |

**File metadata:**
- **EXM** (`{"exams": [...]}`): `exam_id`, `exam_type`, `language`, `course`, `week`, `topic`, `learning_outcome`, `bloom_level`, `programming_skill`, `problem_type`, `difficulty_teacher`, `estimated_time_min`, `statement` (markdown đề đầy đủ), `problems[]`, `grading_policy`, `has_hidden_testcases`, `note`.
  - `problems[]` của multi: {`pid`, `max_score`, `prototype`, `is_prerequisite_for`}.
  - `problems[]` của single: {`pid`, `max_score`, `io_mode`, `constraints`}.
- **LS:** các chiều rubric và miền; 10 chuỗi `taxonomy_error`; 4 `feedback_level`, mỗi level có định nghĩa và điều cấm (BR-18).
- **Mỗi file task:** {`task`, `description`, `input_fields`, `output_fields`, `metric`, `n_samples`, `samples`}; T2 có thêm `label_space`, T3 có thêm `level_definition_ref`.

**Sự thật dữ liệu ảnh hưởng tới yêu cầu (mẫu, EDA §1, §2, §5):**
- Bộ mẫu có 32 bài: EX01 15 bài (multi), EX02 17 bài (single).
- EX01: bài nộp không có `main`, `test_report = null`, không kèm `generals.csv`, nên không chạy được (FR-EVD-08).
- EX02: có 8 test công khai (6 True, 2 False), `has_hidden_testcases = true`.
- Trường `compiles_locally` mà README nhắc tới **không có** trong dữ liệu (Q-17).
- Token: prompt T1/T2 có p95 < 2k, max 2309 (EDA §1).

### 6.2 Artefact nội bộ

| Artefact | Khoá | Nội dung chính | Ràng buộc |
|---|---|---|---|
| Evidence card | (`code_sha`, `toolchain_id`) | Như FR-EVD-01 | Không chứa feedback; JSON Schema có version |
| Problem card | `exam_id` + sha của mục EXM | Như FR-POL-01 | Sinh tự động; cờ `weight_conflict`, `policy_failed` |
| `folds.csv` | `sample_id` | fold, `cluster_id`, exam, khoảng total | Có version; sha ghi trong manifest |
| OOF | (task, run_id, sample_id) | Xác suất hoặc điểm theo từng lớp/nhãn | Định dạng chung `oof/{task}/{run_id}.parquet` |
| Cache LLM/API | sha1(prompt, tham số, model revision) | Response, logprobs, token, chi phí | Chỉ thêm, không sửa; ngoài git |
| Run manifest | run_id | Như FR-REP-04 | Bắt buộc cho mọi run |
| Bảng lineage | id artefact | Đã chạm feedback? Dùng cho task nào? | FR-SEC-01: artefact đã chạm feedback bị cấm cho T1/T2 |
| `constraints.json` / `audit_feedback.md` | run_id của audit | Ràng buộc từ nhãn T1/T2 + evidence / chỉ số rút từ feedback và level | FR-EVL-10: chỉ `constraints.json` (`touched_feedback=false`) vào train T1/T2 |
| Nhật ký nộp | (ngày, upload) | task, run_id, sha file, lý do, điểm public | FR-VAL-03, FR-VAL-05 |

### 6.3 Định dạng `predictions.json` đề xuất

- Đề chỉ nói "mỗi phần tử gồm `sample_id` và trường `output` tương ứng với task". Bộ mẫu không mô tả định dạng (EDA §1). → **Q-01**.
- **Đề xuất mặc định:** `output` lặp lại đúng trường `output` của file task, vì README ghi 3 file "chỉ khác trường output". Mỗi task một file, top-level là mảng.

```json
[{"sample_id": "EX02-S2xx", "output": {"rubric": {"compilable": 1, "io_format": 1, "logic": 2, "edge_case": 1, "complexity": 1, "code_quality": 1}, "total_score": 7}}]
[{"sample_id": "EX02-S2xx", "output": {"taxonomy_error": ["Lỗi logic"]}}]
[{"sample_id": "EX02-S2xx", "output": {"feedback": "<phản hồi tiếng Việt>"}}]
```

| Quy tắc | Chi tiết | Yêu cầu |
|---|---|---|
| Phạm vi id | Đủ sample_id của split, mỗi id đúng 1 lần, thứ tự như input | FR-VAL-01 |
| Kiểu số | `int` của Python; `total_score` = Σ 6 chiều | FR-VAL-04, BR-01 |
| Nhãn | NFC, đúng chuỗi LS, thứ tự LS, không trùng; `[]` hợp lệ | BR-12, BR-13 |
| Mã hoá | UTF-8 không BOM, `ensure_ascii=False` | FR-VAL-04 |
| Biến thể | Validator cấu hình được: output phẳng hay lồng, mảng hay object bọc ngoài, 1 file hay 3 file, để đổi nhanh khi Q-01 có trả lời | FR-VAL-01 |

---

## 7. Ma trận truy vết: điều khoản đề → C / BR / FR / NFR

| Điều khoản đề | C | BR | FR / NFR |
|---|---|---|---|
| §1 Bối cảnh: phản hồi để sinh viên học; hai giảng viên có thể chấm lệch nhau | — | BR-08, BR-24 | FR-T3-07, FR-RPT-05, FR-EVL-09 |
| §2 Dữ liệu: rubric 6 chiều, 10 nhãn, Level 1–4, đường dẫn code | C-18 | BR-01, BR-02, BR-12, BR-18 | FR-ING-01..06 |
| §2 Hai loại đề, tín hiệu phụ `compile_log`/`test_report`, trọng số và tiên quyết | — | BR-03..07, BR-11 | FR-EVD-01..08, FR-POL-01 |
| §3 T1: output và metric (QWK, MAE, exact-match) | C-01 | BR-01..06, BR-10 | FR-T1-01..09, FR-EVL-01, FR-EVL-02, FR-EVL-04 |
| §3 T1: tôn trọng chính sách đề multi_problem | C-16 | BR-07, BR-08, BR-09 | FR-POL-01..05 |
| §3 T2: đa nhãn, rỗng hợp lệ, macro-F1 | C-01 | BR-12..17 | FR-T2-01..08, FR-EVL-01 |
| §3 T3: level compliance, chẩn đoán, tính sư phạm | C-01, C-17 | BR-18..20, BR-22..24 | FR-T3-01..08, FR-GRD-01..07, FR-EVL-06, FR-EVL-07 |
| §4 ≥2 hướng: prompting + open-weight fine-tune | C-02 | — | FR-T3-04 (fine-tune bắt buộc), FR-T1-04, FR-T2-03, FR-EVL-08, FR-RPT-06; FR-T1-05, FR-T2-04 khi là LoRA |
| §4 ≤10B, chạy cục bộ, báo phần cứng | C-03 | — | FR-RPT-03, NFR-PERF-04 |
| §4 API thương mại, báo chi phí | C-04 | — | FR-SEC-04, FR-RPT-03, NFR-COST-01 |
| §4 Tự động 100% lúc suy luận | C-05 | — | FR-POL-01, FR-GRD-06, FR-SEC-05 |
| §4 Seed, phiên bản model, tái lập | C-06 | — | FR-REP-01..06, NFR-REPRO-01..04 |
| §4 Cấm feedback làm input T1/T2 | C-07 | BR-21, BR-22 | FR-ING-05, FR-SEC-01, FR-T3-05 (lineage) |
| §5 Public/private LB, 2 lượt/ngày | C-08 | — | FR-VAL-03, FR-VAL-05, FR-EVL-05 |
| §5 `predictions.json`, script kiểm định dạng, file sai vẫn trừ lượt | C-09 | BR-12 | FR-VAL-01, FR-VAL-02, FR-VAL-04, FR-VAL-06 |
| §6 Trọng số chấm (phương pháp, ablation, báo cáo, tái lập) | C-19 | — | FR-RPT-01, FR-RPT-02, FR-EVL-01..10, NFR-MAINT-03 |
| §6 Bonus: vấn đề thật của dữ liệu/metric | C-19 | BR-08, BR-10, BR-14, BR-19 | FR-RPT-04, FR-POL-03, FR-POL-05, FR-EVL-07, FR-EVL-09, FR-EVL-10 |
| §6 Bảng đóng góp | C-20 | — | FR-RPT-07 |
| §8 Repo, hướng dẫn cài, tái lập 1 lệnh | C-10 | — | FR-REP-01, NFR-USAB-01 |
| §8 Báo cáo 10–15 trang (6 mục), model card, slide 15 + 10 phút | C-10 | — | FR-RPT-01..08 |
| §8 predictions.json cho test riêng tư | C-09, C-10 | — | FR-VAL-06, FR-SEC-05, NFR-PERF-01 |
| §9 Không phát tán, không repo công khai | C-11 | — | FR-SEC-03, FR-SEC-04, NFR-PRIV-01..03 |
| §9 Không gán nhãn thủ công test | C-12 | — | FR-SEC-05 |
| §9 Không truy ngược danh tính | C-13 | BR-25 | FR-SEC-06 |
| §9 Không chia sẻ mã hay dự đoán giữa các nhóm | C-14 | — | FR-SEC-03 |
| §9 Khai báo công cụ LLM hỗ trợ | C-15 | — | FR-RPT-07 |

**Truy vết ngược:**
- Các FR không gắn trực tiếp với một điều khoản đề đều xuất phát từ phát hiện dữ liệu của EDA (FR-EVD-02..07, FR-T2-07, FR-GRD-03..05, FR-EVL-03, FR-EVL-10). Chúng phục vụ C-19 (phương pháp, bonus).
- Cột "Nguồn / lý do" ở §4 ghi mục EDA cụ thể của từng FR.

---

## 8. Câu hỏi mở (Q-xx), xếp theo tác động

- Gộp và khử trùng từ BS §3, EDA §9, câu hỏi phát sinh ở tài liệu scenario §5 và test §6.
- Q-01..Q-21, Q-24..Q-28 gửi giảng viên. Q-22, Q-23, Q-29..Q-33 do nhóm tự chốt.
- Cột "Mặc định" là cách hệ thống chạy khi chưa có trả lời.

| ID | Câu hỏi | Tác động | Ảnh hưởng | Nguồn | Mặc định khi chưa có trả lời |
|---|---|---|---|---|---|
| Q-01 | Định dạng `predictions.json` của từng task: top-level là mảng hay object; `output` lồng như file task hay phẳng; total là int; một file chung cho 3 task hay 3 file? | Rất cao: sai định dạng là mất lượt; private chỉ có 1 cơ hội | §6.3, FR-VAL-01..04 | Đề §5; EDA §9.1; BS §3.7 | Lặp lại `output` của file task; 1 file/task; top-level là mảng |
| Q-02 | Scorer T1: QWK có cố định `labels=0..10` không? Tính gộp hay theo exam? Làm tròn thế nào? Tính trên `total_score` hay tự cộng 6 chiều? Có phát scorer không? | Rất cao: theo EDA §2.5, chỉ đổi `labels` đã làm QWK chênh 0.738 so với 0.931 | FR-EVL-01, FR-T1-07 | EDA §9.2; BS §3.1 | `labels=0..10`; báo cả gộp lẫn theo exam; total = Σ |
| Q-03 | Scorer T2: dùng `zero_division` nào? Tính trên 10 nhãn cố định hay chỉ nhãn có support? Có chuẩn hoá Unicode không? | Rất cao: oracle được 0.90 hay 1.00 tuỳ zd; đổi chiến lược cho nhãn hiếm | FR-T2-05, FR-T2-07, FR-EVL-01 | EDA §9.2; BS §3.1 | Cấu hình bền với cả 3 quy ước zd; không đoán nhãn 0-support |
| Q-04 | Metric T3: detector compliance là regex, classifier hay LLM? Có phạt dưới mức không? Trích code sinh viên ở L2 có tính là "đưa code" không? Điều cấm của L1/L3 có được kiểm không? Judge (b) dùng gold làm reference cứng hay chỉ để gợi ý? Gold không có chẩn đoán thì xử lý sao? (a) và (b) gộp thế nào để xếp hạng? | Cao: quyết định guard chặt tới đâu, và thiết kế template | FR-GRD-01, FR-GRD-03, FR-GRD-07, FR-EVL-06 | EDA §9.7, §9.8; BS §3.3 | Áp toàn bộ điều cấm của LS; L2 không đưa code mới; tự đo dưới mức |
| Q-05 | Phạm vi cấm feedback: level, exemplar/RAG, target phụ, rationale distill có bị tính là input T1/T2 không? Định nghĩa nhãn hoặc prompt T1/T2 do người đã đọc feedback train soạn có bị tính là input không? Test T2 và test T3 có dùng chung sample_id không? | Cao: rủi ro kỷ luật | BR-21, BR-22, FR-SEC-01, FR-EVL-09 | BS §3.2; EDA §9.9 | Hiểu rộng nhất (BR-21); `label_defs.yaml` và prompt T1/T2 soạn chỉ từ LS và code, ghi lineage tác giả; không khai thác phần giao nhau |
| Q-06 | Dữ liệu được phép đi đâu: API thương mại (lúc train, lúc suy luận), GPU thuê, Colab/Kaggle, HF Hub private, W&B cloud, AI coding assistant đọc repo? Bộ mẫu đã được xử lý bằng AI agent khi làm EDA (2026-10-03) và soạn bộ tài liệu này: xin xác nhận được phép, hoặc hướng xử lý; tập train/test chính thức có áp cùng quy tắc không? | Cao: rủi ro mức S (huỷ kết quả) | FR-SEC-04, NFR-PRIV-02, FR-T3-06, FR-RPT-07 | BS §3.4, §8.4; EDA (phần "Cách làm") | Mọi thứ chạy local; teacher local; không gửi code ra ngoài; khai báo việc dùng agent trên bộ mẫu (FR-RPT-07, A-16 của BA) |
| Q-07 | Kích thước tập đầy đủ (≈ 121 bài [GT]) và cách chia train/dev/test? Dev/test có đề mới không? Dev có nhãn không? Có được đọc hoặc gán tay output trên dev không? Có được dùng input dev/test (không nhãn) cho self-training không? | Cao: quyết định sơ đồ CV, kNN và có fine-tune T1/T2 hay không | FR-EVL-03, FR-T1-05 | EDA §9.12; BS §3.6 (câu 6), §10 (D0, D2) | Group theo near-dup; nếu có đề mới thì group theo exam; không đọc hay gán tay output dev; không self-training trên test |
| Q-08 | EX01: trọng số đúng là 1/4/2/3 hay 2.5 × 4? Rubric 6 chiều được suy từ kết quả từng câu thế nào? Vì sao EX01-S108/S110/S112 có điểm dù P1 sai? Có điểm từng câu không? | Cao: thiết kế policy engine; phát hiện bonus #1 | BR-07, BR-08, FR-POL-* | EDA §9.3; BS §3.5 | Policy chỉ là feature; báo cả 2 phiên bản (FR-POL-03) |
| Q-09 | Tái lập: giảng viên có chạy lại không, trên phần cứng nào, dung sai ε bao nhiêu? Nộp weights và cache ở đâu khi repo phải private? | Cao: tái lập là cổng của 30% điểm thứ hạng | FR-REP-*, NFR-REPRO-02 | BS §3.8 | ε = 0.01; cache và adapter gửi qua kênh nội bộ đã được duyệt |
| Q-10 | "Chạy cục bộ" có tính VM thuê không? "≤10B" là tổng tham số hay tham số active? Yêu cầu ≥2 hướng áp cho từng task hay cho toàn hệ thống? | Trung bình–cao | C-02, C-03, FR-T1-05, FR-T3-04 | BS §3.9, §3.10 | Tổng tham số ≤10B; mỗi task đều có ≥2 hướng (phương án an toàn) |
| Q-11 | Quota 2 lượt/ngày tính chung hay theo task? Đơn vị đếm lượt là một upload hay một file task? Ranh giới ngày theo múi giờ nào? Bản nộp nào được tính cho private? Khi nào phát input private test, cửa sổ chạy dài bao lâu? | Trung bình | FR-VAL-03, NFR-PERF-01 | BS §3.7; scenario §5 #1 | 2 lượt chung cho cả 3 task; đơn vị là một upload; múi giờ `Asia/Ho_Chi_Minh`, cảnh báo gần ranh giới; private chạy xong ≤ 60 phút |
| Q-12 | io_format theo chuẩn chính thức nào: in prompt có bị trừ không, thừa khoảng trắng có bị trừ không? | Trung bình: ảnh hưởng feature T1; phát hiện bonus #2 | BR-10, FR-EVD-04 | EDA §9.4 | Học ánh xạ từ gold, không áp luật cứng |
| Q-13 | Level ở tập test gán thế nào? Có L4 không? Có L1 cho bài có lỗi không? | Trung bình | BR-20, FR-T3-06 | EDA §9.6 | Hỗ trợ đủ 4 level cho mọi exam |
| Q-14 | T3 có được dùng `test_report` (ghép qua sample_id) không, kể cả bộ test công khai gom từ T1 để chạy lại cho T3/GRD (repair, kiểm leak)? | Trung bình | BR-23, FR-T3-02, FR-T3-03, FR-GRD-05 | EDA §9.9, §7.2.4 | Không ghép; `public_tests` lấy từ statement/EXM nếu đề có liệt kê, không có thì tắt repair và kiểm leak bằng thực thi |
| Q-15 | EX02-S214 (file bị comment toàn bộ, 10/10) và các nhãn nghi nhiễu: chấm theo ý định hay lỗi nhãn? Có loại khỏi tập đánh giá không? | Trung bình | FR-EVL-09, FR-RPT-04 | EDA §9.5 | Giữ nguyên nhãn; báo phân tích độ nhạy |
| Q-16 | Có bổ sung `generals.csv` và test cho EX01 không? Có kết quả test ẩn của EX02 không? `test_report` có thể thêm returncode và stderr không? | Trung bình | FR-EVD-04, FR-EVD-08 | EDA §9.10 | EX01 chỉ phân tích tĩnh + LLM đọc code |
| Q-17 | Toolchain: tập đầy đủ có `compiles_locally` không, định nghĩa bằng lệnh nào? `compile_log` có luôn sinh bằng GCC 13 không, có luôn cắt ở 2000 ký tự không? | Thấp–trung bình | FR-ING-06, FR-EVD-04 | EDA §9.11 | Dùng log gốc; rebuild bằng gcc 13 chỉ làm feature |
| Q-18 | Có định nghĩa chính thức cho 10 nhãn không (ví dụ "Lỗi hard-code", "Lỗi edge case")? | Thấp–trung bình | BR-17, FR-T2-03 | BS §3.11 | Định nghĩa vận hành rút từ train |
| Q-19 | Có được dùng dữ liệu hay model bên ngoài không? Có được tự biên dịch, chạy code sinh viên và sinh test lúc suy luận không? | Trung bình | FR-EVD-04, FR-GRD-05, FR-REP-06 | BS §3.12 | Được chạy trong sandbox; luôn giữ chế độ chỉ dùng tín hiệu được cấp |
| Q-20 | Báo cáo: mục 7 bị thiếu trong đề là gì? Thứ hạng quy ra 30% điểm thế nào? Giới hạn 10–15 trang có tính phụ lục không? | Thấp | FR-RPT-* | BS §3.13; Đề (không có §7) | Phụ lục nằm ngoài 15 trang |
| Q-21 | Bonus: tiêu chí của "vấn đề thật" là gì? Nhiều nhóm cùng báo một lỗi thì tính thế nào? Nếu scorer hoặc dữ liệu được sửa giữa kỳ thì leaderboard có chấm lại không? | Thấp | FR-RPT-04 | BS §3.14 | Gửi hồ sơ kèm timestamp khoảng W5 |
| Q-22 | (Nhóm) Codebase viết sync hay async? | Thấp, nhưng phải chốt trước khi init repo | NFR-MAINT-02 | PY-00 | Không có mặc định; nhóm phải quyết |
| Q-23 | (Nhóm) Trần chi phí API, ngân sách GPU-giờ, và ε tái lập khác máy là bao nhiêu? | Trung bình | NFR-COST-01, NFR-COST-02, NFR-REPRO-02 | BS §8.2, §8.3 | Chốt ở W1; trước đó không gọi API trả phí |
| Q-24 | Nếu lượt chạy private crash giữa chừng, nhóm được chạy lại toàn bộ bao nhiêu lần, và lần nào là lần hợp lệ? | Trung bình–cao | FR-SEC-05, FR-REP-03 | Scenario §5 #2 (SC-21) | Chạy lại toàn bộ bằng cùng lệnh; nộp output của lần hoàn chỉnh đầu tiên; audit log ghi mọi lần |
| Q-25 | Máy giảng viên dùng để tái lập có Docker và GPU không? Có chấp nhận reproduce chỉ từ cache không? | Cao | FR-REP-01, FR-REP-03, NFR-REPRO-02 | Scenario §5 #3 (SC-02, SC-32) | Mặc định replay từ cache, không cần GPU hay Docker; `FULL=1` cần cả hai |
| Q-26 | Với exam mới ở test, BTC có công bố quy ước cổng biên dịch (vector khi CE) không? | Trung bình | FR-T1-03, FR-T1-02 | Scenario §5 #4 (SC-19, SC-59) | Vector cổng mặc định theo `exam_type`, ghi `gate_source` |
| Q-27 | Khi `taxonomy_error = []` mà level yêu cầu là L2–L4, phản hồi kỳ vọng là gì? | Trung bình | FR-T3-08, FR-GRD-07 | Scenario §5 #5 (SC-22, SC-69) | Nhận xét tích cực hoặc gợi ý nhẹ, không bịa lỗi; template `no_error` theo level |
| Q-28 | Có được tạo bộ prompt injection tổng hợp từ code train để báo trong model card không? | Thấp | FR-SEC-07 | Scenario §5 #6 (SC-05) | Chỉ dùng chương trình do nhóm tự viết, không dẫn xuất từ code train |
| Q-29 | (Nhóm) CI-cloud có được chạy trên runner của host repo khi chỉ có code + dữ liệu tổng hợp không? | Thấp | NFR-PRIV-02, FR-SEC-03 | Test §6 #9 | Có, không có dữ liệu thật; mọi job dùng dữ liệu chạy ở CI-local |
| Q-30 | (Nhóm) Pre-commit chặn `*.cpp`: có chấp nhận lưu chương trình C++ tổng hợp dạng chuỗi trong generator Python không? | Thấp | FR-SEC-03 | Test §6 #10 | Có; không có ngoại lệ đường dẫn cho `*.cpp` |
| Q-31 | (Nhóm) Giá trị `t2.tau_strong`, support tối thiểu k để không co ngưỡng, trần stdout sandbox? | Thấp | FR-T2-05, FR-T2-07, FR-SEC-02 | Test §6 #11 | `tau_strong = 0.90`; k chốt từ OOF; `stdout_cap_bytes = 65536` (theo `Settings` ở đặc tả kỹ thuật §5.1) |
| Q-32 | (Nhóm) Ai gán ≥ 50 phản hồi cho bộ kiểm guard, lấy từ đâu? | Trung bình | FR-GRD-01 | Test §6 #12 | 2 thành viên không viết guard; lấy từ train hoặc tổng hợp, không dùng test; xong trước W5 |
| Q-33 | (Nhóm) Δ_min chọn mô hình cho macro-F1 (T2) và điểm judge (T3) là bao nhiêu? | Trung bình | FR-EVL-05, FR-VAL-05 | EDA §7.2.6 (chỉ có ngưỡng cho QWK) | Chốt ở W2; trước đó bằng nửa độ rộng CI95 bootstrap của baseline cùng task |
