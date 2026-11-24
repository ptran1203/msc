# Mục lục bộ tài liệu: hệ thống LLM chấm rubric, phân loại lỗi, sinh phản hồi bài C++

| Mục | Giá trị |
|---|---|
| Phiên bản | 0.1, bản nháp để nhóm review |
| Ngày | 2026-10-04 |
| Căn cứ chung | Đề đồ án (§1–§9, không có §7); bộ mẫu 32 bài (EX01 15, EX02 17); [báo cáo EDA](../plans/reports/eda-261003-2335-sample-dataset.md) (ưu tiên khi mâu thuẫn, nhất là §7); [brainstorm](../../plans/reports/brainstorm-261003-2159-llm-grading-challenge.md) |
| ID | Tài liệu này không định nghĩa ID; chỉ đếm và trỏ |

**Hai "hệ thống" xuyên suốt bộ tài liệu:**
- **(1) Hệ thống challenge** (ký hiệu [CH] ở BA, HT1 ở scenario): pipeline dự đoán T1/T2/T3, đánh giá nội bộ, nộp leaderboard, tái lập. Là đối tượng của SRS, đặc tả kỹ thuật và test.
- **(2) Bối cảnh thật: trợ lý chấm bài cho giảng viên** ([NV], HT2): chỉ ở mức thiết kế khái niệm, dùng cho model card (FR-RPT-05) và mục "khả năng triển khai" của báo cáo.

---

## 1. Năm tài liệu

| # | Tài liệu | Nội dung | ID định nghĩa | Dòng |
|---|---|---|---|---|
| 01 | [Phân tích nghiệp vụ](01-business-analysis.md) | Bối cảnh, pain point có bằng chứng từ mẫu, stakeholder, mục tiêu và KPI cho cả (1) và (2), use case, user story, vấn đề dữ liệu ở góc nghiệp vụ, rủi ro, giả định | STK, G, KPI, UC, US, RSK, A | 628 |
| 02 | [SRS](02-requirements-srs.md) | Ràng buộc từ đề, quy tắc nghiệp vụ (kèm độ chắc chắn [GT]), yêu cầu chức năng theo 12 module, NFR, schema dữ liệu, truy vết điều khoản đề → yêu cầu, câu hỏi mở kèm mặc định | C, BR, FR, NFR, Q | 530 |
| 03 | [Đặc tả kỹ thuật](03-technical-specification.md) | Kiến trúc, hợp đồng dữ liệu pydantic, đặc tả từng module, make target, `Settings` mặc định, model đề xuất và cổng D0, xử lý lỗi, truy vết FR → symbol | Không (chỉ trỏ SRS) | 730 |
| 04 | [Kịch bản](04-scenarios.md) | 70 kịch bản biên trên 14 chiều (12 chiều chuẩn + metric gaming + sư phạm/công bằng), severity, kết hợp chiều nguy hiểm | SC | 341 |
| 05 | [Test case](05-test-cases.md) | Chiến lược, môi trường, dữ liệu test, 131 test case theo module, ma trận FR/NFR/BR/SC → TC, smoke trước nộp, gate trước private | TC | 654 |
| 06 | [Hợp đồng dữ liệu, CLI, API](06-api-contract.md) | Bốn lớp hợp đồng (file với BTC, CLI/make, interface nội bộ, REST tuỳ chọn), nguồn test của evidence card theo task, versioning, bảng exception ↔ exit code ↔ HTTP, luật validator predictions, test hợp đồng đề xuất, lệch so với đặc tả kỹ thuật | Không (chỉ trỏ; câu hỏi mới ghi "đề xuất Q mới") | 537 |
| 07 | [Thư viện và phụ thuộc](07-libraries.md) | Quyết định phụ thuộc (Python 3.12, pin vllm/torch/transformers), inventory theo nhóm core/infer/train/api/teacher/dev/experimental, công cụ hệ thống, model weights, tương thích, khoá phiên bản, bảo mật, license, ánh xạ thư viện ↔ FR/TC | Không ("QĐ n" chỉ là số thứ tự dòng) | 315 |

**Thư mục `contracts/` và `deps/`** (sản phẩm kèm của 06 và 07):

| Thư mục | File | Ghi chú |
|---|---|---|
| `contracts/` | `openapi.yaml` | Lớp REST (iv) tuỳ chọn, OpenAPI 3.1 |
| `contracts/dataset/` | `task1-grading`, `task2-error-taxonomy`, `task3-feedback`, `exams`, `label-space` (`.schema.json`) | Schema quan sát trên bộ mẫu |
| `contracts/predictions/` | `predictions-t1`, `predictions-t2`, `predictions-t3` (`.schema.json`) | Định dạng nộp đề xuất, chờ Q-01 |
| `contracts/internal/` | `blind-sample`, `cond-sample`, `t1-output`, `t2-output`, `t3-output`, `evidence-card`, `problem-card`, `run-manifest`, `run-result` (`.schema.json`) | Hợp đồng nội bộ giữa module, manifest và dòng stdout của CLI |
| `contracts/examples/` | `predictions-t{1,2,3}.json`, `evidence-card-ex0{1,2}.json`, `problem-card-ex01.json`, `api-{grade,classify,feedback,evidence,assess}-{request,response}.json`, `api-info-response.json`, `api-healthz-response.json`, `api-problem-422.json`, `run-result-quota.json` | Dữ liệu tổng hợp (dải `S9xx`), không có code sinh viên |
| `deps/` | `pyproject.draft.toml` | Bản nháp `pyproject.toml`; `uv lock` thử đạt 255 gói (07, mục "Kết quả kiểm thật") |
| `../validate_contracts.py` | script kiểm hợp đồng | Validate bộ dữ liệu, ví dụ, ca âm, OpenAPI; kết quả ở 06, mục "Kết quả kiểm thật" |

---

## 2. Đọc theo vai

| Vai | Thứ tự đọc | Mục tiêu khi đọc |
|---|---|---|
| Giảng viên / BTC | SRS §1.2 (phạm vi 2 hệ thống) → SRS §2 (C-01..C-21) và §7 (điều khoản đề → yêu cầu) → SRS §8: Q-01..Q-21, Q-24..Q-28 (cần trả lời) → BA §8 (vấn đề dữ liệu/metric, ứng viên bonus) → BA §10.2 A-16 → Test §1.7 (acceptance theo C) | Xác nhận cách hiểu đề, trả lời câu hỏi mở, thấy cơ chế tuân thủ (feedback firewall, cô lập test, tái lập) |
| Trưởng nhóm | BA §3 (G, KPI) → BA §9.2 (nguồn lực), §10.1 (RSK), §11 (quyết định cần chốt) → SRS §4 bảng "Hai hướng tiếp cận", FR-POL-03, FR-EVL-05, FR-VAL-03/05 → Scenario §4.3–§4.5 → Test §1.4 (tiêu chí theo mốc), §5 (lịch đưa TC vào CI) | Lập kế hoạch tuần, chính sách nộp, thứ tự cắt giảm, phân vai |
| Dev | SRS §4 (FR của module mình), §5.1, §6 → Đặc tả kỹ thuật §1 (kiến trúc, repo), §2 (hợp đồng dữ liệu), §3.x (module), §4 (make), §5 (`Settings`, model, D0), §6 (exception) → TC của module trong Test §2 → API contract §1.5 (interface nội bộ, bất biến liên module), §3.4, §4 (exception ↔ exit code), `contracts/internal/*` → Thư viện §1, §2 (nhóm cài đặt), `deps/pyproject.draft.toml` | Hiện thực đúng tên symbol, cấu hình, lỗi mà TC kiểm |
| QA | Test §1 (chiến lược, marker, môi trường) → Test §2, §3 (truy vết), §4 (smoke, gate private) → Scenario §2 và §4.3 → cột "Tiêu chí chấp nhận" của SRS §4 → API contract §5 (luật validator predictions), §6 (hợp đồng ↔ TC, test hợp đồng đề xuất) → Thư viện §5.4 (khoá phiên bản), §6 (thư viện ↔ TC) | Dựng CI, giữ độ phủ FR MUST và SC Critical/High, chạy smoke trước mỗi lượt nộp |

---

## 3. Thống kê

### 3.1 Tài liệu BA

| Loại | Số lượng | Ghi chú |
|---|---|---|
| Stakeholder (STK) | 12 | |
| Mục tiêu (G) | 11 | 7 cho (1), 4 cho (2) |
| KPI | 24 | KPI-01..16 cho (1), KPI-17..24 cho (2) |
| Use case (UC) | 12 | UC-01..06 cho (1), UC-07..12 cho (2) |
| User story (US) | 20 | US-01..12 cho (1) (US-08 dùng cả hai), US-13..20 cho (2) |
| Rủi ro (RSK) | 19 | |
| Giả định, phụ thuộc (A) | 16 | A-16: bộ mẫu đã được AI agent xử lý, chờ Q-06 |

### 3.2 SRS

| Loại | Số lượng | Chi tiết |
|---|---|---|
| Ràng buộc (C) | 21 | |
| Quy tắc nghiệp vụ (BR) | 25 | Nhiều BR là [GT] từ mẫu n = 32, chỉ áp cứng sau `make audit` trên train chính thức (FR-EVL-10) |
| FR | 88 | M 70 · S 15 · C 2 · W 1 |
| NFR | 22 | REPRO 4 · PERF 4 · SEC 2 · PRIV 3 · COST 2 · MAINT 3 · OBS 2 · USAB 2 |
| Câu hỏi mở (Q) | 33 | Q-01..Q-21, Q-24..Q-28 gửi giảng viên; Q-22, Q-23, Q-29..Q-33 nhóm tự chốt |

**FR theo module:**

| MOD | Tổng | M | S | C | W |
|---|---|---|---|---|---|
| ING (nạp dữ liệu) | 6 | 6 | 0 | 0 | 0 |
| EVD (evidence, sandbox) | 8 | 6 | 1 | 0 | 1 |
| POL (chính sách multi_problem) | 5 | 4 | 0 | 1 | 0 |
| T1 (chấm rubric) | 9 | 7 | 2 | 0 | 0 |
| T2 (phân loại lỗi) | 8 | 7 | 0 | 1 | 0 |
| T3 (sinh phản hồi) | 8 | 5 | 3 | 0 | 0 |
| GRD (guard compliance T3) | 7 | 5 | 2 | 0 | 0 |
| EVL (metric, đánh giá nội bộ) | 10 | 8 | 2 | 0 | 0 |
| VAL (validator, nộp bài) | 6 | 5 | 1 | 0 | 0 |
| REP (tái lập, hạ tầng) | 6 | 4 | 2 | 0 | 0 |
| SEC (bảo mật, riêng tư, tuân thủ) | 7 | 6 | 1 | 0 | 0 |
| RPT (báo cáo, ablation, artefact) | 8 | 7 | 1 | 0 | 0 |
| **Tổng** | **88** | **70** | **15** | **2** | **1** |

### 3.3 Kịch bản (SC)

| Severity | Số SC |
|---|---|
| Critical | 13 |
| High | 31 |
| Medium | 24 |
| Low | 2 |
| **Tổng** | **70** (14 chiều × 5) |

### 3.4 Test case (TC)

| MOD | Số TC | P0 | P1 | P2 | Tự động |
|---|---|---|---|---|---|
| ING | 10 | 7 | 3 | 0 | 10 |
| EVD | 15 | 5 | 9 | 1 | 15 |
| POL | 9 | 4 | 4 | 1 | 8 |
| T1 | 12 | 7 | 5 | 0 | 11 |
| T2 | 10 | 6 | 4 | 0 | 10 |
| T3 | 10 | 5 | 5 | 0 | 9 |
| GRD | 12 | 9 | 3 | 0 | 11 |
| EVL | 13 | 7 | 4 | 2 | 12 |
| VAL | 9 | 9 | 0 | 0 | 8 |
| REP | 14 | 8 | 6 | 0 | 10 |
| SEC | 10 | 9 | 1 | 0 | 8 |
| RPT | 7 | 1 | 6 | 0 | 2 |
| **Tổng** | **131** | **77** | **50** | **4** | **114 (87%)** |

---

## 4. Truy vết rút gọn

### 4.1 MOD → FR → TC

"FR MUST có TC" đếm theo bảng Test §3.1; mọi FR MUST đều có ít nhất một TC mức P0 hoặc P1.

| MOD | FR (M) | TC của MOD | FR MUST có TC | Ghi chú phủ |
|---|---|---|---|---|
| ING | 6 (6) | 10 | 6/6 | |
| EVD | 8 (6) | 15 | 6/6 | FR-EVD-08 (W) có test âm TC-EVD-15 |
| POL | 5 (4) | 9 | 4/4 | FR-POL-04 nâng lên M, phủ bởi TC-POL-07, TC-POL-08 |
| T1 | 9 (7) | 12 | 7/7 | FR-T1-05 chỉ có TC bán tự động (TC-T1-12) |
| T2 | 8 (7) | 10 | 7/7 | |
| T3 | 8 (5) | 10 | 5/5 | FR-T3-04 (fine-tune bắt buộc, C-02) phủ bởi TC-T3-10 mức P0 |
| GRD | 7 (5) | 12 | 5/5 | |
| EVL | 10 (8) | 13 | 8/8 | FR-EVL-06 chỉ có TC bán tự động (TC-EVL-09) |
| VAL | 6 (5) | 9 | 5/5 | |
| REP | 6 (4) | 14 | 4/4 | |
| SEC | 7 (6) | 10 | 6/6 | |
| RPT | 8 (7) | 7 | 7/7 | FR-RPT-03..08 kiểm bằng review/checklist; FR-RPT-08 mới, phủ bởi TC-RPT-07 |
| **Tổng** | **88 (70)** | **131** | **70/70** | |

### 4.2 Các tầng truy vết khác

| Tầng | Kết quả | Ở đâu |
|---|---|---|
| Điều khoản đề → C/BR/FR/NFR | Mọi điều khoản §1–§9 có yêu cầu tương ứng; §4 "≥ 1 model fine-tune" bảo đảm bởi FR-T3-04 (M) | SRS §7 |
| C → TC (acceptance) | 20/21 C có TC; C-21 (8 tuần, 5–7 người) không test trực tiếp | Test §1.7 |
| NFR → TC | 22/22 | Test §3.2 |
| BR → TC | 25/25 | Test §3.4 |
| SC Critical → TC P0 | 13/13 | Test §3.5 |
| SC High → TC P0/P1 | 31/31 | Test §3.5 |
| FR → symbol hiện thực | 88/88 | Đặc tả kỹ thuật §7.1 |
| ID treo | 0 (mọi STK/G/KPI/UC/US/RSK/A/C/BR/FR/NFR/Q/SC/TC được tham chiếu đều có định nghĩa) | Kiểm chéo 5 tài liệu |

---

## 5. Câu hỏi mở quan trọng nhất

Đầy đủ ở SRS §8, mỗi câu có cột "Mặc định khi chưa có trả lời".

| Ưu tiên | Q | Vì sao quan trọng | Mặc định đang chạy |
|---|---|---|---|
| 1 | Q-01 | Định dạng `predictions.json`; sai định dạng vẫn mất lượt, private chỉ có 1 cơ hội | Lặp lại `output` của file task; 1 file/task; mảng |
| 2 | Q-02, Q-03 | Quy ước scorer (`labels` của QWK, `zero_division` của macro-F1) đổi metric đáng kể (0.738 so với 0.931; oracle 0.90 so với 1.00) | `labels=0..10`; cấu hình bền với 3 quy ước zd |
| 3 | Q-05 | Phạm vi cấm feedback cho T1/T2 (level, exemplar, rationale, người soạn prompt đã đọc feedback); rủi ro kỷ luật | Hiểu rộng nhất (BR-21) |
| 4 | Q-06 | Dữ liệu được đi đâu; bộ mẫu đã được AI agent xử lý trước khi có phép (A-16) | Mọi thứ chạy local; khai báo việc dùng agent (FR-RPT-07) |
| 5 | Q-07 | Cỡ tập đầy đủ, dev/test có đề mới không, dev có nhãn không; quyết định sơ đồ CV và cổng D0 | Group theo near-dup; theo exam nếu có đề mới |
| 6 | Q-08 | Trọng số EX01 và việc áp cổng P1; quyết định điểm vận hành nộp (FR-POL-03) và phát hiện bonus #1 | Policy là feature; chọn (a)/(b)/lai trên OOF |
| 7 | Q-04 | Detector compliance T3 và cách gộp (a)/(b); quyết định guard chặt tới đâu | Áp toàn bộ điều cấm của label_space |
| 8 | Q-09, Q-25 | Giảng viên tái lập thế nào, trên phần cứng nào, có chấp nhận replay từ cache | ε = 0.01; replay từ cache không cần GPU/Docker |
| 9 | Q-10 | "≥ 2 hướng" áp cho từng task hay toàn hệ thống; "≤ 10B" là tổng hay active | Mỗi task đều có ≥ 2 hướng; tổng tham số |
| 10 | Q-11 | Đơn vị quota (upload hay file task), múi giờ ranh giới ngày | 1 upload = 1 lượt; `Asia/Ho_Chi_Minh` |
| 11 | Q-14, Q-19 | T3 có được dùng `test_report`/test công khai; có được chạy code SV lúc suy luận | Không ghép; chạy trong sandbox, luôn có chế độ `given_signals_only` |
| Nhóm | Q-22, Q-23, Q-33 | Sync/async (chốt trước khi init repo); trần chi phí, GPU-giờ; Δ_min chọn mô hình cho T2/T3 | Chốt ở W1–W2 |
