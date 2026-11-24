# Test cases: pipeline chấm rubric, phân loại lỗi, sinh phản hồi bài C++

| Mục | Giá trị |
|---|---|
| Phiên bản | 0.1, bản nháp để nhóm review |
| Ngày | 2026-10-04 |
| Vai trò soạn | QA Lead |
| Căn cứ | [SRS](02-requirements-srs.md) (C, BR, FR, NFR, Q); [Scenarios](04-scenarios.md) (SC-01..SC-70); [báo cáo EDA](../plans/reports/eda-261003-2335-sample-dataset.md) (ưu tiên khi mâu thuẫn); [brainstorm](../../plans/reports/brainstorm-261003-2159-llm-grading-challenge.md) (§8.3, §9); quy ước Python PY-00..PY-13 |
| ID định nghĩa ở đây | Chỉ TC-<MOD>-xx. Mọi C/BR/FR/NFR/Q tham chiếu từ SRS, SC từ tài liệu scenario |
| Quy mô | 131 test case, 12 module. P0 77 · P1 50 · P2 4. Tự động 114/131 (87%) |

**Đối tượng test:** hệ thống (1), pipeline của nhóm cho challenge (SRS §1.2). Hệ thống (2), trợ lý chấm cho giảng viên, chỉ được kiểm gián tiếp qua: phiên bản (b) đúng policy (TC-POL-07), bộ injection (TC-SEC-10), model card (TC-RPT-06).

**Nguyên tắc dữ liệu:** tài liệu chỉ dẫn sample_id, không trích code sinh viên hay feedback. Mọi khẳng định số liệu lấy từ EDA (dẫn §). Số liệu chưa kiểm chứng ghi [GT].

---

## 1. Chiến lược test

### 1.1 Các mức test

| Mức | Mục tiêu | Kỹ thuật chính | Marker pytest | Môi trường | Khi chạy |
|---|---|---|---|---|---|
| Unit | Hàm thuần: hậu xử lý T1/T2, scorer, `aggregate()`, guard, validator, mapper nhãn, writer | pytest; property-based (Hypothesis); test double cho LLM (chỉ trong `tests/`, PY-12) | `unit` | Dev-CPU, CI-cloud | Mỗi commit |
| Integration | Module + dữ liệu thật, cache-replay LLM, sandbox | Fixture 32 bài theo sample_id; replay cache theo sha1(prompt, tham số, revision) | `fixture32`, `llm_cache`, `sandbox` | CI-local | Trước merge vào `main`; nightly |
| E2E pipeline | `make predict`, `make reproduce`, `make predict-private` chạy trọn | Dev đóng gói như private; bộ 36 sample tổng hợp | `e2e` | CI-local, GPU-24G | Nightly; W6, W7, W8 |
| Acceptance theo đề | Mỗi ràng buộc C-xx có bằng chứng (§1.7) | Checklist + artefact + test tự động liên quan | `acceptance` | — | Review W4, W7 |
| Metric / golden | Scorer và baseline tái tạo đúng số EDA | So với số tham chiếu, có dung sai | `golden` | CI-local | Mỗi lần đổi scorer, feature, decoder |
| Compliance / quy chế | Feedback firewall, lineage, cô lập test, không dùng mã SV, không viết tay card | Metamorphic (đổi feedback, output không đổi), scanner chuỗi con, import-linter | `compliance` | CI-cloud (dữ liệu tổng hợp) + CI-local | Mỗi commit; trước mỗi lượt nộp |
| Bảo mật sandbox | Code sinh viên không thoát được sandbox | Chương trình độc hại tổng hợp | `security`, `sandbox` | Sandbox-g++13 | Mỗi lần đổi image; nightly |
| Riêng tư | Không lộ dữ liệu qua log, tài liệu, git, egress | Canary string, linter tài liệu, gitleaks, allowlist | `privacy` | CI-cloud, CI-local | Mỗi commit; trước mỗi mốc |
| Tái lập | Diff 0 cùng máy; ε khác máy; replay không GPU/mạng; phương sai seed | Chạy 2 lần, so sha256; máy sạch | `repro` | CI-local, Máy sạch | Gate W3; W6, W7; trước nộp |
| Hiệu năng | Private ≤ 60 phút (final), ≤ 15 phút (nhanh); throughput | Đo thời gian theo stage từ log có cấu trúc | `perf` | GPU-24G | W6, W7, W8 |
| Ablation-harness | Bảng ablation và so sánh 2 hướng sinh tự động, truy được run_id | `make ablation`; paired bootstrap; Holm–Bonferroni | `ablation` | GPU-24G + CI-local | W5–W7 |

### 1.2 Môi trường

| Môi trường | Cấu hình | Dùng cho |
|---|---|---|
| Dev-CPU | macOS/Linux, Python theo lockfile, không GPU, Docker không bắt buộc | Unit, compliance tổng hợp, chế độ `given_signals_only` |
| Sandbox-g++13 | Docker image GNU g++ 13 (pin theo digest); `-std=c++11 -O0`, trap khi hàm non-void thiếu return; `--network none`; cgroup ≤ 1 CPU, ≤ 512 MB, ≤ 64 tiến trình; rootfs chỉ đọc + tmpfs; user không phải root; seccomp mặc định; chạy với `setarch -R`; mỗi bài 1 container (NFR-SEC-01) | EVD, sandbox của SEC, TC-GRD-07, TC-T3-05 |
| GPU-24G | 1 GPU 24 GB; driver/CUDA ghi vào manifest; vLLM và model revision pin | Marker `gpu`, `perf`; P-arm, F-arm, sinh T3 |
| Máy sạch | Clone mới; không GPU; mạng tắt; cache/adapter lấy qua kênh nội bộ đã duyệt, có checksum | TC-REP-02, TC-REP-03; diễn tập W6, W7 |
| CI-cloud | Runner của host repo private. **Không có dữ liệu thật**, chỉ code + fixture tổng hợp, vì runner là bên thứ ba (C-11, NFR-PRIV-02) | ruff, mypy, gitleaks, pre-commit, `unit`, `compliance`/`privacy` tổng hợp |
| CI-local | Self-hosted runner trên máy nhóm; data-root mount chỉ đọc; job test chạy khi mạng tắt | `fixture32`, `golden`, `sandbox`, `llm_cache`, `repro`, `e2e` |

### 1.3 Dữ liệu test

| Loại | Nguồn | Dùng ở | Quy tắc |
|---|---|---|---|
| Bộ mẫu 32 bài | `sample_dataset/` nằm **ngoài** repo; đường dẫn đọc từ `Settings` | `fixture32`, `golden` | Tham chiếu bằng sample_id; không copy code hay feedback vào `tests/`; thiếu data-root thì skip kèm lý do; verify MANIFEST trước khi chạy |
| Tổng hợp tự sinh | `tests/synth/`: generator Python sinh chương trình C++ do nhóm tự viết (Armstrong đúng/sai, in prompt, thừa khoảng trắng, thiếu return, lặp vô hạn, fork bomb, socket, ghi file, output lớn, comment injection), JSON theo schema SRS §6.1, compile_log, nhãn NFD, feedback có canary, exam `EX99` | `unit`, `security`, `compliance`, `privacy` | Chương trình C++ lưu dạng chuỗi trong generator và ghi ra tmp lúc chạy (pre-commit chặn `*.cpp`, FR-SEC-03); seed cố định |
| Train chính thức | data-root | OOF, `make audit`, golden mở rộng, bộ kiểm guard (TC-GRD-11) | Không commit. Test đọc feedback gắn marker `touches_feedback` và chỉ nằm ở luồng T3/EVL |
| Dev | data-root | Dry-run E2E, kiểm bất biến output, public LB | Không chỉnh prompt, ngưỡng hay rule theo từng bài dev |
| Test chính thức (private) | Chỉ đọc trong `make predict-private` | **Không dùng cho bất kỳ test nào** (C-12, FR-SEC-05) | Cơ chế cô lập được kiểm bằng archive giả (TC-SEC-06) |

### 1.4 Tiêu chí vào/ra

| Mức / mốc | Tiêu chí vào | Tiêu chí ra |
|---|---|---|
| Unit | Module có type hint và interface ổn định | 100% pass; coverage ≥ 80% cho scorer, `aggregate()`, validator, guard, loader, firewall (NFR-MAINT-03) |
| Integration / golden | Unit pass; image sandbox pin digest; cache có; MANIFEST data-root khớp | 100% P0 pass; P1 ≥ 95% pass, phần còn lại có ticket. Golden: metric ± 0.001, Spearman ± 0.01, số đếm khớp tuyệt đối |
| E2E / tái lập | Integration pass; run manifest đầy đủ | Cùng máy diff = 0; khác máy trong ε (NFR-REPRO-02); file qua cả 2 validator |
| Trước mỗi lượt nộp public | Bộ smoke §4.1 | 100% pass; fail thì không nộp, không mất lượt |
| Gate W3 (BS §9) | — | TC-REP-01 diff = 0; TC-POL-07 có bảng (a)/(b)/lai |
| MVP W4 | — | 100% P0 pass, trừ {TC-REP-02, TC-REP-03, TC-REP-13, TC-VAL-09, TC-SEC-05, TC-RPT-01, TC-T3-09, TC-T3-10}; các TC này xanh trước W7 |
| Feature freeze W7 | 100% P0; P1 ≥ 95% | 0 SC Critical còn mở; mọi FR MUST có TC pass (§3.1) |
| Chạy private W8 | Tag `v1.0`, tree sạch | Gate §4.2 pass 100% |

### 1.5 Tự động hoá

**Marker pytest** (khai báo trong `pyproject.toml`, `--strict-markers`):

```toml
[tool.pytest.ini_options]
markers = [
  "unit", "fixture32", "golden", "llm_cache", "sandbox", "gpu", "e2e", "repro", "perf",
  "compliance", "security", "privacy", "acceptance", "ablation", "touches_feedback",
  "p0", "p1", "p2", "smoke",
]
```

| Pipeline | Kích hoạt | Lệnh | Ghi chú |
|---|---|---|---|
| CI-cloud | Mỗi push/PR | `ruff check`, `ruff format --check`, `mypy`, `gitleaks`, `pre-commit run -a`, `pytest -m "unit or ((compliance or privacy) and not fixture32)" --cov` | Không có dữ liệu thật |
| CI-local | Nightly; trước merge `main`; trước nộp | `pytest -m "fixture32 or golden or sandbox or llm_cache or repro or e2e"` | Job chạy khi mạng tắt; xuất JUnit XML + `runs/<run_id>/summary.md` |
| GPU | Thủ công hoặc nightly khi có máy | `pytest -m "gpu or perf"` | Ghi phần cứng vào manifest |
| Smoke trước nộp | Người giữ quyền nộp | `make smoke-submit TASK=t1\|t2\|t3` (§4.1) | ≤ 10 phút trên CPU, chạy từ cache |
| Gate private | W8 | `make gate-private` (§4.2) | Bắt buộc trước `make predict-private` |

**Quy ước tên và truy vết:**
- Tên test mô tả hành vi, ví dụ `test_link_error_does_not_trigger_compile_gate`. Không đặt TC ID trong tên test hay comment code (quy tắc của nhóm về artefact ổn định).
- Ánh xạ TC → node id pytest nằm trong `docs/test-traceability.csv`. Script CI kiểm: mọi TC có cột "Tự động?" = Có thì phải có ≥ 1 node id tồn tại.
- LLM trong `unit` là test double trả response cố định. `integration` dùng cache-replay (tất định). Chỉ marker `gpu` gọi model thật.

### 1.6 Quy ước bảng test case

| Cột | Nghĩa |
|---|---|
| Loại | Unit · Integration · E2E · Acceptance · Golden · Compliance · Security · Privacy · Repro · Perf · Ablation · CI · Checklist |
| Ưu tiên | **P0:** chặn nộp hoặc chặn release; gắn SC Critical, quy chế, tính hợp lệ output, tái lập, scorer lõi. **P1:** SC High, FR MUST còn lại, hoặc FR Should quan trọng; phải pass trước W7. **P2:** chỉ gắn FR Could/Won't, hoặc FR Should không gắn SC Critical/High; làm khi còn thời gian |
| Fixture | sample_id của bộ mẫu, hoặc mô tả ngắn input tổng hợp |
| Tự động? | **Có** (marker) · **Bán tự động** (script sinh số, người đối chiếu) · **Thủ công** (checklist) |

### 1.7 Acceptance theo đề: ràng buộc C-xx → TC

| C | Nội dung gọn | TC chứng minh |
|---|---|---|
| C-01 | Input/output cố định 3 task | TC-T1-01, TC-T2-01, TC-T3-01 |
| C-02 | ≥ 1 prompting + ≥ 1 fine-tune (bắt buộc ở T3, FR-T3-04) | TC-T3-10 (fine-tune A3), TC-T1-10, TC-T1-12, TC-T2-08, TC-EVL-10 |
| C-03 | ≤ 10B, local, báo phần cứng | TC-REP-13, TC-RPT-04, TC-EVL-10, TC-T3-10 |
| C-04 | API được dùng khi train, báo chi phí | TC-SEC-04, TC-RPT-04 |
| C-05 | Tự động 100% lúc suy luận | TC-POL-03, TC-POL-04, TC-GRD-08, TC-SEC-06 |
| C-06 | Seed, phiên bản model, tái lập | TC-REP-01..TC-REP-08 |
| C-07 | Cấm feedback làm input T1/T2 | TC-ING-09, TC-T1-09, TC-T2-10, TC-SEC-01, TC-SEC-02 |
| C-08 | 2 lượt/ngày, private quyết định | TC-VAL-07 |
| C-09 | `predictions.json`, file sai vẫn mất lượt | TC-VAL-01..TC-VAL-06, TC-VAL-09 |
| C-10 | Repo + tái lập 1 lệnh + báo cáo 10–15 trang + model card + slide | TC-REP-02, TC-REP-14, TC-RPT-06, TC-RPT-07, TC-VAL-09 |
| C-11 | Không phát tán dữ liệu | TC-SEC-03, TC-SEC-04, TC-SEC-08, TC-SEC-09 |
| C-12 | Không gán nhãn tay test | TC-SEC-06 |
| C-13 | Không truy danh tính | TC-SEC-07 |
| C-14 | Không chia sẻ mã/dự đoán | TC-SEC-03 |
| C-15 | Khai báo công cụ LLM | TC-RPT-06 |
| C-16 | Tôn trọng chính sách multi_problem | TC-POL-01, TC-POL-05, TC-POL-07 |
| C-17 | L1/L2 không có code sửa hay lời giải | TC-GRD-01, TC-GRD-03, TC-GRD-06, TC-GRD-10, TC-GRD-12 |
| C-18 | Đường dẫn code qua `code_file` | TC-ING-01, TC-ING-06 |
| C-19 | Trọng số chấm, bonus | TC-RPT-01..TC-RPT-05, TC-RPT-07, TC-EVL-10, TC-EVL-13 |
| C-20 | Bảng đóng góp | TC-RPT-06 |
| C-21 | 8 tuần, 5–7 người | Không test trực tiếp; thứ tự P0 → P1 → P2 bám thứ tự cắt giảm BS §9 |

---

## 2. Test case theo module

Dung sai mặc định: metric ± 0.001; Spearman ± 0.01; số đếm khớp tuyệt đối; thời gian ± 0.5 s.

### 2.1 ING: nạp dữ liệu

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-ING-01 | Nạp 32 mẫu, join 3 file task khớp | FR-ING-01, FR-ING-04, C-18 · SC-45 | Integration | P0 | Bộ mẫu 32 bài | Gọi `load_dataset(data_root)`; join T1/T2/T3 theo sample_id | 0 lỗi schema; mỗi task n = 32 (EX01 15, EX02 17); 32/32 sample_id duy nhất và có ở cả 3 file; input T1 ≡ input T2 ở 32/32; input T3 = input T1 − `test_report` + `taxonomy_error` + `target_feedback_level` ở 32/32 | Có (`fixture32`) |
| TC-ING-02 | JSON hỏng cố ý → dừng, nêu đúng vị trí | FR-ING-01, NFR-USAB-02 · SC-45 | Unit | P0 | 5 file task tổng hợp: thiếu `compile_log`; `logic: "2"`; `exam_type: "multi"`; `logic: 5`; sample_id trùng | Nạp từng file | 5/5 raise `DataSchemaError`, exit ≠ 0; thông điệp có tên file, sample_id, đường dẫn trường và giá trị thấy được; không có câu phỏng đoán nguyên nhân (PY-08) | Có (`unit`) |
| TC-ING-03 | EX02-S214: lỗi link không phải CE | FR-ING-06, BR-15 · SC-06 | Unit | P0 | EX02-S214 | Phân tích `compile_log` | `log_build_failed=True`, `log_has_link_error=True`, `log_has_error=False` | Có (`fixture32`) |
| TC-ING-04 | Thống kê cờ log 32 bài; log bị cắt | FR-ING-06, BR-11 · SC-10, SC-30 | Golden | P0 | Bộ mẫu 32 bài; 1 log tổng hợp dài đúng 2000 ký tự, kết thúc giữa một dòng diagnostic `error:` | Phân tích 32 log và log tổng hợp | 32 bài: 22 log rỗng; `log_has_error` 6; lỗi link 1; chỉ warning 3; `log_truncated` 1 (EX01-S111, độ dài 2000); `log_has_error` ⇔ gold `compilable = 0` ở 6/6 (EDA §3.2, §5). Log tổng hợp: dòng dở cuối không được đếm là error; `log_truncated=True`; `n_errors_is_lower_bound=True` | Có (`fixture32`, `golden`) |
| TC-ING-05 | MANIFEST.sha256; bỏ qua `.DS_Store`; phát hiện file thừa | FR-ING-03 · SC-24, SC-45 | Integration | P1 | Bản sao tmp của bộ mẫu (có 2 `.DS_Store`) | (1) tạo manifest; (2) verify; (3) lật 1 byte trong một `.cpp`; (4) thêm 1 `.cpp` không JSON nào tham chiếu | (1) 0 entry `.DS_Store`, 0 cảnh báo về `.DS_Store`; (2) pass; (3) lỗi checksum nêu đúng đường dẫn file, exit ≠ 0; (4) 1 warning nêu đường dẫn file thừa, pipeline vẫn chạy | Có (`fixture32`) |
| TC-ING-06 | Thiếu file code: dừng ở train/audit, không dừng ở predict | FR-ING-03, NFR-USAB-02, C-18, C-09 · SC-45 | Integration | P0 | Bản sao tmp của bộ mẫu, xoá file code của EX02-S201 | (a) `make audit` (chế độ train/audit); (b) `make predict SPLIT=sample` (chế độ predict) | (a) `DataIntegrityError`, exit 2; thông điệp nêu `sample_id=EX02-S201` và `code_file` tương đối; 0 file output được ghi. (b) exit 0; 3 file output đủ 32 id, qua validator; card của S201 có `missing_code=True`, `evidence_mode="given_signals_only"`; output của S201 là fallback hợp lệ | Có (`fixture32`) |
| TC-ING-07 | BOM (EX01-S107): bỏ BOM, giữ bytes gốc, giữ số dòng | FR-ING-02 · SC-08, SC-50 | Unit | P1 | EX01-S107 | Đọc code | Text không bắt đầu bằng U+FEFF; `code_sha` = sha1(bytes gốc, kể cả BOM); số dòng của text = số dòng file gốc (grounding T3 không lệch); text ở dạng NFC | Có (`fixture32`) |
| TC-ING-08 | Decode fallback và file biên | FR-ING-02, FR-EVD-06 · SC-07, SC-08 | Unit | P1 | 5 file tổng hợp: cp1258 có chuỗi tiếng Việt, latin-1, 0 byte, chỉ BOM, chỉ khoảng trắng | Đọc; dựng card; chạy baseline 3 task | 5/5 không crash; 2 file mã hoá khác decode đúng chuỗi tham chiếu; bộ đếm fallback = 2 trong log; 3 file rỗng có `effective_code_lines=0`; output 3 task qua validator | Có (`unit`) |
| TC-ING-09 | Whitelist input T1/T2 | FR-ING-05, BR-21, C-07 · SC-41 | Unit | P0 | Sample tổng hợp có đủ trường T1/T2/T3 và `feedback` | Lấy input T1, T2; truy cập `feedback`, `target_feedback_level`, `taxonomy_error` | Tập key ⊆ whitelist cấu hình; 3/3 truy cập thuộc tính trên `BlindSample` raise `AttributeError`; dựng `BlindSample` kèm trường cấm → `pydantic.ValidationError` (`extra="forbid"`); không hàm public nào của loader T1/T2 trả về `feedback` (kiểm bằng introspection) | Có (`unit`, `compliance`) |
| TC-ING-10 | Báo giao sample_id T2/T3; test chỉ đếm trong `PrivateRun`; không có đường T3 → T2 | FR-ING-04, BR-22, FR-SEC-05 · SC-42 | Unit | P0 | (a) split dev tổng hợp: T2 và T3 chung 2 sample_id; (b) archive test tổng hợp giao 2 id; bộ mẫu | (a) chạy bước join; (b) chạy `predict-private` trên archive (b), rồi gọi join trên split test bên ngoài `PrivateRun`; chạy import-linter | (a) bảng giao ghi `overlap=2` + 1 warning; bộ mẫu: bảng giao 32/32 (cùng file). (b) `private_audit.jsonl` có `overlap=2`; 0 sample_id hay nội dung được in ra log, stdout hay audit; exit kèm cảnh báo; gọi ngoài `PrivateRun` → `TestIsolationError`. Import-linter: `t2` không import loader T3 (0 vi phạm) | Có (`unit`, `compliance`) |

### 2.2 EVD: evidence và sandbox biên dịch-chạy

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-EVD-01 | Rebuild GNU g++ 13 khớp trạng thái log 32/32; log gốc vẫn là tín hiệu chính | FR-EVD-04, FR-ING-06, FR-T1-03, BR-11 · SC-48, SC-26 | Integration | P1 | Bộ mẫu 32 bài; Sandbox-g++13 | EX01: `g++ -std=c++11 -fsyntax-only`; EX02: `g++ -std=c++11` (build + link); thêm 1 lượt EX01 bằng lệnh của statement (`-o id`) | `build_failed` (rebuild) = `log_build_failed` (log gốc) ở 32/32; `has_error` (rebuild) = `log_has_error` ở 32/32. Lệch thì test đỏ và liệt kê sample_id (EDA mới xác nhận 32/32 bằng clang; g++ 13 là [GT]). Lượt `-o id`: kết quả ghi vào `rebuild.*`, không đè `compile.*`; số bài bị cổng CE vẫn bằng 6 | Có (`sandbox`, `fixture32`) |
| TC-EVD-02 | UB (EX02-S206, EX02-S210): ghi returncode, stderr, cờ crash | FR-EVD-03, FR-EVD-04 · SC-26 | Integration | P1 | EX02-S206, EX02-S210; toolchain pin `-O0` + trap thiếu return | Chạy 8 test công khai trong sandbox | Mọi test có `actual=''` mà rebuild tái lập được (stdout rỗng) đều có returncode ≠ 0 hoặc signal, stderr được lưu, `verdict ∈ {nonzero_exit, signal}`, `fail_class=empty_crash`; bài có `crash_suspected_ub=True`. Test không tái lập được (toolchain, EDA §5: gcc 12 -O2 chỉ 96/104) có `agrees_with_report=False`, bài có `rerun_agreement < 1`, và không tính là fail của test | Có (`sandbox`) |
| TC-EVD-03 | Tái lập `actual` của test_report | FR-EVD-04 · SC-26 | Golden | P1 | 13 bài EX02 có test (104 test) | Chạy lại; so `actual` (sau strip) với report | ≥ 96/104 khớp (EDA §5) với profile `gcc13-*` đã calibrate; các test lệch có `agrees_with_report=False` và `rerun_agreement` của bài < 1 trong card | Có (`sandbox`, `golden`) |
| TC-EVD-04 | Timeout (EX02-S209) | FR-EVD-03, NFR-PERF-03 · SC-11 | Integration | P0 | EX02-S209; chương trình tổng hợp lặp vô hạn | Chạy test | Mỗi test treo bị kill sau 2 s, verdict `timeout`; S209: 8/8 test `timeout` (khớp 8 timeout ở EDA §2.3); tổng thời gian bài ≤ 30 s; card được ghi; pipeline chạy bài kế; exit 0 | Có (`sandbox`) |
| TC-EVD-05 | Phát hiện `prompt_prefix` từ stdout thô (EX02-S211, EX02-S216) | FR-EVD-04, FR-EVD-02, BR-10 · SC-58 | Integration | P1 | EX02-S211, EX02-S216 | Chạy lại; tính pass strict và pass nới | `prompt_prefix=True` ở cả 2; S211: strict 0/8, nới 8/8, `io_gap` = 8/8 (suy từ EDA §2.3: 11 test chỉ sai vì prompt); S216: nới 3/8 | Có (`sandbox`) |
| TC-EVD-06 | Phát hiện `leading_ws` (EX02-S205) | FR-EVD-04 · SC-58 | Integration | P1 | EX02-S205 (report pass 8/8, gold `io_format = 0`); EX02-S211 làm đối chứng | Chạy lại; phân tích stdout thô | S205: `leading_ws=True` dù report pass 8/8 (report đã strip); S211: `leading_ws=False` | Có (`sandbox`) |
| TC-EVD-07 | Phân loại test fail, pass strict/nới, output hằng | FR-EVD-02, FR-EVD-03, FR-T1-08 · SC-65 | Golden | P1 | test_report của 17 bài EX02 (không cần sandbox); 1 test_report tổng hợp có `actual` luôn "True" | Tính từ report | 57/104 test fail phân đúng: sai đáp án 17, output rỗng 16, chỉ prompt 11, timeout 8, prompt + sai 5; Spearman(pass nới, total) = 0.89, strict 0.57 (n = 13); report tổng hợp: pass 6/8, `constant_output=True`; EX02-S208: `constant_output=True` | Có (`fixture32`, `golden`) |
| TC-EVD-08 | Phân tích tĩnh EX01 | FR-EVD-05, BR-17 · SC-57 | Golden | P1 | 15 bài EX01; problem card EX01 | Chạy phân tích tĩnh | Đủ 4 hàm: 6/15; thiếu P2/P3/P4: 4/8/6 bài; còn `main`: 10/15; 3/3 bài có nhãn "Lỗi hard-code" bật ≥ 1 trong 3 cờ `hardcode_fixed_path`, `hardcode_cin_instead_of_file`, `hardcode_fixed_array_size`; 0 bài đổi prototype | Có (`fixture32`, `golden`) |
| TC-EVD-09 | File toàn comment và khối comment dài | FR-EVD-06 · SC-06 | Unit | P1 | EX02-S214, EX02-S209 | Tính `effective_code_lines` và khối comment | S214: `all_commented=True`, `effective_code_lines=0`; S209: `max_commented_block=26` ≥ `evd.commented_block_min_lines` (20); 16 bài EX02 còn lại: `all_commented=False` | Có (`fixture32`) |
| TC-EVD-10 | Cụm near-dup ổn định | FR-EVD-07, NFR-REPRO-04 · SC-29, SC-43 | Unit | P0 | 17 bài EX02 (đã bỏ code mẫu của đề) | Tính cụm ở 2 tiến trình `PYTHONHASHSEED=1` và `=2`, và trên máy thứ hai | EX02-S203..S206 cùng `cluster_id`; kỳ vọng âm: 13 bài EX02 ngoài S203–S206 có `cluster_size = 1`, không cụm nào > 25% số bài của exam (`evd.neardup_max_cluster_frac`); `cluster_id` của mọi bài giống nhau giữa 2 tiến trình và 2 máy; lint: 0 lời gọi `hash(` trong package | Có (`fixture32`, `repro`) |
| TC-EVD-11 | Evidence card: schema, tất định, không có feedback, gọn | FR-EVD-01 · SC-26 | Integration | P0 | 32 bài; evidence từ cache | Dựng card 2 lần; validate JSON Schema; đếm token bản text gọn | 32/32 hợp lệ; 2 lần cho bytes giống hệt; 0 key thuộc {`feedback`, `target_feedback_level`, `taxonomy_error`}; text gọn ≤ 1.5k token (tokenizer pin) ở 32/32 | Có (`fixture32`) |
| TC-EVD-12 | Cache hit ở lần chạy thứ 2 | FR-REP-03, NFR-PERF-03 · SC-26, SC-14 | Integration | P1 | Cache evidence rỗng; 17 bài EX02 | Chạy evidence 2 lần; lần 3 đổi `toolchain_id` | Lần 1: miss 100%. Lần 2: hit 100%, 0 container được tạo, card giống byte, thời gian ≤ 10% lần 1. Lần 3: miss 100%, không đọc card của toolchain cũ | Có (`sandbox`) |
| TC-EVD-13 | Sandbox: chặn mạng, không lộ secret, không chạy root | FR-SEC-02, NFR-SEC-01, NFR-SEC-02 · SC-36 | Security | P0 | Chương trình tổng hợp: mở TCP tới 1.1.1.1:80, resolve DNS, đọc biến canary `GRADER_API_KEY` (đặt ở host), in `getuid()` | Chạy trong sandbox | connect và DNS đều thất bại; biến canary đọc ra rỗng; uid ≠ 0; verdict ghi vào card; pipeline chạy tiếp | Có (`sandbox`, `security`) |
| TC-EVD-14 | Sandbox: fork bomb, bộ nhớ, ghi ngoài tmp, output khổng lồ | FR-SEC-02, NFR-SEC-01 · SC-36 | Security | P0 | 4 chương trình tổng hợp tương ứng; data-root và cache có MANIFEST | Chạy từng chương trình | Fork bomb: số tiến trình ≤ 64, bị kill trong timeout bài, số tiến trình của host trở về mức trước. Cấp phát 2 GB: OOM-kill ở ≤ 512 MB, verdict `memory_limit`. Ghi `/etc`, `/data`, `/cache`, `$HOME`: EROFS/EACCES, MANIFEST data-root và cache không đổi. In 1 GB: stdout bị cắt ở `sandbox.stdout_cap_bytes` (mặc định 64 KB), `output_truncated=True`, dung lượng đĩa host tăng ≤ trần | Có (`sandbox`, `security`) |
| TC-EVD-15 | Không chạy harness EX01 khi chưa có dữ liệu chạy được | FR-EVD-08 · SC-48 | Unit | P2 | 15 bài EX01; cấu hình mặc định | Gọi runner cho EX01 | 15/15 trả `skipped: harness_disabled`; 0 container; chỉ có phân tích tĩnh | Có (`unit`) |

### 2.3 POL: chính sách multi_problem

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-POL-01 | Problem card EX01: tiên quyết, trọng số mâu thuẫn | FR-POL-01, BR-07, C-16 · SC-57 | Integration | P0 | `exams.json` của bộ mẫu; bước LLM chạy qua cache-replay (self-consistency) | Trích card EX01 | Có P1..P4; P1 → {P2, P3, P4}; trọng số statement 1/4/2/3, `max_score` 2.5 × 4, `weight_conflict=True`; 4 điều khoản 0 điểm (vi phạm quy định, giống bài khác, không biên dịch, lặp vô hạn); đúng 6 thư viện cho phép (iostream, fstream, sstream, string, cstring, cmath); đồ thị không có chu trình | Có (`fixture32`, `llm_cache`) |
| TC-POL-02 | Problem card EX02 | FR-POL-01 · SC-19 | Integration | P1 | Như TC-POL-01 | Trích card EX02 | 1 câu; `io_mode` stdin/stdout; cấm `pow`; ràng buộc 1 ≤ n ≤ 10^9; không có tiên quyết; `weight_conflict=False`; `policy_failed=False` | Có (`fixture32`, `llm_cache`) |
| TC-POL-03 | Trích thất bại → trọng số đều, pipeline chạy hết | FR-POL-01, C-05 · SC-23 | Unit | P0 | 3 exam tổng hợp: không có `grading_policy`; LLM (test double) trả đồ thị P1 → P2 → P1; Σ trọng số ≠ tổng điểm | Trích card; chạy pipeline baseline trên 5 sample tổng hợp mỗi exam | 3/3 `policy_failed=True`, trọng số đều; pipeline exit 0, ra 15/15 output hợp lệ; mỗi exam 1 warning có cấu trúc | Có (`unit`) |
| TC-POL-04 | Cấm card viết tay cho exam ngoài train | FR-POL-01, C-05 · SC-23 | Compliance | P1 | Thêm 1 file card viết tay cho exam tổng hợp `EX99` chỉ có ở dev | Chạy bước kiểm của CI | CI đỏ: card của exam không thuộc train phải do `ProblemCard.from_exam` sinh (có `extraction_method`, `llm_revision`) và sha khớp output của bộ trích; xoá file → CI xanh | Có (`compliance`) |
| TC-POL-05 | `aggregate()` bản cứng | FR-POL-02, BR-07, BR-09 · SC-57 | Unit | P0 | Trọng số 1/4/2/3 | Chạy ≥ 10 case tham số hoá | p = [1,1,1,1] → 10; P1 sai → 0 (P2–P4 bị chặn); P3 không làm → 8; P3 lặp vô hạn: biến thể theo câu → 8, theo file → 0; chuỗi tiên quyết 2 tầng (tổng hợp) chặn đúng mọi hậu duệ; trọng số 2.5 × 4, p = [1,0,1,1] → 7.5 | Có (`unit`) |
| TC-POL-06 | `aggregate()` bản mềm; policy chỉ là feature | FR-POL-02, BR-08 · SC-56 | Unit | P1 | Trọng số 1/4/2/3 | Gọi bản mềm Σ w_q · p_q · Π p_tiên quyết; kiểm luồng dữ liệu | p = [0.5,1,1,1] → 5.0; p = [1,0.5,1,1] → 8.0; giá trị nằm trong card (`policy.hard_score`, `policy.soft_score`); phiên bản (a) không có đường gán thẳng `aggregate()` → `total_score` (đổi feature thì total chỉ đổi qua model) | Có (`unit`) |
| TC-POL-07 | Ba phương án T1: (a) theo gold, (b) đúng policy, lai | FR-POL-03, FR-POL-04, BR-07, BR-08, C-16 · SC-56, SC-04 | Integration | P0 | OOF trên bộ mẫu hoặc train; trạng thái câu từ FR-POL-04 | Chạy (a), (b), lai; sinh bảng; chạy bước chọn điểm vận hành | (b), kiểm cấp vector: `prereq_failed=True` ⇒ vector = (compilable, io_format, 0, 0, 0, 0) và total ≤ 2; zero-clause phạm vi file ⇒ `000000`. (a): EX01-S108/S110/S112 không bị ép về 0 bởi cổng P1. EX02: (a) ≡ (b) từng bài, Δ = 0. Bảng 3 dòng (EX01, EX02, pooled) × {QWK, MAE} cho (a), (b), lai, Δ so với (a), CI95; `runs/decisions.jsonl` có dòng chọn điểm vận hành kèm Δ, Δ_min, CI | Có (`fixture32`) |
| TC-POL-08 | Tách câu và trạng thái từng câu (EX01) | FR-POL-04, FR-EVD-05 · SC-57 | Integration | P1 | 15 bài EX01; exam tổng hợp không khớp prototype | Tách theo prototype; ước lượng trạng thái | 15/15 bài có 4 trạng thái ∈ {đúng, một phần, sai, không làm}; câu thiếu hàm → "không làm" (P2/P3/P4: 4/8/6 bài); exam tổng hợp → `split_failed=True`; scanner firewall trên prompt trạng thái: 0 hit | Có (`fixture32`, `llm_cache`) |
| TC-POL-09 | Gating mềm khả vi (τ) | FR-POL-05 · SC-56 | Ablation | P2 | OOF EX01 trên train chính thức | Fit τ, T; bootstrap B = 2000 | Báo τ kèm CI95; 2 lần chạy cùng seed cho τ giống tới 1e-9 | Bán tự động |

### 2.4 T1: chấm rubric

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-T1-01 | Đúng miền, số nguyên, total = Σ | FR-T1-01, BR-01, BR-02, C-01 · SC-60 | Unit | P0 | Hậu xử lý T1; Hypothesis ≥ 1000 input ngẫu nhiên (xác suất, giá trị lẻ, âm, NaN) + predictions của 32 bài | Chạy hậu xử lý và validator | 100% output: compilable, io_format, complexity, code_quality ∈ {0, 1}; logic ∈ 0..4; edge_case ∈ 0..2; `total_score` ∈ 0..10; mọi giá trị kiểu `int` (không `bool`, không float); `total_score` = Σ 6 chiều ở 100% | Có (`unit`) |
| TC-T1-02 | Cổng CE theo exam | FR-T1-03, BR-03, BR-04, BR-15 · SC-06, SC-59 | Golden | P0 | 6 bài CE của bộ mẫu (EX01 3, EX02 3) + EX02-S214 | Chạy T1 | 3 bài CE của EX02 → `000001` (total 1); 3 bài CE của EX01 → `000000` (total 0); EX02-S214 không bị cổng | Có (`fixture32`) |
| TC-T1-03 | Exam mới chưa có vector cổng | FR-T1-03, FR-T1-02 · SC-19, SC-59 | Unit | P1 | Exam tổng hợp `EX99` (single_problem, không có ở train), 1 bài có `log_has_error=True` | Chạy T1 | Vector cổng = mặc định theo `exam_type`, OOF/log ghi `gate_source=exam_type_default` (Q-26); không áp ràng buộc riêng của EX01/EX02; vector thuộc miền BR-02, total = Σ | Có (`unit`) |
| TC-T1-04 | Ràng buộc rubric EX02 | FR-T1-02, BR-05 · SC-60 | Unit | P0 | Xác suất tổng hợp cho EX02, gồm ca argmax logic = 3 | Giải mã MAP có ràng buộc | 100% output EX02: logic ∈ {0, 2, 4}, edge_case = logic/2, code_quality = 1; ca argmax logic = 3 → vector hợp lệ có xác suất cao nhất | Có (`unit`) |
| TC-T1-05 | Ràng buộc rubric EX01 | FR-T1-02, BR-06 | Unit | P0 | Xác suất tổng hợp cho EX01 | Giải mã MAP có ràng buộc | 100% output EX01: complexity = code_quality = [logic > 0] | Có (`unit`) |
| TC-T1-06 | Ràng buộc đúng < 99% trên train → hạ thành feature | FR-T1-02, FR-EVL-10 · SC-30 | Unit | P1 | 2 train tổng hợp: BR-05 đúng 95% và 100% | Chạy `make audit`, rồi fit | 95%: ràng buộc tắt, feature tương ứng bật, audit ghi 95.0%; 100%: ràng buộc bật | Có (`unit`) |
| TC-T1-07 | Decoder không co dự đoán về giữa | FR-T1-07 · SC-60, SC-63 | Golden | P0 | (1) OOF tổng hợp: gold hai đỉnh, dự đoán = 0.6 · gold + nhiễu (seed cố định); (2) OOF thật trên train | Fit decoder bằng nested CV; áp lên OOF | std(pred)/std(gold) ∈ [0.8; 1.2] ở (1) và (2); QWK sau decoder ≥ trước ở (1); 0 total lẻ; hàm làm tròn nội bộ: 2.5 → 3, 6.5 → 7 (không banker's rounding); lint: 0 lời gọi `round(` trong module decoder | Có (`unit`, `golden`) |
| TC-T1-08 | Tái tạo baseline T1 | FR-T1-06, FR-EVL-01 · SC-63 | Golden | P1 | 32 bài | Chạy 4 baseline (luật biên dịch đọc `compile_log`, không dùng gold `compilable`) | Luật biên dịch: QWK pooled 0.512 (EX01 0.413, EX02 0.636), MAE 2.91. Median theo exam: pooled 0.219, từng exam 0.000, MAE 2.94. `rubric_prior` (EX02): pooled 0.487, MAE 2.47. Hằng số: QWK 0 (EDA §2.5) | Có (`fixture32`, `golden`) |
| TC-T1-09 | Không dùng feedback (metamorphic) | FR-SEC-01, FR-ING-05, BR-21, C-07 · SC-41 | Compliance | P0 | 32 bài + train (nếu có); 4 biến thể: xoá feedback, xáo feedback giữa các bài, thay feedback bằng chuỗi canary, xáo `target_feedback_level` | Chạy `make audit`, dựng dữ liệu train T1 và chạy suy luận T1 (P-arm qua cache, F-arm, GBDT) cho bản gốc và 4 biến thể | sha256 của `constraints.json`, dữ liệu train T1 và predictions T1 giống hệt ở cả 4 biến thể; 0 cache miss mới (khoá prompt không đổi); canary không xuất hiện trong prompt hay feature | Có (`compliance`, `llm_cache`) |
| TC-T1-10 | P-arm forced-choice | FR-T1-04, C-02 · SC-28 | Integration | P1 | GPU-24G, vLLM pin; 32 bài | Chạy P-arm | Mỗi chiều: Σ xác suất = 1 ± 1e-6; logprob raw của token kế tiếp sau tiền tố `{"logic": `, đọc khi guided decoding tắt; mọi giá trị hợp lệ có mặt trong top-k (smoke: là token đơn); EV nằm trong miền của chiều; parse-fail sau retry ≤ 1%; manifest ghi model revision và seed | Có (`gpu`) |
| TC-T1-11 | JSON hỏng từ LLM → retry → fallback | FR-T1-04, FR-T1-01 · SC-31 | Unit | P0 | LLM (test double) trả lần lượt: thiếu trường, `logic: 5`, `"hai"`, nhãn ngoài vocab | Chạy T1 trên 4 sample tổng hợp | Mỗi sample retry theo danh sách seed cố định (≤ 3 lần); hết lượt → fallback baseline tất định cho đúng sample đó; file có đủ 4 sample_id và qua validator; summary ghi `parse_fail=4`, `fallback=4` | Có (`unit`) |
| TC-T1-12 | Hướng fine-tune cho T1, hoặc lý do dồn sang T3 | FR-T1-05, C-02 · SC-17 | Acceptance | P1 | Run final | Kiểm artefact | Có `oof/t1/{run_id}.parquet` của F-arm là LoRA trên model open-weight ≤ 10B (trainable params > 0; classifier trên embedding đóng băng không được tính), có mặt trong bảng so sánh (TC-EVL-10); hoặc báo cáo có mục lý do dồn fine-tune sang T3, kèm learning curve | Bán tự động |

### 2.5 T2: phân loại lỗi đa nhãn

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-T2-01 | `[]` hợp lệ từ đầu tới cuối | FR-T2-01, BR-13, C-01 | Unit | P0 | Predictions toàn `[]` cho 32 sample_id | Ghi file; validate; chấm | Writer và validator chấp nhận; macro-F1 = 0.000 / 0.100 / 0.000 (zd = 0 / 1 / nan), micro-F1 = 0 (EDA §3.3) | Có (`fixture32`, `golden`) |
| TC-T2-02 | Map nhãn NFC/NFD/lệch dấu về đúng chuỗi LS | FR-T2-01, BR-12 · SC-09 | Unit | P0 | 10 nhãn ở dạng NFD; biến thể "Loi logic", "Lỗi edge-case", "LỖI LOGIC"; nhãn lạ "Lỗi bộ nhớ"; danh sách có trùng và sai thứ tự | Chạy mapper | 10/10 NFD → đúng bytes NFC của LS; 3 biến thể → nhãn LS tương ứng; "Lỗi bộ nhớ" bị bỏ + 1 warning; output không trùng, theo thứ tự LS | Có (`unit`) |
| TC-T2-03 | Luật "Lỗi biên dịch" chỉ dùng diagnostic error | FR-T2-02, BR-15 · SC-06 | Golden | P0 | 32 bài | Chạy luật | 6/6 bài CE có "Lỗi biên dịch"; EX02-S214 không có; 0 FP trên 26 bài còn lại | Có (`fixture32`) |
| TC-T2-04 | Chỉ override CE khi tam giác khớp ≥ 99% trên train | FR-T2-02, FR-EVL-10 · SC-30 | Unit | P1 | 2 train tổng hợp: tam giác (compilable, log, nhãn) khớp 97% và 100% | Fit | 97%: chế độ feature (không override), audit ghi 97.0%; 100%: override | Có (`unit`) |
| TC-T2-05 | Ràng buộc chéo: total < 10 ⇒ ≥ 1 nhãn | FR-T2-06, BR-14 · SC-34 | Unit | P0 | Hypothesis: (total dự đoán, xác suất 10 nhãn) ngẫu nhiên | Chạy hậu xử lý ở chế độ bật và tắt | Bật: 100% output thoả (total < 10 và tập rỗng → thêm nhãn xác suất cao nhất; chỉ total = 10 mới được `[]`). Tắt: output = kết quả ngưỡng thuần. Báo cáo có Δmacro-F1 bật/tắt kèm CI | Có (`unit`) |
| TC-T2-06 | Không đoán "Lỗi edge case" khi xác suất thấp hoặc thiếu bằng chứng | FR-T2-07 · SC-18 | Unit | P0 | τ_strong đọc từ `Settings` (`t2.tau_strong`, mặc định 0.90); 4 ca: (p = τ_strong − 0.01, có test biên fail), (p = τ_strong, không có), (p = τ_strong, có), (p = 0.6, có) | Chạy chính sách nhãn hiếm; chạy pipeline trên 32 bài | 4 ca → không / không / có / không; 32 bài: 0 dự đoán "Lỗi edge case"; báo cáo có macro-F1 ở zd ∈ {0, 1, nan} | Có (`unit`, `fixture32`) |
| TC-T2-07 | Ngưỡng riêng từng nhãn fit trên OOF | FR-T2-05 · SC-62 | Golden | P1 | OOF T2 (train hoặc bộ mẫu) | Fit ngưỡng: argmax F1, lấy median qua bootstrap (seed cố định) | Bảng 10 ngưỡng + F1 OOF + mốc 2π/(1+π); 2 lần chạy cho ngưỡng giống hệt; nhãn có support < k (Settings) co về ngưỡng chung; 0 nhãn thua mốc mà vẫn dùng model | Có (`golden`) |
| TC-T2-08 | P-arm hỏi yes/no từng nhãn | FR-T2-03, C-02 | Integration | P1 | GPU-24G, vLLM pin; 32 bài | Chạy P-arm | Xác suất 10 nhãn cho 32/32 bài, mỗi giá trị ∈ [0, 1]; prompt theo exam có định nghĩa, phản ví dụ và câu "có thể không có lỗi nào"; manifest ghi seed và revision | Có (`gpu`) |
| TC-T2-09 | Bộ phân loại học được; không suy nhãn từ rubric | FR-T2-04, BR-16 | Integration | P1 | OOF T2 | Train GBDT/LR; chạy ablation bỏ feature rubric; kiểm hậu xử lý | `oof/t2/{run_id}.parquet` đủ 10 cột; có dòng ablation bỏ feature rubric; ngoài luật CE và ràng buộc chéo, 0 luật gán nhãn từ chiều rubric (đặt io_format = 0 không tự sinh "Lỗi nhập/xuất") | Có (`fixture32`) |
| TC-T2-10 | Không dùng input T3 hay feedback cho T2 (metamorphic) | FR-ING-05, FR-SEC-01, BR-22, C-07 · SC-42, SC-41 | Compliance | P0 | 32 bài; biến thể: xáo `taxonomy_error` trong file T3; xoá feedback | Chạy T2 cho bản gốc và các biến thể | sha256 predictions T2 giống hệt; 0 cache miss mới | Có (`compliance`, `llm_cache`) |

### 2.6 T3: sinh phản hồi theo level

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-T3-01 | Phản hồi tiếng Việt, không rỗng | FR-T3-01, BR-24, C-01 | Integration | P0 | 32 bài × 4 level (128 lượt, qua cache) | Chạy T3 | 128/128 không rỗng sau strip; detector ngôn ngữ (pin) nhận tiếng Việt ≥ 99% | Có (`llm_cache`) |
| TC-T3-02 | Diagnoser JSON + kiểm grounding | FR-T3-02 · SC-50 | Integration | P1 | Chẩn đoán tổng hợp trỏ dòng 999 và định danh không có trong code; feedback gold EX01-S109 | Chạy grounding | Ca tổng hợp: `grounding_fail`, claim bị loại hoặc confidence bị hạ; grounding pass ≥ 95% trên tập đánh giá nội bộ; gold EX01-S109 bị gắn lệch dòng | Có (`fixture32`, `touches_feedback`) |
| TC-T3-03 | Không bịa dòng, định danh trong phản hồi | FR-T3-02 · SC-50 | Integration | P0 | Output T3 của 32 × 4 | Trích số dòng và định danh dạng code từ phản hồi | Mọi số dòng ≤ số dòng file và nằm trong hàm được nêu; mọi định danh có trong code sinh viên hoặc statement; 0 vi phạm | Có (`llm_cache`) |
| TC-T3-04 | Che thông tin theo level; không ghép `test_report` khi chưa được phép | FR-T3-03, BR-18, BR-23 | Unit | P0 | Chẩn đoán tổng hợp có `fix_sketch` chứa canary và có diff; sample tổng hợp có `test_report` ở T1 | Dựng prompt L1..L4 | Prompt L1/L2: 0 canary của `fix_sketch`, 0 diff; L3: có hướng sửa bằng lời, 0 diff; L4: có code sửa; prompt T3 không chứa `test_report` khi cờ ghép (Q-14) tắt, là mặc định | Có (`unit`) |
| TC-T3-05 | L4 kèm code đã pass test | FR-T3-03 · SC-69 | Integration | P1 | 17 bài EX02 và 15 bài EX01 ở L4 | Sinh L4; build và chạy test công khai trong sandbox | EX02: 100% phản hồi có code thì code build được và pass 8/8 test; bản sửa không pass → không kèm code, dùng template L4. EX01: code qua `-fsyntax-only` với prototype của đề. Card không có `public_tests` hoặc Q-19/Q-14 không cho thực thi: 100% L4 có cờ `unverified`, tỷ lệ có trong summary | Có (`sandbox`, `llm_cache`) |
| TC-T3-06 | Văn phong nói với sinh viên, xưng "em" | FR-T3-07, BR-24 · SC-68 | Integration | P1 | Output 32 × 4 | Kiểm regex; thống kê độ dài | 0 output gọi "SV" hay "sinh viên này" ở ngôi thứ ba; 0 token trong danh sách viết tắt cấm (ví dụ "ko", "dc", "đc"); 100% có "em"; ≥ 90% độ dài nằm trong median ± IQR theo (exam × level), đo trên train đã làm sạch | Có (`llm_cache`) |
| TC-T3-07 | `taxonomy_error = []`: không bịa lỗi | FR-T3-08 · SC-22, SC-69 | Integration | P1 | 4 bài mẫu có `[]` × 4 level | Sinh; chấm bằng judge nội bộ | 16/16: 0 claim lỗi cụ thể; qua guard ở đúng level | Có (`llm_cache`) |
| TC-T3-08 | Level là biến điều khiển | FR-T3-03, FR-T3-06, BR-20 · SC-69 | Integration | P1 | 8 bài (4 EX01, 4 EX02) × 4 level, gồm L1 cho bài có lỗi và L4 | Sinh | 4 output của mỗi bài khác nhau; mỗi output qua guard ở level của nó; L1: không có số dòng; L4: có code; EX01 không bị ép về L2 | Có (`llm_cache`) |
| TC-T3-09 | Dữ liệu huấn luyện T3: đã làm sạch, level cân bằng | FR-T3-05, FR-T3-06 · SC-66, SC-68, SC-49 | Integration | P0 | Bộ SFT/DPO đã dựng | Chạy guard trên mọi target; đọc lineage; đếm level | 0 target vi phạm guard; 100% mẫu có lineage (gốc → lọc / viết lại / gán lại level); bảng số mẫu bị lọc hoặc viết lại theo exam × level; tỷ lệ mỗi level ∈ [15%; 35%]; chi phí teacher có trong nhật ký | Có (`touches_feedback`) |
| TC-T3-10 | Fine-tune T3 bắt buộc (adapter A3) và so sánh với prompting | FR-T3-04, FR-EVL-08, C-02, C-03 · SC-55 | Acceptance | P0 | 2 hướng, cùng split, cùng base `model_id` + revision; manifest run final | Kiểm manifest và adapter; chấm bằng judge nội bộ; paired bootstrap | Adapter A3 tồn tại, sha256 khớp manifest, số tham số trainable > 0, base ≤ 10B chạy local, phần cứng ghi trong manifest; bảng có Δ compliance, Δ chẩn đoán, CI95, p đã hiệu chỉnh Holm | Bán tự động |

### 2.7 GRD: guard compliance cho T3

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-GRD-01 | L1/L2 có code fence → reject + sinh lại | FR-GRD-02, FR-GRD-06, C-17 · SC-67 | Unit | P0 | Generator (test double): lần 1 trả phản hồi có khối code fence C++, lần 2 sạch; level L1 và L2 | Chạy render + guard | Lần 1: violation `code`; gọi lại với seed kế tiếp trong danh sách cố định; output cuối = lần 2 và qua guard; log `retry=1`, `fallback=0` | Có (`unit`) |
| TC-GRD-02 | Bộ phát hiện từ vựng và cú pháp | FR-GRD-02 | Unit | P0 | ≥ 30 case. Dương: dấu backtick, `;`, `{}`, `#include`, `cout <<`, `cin >>`, `i <= n`, `a = b + 1`, `for (int i`. Âm: câu tiếng Việt "dấu chấm phẩy", "vòng lặp for", số "2.5", ngoặc trong câu. Biến thể Unicode: dấu chấm phẩy full-width, zero-width space. Thêm 2 case L3 trên chương trình tổng hợp có diff nội bộ: (c1) phản hồi có 1 dòng code sửa một phần; (c2) phản hồi có code thay đủ mọi dòng `+` của diff | Chạy detector; xác nhận bằng tree-sitter; chạy guard ở L1, L2, L3 | Recall 100% trên case dương ở L1/L2 (0 bỏ sót code nguyên văn); FP ≤ 1 trên case âm; biến thể Unicode được chuẩn hoá rồi bắt. L3 (Lớp 1–2 chỉ là tín hiệu): (c1) pass; (c2) violation `full_solution` | Có (`unit`) |
| TC-GRD-03 | L2 "đổi `<=` thành `<`" → reject | FR-GRD-04, BR-18, C-17 · SC-67 | Unit | P0 | 4 phản hồi tổng hợp: "đổi `<=` thành `<`", "chỉ cần thêm … vào …", "bỏ … là được", "hãy sửa … thành …" | Chạy guard ở L1, L2, L3 | L1/L2: 4/4 violation `fix_instruction` → reject; cùng câu ở L3: pass | Có (`unit`) |
| TC-GRD-04 | Guard gắn đúng các gold vi phạm | FR-GRD-04 · SC-66, SC-67 | Golden | P1 | Feedback gold EX01-S104, EX01-S110 (L2), EX02-S207 (L3) | Chạy guard | S104, S110: vi phạm L2 (`fix_instruction`); S207: "L3 gần như lời giải" | Có (`fixture32`, `touches_feedback`) |
| TC-GRD-05 | Trích nguyên dòng code sinh viên ở L2 theo quy tắc đã chốt | FR-GRD-03 · SC-67 | Unit | P0 | Chương trình tổng hợp; phản hồi L2 (a) trích nguyên 1 dòng có sẵn trong hàm đang nói tới, (b) đưa 1 dòng không có trong hàm đó | Chạy guard với `grd.l2_quote_max_lines` = 1 và = 0 | `= 1`: (a) pass, (b) violation `new_code`. `= 0`: (a) và (b) đều vi phạm. Giá trị mặc định ghi trong Settings theo Q-04 (mặc định: 1, L2 không đưa code mới) | Có (`unit`) |
| TC-GRD-06 | Câu lệnh còn thiếu có ở chỗ khác trong file vẫn bị coi là code mới (kiểu EX01-S106) | FR-GRD-03, C-17 · SC-67 | Unit | P0 | Chương trình tổng hợp: hàm A thiếu câu lệnh duyệt node, hàm B có đúng câu đó; phản hồi L2 về hàm A nêu câu đó; feedback gold EX01-S106 | Chạy guard theo ngữ cảnh; chạy bộ so chuỗi con toàn file làm đối chứng | Guard theo ngữ cảnh: violation `new_code_in_context`; bộ so chuỗi con toàn file: bỏ sót (chứng minh cần so theo ngữ cảnh); gold S106 bị gắn "đề xuất sửa" | Có (`unit`, `fixture32`) |
| TC-GRD-07 | Kiểm leak bằng thực thi (kiểu EX02-S207) | FR-GRD-05 · SC-66 | Integration | P1 | Chương trình Armstrong lỗi do nhóm tự viết (fail 4/8); phản hồi L3 chỉ đúng thao tác sửa; phản hồi L3 chỉ nêu hướng chung; feedback gold EX02-S207 | LLM áp phản hồi vào code; chạy 8 test trong sandbox | Phản hồi chỉ thao tác: từ fail lên 8/8, `leak=True`, reject ở L1–L3; phản hồi hướng chung: `leak=False`; gold S207: `leak=True` (4/8 → 8/8, EDA §4.2); báo tỷ lệ leak theo level; card không có `public_tests` (Q-14) → Lớp 5 tắt, không chặn output | Có (`sandbox`, `llm_cache`) |
| TC-GRD-08 | Hết lượt sinh lại → template đúng level | FR-GRD-06, C-05 · SC-22 | Unit | P0 | Generator (test double) luôn vi phạm; 4 level | Chạy render + guard | Sau đúng 3 lần (danh sách seed cố định, có log) → template (level × nhãn đầu tiên theo thứ tự LS, hoặc `[]`); level của template = level yêu cầu ở 4/4; template qua guard; log `retry=3`, `fallback=1` | Có (`unit`) |
| TC-GRD-09 | Phủ template 4 level × (10 nhãn + `[]`) | FR-GRD-06, FR-GRD-01 · SC-22, SC-69 | Unit | P0 | Thư mục template | Khởi động hệ thống; chạy guard trên từng template | 44/44 template tồn tại, viết tiếng Việt, xưng "em", qua guard ở đúng level; xoá 1 template → khởi động fail ngay và nêu cặp (level, nhãn) thiếu | Có (`unit`) |
| TC-GRD-10 | Điều cấm của L1 và L3 | FR-GRD-01, BR-18, C-17 | Unit | P0 | L1: "lỗi ở dòng 12 do biến i chưa khởi tạo"; L3: phản hồi liệt kê đủ các bước kèm code hoàn chỉnh; L4: code đầy đủ | Chạy guard | L1: violation `location` + `cause`; L3: violation `full_solution`; L4: pass | Có (`unit`) |
| TC-GRD-11 | Đo chất lượng guard trên bộ có nhãn | FR-GRD-01 | Golden | P1 | ≥ 50 phản hồi lấy từ train hoặc tổng hợp (không dùng test), 2 người gán độc lập | Chạy guard; tính κ, recall, precision | κ giữa 2 người được báo; recall vi phạm ≥ 0.90; precision ≥ 0.80 | Bán tự động |
| TC-GRD-12 | 100% output cuối qua guard; đo dưới mức | FR-GRD-06, FR-GRD-07, NFR-OBS-02, BR-19, C-17 · SC-61 | Integration | P0 | Run T3 trên 32 × 4; output hằng "Tốt." ở L2–L4 | Chạy guard trên output cuối; đọc summary | `guard_pass_final` = 100%; summary có tỷ lệ retry, fallback, dưới mức; "Tốt." ở L2–L4 bị gắn dưới mức nhưng không bị chặn | Có (`llm_cache`) |

### 2.8 EVL: metric và đánh giá nội bộ

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-EVL-01 | QWK ví dụ đồ chơi: tham số `labels` | FR-EVL-01 · SC-47 | Unit | P0 | y = [0, 0, 10, 10], p = [1, 1, 9, 9] | Chấm với `labels=list(range(11))` và `labels=None` | 0.976 và 0.600; scorer nội bộ mặc định cho 0.976 | Có (`unit`) |
| TC-EVL-02 | QWK ví dụ EDA trên 17 bài EX02 | FR-EVL-01 · SC-47 | Golden | P1 | 17 bài EX02: dự đoán = gold, riêng 7 bài có total 7 đổi thành 3 | Chấm 2 cách | `labels=None`: 0.931; `labels=0..10`: 0.738 (EDA §2.5) | Có (`fixture32`, `golden`) |
| TC-EVL-03 | macro/micro-F1 tham chiếu; không gian nhãn cố định | FR-EVL-01 · SC-62, SC-18, SC-09 | Golden | P0 | 32 bài; 4 bộ dự đoán: oracle, cả 10 nhãn, luôn `[]`, luật "CE, ngược lại {Lỗi logic}"; toy 1 bài y = ŷ = {"Lỗi logic"} | Chấm ở zd ∈ {0, 1, nan} | Oracle 0.900 / 1.000 / 1.000, micro 1.000. Cả 10 nhãn 0.214 ở cả 3 zd, micro 0.227. Luôn `[]` 0.000 / 0.100 / 0.000. Luật 0.158 / 0.258 / 0.175, micro 0.466 (EDA §3.3). Toy: 0.100 / 1.000 / 1.000 (chứng minh không gian 10 nhãn cố định, không dựng từ y ∪ ŷ) | Có (`fixture32`, `golden`) |
| TC-EVL-04 | Lát cắt suy biến | FR-EVL-01, FR-EVL-02 · SC-20 | Unit | P1 | Lát có σ = 0 (mọi gold và dự đoán bằng 7); nhãn không có positive trong fold | Chấm; chạy bootstrap | QWK trả `nan` + 1 cảnh báo có cấu trúc (không crash, không thay bằng 0); F1 của nhãn theo quy ước zd; bootstrap đếm số resample suy biến; mỗi ô có n | Có (`unit`) |
| TC-EVL-05 | Báo theo exam + pooled, CI bootstrap tái lập, cột baseline | FR-EVL-02, FR-EVL-04 · SC-16, SC-63 | Unit | P0 | Kết quả 32 bài; `cluster_id` | Sinh bảng; bootstrap B = 2000 hai lần cùng seed, một lần khác seed | Mọi bảng có cột exam ∈ {EX01, EX02, pooled}, n, CI_low, CI_high; cùng seed → CI giống tuyệt đối; seed ghi trong manifest; resample theo cụm (S203–S206 luôn cùng vào hoặc cùng ra); bảng exact-match có cột mode (EX02: code_quality 17/17, compilable 14/17); bảng F1 nhãn có mốc 2π/(1+π) | Có (`fixture32`) |
| TC-EVL-06 | Group CV theo cụm near-dup | FR-EVL-03 · SC-43, SC-29, SC-19 | Unit | P0 | 32 bài + `cluster_id`; train tổng hợp có exam mới | Sinh `folds.csv` | 0 cụm bị chia qua 2 fold (S203–S206 cùng fold); stratify theo exam × khoảng total; sha `folds.csv` ghi trong manifest; có exam mới → group theo exam | Có (`unit`) |
| TC-EVL-07 | Loại exemplar kNN/RAG | FR-EVL-03, BR-21 · SC-43, SC-41 | Unit | P0 | Index tổng hợp + 32 bài | Truy vấn exemplar cho từng bài | Tập exemplar không chứa chính bài, bài cùng cụm, bài cùng fold; 0 exemplar mang trường feedback | Có (`unit`) |
| TC-EVL-08 | Quy tắc chọn mô hình theo Δ_min từng task | FR-EVL-05, FR-VAL-05 · SC-16, SC-64 | Unit | P1 | T1: A (CV 0.70, SE 0.05), B (CV 0.72, SE 0.10). T2: Δ macro-F1 nhỏ hơn Δ_min đọc từ `evl.select_min_delta.t2`. T3: Δ điểm judge tương tự, và 1 ca judge chưa qua cổng meta-eval | Gọi `select_model` cho từng task | T1: chọn A (0.675 > 0.670); Δ = 0.02 < 0.1 → "không đủ bằng chứng". T2, T3: Δ < Δ_min → giữ bản đơn giản hơn. Judge chưa qua cổng → điểm judge không được dùng. Nhật ký ghi task, Δ, Δ_min, CI, lý do; hàm không nhận điểm public LB làm input | Có (`unit`) |
| TC-EVL-09 | Judge T3 nội bộ và cổng meta-eval | FR-EVL-06 | Integration | P1 | Judge pin (model, revision, sha prompt); mẫu con của train có 2 người chấm | Chạy judge 2 lần; meta-eval; nếu judge cùng họ generator thì hoán vị nguồn | 2 lần cho điểm giống hệt; báo κ judge–người và κ người–người; cổng đạt khi κ judge–người ≥ 0.6 và thấp hơn κ người–người không quá 0.1 [GT], không đạt thì judge bị gắn "chỉ báo cáo"; judge không phải A3, khác họ generator hoặc có báo Δ self-preference | Bán tự động |
| TC-EVL-10 | So sánh có kiểm soát giữa 2 hướng | FR-EVL-08, C-02, C-03 · SC-55 | Acceptance | P0 | OOF của P-arm và F-arm cho từng task | Kiểm manifest; paired bootstrap hoặc approximate randomization + Holm–Bonferroni | Hai run cùng sha `folds.csv`, cùng sha evidence; manifest hai arm có cùng `model_id` + revision base, chỉ khác adapter/head; P-arm 8B của T1/T2 (nếu có) là dòng phụ ghi "lẫn kích thước"; mỗi task có 1 bảng Δ, CI, p đã hiệu chỉnh; F-arm ≤ 10B chạy local | Có (`golden`) |
| TC-EVL-11 | Hệ suy biến T3 chỉ chạy local; đường Pareto | FR-EVL-07, FR-GRD-07 · SC-61 | Golden | P2 | 3 hệ: "Tốt.", rỗng, template | Chấm compliance và chẩn đoán | "Tốt." có compliance 1.00; có hình Pareto compliance × chẩn đoán; nhật ký nộp có 0 dòng gắn với run suy biến | Có (`golden`) |
| TC-EVL-12 | Ma trận bền vững và độ nhạy với nhãn nghi nhiễu | FR-EVL-09 · SC-44, SC-47 | Golden | P2 | 32 bài | Chạy ma trận (labels × zd × gộp/tách exam); loại EX02-S214, EX01-S108, EX01-S110, EX01-S112 | `rubric_prior` 0.487 → 0.556; sha file nhãn train không đổi; ma trận đủ ô | Có (`fixture32`, `golden`) |
| TC-EVL-13 | `make audit` trước khi áp cứng BR [GT] | FR-EVL-10, C-19 · SC-30, SC-45, SC-24 | Golden | P0 | Bộ mẫu (golden); train chính thức khi có | Chạy `make audit` | `constraints.json` có % thoả BR-01, BR-03..06, BR-14, BR-15, support từng nhãn, cổng D0, số bài in prompt/thừa khoảng trắng, số `actual=''`, giao T2/T3 (train/dev), số bài 0 LOC; **không** có khoá `p1_gate*`, `level*`, `feedback*`; lineage `touched_feedback=false`. `audit_feedback.md` có cổng P1 theo feedback và BR-20, lineage `touched_feedback=true`. Trên mẫu: BR-01 32/32; BR-03 3/3; BR-04 3/3; BR-05 17/17; BR-06 15/15; BR-15 6/6; BR-14 theo chiều: total < 10 ⇒ có nhãn 27/27, `[]` ⇒ total = 10 4/4, total = 10 ⇒ `[]` 4/5 (ngoại lệ S214) → chiều này bị đánh dấu "không áp cứng"; `actual=''` 16; "Lỗi edge case" 0/32; 0 LOC 1 bài | Có (`fixture32`, `golden`) |

### 2.9 VAL: validator và nộp bài

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-VAL-01 | Bộ 12 file hỏng cố ý | FR-VAL-01, C-09 · SC-46 | Unit | P0 | 12 file tổng hợp: (1) thiếu 1 id; (2) thừa id lạ; (3) trùng id; (4) `logic: "2"`; (5) `total_score: 7.0`; (6) nhãn NFD; (7) total ≠ Σ; (8) feedback rỗng hoặc chỉ khoảng trắng; (9) `logic: 5`; (10) nhãn ngoài LS; (11) top-level object khi cấu hình là mảng; (12) UTF-8 có BOM hoặc byte không hợp lệ. Thêm 1 file đúng | Chạy validator | 12/12 bị chặn, mỗi lỗi nêu sample_id, trường, giá trị thấy được; file đúng pass; exit ≠ 0 khi có lỗi | Có (`unit`) |
| TC-VAL-02 | Thiếu, thừa, trùng sample_id | FR-VAL-01, C-09 · SC-46, SC-31 | Unit | P0 | Predictions tổng hợp cho split 32 id: thiếu EX02-S203, EX01-S101 xuất hiện 2 lần, thêm `EX09-S999` | Chạy validator | 3 lỗi, liệt kê đúng 3 id; không tạo file nộp | Có (`unit`) |
| TC-VAL-03 | Nhãn NFD bị chặn trước khi nộp | FR-VAL-01, FR-VAL-04, BR-12 · SC-09 | Unit | P0 | File T2 có đủ 10 nhãn ở dạng NFD (cấu trúc hợp lệ) | Chạy validator; chạy writer rồi validate lại | 10 lỗi, mỗi lỗi nêu sample_id và nhãn; sau khi writer chuẩn hoá NFC → pass | Có (`unit`) |
| TC-VAL-04 | Điểm ngoài miền, float, total ≠ Σ | FR-VAL-01, BR-01, BR-02 · SC-60 | Unit | P0 | 6 ca: total = 11; total = −1; logic = 5; edge_case = 3; total = 7.0; total ≠ Σ | Chạy validator | 6/6 bị chặn, nêu đúng trường | Có (`unit`) |
| TC-VAL-05 | Writer chuẩn hoá; đọc lại khớp | FR-VAL-04, BR-12 | Unit | P0 | Object predictions có numpy int64, chuỗi NFD, ký tự điều khiển, thứ tự bị xáo | Ghi file; đọc lại | Số là `int` (không có `7.0`); chuỗi NFC; `ensure_ascii=False`; 0 ký tự điều khiển (trừ xuống dòng trong feedback); không BOM; thứ tự = thứ tự input; đọc lại == object đã chuẩn hoá | Có (`unit`) |
| TC-VAL-06 | Cổng script chính thức; biến thể định dạng | FR-VAL-02, FR-VAL-01, C-09 · SC-46 | Integration | P0 | Stub script BTC trả reject; cấu hình chưa có script; 3 biến thể định dạng (lồng/phẳng, 1 file/3 file) | Chạy `make submit TASK=t1` | Nội bộ pass mà BTC reject → chặn, exit ≠ 0, ghi cả 2 kết luận; chưa cấu hình script → chặn, nêu đường dẫn thiếu; đổi biến thể chỉ qua Settings, mỗi biến thể có test pass | Có (`unit`) |
| TC-VAL-07 | Nhật ký quota (đơn vị upload) và lý do nộp | FR-VAL-03, FR-VAL-05, C-08 · SC-01, SC-15, SC-64 | Unit | P0 | Nhật ký dùng chung đã có 2 lượt hôm nay; múi giờ quota cấu hình là UTC | Gọi `make submit` lần 3; gọi không có lý do; gọi lý do (i) với Δ < max(Δ_min(task), 1 SE); gọi lúc 06:00 giờ Việt Nam; ở ngày mới, 1 upload gồm 3 file task (`TASK=all`) | Lần 3 bị từ chối ngay tại máy (`QuotaExceeded`), không tạo file; thiếu lý do ∈ {i, ii, iii} → từ chối; lý do (i) không đủ Δ → từ chối, ghi Δ và Δ_min; 06:00 giờ Việt Nam được tính vào ngày UTC trước, kèm cảnh báo gần ranh giới; upload 3 file task tính là 1 lượt (cùng `upload_id`); mỗi dòng nhật ký có timestamp kèm múi giờ, upload_id, task, run_id, sha file, lý do, chỗ ghi điểm public | Có (`unit`) |
| TC-VAL-08 | `make submit` không upload; kiểm run_id | FR-VAL-03 · SC-33 | Unit | P0 | Socket bị chặn bằng monkeypatch; predictions T1 mang run_id cũ, khác manifest hiện tại | Chạy `make submit` | 0 kết nối mạng; file có run_id lệch hoặc thuộc run chưa hoàn tất → từ chối, nêu cả 2 run_id | Có (`unit`) |
| TC-VAL-09 | Dry-run `make predict-private` trên dev | FR-VAL-06, NFR-PERF-01, C-09, C-10 · SC-13, SC-25 | E2E | P0 | Tag release ứng viên; input dev đóng gói như private | Chạy đúng lệnh private ở W6 và W7 | Exit 0; 3 file qua validator nội bộ và script BTC; thời gian từng stage ghi vào summary; cấu hình nhanh ≤ 15 phút | Bán tự động |

### 2.10 REP: tái lập, hạ tầng, hiệu năng

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-REP-01 | Chạy predict 2 lần → diff 0 | NFR-REPRO-01, FR-REP-01, C-06 | Repro | P0 | Cùng máy, cùng cache, cùng tag | Chạy `make reproduce` 2 lần | sha256 của 3 file predictions giống hệt; OOF giống hệt | Có (`repro`) |
| TC-REP-02 | Reproduce từ cache, không GPU, không API, không mạng | FR-REP-03, FR-REP-01, C-06, C-10 · SC-02, SC-32 | Repro | P0 | Máy sạch hoặc container: `CUDA_VISIBLE_DEVICES=""`, mạng tắt, không có biến API key, không Docker | Chạy `make reproduce` | sha predictions = bản tham chiếu đã nộp; bộ đếm trong summary `llm_live_calls`, `f_arm_live_calls`, `judge_live_calls`, `sandbox_runs` (gồm repair T3 và exec-leak GRD) đều = 0 | Có (`repro`) |
| TC-REP-03 | Thiếu artefact → dừng, nêu đúng file | FR-REP-01, NFR-USAB-02 · SC-02 | Repro | P0 | Xoá 1 entry cache LLM và 1 adapter | Chạy `make reproduce` | exit ≠ 0; thông điệp nêu path và sha mong đợi của từng file thiếu; không tự chuyển sang `FULL=1`; 0 call LLM | Có (`repro`) |
| TC-REP-04 | Khác máy, chạy lại từ đầu (`FULL=1`) | NFR-REPRO-02, C-06 · SC-27 | Repro | P1 | Máy thứ hai cùng image digest, GPU khác | Chạy `make reproduce FULL=1` | Chênh lệch tuyệt đối của QWK, macro-F1, compliance ≤ 0.01 (ε chờ Q-09); argmax T1/T2 trùng ≥ 99% số bài; manifest ghi phần cứng của 2 máy | Bán tự động |
| TC-REP-05 | Seed khác → ghi nhận phương sai | NFR-REPRO-03, C-06 · SC-17 | Repro | P1 | Fine-tune 3 seed với cấu hình final | Train 3 lần; suy luận | Bảng mean ± std metric theo seed; model final nằm trong mean ± 2·std; seed sampling khác cho T3 → Δ metric được ghi lại | Bán tự động |
| TC-REP-06 | Manifest, run_id, seed từ Settings, log có cấu trúc | FR-REP-02, FR-REP-04, NFR-OBS-01, C-06 · SC-33 | Unit | P0 | 1 run nhỏ trên bộ tổng hợp | Chạy; đổi 1 mục pin (model revision) rồi chạy lại; tiêm 1 exception | Manifest có: config, git sha, sha data manifest, sha folds, model revision, checksum adapter, hash lockfile, image digest, phiên bản vLLM, seed, phần cứng, thời gian, chi phí; đổi pin → manifest đổi; 100% file trong `runs/`, `oof/` và predictions mang run_id; lint: 0 literal seed ngoài `Settings`; mỗi dòng log là JSON có run_id, sample_id, task, stage, duration, cache hit/miss, retry/fallback; exception được log đúng 1 lần | Có (`unit`) |
| TC-REP-07 | Private chỉ chạy từ tag và working tree sạch | FR-REP-04, FR-VAL-06, C-06 · SC-25 | Unit | P0 | Repo tạm: (a) có file sửa chưa commit; (b) HEAD không phải tag; (c) sạch và đúng tag | Chạy `make predict-private` ở chế độ kiểm | (a), (b): từ chối, exit ≠ 0, nêu điều thấy được; (c): chạy, manifest ghi tag và git sha | Có (`unit`) |
| TC-REP-08 | Không dùng `hash()`; code UB chạy với ASLR tắt | NFR-REPRO-04, C-06 · SC-29, SC-26 | Repro | P0 | 32 bài; EX02-S206 | Sinh `folds.csv` ở 2 tiến trình có `PYTHONHASHSEED` khác nhau; chạy S206 2 lần với `setarch -R` | `folds.csv` và `cluster_id` giống byte; S206: stdout và returncode giống nhau ở 2 lần; lint cấm `hash(` pass | Có (`repro`, `sandbox`) |
| TC-REP-09 | Ghi cache atomic; entry hỏng được coi là miss | FR-REP-03 · SC-14, SC-35 | Unit | P1 | 2 tiến trình ghi cùng một khoá; 1 entry bị cắt cụt (giả lập đĩa đầy) | Ghi đồng thời; đọc lại | 1 entry hợp lệ, 0 file tạm sót lại; entry cụt: checksum fail → miss + 1 warning, giá trị mới được ghi, predictions không đổi | Có (`unit`) |
| TC-REP-10 | Crash giữa chừng rồi chạy lại toàn bộ | FR-SEC-05, FR-REP-03, NFR-REPRO-01 · SC-21, SC-33 | E2E | P1 | Bộ 36 sample tổng hợp hoặc dev | Kill tiến trình ở sample thứ 20; chạy lại cùng lệnh | sha predictions = run liền mạch tham chiếu; audit log có 2 dòng (run hỏng, run hoàn chỉnh) kèm hash input/output; `make submit` chỉ nhận output của run hoàn chỉnh | Có (`e2e`) |
| TC-REP-11 | `make smoke` môi trường; pin phiên bản | FR-REP-05, FR-REP-02 · SC-28, SC-49 | Integration | P1 | GPU-24G, vLLM pin; môi trường lệch lockfile (giả lập) | Chạy `make smoke` | Bản pin: structured output, prompt_logprobs, multi-LoRA, `bad_words` đều pass; môi trường lệch: fail và chặn run; model id teacher dạng alias ("latest") bị `Settings` từ chối lúc khởi động | Có (`gpu`) |
| TC-REP-12 | Chế độ chỉ dùng tín hiệu được cấp; baseline trên CPU | FR-REP-06, NFR-USAB-01, NFR-USAB-02 · SC-03, SC-32 | E2E | P1 | Laptop macOS không GPU, không Docker; bộ mẫu | `make setup`; `make predict SPLIT=sample GRADER_EVD__MODE=given_signals_only` | Ra 3 file qua validator; 0 lần biên dịch hay chạy code; báo cáo có Δmetric so với chế độ đầy đủ; gọi P-arm thì thông báo nêu "không tìm thấy thiết bị CUDA", không phỏng đoán nguyên nhân | Bán tự động |
| TC-REP-13 | Hiệu năng suy luận private | NFR-PERF-01, NFR-PERF-02, NFR-PERF-04, C-03 · SC-13 | Perf | P0 | 1 GPU 24 GB; input dev đóng gói như private (≈ 36 bài [GT]); image đã build sẵn | Chạy `make predict-private` với cấu hình final và cấu hình nhanh | Final ≤ 60 phút, nhanh ≤ 15 phút (không tính build image); throughput (bài/phút) từng task có trong summary; p95 prompt < 2.5k token, 0 prompt > 8k; VRAM đỉnh khi fine-tune ≤ 24 GB, ghi trong manifest | Bán tự động |
| TC-REP-14 | Chất lượng mã; mỗi luồng một lệnh | NFR-MAINT-01, NFR-MAINT-02, NFR-MAINT-03, NFR-USAB-01, C-10 | CI | P0 | Repo | CI chạy ruff, mypy, pytest + coverage; kiểm README | ruff và mypy 0 lỗi; coverage ≥ 80% cho scorer, `aggregate()`, validator, guard, loader, firewall; README có quyết định sync/async (Q-22) và các lệnh `make setup`, `audit`, `train`, `predict`, `reproduce`, `submit`; một thành viên khác chạy được baseline (gate W1) | Có (CI) |

### 2.11 SEC: bảo mật, riêng tư, tuân thủ

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-SEC-01 | Firewall CI: quét chuỗi feedback trong prompt và feature T1/T2 | FR-SEC-01, BR-21, C-07 · SC-41 | Compliance | P0 | Prompt + feature T1/T2 của run; feedback train làm haystack; 1 prompt T1 tổng hợp bị chèn 1 feedback | Chạy scanner (chuỗi con ≥ 30 ký tự, trừ phần trùng code sinh viên) | Run thật: 0 hit; prompt bị chèn: test đỏ, nêu sample_id và vị trí; chèn đoạn 29 ký tự: xanh (biên); chuỗi trùng code sinh viên: không tính | Có (`compliance`) |
| TC-SEC-02 | Bảng lineage chặn artefact đã chạm feedback | FR-SEC-01, BR-21, C-07 · SC-41, SC-55 | Compliance | P0 | Bảng lineage; cấu hình T1 tổng hợp dùng output của adapter T3 làm feature | Chạy bước kiểm của CI | Adapter T3, dữ liệu SFT T3, output judge có `touched_feedback=True`; cấu hình T1/T2 tham chiếu chúng → CI đỏ; bảng lineage được xuất vào báo cáo | Có (`compliance`) |
| TC-SEC-03 | Pre-commit, gitleaks, lịch sử git, repo private | FR-SEC-03, NFR-PRIV-01, NFR-SEC-02, C-11, C-14 · SC-37, SC-39 | Security | P0 | Repo tạm | Commit thử `x.cpp`, `task1_grading.json`, `cache/…`, `adapter_model.safetensors`, một chuỗi giống API key; quét lịch sử; gọi API của host repo | 5/5 commit bị chặn; gitleaks 0 phát hiện trên `main`; lịch sử git có 0 file dữ liệu hay weights; repo có `private=true` | Có (CI) |
| TC-SEC-04 | Allowlist egress, trần chi phí, retry API | FR-SEC-04, NFR-COST-01, NFR-PRIV-02, C-04, C-11 · SC-38, SC-53, SC-12 | Security | P0 | Server HTTP giả lập chạy local; allowlist chỉ chứa host giả lập | Gọi host ngoài allowlist; gọi host hợp lệ; đặt trần 0.01 USD rồi vượt; server trả 429 liên tục | Ngoài allowlist: `EgressBlocked` nêu host, 0 byte gửi đi; hợp lệ: log đích, token vào/ra, chi phí; vượt trần: `CostCapExceeded`, dừng; 429: retry có backoff, dừng sau N lần, giữ status và body gốc, khoá đã cache không bị gọi lại | Có (`unit`) |
| TC-SEC-05 | Suy luận private chạy offline; không đẩy adapter lên hub | FR-SEC-04, FR-SEC-03, NFR-PRIV-02 · SC-53, SC-39 | E2E | P0 | Dry-run trên dev trong container `--network none`; image app | Chạy `make predict-private`; assert biến môi trường image; gọi `huggingface_hub.upload_file` trong image; CI grep `src/` | Exit 0; 3 file hợp lệ; nhật ký egress rỗng (0 call API thương mại); image có `HF_HUB_OFFLINE=1`; `upload_file` raise; 0 hit `push_to_hub`, `upload_folder`, `create_repo` trong `src/` | Bán tự động |
| TC-SEC-06 | Cô lập input test; toàn vẹn output | FR-SEC-05, C-12, C-05 · SC-40, SC-21 | Compliance | P0 | Archive "test" tổng hợp (không phải test chính thức) | Chạy `make predict SPLIT=test`; chạy `make predict-private`; tiêm 1 bài sai schema có canary trong archive; sửa 1 byte output rồi `make submit` | Lệnh khác `predict-private` bị từ chối đọc archive (`TestIsolationError`); `predict-private` giải nén vào tmp, xoá sau run, ghi hash input/output vào audit log; exception trong `PrivateRun` chỉ mang kiểu, độ dài, sha1 của giá trị, 0 canary trong message, log và stdout; output bị sửa tay → submit từ chối (sha không khớp run nào) | Có (`compliance`) |
| TC-SEC-07 | Không dùng mã SV làm feature; PII scan chỉ đếm | FR-SEC-06, BR-25, C-13 · SC-52 | Compliance | P0 | Schema feature; bộ tổng hợp có chuỗi giống email và mã số sinh viên | Kiểm schema; chạy PII scan | 0 cột khớp mã SV (`S\d{3}`, `student_id`); PII scan chỉ in số đếm theo loại, 0 chuỗi khớp bị in ra | Có (`compliance`) |
| TC-SEC-08 | Log và tài liệu không chứa code sinh viên hay feedback | FR-SEC-06, NFR-PRIV-03, BR-25, C-11 · SC-51 | Privacy | P0 | Bộ tổng hợp có canary trong code và feedback; thư mục báo cáo và slide | Chạy pipeline ở mức INFO; chạy linter tài liệu (so n-gram ≥ 8 token với code sinh viên và feedback) | 0 canary trong log INFO; linter tài liệu 0 hit; tài liệu chỉ tham chiếu sample_id | Có (`privacy`) |
| TC-SEC-09 | Dữ liệu ngoài workspace; tracking offline; văn bản luồng dữ liệu | NFR-PRIV-02, C-11 · SC-38, SC-39 | Checklist | P0 | Máy từng thành viên | Kiểm theo checklist ở W1 và trước mỗi mốc | data-root không nằm dưới thư mục repo; W&B ở chế độ offline hoặc self-host; adapter chỉ chuyển qua kênh nội bộ, có checksum; văn bản "dữ liệu được đi đâu" có xác nhận của giảng viên | Thủ công |
| TC-SEC-10 | Bộ prompt injection tổng hợp | FR-SEC-07 · SC-05 | Security | P1 | ≥ 20 chương trình tự viết có comment injection | Chạy T1, T3 | Tỷ lệ đổi total T1 và tỷ lệ T3 khen sai được ghi vào model card; cổng CE và feature tất định không đổi ở 100% bài | Có (`llm_cache`) |

### 2.12 RPT: báo cáo, ablation, artefact

| ID | Tiêu đề | Trace (FR/NFR/BR · SC) | Loại | Ưu tiên | Tiền điều kiện / Fixture | Bước | Kết quả mong đợi | Tự động? |
|---|---|---|---|---|---|---|---|---|
| TC-RPT-01 | Bảng ablation sinh tự động từ log | FR-RPT-01, FR-T1-08, FR-T1-09, FR-T2-08, C-19 | Ablation | P0 | `runs/*/manifest.json` và `summary.md` | Chạy `make ablation` | Có bảng build-up và leave-one-out; mỗi thành phần lõi (cổng CE, ràng buộc rubric, policy feature/cứng, decoder, ngưỡng nhãn, ràng buộc chéo, guard, làm sạch gold, level-swap) có ≥ 1 dòng kèm CI; dòng fine-tune có 3 seed; có dòng nhóm feature T1, stacking, augmentation (nếu làm); sửa tay bảng → hash lệch, CI đỏ | Có (`golden`) |
| TC-RPT-02 | Mọi số trong báo cáo truy được về run_id | FR-RPT-01, FR-RPT-02, NFR-OBS-02 | Compliance | P1 | Nguồn báo cáo (markdown hoặc LaTeX) | Chạy linter báo cáo | Mỗi số trong bảng kết quả gắn run_id có trong `runs/` và khớp `summary.md` (theo làm tròn); 0 số mồ côi | Có (CI) |
| TC-RPT-03 | Phân tích lỗi theo lát cắt | FR-RPT-02, C-19 · SC-65, SC-70, SC-34 | Acceptance | P1 | OOF final | Chạy script phân tích lỗi | ≥ 5 thẻ lỗi "kiểu bài nào, vì sao"; có lát exam, khoảng total, CE, số nhãn, level, cụm near-dup, có/không có test, toolchain/template; Σ đóng góp loss = loss tổng (± 1e-6); có oracle-fix | Bán tự động |
| TC-RPT-04 | Bảng phần cứng và chi phí | FR-RPT-03, NFR-COST-01, NFR-COST-02, C-03, C-04 · SC-49 | Acceptance | P1 | Nhật ký egress; manifest | Sinh bảng | GPU, VRAM, GPU-giờ theo run, tổng ≤ ngân sách (Q-23); tổng chi phí API = tổng nhật ký; khớp hoá đơn ± 5% | Bán tự động |
| TC-RPT-05 | Hồ sơ bonus tái tạo được | FR-RPT-04, C-19 · SC-44, SC-42 | Acceptance | P1 | Hồ sơ bonus | Chạy từng lệnh `make audit-*` | ≥ 2 phát hiện đủ 8 bước; số liệu chạy lại khớp hồ sơ (diff 0); đã chạy trên train chính thức (+ dev nếu có nhãn), log cho thấy 0 lần `run-eda.sh` hay `make audit-*` trên input test; có mục loại giả thuyết "chấm theo ý định" trước khi claim nhãn nhiễu | Bán tự động |
| TC-RPT-06 | Model card, khai báo LLM, bảng đóng góp, learning curve | FR-RPT-05, FR-RPT-06, FR-RPT-07, C-15, C-20 · SC-04, SC-54, SC-70, SC-17 | Checklist | P1 | Bản báo cáo W7 | Review theo checklist | Model card ≤ 2 trang đủ mục (dữ liệu, giới hạn, rủi ro, yêu cầu người duyệt, bảng khác biệt hệ thống (1)/(2), tỷ lệ injection, thiên lệch toolchain/template, không tự gửi L4 khi bài tập còn mở); mục khai báo công cụ LLM ở cuối báo cáo, có dòng dùng AI agent trên dữ liệu (ngày, phạm vi); bảng đóng góp khớp lịch sử commit và có xác nhận cả nhóm; learning curve 10/25/50/100% có CI | Thủ công |
| TC-RPT-07 | Báo cáo 10–15 trang đủ 6 mục, slide và diễn tập | FR-RPT-08, C-10, C-19 · SC-54 | Checklist | P1 | Bản báo cáo và slide W7; `report_check.py` | Chạy `report_check.py`; review biên bản | Đủ 6 heading của đề §8(2) (dữ liệu & EDA, các hướng, ablation, phân tích lỗi, chi phí & khả năng triển khai, hạn chế); số trang ∈ [10; 15] không tính phụ lục (Q-20); slide cho 15 phút + 10 phút hỏi đáp; biên bản ≥ 2 lần diễn tập; mỗi mục có 2 người review không viết mục đó | Thủ công |

---

## 3. Ma trận truy vết

### 3.1 FR → TC (88 FR; MoSCoW theo SRS §4)

| FR | MoSCoW | TC |
|---|---|---|
| FR-ING-01 | M | TC-ING-01, TC-ING-02 |
| FR-ING-02 | M | TC-ING-07, TC-ING-08 |
| FR-ING-03 | M | TC-ING-05, TC-ING-06 |
| FR-ING-04 | M | TC-ING-01, TC-ING-10 |
| FR-ING-05 | M | TC-ING-09, TC-T1-09, TC-T2-10 |
| FR-ING-06 | M | TC-ING-03, TC-ING-04, TC-EVD-01 |
| FR-EVD-01 | M | TC-EVD-11 |
| FR-EVD-02 | M | TC-EVD-05, TC-EVD-07 |
| FR-EVD-03 | M | TC-EVD-02, TC-EVD-04, TC-EVD-07 |
| FR-EVD-04 | S | TC-EVD-01, TC-EVD-02, TC-EVD-03, TC-EVD-05, TC-EVD-06 |
| FR-EVD-05 | M | TC-EVD-08, TC-POL-08 |
| FR-EVD-06 | M | TC-EVD-09, TC-ING-08 |
| FR-EVD-07 | M | TC-EVD-10 |
| FR-EVD-08 | W | TC-EVD-15 (test âm: harness phải tắt) |
| FR-POL-01 | M | TC-POL-01, TC-POL-02, TC-POL-03, TC-POL-04 |
| FR-POL-02 | M | TC-POL-05, TC-POL-06 |
| FR-POL-03 | M | TC-POL-07 |
| FR-POL-04 | M | TC-POL-08, TC-POL-07 |
| FR-POL-05 | C | TC-POL-09 |
| FR-T1-01 | M | TC-T1-01, TC-T1-11 |
| FR-T1-02 | M | TC-T1-03, TC-T1-04, TC-T1-05, TC-T1-06 |
| FR-T1-03 | M | TC-T1-02, TC-T1-03, TC-EVD-01 |
| FR-T1-04 | M | TC-T1-10, TC-T1-11 |
| FR-T1-05 | M | TC-T1-12 |
| FR-T1-06 | M | TC-T1-08 |
| FR-T1-07 | M | TC-T1-07 |
| FR-T1-08 | S | TC-EVD-07, TC-RPT-01 |
| FR-T1-09 | S | TC-RPT-01 |
| FR-T2-01 | M | TC-T2-01, TC-T2-02 |
| FR-T2-02 | M | TC-T2-03, TC-T2-04 |
| FR-T2-03 | M | TC-T2-08 |
| FR-T2-04 | M | TC-T2-09 |
| FR-T2-05 | M | TC-T2-07 |
| FR-T2-06 | M | TC-T2-05 |
| FR-T2-07 | M | TC-T2-06 |
| FR-T2-08 | C | TC-RPT-01 (dòng augmentation, nếu làm) |
| FR-T3-01 | M | TC-T3-01 |
| FR-T3-02 | M | TC-T3-02, TC-T3-03 |
| FR-T3-03 | M | TC-T3-04, TC-T3-05, TC-T3-08 |
| FR-T3-04 | M | TC-T3-10 |
| FR-T3-05 | M | TC-T3-09 |
| FR-T3-06 | S | TC-T3-08, TC-T3-09 |
| FR-T3-07 | S | TC-T3-06 |
| FR-T3-08 | S | TC-T3-07 |
| FR-GRD-01 | M | TC-GRD-09, TC-GRD-10, TC-GRD-11 |
| FR-GRD-02 | M | TC-GRD-01, TC-GRD-02 |
| FR-GRD-03 | M | TC-GRD-05, TC-GRD-06 |
| FR-GRD-04 | M | TC-GRD-03, TC-GRD-04 |
| FR-GRD-05 | S | TC-GRD-07 |
| FR-GRD-06 | M | TC-GRD-01, TC-GRD-08, TC-GRD-09, TC-GRD-12 |
| FR-GRD-07 | S | TC-GRD-12, TC-EVL-11 |
| FR-EVL-01 | M | TC-EVL-01, TC-EVL-02, TC-EVL-03, TC-EVL-04, TC-T1-08 |
| FR-EVL-02 | M | TC-EVL-04, TC-EVL-05 |
| FR-EVL-03 | M | TC-EVL-06, TC-EVL-07 |
| FR-EVL-04 | M | TC-EVL-05 |
| FR-EVL-05 | M | TC-EVL-08 |
| FR-EVL-06 | M | TC-EVL-09 |
| FR-EVL-07 | S | TC-EVL-11 |
| FR-EVL-08 | M | TC-EVL-10, TC-T3-10 |
| FR-EVL-09 | S | TC-EVL-12 |
| FR-EVL-10 | M | TC-EVL-13, TC-T1-06, TC-T2-04 |
| FR-VAL-01 | M | TC-VAL-01, TC-VAL-02, TC-VAL-03, TC-VAL-04, TC-VAL-06 |
| FR-VAL-02 | M | TC-VAL-06 |
| FR-VAL-03 | M | TC-VAL-07, TC-VAL-08 |
| FR-VAL-04 | M | TC-VAL-03, TC-VAL-05 |
| FR-VAL-05 | S | TC-VAL-07, TC-EVL-08 |
| FR-VAL-06 | M | TC-VAL-09, TC-REP-07 |
| FR-REP-01 | M | TC-REP-01, TC-REP-02, TC-REP-03 |
| FR-REP-02 | M | TC-REP-06, TC-REP-11 |
| FR-REP-03 | M | TC-REP-02, TC-REP-09, TC-REP-10, TC-EVD-12 |
| FR-REP-04 | M | TC-REP-06, TC-REP-07 |
| FR-REP-05 | S | TC-REP-11 |
| FR-REP-06 | S | TC-REP-12 |
| FR-SEC-01 | M | TC-SEC-01, TC-SEC-02, TC-T1-09, TC-T2-10 |
| FR-SEC-02 | M | TC-EVD-13, TC-EVD-14 |
| FR-SEC-03 | M | TC-SEC-03, TC-SEC-05 |
| FR-SEC-04 | M | TC-SEC-04, TC-SEC-05 |
| FR-SEC-05 | M | TC-SEC-06, TC-REP-10, TC-ING-10 |
| FR-SEC-06 | M | TC-SEC-07, TC-SEC-08 |
| FR-SEC-07 | S | TC-SEC-10 |
| FR-RPT-01 | M | TC-RPT-01, TC-RPT-02 |
| FR-RPT-02 | M | TC-RPT-02, TC-RPT-03 |
| FR-RPT-03 | M | TC-RPT-04 |
| FR-RPT-04 | M | TC-RPT-05 |
| FR-RPT-05 | M | TC-RPT-06 |
| FR-RPT-06 | S | TC-RPT-06 |
| FR-RPT-07 | M | TC-RPT-06 |
| FR-RPT-08 | M | TC-RPT-07 |

### 3.2 NFR → TC (22 NFR)

| NFR | TC |
|---|---|
| NFR-REPRO-01 | TC-REP-01, TC-REP-10 |
| NFR-REPRO-02 | TC-REP-04 |
| NFR-REPRO-03 | TC-REP-05 |
| NFR-REPRO-04 | TC-EVD-10, TC-REP-08 |
| NFR-PERF-01 | TC-REP-13, TC-VAL-09 |
| NFR-PERF-02 | TC-REP-13 |
| NFR-PERF-03 | TC-EVD-04, TC-EVD-12 |
| NFR-PERF-04 | TC-REP-13 |
| NFR-SEC-01 | TC-EVD-13, TC-EVD-14 |
| NFR-SEC-02 | TC-EVD-13, TC-SEC-03 |
| NFR-PRIV-01 | TC-SEC-03 |
| NFR-PRIV-02 | TC-SEC-04, TC-SEC-05, TC-SEC-09 |
| NFR-PRIV-03 | TC-SEC-08 |
| NFR-COST-01 | TC-SEC-04, TC-RPT-04 |
| NFR-COST-02 | TC-RPT-04 |
| NFR-MAINT-01 | TC-REP-14 |
| NFR-MAINT-02 | TC-REP-14 |
| NFR-MAINT-03 | TC-REP-14 |
| NFR-OBS-01 | TC-REP-06 |
| NFR-OBS-02 | TC-GRD-12, TC-RPT-02 |
| NFR-USAB-01 | TC-REP-12, TC-REP-14 |
| NFR-USAB-02 | TC-ING-02, TC-ING-06, TC-REP-03, TC-REP-12 |

### 3.3 Độ phủ và khoảng trống

| Nhóm | Số yêu cầu | Có ≥ 1 TC | Chưa có TC |
|---|---|---|---|
| FR Must | 70 | 70 | **0** |
| FR Should | 15 | 15 | 0 |
| FR Could | 2 | 2 | 0 |
| FR Won't | 1 (FR-EVD-08) | 1 (test âm) | 0 |
| NFR | 22 | 22 | **0** |
| BR | 25 | 25 (§3.4) | 0 |
| SC Critical / High | 13 / 31 | 13 / 31 | **0** |
| SC Medium / Low | 24 / 2 | 24 / 2 | 0 |

- Mọi FR Must có ít nhất một TC mức P0 hoặc P1. Các TC chỉ ở P2 thuộc FR Could/Won't hoặc Should (FR-POL-05, FR-EVD-08, FR-EVL-07, FR-EVL-09).
- Mọi SC Critical có ít nhất một TC P0. Mọi SC High có ít nhất một TC P0 hoặc P1.
- Phủ yếu (chỉ 1 TC, và TC đó là Bán tự động hoặc Thủ công): FR-T1-05, FR-T3-04 (TC-T3-10 đã nâng P0, có kiểm tự động sha adapter), FR-EVL-06, FR-RPT-03..08. Đây là yêu cầu về artefact báo cáo, kiểm bằng review ở W4 và W7.

### 3.4 BR → TC

| BR | TC | BR | TC |
|---|---|---|---|
| BR-01 | TC-T1-01, TC-VAL-04, TC-EVL-13 | BR-14 | TC-T2-05, TC-EVL-13 |
| BR-02 | TC-T1-01, TC-T1-03, TC-VAL-04 | BR-15 | TC-ING-03, TC-T1-02, TC-T2-03 |
| BR-03 | TC-T1-02, TC-EVL-13 | BR-16 | TC-T2-09 |
| BR-04 | TC-T1-02, TC-EVL-13 | BR-17 | TC-EVD-08 |
| BR-05 | TC-T1-04, TC-T1-06, TC-EVL-13 | BR-18 | TC-T3-04, TC-GRD-03, TC-GRD-10 |
| BR-06 | TC-T1-05, TC-EVL-13 | BR-19 | TC-GRD-12, TC-EVL-11 |
| BR-07 | TC-POL-01, TC-POL-05, TC-POL-07 | BR-20 | TC-T3-08, TC-EVL-13 |
| BR-08 | TC-POL-06, TC-POL-07 | BR-21 | TC-ING-09, TC-T1-09, TC-SEC-01, TC-SEC-02, TC-EVL-07 |
| BR-09 | TC-POL-05 | BR-22 | TC-ING-10, TC-T2-10 |
| BR-10 | TC-EVD-05, TC-EVD-06 | BR-23 | TC-T3-04 |
| BR-11 | TC-ING-04, TC-EVD-01 | BR-24 | TC-T3-01, TC-T3-06 |
| BR-12 | TC-T2-02, TC-VAL-03, TC-VAL-05 | BR-25 | TC-SEC-07, TC-SEC-08 |
| BR-13 | TC-T2-01 | | |

### 3.5 SC → TC (70 kịch bản)

| SC | Mức | TC |
|---|---|---|
| SC-02 | Critical | TC-REP-02, TC-REP-03 |
| SC-09 | Critical | TC-T2-02, TC-VAL-03, TC-EVL-03 |
| SC-25 | Critical | TC-REP-07, TC-VAL-09 |
| SC-36 | Critical | TC-EVD-13, TC-EVD-14 |
| SC-37 | Critical | TC-SEC-03 |
| SC-38 | Critical | TC-SEC-04, TC-SEC-09 |
| SC-39 | Critical | TC-SEC-03, TC-SEC-05, TC-SEC-09 |
| SC-40 | Critical | TC-SEC-06 |
| SC-41 | Critical | TC-ING-09, TC-T1-09, TC-T2-10, TC-EVL-07, TC-SEC-01, TC-SEC-02 |
| SC-42 | Critical | TC-ING-10, TC-T2-10, TC-RPT-05 |
| SC-46 | Critical | TC-VAL-01, TC-VAL-02, TC-VAL-06 |
| SC-52 | Critical | TC-SEC-07 |
| SC-53 | Critical | TC-SEC-04, TC-SEC-05 |
| SC-04 | High | TC-POL-07, TC-RPT-06 |
| SC-05 | High | TC-SEC-10 |
| SC-06 | High | TC-ING-03, TC-EVD-09, TC-T1-02, TC-T2-03 |
| SC-11 | High | TC-EVD-04 |
| SC-13 | High | TC-VAL-09, TC-REP-13 |
| SC-16 | High | TC-EVL-05, TC-EVL-08 |
| SC-17 | High | TC-T1-12, TC-REP-05, TC-RPT-06 |
| SC-18 | High | TC-T2-06, TC-EVL-03 |
| SC-19 | High | TC-POL-02, TC-T1-03, TC-EVL-06 |
| SC-21 | High | TC-REP-10, TC-SEC-06 |
| SC-22 | High | TC-T3-07, TC-GRD-08, TC-GRD-09 |
| SC-23 | High | TC-POL-03, TC-POL-04 |
| SC-26 | High | TC-EVD-01, TC-EVD-02, TC-EVD-03, TC-EVD-11, TC-EVD-12, TC-REP-08 |
| SC-27 | High | TC-REP-04 |
| SC-28 | High | TC-T1-10, TC-REP-11 |
| SC-29 | High | TC-EVD-10, TC-EVL-06, TC-REP-08 |
| SC-30 | High | TC-ING-04, TC-T1-06, TC-T2-04, TC-EVL-13 |
| SC-31 | High | TC-T1-11, TC-VAL-02 |
| SC-32 | High | TC-REP-02, TC-REP-12 |
| SC-43 | High | TC-EVD-10, TC-EVL-06, TC-EVL-07 |
| SC-47 | High | TC-EVL-01, TC-EVL-02, TC-EVL-12 |
| SC-48 | High | TC-EVD-01, TC-EVD-15 |
| SC-51 | High | TC-SEC-08 |
| SC-55 | High | TC-T3-10, TC-EVL-10, TC-SEC-02 |
| SC-56 | High | TC-POL-06, TC-POL-07, TC-POL-09 |
| SC-58 | High | TC-EVD-05, TC-EVD-06 |
| SC-64 | High | TC-EVL-08, TC-VAL-07 |
| SC-66 | High | TC-T3-09, TC-GRD-04, TC-GRD-07 |
| SC-67 | High | TC-GRD-01, TC-GRD-03, TC-GRD-04, TC-GRD-05, TC-GRD-06 |
| SC-69 | High | TC-T3-05, TC-T3-07, TC-T3-08, TC-GRD-09 |
| SC-70 | High | TC-RPT-03, TC-RPT-06 |
| SC-01 | Medium | TC-VAL-07 |
| SC-03 | Medium | TC-REP-12 |
| SC-07 | Medium | TC-ING-08 |
| SC-08 | Medium | TC-ING-07, TC-ING-08 |
| SC-12 | Medium | TC-SEC-04 |
| SC-14 | Medium | TC-EVD-12, TC-REP-09 |
| SC-20 | Medium | TC-EVL-04 |
| SC-24 | Medium | TC-ING-05, TC-EVL-13 |
| SC-33 | Medium | TC-VAL-08, TC-REP-06, TC-REP-10 |
| SC-34 | Medium | TC-T2-05, TC-RPT-03 |
| SC-35 | Medium | TC-REP-09 |
| SC-44 | Medium | TC-EVL-12, TC-RPT-05 |
| SC-45 | Medium | TC-ING-01, TC-ING-02, TC-ING-05, TC-ING-06, TC-EVL-13 |
| SC-49 | Medium | TC-T3-09, TC-REP-11, TC-RPT-04 |
| SC-50 | Medium | TC-ING-07, TC-T3-02, TC-T3-03 |
| SC-54 | Medium | TC-RPT-06, TC-RPT-07 |
| SC-57 | Medium | TC-EVD-08, TC-POL-01, TC-POL-05, TC-POL-08 |
| SC-59 | Medium | TC-T1-02, TC-T1-03 |
| SC-60 | Medium | TC-T1-01, TC-T1-04, TC-T1-07, TC-VAL-04 |
| SC-61 | Medium | TC-GRD-12, TC-EVL-11 |
| SC-62 | Medium | TC-T2-07, TC-EVL-03 |
| SC-63 | Medium | TC-T1-07, TC-T1-08, TC-EVL-05 |
| SC-65 | Medium | TC-EVD-07, TC-RPT-03 |
| SC-68 | Medium | TC-T3-06, TC-T3-09 |
| SC-10 | Low | TC-ING-04 |
| SC-15 | Low | TC-VAL-07 |

**Kết hợp chiều (scenario §3)** được kiểm bằng TC tích hợp sẵn có: SC-21 + SC-40 → TC-REP-10 + TC-SEC-06 (diễn tập W8); SC-30 + SC-19 → TC-T1-03 + TC-EVL-13; SC-09 + SC-46 → TC-VAL-03; SC-41 + SC-55 → TC-SEC-02; SC-22 + SC-69 → TC-GRD-09 + TC-T3-08.

---

## 4. Bộ smoke P0 và gate trước private

### 4.1 Smoke trước mỗi lượt nộp leaderboard (`make smoke-submit TASK=…`)

- **Người chạy:** người giữ quyền nộp (FR-VAL-05), trên CI-local hoặc máy có data-root.
- **Thời gian:** ≤ 10 phút trên CPU; mọi bước LLM và sandbox đọc từ cache.
- **Quy tắc:** chạy tuần tự, dừng ở lỗi đầu tiên. Có bất kỳ fail nào thì **không nộp** (file sai vẫn mất lượt, C-09). Pass hết thì in đường dẫn file + sha256 để người nộp upload thủ công.
- Các TC áp lên predictions dev chạy ở **chế độ bất biến**: kiểm ràng buộc trên output, không cần gold.

| Bước | TC (đều là P0) | Áp cho task | Kiểm gì |
|---|---|---|---|
| 1. Tĩnh | TC-REP-14, TC-SEC-03 | Cả 3 | ruff, mypy, `pytest -m "unit and p0"`; gitleaks, pre-commit trên HEAD; tree sạch |
| 2. Firewall | TC-SEC-01, TC-SEC-02, TC-ING-09; TC-T1-09 (t1); TC-T2-10 (t2) | t1, t2 (T3 chỉ bước SEC-02) | 0 chuỗi feedback trong prompt/feature; lineage sạch; metamorphic: đổi feedback, predictions không đổi |
| 3. Scorer | TC-EVL-01, TC-EVL-03 | Cả 3 | Bản sao scorer chưa bị hồi quy |
| 4. Bất biến output T1 | TC-T1-01, TC-T1-02, TC-T1-04, TC-T1-05 | t1 | Miền, số nguyên, total = Σ; bài có `log_has_error` ra đúng vector cổng của exam; ràng buộc rubric |
| 5. Bất biến output T2 | TC-T2-01, TC-T2-02, TC-T2-03, TC-T2-05, TC-T2-06 | t2 | Nhãn NFC đúng LS, thứ tự LS; "Lỗi biên dịch" ⇔ `log_has_error`; ràng buộc chéo (nếu bật); 0 "Lỗi edge case" thiếu bằng chứng |
| 6. Bất biến output T3 | TC-T3-01, TC-GRD-09, TC-GRD-12 | t3 | Không rỗng, tiếng Việt; đủ 44 template; 100% output cuối qua guard |
| 7. Tái lập | TC-REP-01, TC-REP-06 | Cả 3 | Chạy lại từ cache cho sha giống hệt; manifest đủ, run_id khớp |
| 8. Validator | TC-VAL-01 (tự kiểm validator), TC-VAL-02..TC-VAL-05 trên file thật, TC-VAL-06 (script BTC), TC-VAL-08 | Cả 3 | Schema, id, nhãn, miền, chuẩn hoá; hai validator cùng pass; run_id đúng, không upload |
| 9. Quota | TC-VAL-07 | Cả 3 | Chưa quá 2 lượt/ngày; có lý do ∈ {i, ii, iii}; ghi nhật ký |

### 4.2 Gate trước khi chạy private (`make gate-private`, W8)

| Điều kiện | TC |
|---|---|
| Smoke §4.1 pass cho cả t1, t2, t3 trên tag `v1.0` | Như §4.1 |
| Tái lập từ cache trên máy sạch; thiếu artefact thì dừng đúng | TC-REP-02, TC-REP-03 |
| Kết quả khác máy W6, W7 trong ε; diễn tập crash rồi chạy lại | TC-REP-04, TC-REP-10 |
| Chỉ chạy từ tag, tree sạch | TC-REP-07 |
| Dry-run đúng lệnh private trên dev, qua 2 validator; thời gian trong ngưỡng | TC-VAL-09, TC-REP-13 |
| Chạy offline, 0 call API; cô lập input test; output thuộc đúng 1 run | TC-SEC-05, TC-SEC-06 |
| Sandbox chặn mạng, secret, fork bomb, ghi ngoài tmp, output lớn | TC-EVD-13, TC-EVD-14 |
| Checklist dữ liệu ngoài workspace, tracking offline, văn bản luồng dữ liệu | TC-SEC-09 |
| 0 SC Critical còn mở; 100% P0 pass; P1 ≥ 95% pass, phần còn lại có waiver ghi lý do | §3.5 |

---

## 5. Thống kê

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

**Lịch đưa TC vào CI** (bám BS §9):

| Tuần | TC phải xanh |
|---|---|
| W1 | TC-ING-*, TC-VAL-01..TC-VAL-08, TC-EVL-01..TC-EVL-03, TC-SEC-01..TC-SEC-04, TC-SEC-07, TC-SEC-09, TC-REP-14 (baseline cả 3 task nộp được qua smoke) |
| W2 | TC-EVD-*, TC-EVL-04..TC-EVL-08, TC-EVL-13, TC-T1-08, TC-T2-03 |
| W3 | TC-POL-*, TC-GRD-01..TC-GRD-10, TC-GRD-12, TC-T1-01..TC-T1-07, TC-T2-01..TC-T2-07, TC-REP-01 (gate diff = 0) |
| W4 (MVP) | Mọi P0 trừ các TC miễn ở §1.4 |
| W5 | TC-GRD-07, TC-GRD-11, TC-T3-09, TC-RPT-05 |
| W6–W7 | TC-REP-02..TC-REP-05, TC-REP-10..TC-REP-13, TC-VAL-09, TC-SEC-05, TC-T3-10, TC-RPT-*, TC-EVL-09..TC-EVL-12; 100% P0, P1 ≥ 95% |
| W8 | Gate §4.2 |

---

## 6. Giả định và câu hỏi mở cho phần test

Không đặt Q ID mới. Các mục phát sinh ở đây đã được SRS §8 gán Q-29..Q-32.

| # | Nội dung | Ảnh hưởng tới TC | Mặc định khi chưa có trả lời |
|---|---|---|---|
| 1 | Q-01: định dạng `predictions.json` | TC-VAL-01, TC-VAL-06 | Validator chạy cả 3 biến thể; mặc định theo SRS §6.3 |
| 2 | Q-02, Q-03: quy ước scorer (`labels`, zd, gộp/tách exam) | TC-EVL-01..TC-EVL-03, TC-EVL-12 | Kiểm mọi quy ước; mặc định `labels=0..10` |
| 3 | Q-04: L2 có được trích nguyên văn code sinh viên không | TC-GRD-05 | `grd.l2_quote_max_lines = 1`: trích ≤ 1 dòng có sẵn trong cùng hàm, cấm code mới |
| 4 | Q-09, Q-23, Q-25: dung sai ε khác máy, phần cứng giảng viên dùng để tái lập | TC-REP-02, TC-REP-04, TC-REP-13 | ε = 0.01; 1 GPU 24 GB; replay từ cache không cần GPU hay Docker |
| 5 | Q-11: đơn vị quota (upload) và múi giờ ranh giới ngày | TC-VAL-07 | 1 upload = 1 lượt; `Asia/Ho_Chi_Minh`, cấu hình được; cảnh báo gần ranh giới |
| 6 | Q-14: T3 có được ghép `test_report` (kể cả bộ test công khai cho repair, exec-leak) không | TC-T3-04, TC-T3-05, TC-GRD-07 | Không ghép; `public_tests` từ statement/EXM, không có thì tắt repair và exec-leak |
| 7 | Q-17: g++ 13 có tái lập trạng thái log 32/32 không (EDA mới kiểm bằng clang) | TC-EVD-01, TC-EVD-03 | Log gốc là tín hiệu chính (BR-11); lệch thì ghi vào card, không đổi cổng CE |
| 8 | Q-19: có được biên dịch, chạy code sinh viên lúc suy luận không | TC-EVD-01..TC-EVD-06, TC-GRD-07, TC-T3-05 | Được chạy trong sandbox; luôn giữ chế độ `given_signals_only` (TC-REP-12) |
| 9 | Q-29: CI-cloud có được chạy trên runner của host repo không, khi chỉ có code + dữ liệu tổng hợp? | §1.2, §1.5 | Có, với điều kiện không có dữ liệu thật; mọi job dùng dữ liệu chạy ở CI-local |
| 10 | Q-30: Pre-commit chặn `*.cpp`: chấp nhận lưu chương trình C++ tổng hợp dạng chuỗi trong generator Python không? | §1.3 | Có; không có ngoại lệ đường dẫn cho `*.cpp` |
| 11 | Q-31: giá trị τ_strong, k (support tối thiểu để không co ngưỡng), trần stdout của sandbox | TC-T2-06, TC-T2-07, TC-EVD-14 | Theo `Settings` của đặc tả kỹ thuật §5.1: `t2.tau_strong = 0.90`; k chốt từ OOF; `sandbox.stdout_cap_bytes = 65536` (64 KB). TC đọc giá trị từ `Settings`, không hard-code |
| 12 | Q-32: ai gán ≥ 50 phản hồi cho bộ kiểm guard và lấy từ đâu (train hoặc tổng hợp, không dùng test) | TC-GRD-11 | 2 thành viên không viết guard; xong trước W5 |
| 13 | [GT] S211 pass nới 8/8 suy từ phân loại EDA §2.3 (11 test "chỉ sai vì prompt" = 8 của S211 + 3 của S216); S209 có 8/8 test timeout | TC-EVD-04, TC-EVD-05 | Xác nhận khi chạy TC-EVD-07 lần đầu; lệch thì sửa số kỳ vọng, không sửa code |
