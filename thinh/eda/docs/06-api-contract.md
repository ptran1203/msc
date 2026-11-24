# Hợp đồng dữ liệu, CLI, interface và API: hệ thống chấm rubric, phân loại lỗi, sinh phản hồi bài C++

| Mục | Giá trị |
|---|---|
| Phiên bản | 0.1, bản nháp để nhóm review |
| `contract_version` | `0.1.0` (semver, quy tắc ở §3) |
| Ngày | 2026-10-04 |
| Vai trò soạn | API/Contract architect |
| Căn cứ | [SRS](02-requirements-srs.md) §5, §6, §8 (Q-01); [Đặc tả kỹ thuật](03-technical-specification.md) §1.6, §2, §3, §4, §6 (**nguồn gốc**; lệch ghi ở §7); [Test case](05-test-cases.md); BA §9.3 (điều kiện duyệt sâu); bộ mẫu 32 bài (chỉ đọc cấu trúc) |
| ID | **Không định nghĩa ID mới.** Chỉ trỏ C/BR/FR/NFR/Q của SRS, SC của scenario, TC của test. Câu hỏi mới ghi "đề xuất Q mới" ở §7.2 |
| Thư mục hợp đồng | [`contracts/`](contracts/) (schema JSON draft 2020-12, OpenAPI 3.1, ví dụ) |

**Quy ước của tài liệu này:**
- Schema JSON và OpenAPI viết tiếng Anh vì sẽ vào repo (`schemas/`, PY-01); tài liệu này viết tiếng Việt.
- Không trích code sinh viên. Mọi ví dụ là dữ liệu tổng hợp, sample_id dải `S9xx` (`EX01-S999`, `EX02-S997..S999`); hash trong ví dụ là giá trị minh hoạ.
- `$id` dùng miền `grader.invalid` (miền dành riêng, không phân giải được) nên không validator nào gọi mạng.
- Mỗi file schema tự đứng được (không `$ref` ra file khác), validate từng file không cần registry. Chỉ OpenAPI `$ref` sang schema bằng đường dẫn tương đối.
- Nhãn, level và tên khoá tiếng Việt (`định nghĩa`, `cấm`) trong schema viết trực tiếp bằng UTF-8, **không** dùng escape `\uXXXX`. Cách viết không bảo đảm NFC; test hợp đồng "NFC trong contracts/" (§6.2) kiểm điều đó. Khi test xanh, `enum` tự chặn nhãn NFD (BR-12).

**Kết quả kiểm thật (2026-10-05):** chạy `../.venv/bin/python ../validate_contracts.py --data-root <sample_dataset>` (script ở thư mục `eda/`). Script không gọi mạng.

| Kiểm | Kết quả |
|---|---|
| 17 schema hợp lệ theo meta-schema draft 2020-12; mọi `$ref` resolve qua registry cục bộ | Đạt |
| Bộ mẫu thật (task1/2/3, `exams.json`, `label_space.json`) validate theo schema quan sát | 5/5 file, 0 lỗi (32 bài) |
| Ví dụ trong `contracts/examples/` validate theo schema hoặc component OpenAPI | 20/20 |
| Ca âm phải bị từ chối: total 11, logic 5, `feedback` trong input T1, nhãn NFD, nhãn lạ, T3 có `test_report`, level sai, request `/v1/grade` có `feedback` hoặc `target_feedback_level`, predictions T2 nhãn NFD, predictions T1 total 10.5 | 11/11 bị từ chối |
| `openapi.yaml` qua `openapi-spec-validator` (đọc `$ref` ngoài từ đĩa) | Đạt |
| Mọi file trong `contracts/` ở dạng NFC | Đạt |

---

## 1. Phạm vi và bốn lớp hợp đồng

### 1.1 Bốn lớp

| Lớp | Ranh giới | Bên cấp → bên dùng | Hiện thực (03) | Đặc tả hợp đồng | Trạng thái |
|---|---|---|---|---|---|
| (i) File với BTC | Data root vào; `predictions` ra | BTC → ING; VAL → BTC | `ing/schemas.py`, `val/formats.py`, `val/writer.py` | `contracts/dataset/*`, `contracts/predictions/*`, §1.3, §5 | observed (vào), proposed (ra, chờ Q-01) |
| (ii) CLI/make | Người, CI → `grader` | `Makefile` → Typer `cli.py` | 03 §4 | §1.4, §4 | đã đặc tả ở 03, bổ sung stdout/stderr |
| (iii) Nội bộ | Module ↔ module | Python `Protocol` + pydantic | `src/grader/*` (03 §1.5) | `contracts/internal/*`, §1.5 | internal |
| (iv) REST tuỳ chọn | Client của hệ thống (2) hoặc demo → service | FastAPI, handler sync | `grader serve` (đề xuất) | `contracts/openapi.yaml`, §1.6 | proposed, tuỳ chọn; **không chuẩn tắc** tới khi có trả lời đề xuất Q mới ở §7.2 |

- Pipeline challenge (hệ thống 1) **chỉ** dùng lớp (i)–(iii), chạy batch qua CLI. Lớp (iv) không được dùng trong `make predict-private`.
- Lớp (iv) tái dùng đúng `Settings`, cache, evidence builder, firewall, guard của lớp (iii). Không thêm hành vi model.

### 1.2 Cây thư mục `contracts/`

```
docs/contracts/
├─ openapi.yaml                                   # lớp (iv)
├─ dataset/       task1-grading · task2-error-taxonomy · task3-feedback · exams · label-space   (.schema.json)
├─ predictions/   predictions-t1 · predictions-t2 · predictions-t3                             (.schema.json)
├─ internal/      blind-sample · cond-sample · t1-output · t2-output · t3-output ·
│                 evidence-card · problem-card · run-manifest · run-result                     (.schema.json)
└─ examples/      predictions-t{1,2,3}.json · evidence-card-ex0{1,2}.json · problem-card-ex01.json ·
                  api-{grade,classify,feedback,evidence,assess}-{request,response}.json ·
                  api-info-response.json · api-healthz-response.json · api-problem-422.json ·
                  run-result-quota.json
```

Khi init repo: `internal/evidence-card` ↔ `schemas/evidence_card.v1.json`, `internal/problem-card` ↔ `schemas/problem_card.v1.json`, `predictions/*` ↔ `schemas/predictions.{t1,t2,t3}.json` (03 §1.5). Bản trong repo được **xuất từ pydantic**; test parity ở §6.2 giữ hai bản khớp nhau.

### 1.3 Lớp (i): file với BTC

**Input (observed, khớp 03 §2.1 và SRS §6.1):**

| Thành phần | Bắt buộc | Kiểu, miền | Null / rỗng | Ràng buộc chỉ quan sát trên mẫu [GT] |
|---|---|---|---|---|
| Envelope task file | `task, description, input_fields, output_fields, metric, n_samples, samples` | `label_space` (T2), `level_definition_ref` (T3) tuỳ chọn | — | `additionalProperties: true` (pydantic `extra="allow"`) |
| `sample` | `sample_id, input` | `output` tuỳ chọn | `output` vắng hoặc `null` ở dev/test | `sample_id` khớp `^EX[0-9]{2}-S[0-9]{3}$` |
| Input T1 ≡ T2 | `exam_id, exam_type, language, code_file, compile_log` | `test_report`: `null` hoặc mảng `{input, expected, actual: str, passed: bool}` | `compile_log = ""` khi build không có diagnostic; `test_report = []` khi single không build được | multi ⇒ `test_report = null`; single ⇒ mảng; `compile_log` ≤ 2000 ký tự (Q-17) |
| Input T3 | như T1 trừ `test_report`, thêm `taxonomy_error`, `target_feedback_level` | nhãn ∈ 10 chuỗi LS, không trùng; level ∈ 4 chuỗi `Level k - …` | `taxonomy_error = []` hợp lệ | không có `test_report` (BR-23) |
| Output T1 | `rubric` (6 chiều), `total_score` | số nguyên trong miền BR-02 | — | Σ = total **không** ép ở gold (BR-01 là [GT], audit đo) |
| Output T2 | `taxonomy_error` | mảng nhãn LS, `uniqueItems` | `[]` hợp lệ (BR-13) | thứ tự nhãn gold không ép |
| Output T3 | `feedback` | chuỗi ≥ 1 ký tự | — | chỉ dùng cho T3, EDA, bonus (BR-21) |
| `exams.json` | `exams[]`, mỗi exam đủ 15 trường bắt buộc | `problems[]` 2 biến thể theo `exam_type` | `grading_policy`, `note` có thể `null` | single_problem có đúng 1 câu |
| `label_space.json` | `rubric, taxonomy_error, feedback_level` | tên chiều, miền, 10 nhãn, 4 level ghim bằng `const` | — | đổi bất kỳ giá trị nào → `ConfigError` (03 §3.1 bước 2) |

- `additionalProperties` của sample, input, output đặt `false` (schema "quan sát"). Pydantic vẫn chạy với mặc định `ignore`. Trường mới ở tập đầy đủ (ví dụ `compiles_locally`, Q-17) làm schema JSON báo lỗi; audit hạ lỗi đó thành **cảnh báo**, loader không dừng (03 §2.1).
- Pydantic tự kiểm thêm những gì JSON Schema không diễn đạt được: `n_samples == len(samples)`, `sample_id` duy nhất, join T1/T2/T3 (FR-ING-01, FR-ING-04).

**Output:** `predictions` theo mặc định của Q-01 (SRS §6.3): mỗi task một file, top-level là mảng, phần tử đúng 2 khoá `sample_id`, `output`; `output` lặp lại đúng `output` của file task. Ghi chú `x-settings` trong schema khớp `val.layout="array"`, `val.output_style="nested"`, `val.files="per_task"`. Quy tắc kiểm ở §5.

### 1.4 Lớp (ii): CLI và make

Bảng lệnh lấy từ 03 §4 và 03 §3.2 (`evd calibrate`, `evd calibrate-neardup`). Cột "Exit" là các mã có thể xảy ra theo bảng §4 của tài liệu này; mã 5 chỉ xuất hiện khi `llm.cache_mode = "replay_only"`.

| make | CLI | Tham số chính | Ghi ra (dưới `RUNS_DIR`, `CACHE_DIR`, ngoài repo) | Exit |
|---|---|---|---|---|
| `setup` | `uv sync --frozen`; build/pull image theo digest; `pre-commit install` | — | `env.json` | 0, 1 |
| `smoke` | `grader smoke` | — | `smoke.json` | 0, 1, 4 |
| `audit` | `grader audit --split train`, rồi `grader bonus feedback-audit` (process riêng) | `--split` | `constraints.json`, `audit.md`; `audit_feedback.md` | 0, 2, 3, 4 (`SandboxError` khi dựng evidence, `UpstreamError` khi POL gọi LLM), 5 |
| `audit-<tên>` | `grader bonus <tên>` | tên phát hiện | bảng 8 bước | 0, 2, 3 |
| — | `grader evd calibrate` | profile | profile chọn → `constraints.json`, manifest (03 §3.2) | 0, 2, 3, 4 |
| — | `grader evd calibrate-neardup` | — | ngưỡng Jaccard, phân bố → `audit.md` (03 §3.2) | 0, 2, 3 |
| `features SPLIT=` | `grader features --split S` | `sample \| train \| dev` | evidence card, problem card, `features/<split>.parquet` | 0, 2, 3, 4, 5 |
| `folds` | `grader folds` | — | `folds.csv` có version | 0, 2 |
| `train TASK= ARM=` | `grader train --task {t1,t2,t3} --arm {p,f,all}` | task, arm | `oof/<task>/<run_id>.parquet`, adapter, decoder, ngưỡng, `lineage.jsonl` | 0, 2, 3, 4, 5 |
| `eval TASK= RUN=` | `grader eval [--degenerate]` | task, run_id | `summary.md`, bảng metric | 0, 2 |
| `predict SPLIT=` | `grader predict --split {sample,train,dev}` | split; `test` bị từ chối | `runs/<run_id>/predictions/{t1,t2,t3}.json` | 0, 2, 3, 4, 5 |
| `predict-private ARCHIVE= [DRY_RUN=1]` | `grader predict-private --archive … [--dry-run]` | archive | predictions + dòng `runs/private_audit.jsonl` | 0, 2, 3, 4, 5 |
| `smoke-submit TASK=` | `pytest -m "smoke and p0"` | task | đường dẫn file + sha256 | mã của pytest |
| `gate-private` | `smoke-submit` cho 3 task + kiểm gate | tag `v1.0` | báo cáo gate | 0 hoặc ≠ 0 |
| `submit TASK= RUN= REASON=` | `grader submit` | `t1\|t2\|t3\|all`, run_id, `i\|ii\|iii` | `submissions/<YYYY-MM-DD>/<upload_id>/<task>-<run_id>.json` + sha256, dòng `submission_log.csv` | 0, 2, 3, 6 |
| `reproduce [FULL=1]` | `grader reproduce [--full]` | — | predictions + so sha256 với `submissions/final.sha256` | 0, 2, 4, 5 |
| `ablation` | `grader ablation --matrix configs/ablation.yaml` | ma trận | bảng ablation | 0, 2, 4, 5 |
| `report`, `check` | `grader report`; ruff, mypy, `lint-imports`, `pytest --cov` | — | bảng, hình, model card | 0, 1, 2 |
| — (đề xuất, lớp iv) | `grader serve` | host, port từ `Settings` | `runs/<run_id>/log.jsonl`; không ghi predictions | 0, 1, 2 (`ConfigError` lúc khởi động) |

**Mọi lệnh `grader`** tạo `run_id` + `runs/<run_id>/manifest.json` theo [`run-manifest.schema.json`](contracts/internal/run-manifest.schema.json), ghi `log.jsonl` và `summary.md` (FR-REP-04, NFR-OBS-01, NFR-OBS-02). `Settings` ghi đè bằng biến `GRADER_<NHÓM>__<KHOÁ>` (03 §4). Không lệnh nào hỏi tương tác (C-05).

**Quy ước stdout/stderr (đề xuất, bổ sung 03 §4):**

| Kênh | Nội dung | Ràng buộc |
|---|---|---|
| stdout | Đúng **một dòng JSON** cuối cùng theo [`run-result.schema.json`](contracts/internal/run-result.schema.json): `{"run_id", "command", "status": "ok"\|"error", "exit_code", "outputs": [{"path", "sha256"}], "error": Problem\|null}`. `error` dùng cùng cấu trúc Problem với REST (§4); lỗi chỉ có ở CLI mang `status` HTTP tương đương (ví dụ `examples/run-result-quota.json`) | Để make và CI đọc máy; không có dữ liệu bài |
| stderr | Log console rút gọn (level, stage, sample_id, thông điệp) | Không có code, feedback, prompt, stdout của chương trình SV (`LogSafeFilter`, NFR-PRIV-03) |
| `runs/<run_id>/log.jsonl` | Log có cấu trúc đầy đủ (03 §6.1) | Cùng ràng buộc với stderr |
| `predict-private` | stdout chỉ có run_id, đường dẫn và sha256; stderr không có sample_id hay nội dung input | FR-SEC-05; mức giao T2/T3 chỉ ghi số đếm vào `private_audit.jsonl` (FR-ING-04) |

### 1.5 Lớp (iii): interface nội bộ (Python `Protocol`)

Kiểu dữ liệu ở 03 §2 (`BlindSample`, `CondSample`, `EvidenceCard`, `ProblemCard`, `T1Prediction`…). Tên kiểu phụ (`ManifestReport`, `ClusterInfo`, `GuardContext`, `ValidationReport`…) là đề xuất, chưa có trong 03. Mọi hàm nhận `ctx: RunContext` (03 §1.6) khi cần `Settings`, seed, cache; không module nào đọc `os.environ` (PY-05).

```python
Split = Literal["sample", "train", "dev", "private"]; Task = Literal["t1", "t2", "t3"]
PolicyVariant = Literal["gold_optimized", "written_policy", "hybrid"]   # FR-POL-03: (a), (b), lai; cùng tên với REST

# ING (ing/) — FR-ING-01..06
class Ingestor(Protocol):
    def load_dataset(self, data_root: Path, split: Split) -> Dataset: ...            # Dataset{exams, label_space, records, gold}
    def verify_manifest(self, root: Path) -> ManifestReport: ...
    def load_label_space(self, path: Path) -> LabelSpace: ...
    def load_task_file(self, path: Path, in_model: type[In], out_model: type[Out]) -> TaskFile[In, Out]: ...
    def read_code(self, data_root: Path, code_file: str) -> CodeText: ...
    def join_tasks(self, t1: TaskFile, t2: TaskFile, t3: TaskFile | None) -> list[SampleRecord]: ...
def parse_compile_log(log: str, trunc_len: int) -> CompileLogInfo: ...                    # thuần
BlindSample.from_record(rec: SampleRecord) -> BlindSample                              # whitelist, không **rec
CondSample.from_record(rec: SampleRecord, *, use_t1_test_report: bool) -> CondSample
GoldStore.view(task: Task) -> GoldT1View | GoldT2View | GoldT3View                     # overload; chỉ T3 có .feedback()

# EVD (evd/) — FR-EVD-01..08
def build_evidence_card(sample: BlindSample, problem: ProblemCard, cluster: ClusterInfo, ctx: RunContext, *,
                        tests_from: Literal["given_report", "public_tests"] = "given_report") -> EvidenceCard: ...
                        # T3 gọi với tests_from="public_tests" và sample.test_report = None (§1.5)
def run_in_sandbox(code: CodeText, tests: Sequence[tuple[str, str]], profile: str, ctx: RunContext) -> SandboxRun: ...
def classify_failure(actual: str, expected: str, *, timed_out: bool, signal: int | None,
                     answer_vocab: Sequence[str]) -> FailClass | None: ...                 # thuần
def cluster_near_duplicates(samples: Sequence[BlindSample], problem: ProblemCard, jaccard: float) -> Mapping[str, ClusterInfo]: ...
def calibrate(samples: Sequence[BlindSample], profiles: Sequence[str], ctx: RunContext) -> CalibrationReport: ...

# POL (pol/) — FR-POL-01..05
ProblemCard.from_exam(exam: Exam, split_inputs: Sequence[BlindSample], llm: LLMEngine) -> ProblemCard
def estimate_problem_status(sample: BlindSample, card: EvidenceCard, problem: ProblemCard, llm: LLMEngine) -> ProblemStatus: ...
def aggregate(status: ProblemStatus, problem: ProblemCard, variant: Literal["hard", "soft"]) -> PolicyFeatures: ...
def apply_written_policy(pred: T1Prediction, status: ProblemStatus, problem: ProblemCard) -> T1Prediction: ...  # bản (b)

# T1, T2, T3 (t1/, t2/, t3/) — FR-T1-*, FR-T2-*, FR-T3-*
class T1Predictor(Protocol):
    def predict(self, samples: Sequence[BlindSample], cards: Mapping[str, EvidenceCard], problems: Mapping[str, ProblemCard],
                constraints: ConstraintSet, variant: PolicyVariant) -> list[T1Prediction]: ...
class T2Predictor(Protocol):
    def predict(self, samples: Sequence[BlindSample], cards: Mapping[str, EvidenceCard],
                t1_totals: Mapping[str, int], constraints: ConstraintSet) -> list[T2Prediction]: ...
class T3Generator(Protocol):
    def generate(self, samples: Sequence[CondSample], cards: Mapping[str, EvidenceCard],   # card tests_source ∈ {public_tests, none}
                 problems: Mapping[str, ProblemCard]) -> list[T3Result]: ...          # T3Result = T3Prediction + GuardOutcome
ConstraintSet.mine(train: GoldT1View, audit: ConstraintsFile) -> ConstraintSet
def apply_ce_gate(card: EvidenceCard, constraints: ConstraintSet, exam_type: ExamType) -> GateDecision: ...
LabelVocab.match(raw: Sequence[str]) -> tuple[str, ...]                                # NFC → casefold → exact → rapidfuzz
def apply_cross_constraint(labels: tuple[str, ...], probs: Mapping[str, float], total: int, ctx: RunContext) -> tuple[str, ...]: ...
def diagnose(sample: CondSample, card: EvidenceCard, llm: LLMEngine) -> Diagnosis: ...
def mask_for_level(diagnosis: Diagnosis, level: Level) -> RenderInput: ...               # thuần

# GRD (grd/) — FR-GRD-01..07
def check(text: str, level: Level, ctx: GuardContext) -> GuardReport: ...                 # passed, violations[layer, rule, span], underlevel
def guard_and_retry(render: Callable[[int | None], str], level: Level, ctx: GuardContext, seeds: Sequence[int]) -> GuardOutcome: ...

# EVL (evl/) — FR-EVL-01..10
def qwk(y: Sequence[int], y_hat: Sequence[int]) -> float: ...                             # labels=range(11)
def macro_f1(y: Sequence[Sequence[str]], y_hat: Sequence[Sequence[str]], zd: Literal[0, 1, "nan"]) -> float: ...
def make_folds(records: Sequence[FoldInput], k: int, seed: int) -> Folds: ...
def select_model(candidates: Sequence[CVResult], task: Task) -> Decision: ...             # không nhận điểm public LB
def run_audit(train: Dataset, ctx: RunContext) -> ConstraintsFile: ...                    # không gọi GoldT3View

# VAL (val/) — FR-VAL-01..06
def write_predictions(preds: Sequence[Prediction], fmt: SubmissionFormat, path: Path) -> FileDigest: ...
def validate_predictions(path: Path, task: Task, split_ids: Sequence[str], fmt: SubmissionFormat) -> ValidationReport: ...
def run_official_check(path: Path, cmd: str) -> OfficialCheckResult: ...
def submit(task: Task | Literal["all"], run_id: str, reason: Literal["i", "ii", "iii"], ctx: RunContext) -> SubmissionRecord: ...

# REP (rep/, llm/cache.py) — FR-REP-01..06
def seed_for(*parts: str) -> int: ...
class LLMCache(Protocol):
    def get(self, key: CacheKey) -> CachedResponse | None: ...
    def put(self, key: CacheKey, value: CachedResponse) -> None: ...                      # chỉ thêm
def write_manifest(ctx: RunContext) -> Path: ...
def reproduce(full: bool, ctx: RunContext) -> ReproReport: ...
def run_smoke(ctx: RunContext) -> SmokeReport: ...

# SEC (sec/) — FR-SEC-01..07
class FeedbackFirewall(Protocol):
    def check(self, prompt: str, task: Literal["t1", "t2"]) -> None: ...
class Lineage(Protocol):
    def record(self, artefact_id: str, kind: str, inputs: Sequence[str]) -> LineageRow: ...   # touched_feedback = OR(inputs)
    def assert_clean(self, artefact_ids: Sequence[str], task: Literal["t1", "t2"]) -> None: ...
class EgressClient(Protocol):
    def post(self, url: str, payload: Mapping[str, object]) -> httpx.Response: ...
PrivateRun(archive: Path, ctx: RunContext) -> ContextManager[PrivateSplit]
def count_pii(texts: Iterable[str]) -> Mapping[str, int]: ...                             # chỉ đếm

# RPT (rpt/) — FR-RPT-01..08
def run_ablation(matrix: Path, ctx: RunContext) -> AblationTable: ...
def slice_errors(oof: Path, cards: Mapping[str, EvidenceCard]) -> SliceReport: ...
def build_model_card(run_ids: Sequence[str]) -> Path: ...
def check_report(path: Path) -> ReportCheck: ...
```

| Module | Exception có thể ném (nơi bắt) | Tất định | FR chính |
|---|---|---|---|
| ING | `DataSchemaError`, `DataIntegrityError`, `ConfigError` (biên CLI, exit 2); `TestIsolationError` khi đọc split `private` ngoài `PrivateRun` (exit 3) | Có, thuần I/O | FR-ING-01..06 |
| EVD | `SandboxError` (hạ `given_signals_only` nếu `sandbox.on_error="given_signals"`, ngược lại biên CLI, exit 4). CE, crash, timeout của code SV là dữ liệu | Theo khoá (`code_sha`, `toolchain_id`, `tests_sha`, `evidence_mode`) + cache evidence; bất định thì bật `nondeterministic` | FR-EVD-01..08 |
| POL | `PolicyExtractionError` (bắt tại `extract.py` → trọng số đều, `policy_failed`); `LLMOutputParseError` (bắt tại stage) | Qua cache LLM + `seed_for("pol", exam_id, i)` | FR-POL-01..05 |
| T1, T2 | `LLMOutputParseError` (retry → prior + cờ `p_arm_missing` / cột NaN); `FirewallViolation` không stage nào bắt | Greedy + cache | FR-T1-01..09, FR-T2-01..08 |
| T3 | `LLMOutputParseError` (diagnose → issue chung theo nhãn); `GuardViolation` (bắt ở GRD) | Lượt 0 greedy; retry theo `t3.retry_seeds`; cache | FR-T3-01..08 |
| GRD | `GuardViolation` chỉ nội bộ → retry → template đúng level → câu an toàn tối thiểu | Có | FR-GRD-01..07 |
| EVL | `DataSchemaError` khi input metric không phải `int` | `seed_for` cho fold, bootstrap | FR-EVL-01..10 |
| VAL | `validate_predictions` **không raise**, trả mọi lỗi; `submit` ném `SubmissionFormatError` (exit 2), `QuotaExceeded` (exit 6) | Writer byte-stable | FR-VAL-01..06 |
| REP | `CacheMissError` (exit 5), `ConfigError` (exit 2) | — | FR-REP-01..06 |
| SEC | `FirewallViolation`, `EgressBlocked`, `TestIsolationError` (exit 3); `CostCapExceeded` (exit 6); `UpstreamError` (exit 4) | — | FR-SEC-01..07 |
| RPT | `DataIntegrityError` khi thiếu artefact hoặc run_id | Có, đọc từ `runs/` | FR-RPT-01..08 |

**Bất biến liên module:**
- Thứ tự online cố định ING → EVD → POL → T1 → T2 → T3 → VAL (03 §1.2). T2 nhận `t1_totals` (FR-T2-06); T3 không đọc output T1/T2.
- `build_evidence_card` trả card với `problem_status = policy = null`; pipeline gắn `estimate_problem_status` và `aggregate` vào bằng `model_copy(update=…)`. Chỉ ghi cache khi card đã hoàn chỉnh (phần lõi lưu theo khoá ở gạch "Khoá cache evidence").
- **Nguồn test của card theo task (BR-23, Q-14).** T1/T2 dùng card dựng từ `test_report` của input (`tests_source = given_report`). T3 chỉ nhận card có `tests_source ∈ {public_tests, none}`: test lấy từ `ProblemCard.public_tests` và chạy lại trong sandbox (mọi `TestRun.source = rerun`), hoặc `run = null` (multi_problem, `given_signals_only`, exam không có `public_tests`, hoặc Q-19 không cho chạy code SV; 03 §3.6 bước 1). Card `given_report` chỉ được đưa vào T3 khi `t3.use_t1_test_report = true`. Card T3 không chứa số test, tỷ lệ hay `text` suy từ `test_report`. Vì `tests_sha` nằm trong khoá cache, card T1/T2 và card T3 của cùng một bài là hai entry riêng.
- **Khoá cache evidence** = (`code_sha`, `toolchain_id`, `tests_sha`, `evidence_mode`); `given_signals_only` dùng `toolchain_id = "000000000000"` (sentinel). Khoá chỉ định danh phần lõi dựng từ sandbox; trường lấy từ chính bài (`sample_id`, `compile_given`, cụm) gắn lại theo từng bài (`x-cache-key-note` trong schema).
- T1/T2 chỉ nhận `BlindSample`. import-linter chỉ kiểm theo **module**, không theo class (07 §7 #4), nên luật là: `grader.t1`, `grader.t2` không import (trực tiếp hay gián tiếp) `grader.t3`, `grader.ing.cond_views`, `grader.ing.gold_t3` (tên module đề xuất, đã dùng trong `deps/pyproject.draft.toml`; lệch với 03 ở §7.1 #22) (FR-SEC-01 (a)).

### 1.6 Lớp (iv): REST tuỳ chọn

| Method, path | Task | Request | Response | Luồng module | FR |
|---|---|---|---|---|---|
| `GET /healthz` | — | — | `HealthResponse` (200/503), không cần token | engine, sandbox, cache, LS | NFR-OBS-01 |
| `GET /v1/info` | — | — | model id + revision, adapter sha256, config sha256, toolchain + digest, contract version, exam đã đăng ký | REP | FR-REP-02, FR-REP-04 |
| `POST /v1/evidence` | — | `BlindRequest` | `EvidenceResponse` (evidence card v1.0) | ING → EVD → POL (status) | FR-EVD-01 |
| `POST /v1/grade` | T1 | `BlindRequest` | `GradeResponse` | ING → EVD → POL → T1 | FR-T1-01, FR-POL-03 |
| `POST /v1/classify` | T2 | `BlindRequest` | `ClassifyResponse` | … → T1 (nội bộ) → T2 | FR-T2-01, FR-T2-06 |
| `POST /v1/feedback` | T3 | `FeedbackRequest` | `FeedbackResponse` | ING → EVD → T3 → GRD | FR-T3-01, FR-GRD-06 |
| `POST /v1/assess` | T1+T2(+T3) | `AssessRequest` | `AssessResponse` | như trên; card T1/T2 và card T3 tách riêng khi có `test_report` (§1.5) | — |

**Ánh xạ request → kiểu nội bộ:**
- `code` (chuỗi inline) → `CodeText`: `raw_sha1` = sha1 của bytes UTF-8 nhận được, bỏ BOM, NFC (FR-ING-02). Mã nguồn chỉ nằm trong bộ nhớ và cache, không vào log.
- `BlindRequest` → `SampleRecord` tạm → `BlindSample.from_record` (whitelist). `sample_id` = `sample_id` của client hoặc `request_id`.
- Một request là một lô 1 bài: `cluster_size = 1`, `cluster_id = "c" + sha1(sample_id)[:10]`.
- `exam_id` phải có trong registry dựng **offline** từ `exams.json` (`ProblemCard.from_exam` chạy ở `make features`). Exam lạ → 422 `data-integrity`. `exam_type` lệch registry → 422.
- `compile_log` vắng → card ghi `compile_given = null` (khác log rỗng `""`), cổng CE dùng rebuild trong sandbox (03 §3.4 bước 2); `EvidenceSummary.compile.log_*` trả `null`. `evidence_mode = given_signals_only` thì `compile_log` **bắt buộc** (schema `CompileLogRequiredWithoutSandbox`).
- `test_report` vắng ở single_problem → chạy `public_tests` của ProblemCard (`tests_source = public_tests`). Số test có thể ít hơn batch (EX02: 2 test trong statement so với 8 test trong report).
- `/v1/feedback` không nhận `test_report` nên luôn dựng card theo cách trên, hoặc `run = null`; không dùng lại card của `/v1/grade` hay `/v1/classify` (khác `tests_sha`). Ví dụ `api-feedback-response.json` chỉ có 2 test công khai, `cache = miss`.
- `/v1/feedback` dùng tên trường của file task T3 (`taxonomy_error`, `target_feedback_level`); server đổi sang `CondSample.level`.

**Tường lửa ở mức schema:** `BlindRequest`, `AssessRequest` đặt `additionalProperties: false` và `not.anyOf` liệt kê `feedback`, `target_feedback_level`, `level`, `taxonomy_error`, `labels` → 422 nêu đúng khoá. `AssessRequest.feedback_target` chỉ T3 đọc. Engine từ chối adapter `touched_feedback = true` cho phần T1/T2 nên `versions.adapter_sha256 = null` ở grade/classify (FR-SEC-01 (c)).

**Định tuyến người duyệt** (`review`, theo BA §9.3; AI không chốt điểm):

| `code` | BA §9.3 | Điều kiện trên evidence/kết quả |
|---|---|---|
| `new_exam` | #1 | `gate_source = exam_type_default` hoặc exam chưa có dữ liệu hiệu chỉnh |
| `link_error`, `zero_effective_code` | #2 | `log_has_link_error`; `effective_code_lines = 0` |
| `compile_signal_disagreement`, `compile_log_truncated` | #3 | `agrees_with_given = false`; `log_truncated` |
| `prerequisite_suspected_failed` | #4 | `policy.prereq_failed`; response trả cả hai điểm khi client gọi 2 `policy_variant` |
| `io_anomaly` | #5 | có `prompt_prefix`, `leading_ws`/`trailing_ws`, hoặc `io_gap > 0` |
| `crash_or_timeout` | #6 | `n_by_fail_class.empty_crash + timeout > 0` |
| `t1_high_uncertainty` | #7 | độ lệch chuẩn của `total_distribution` ≥ τ (cấu hình) |
| `t1_t2_inconsistent` | #8 | trước hậu xử lý: `[]` mà total < 10 |
| `near_pass_threshold` | #9 | total trong khoảng cấu hình (ví dụ 4–5) |
| `injection_suspected` | #10 | detector FR-SEC-07 trên comment |
| `t3_template_fallback` | #11 | `guard.fallback ≠ none` |
| `blind_audit_sample` | #12 | lấy mẫu ngẫu nhiên có seed; `blind_audit = true` |
| `missing_code`, `sandbox_degraded`, `policy_extraction_failed`, `l4_unverified`, `model_fallback` | (03) | `missing_code`; `SandboxError` hạ chế độ; `policy_failed`; L4 chưa kiểm; cờ `p_arm_missing` |

`needs_review = true` ⇔ `route = deep` ⇔ có ít nhất một lý do. Ngưỡng τ và khoảng điểm là đề xuất [GT] của BA, đặt trong `Settings`.

**Vận hành:** token bearer do trường cấp (lưu `SecretStr`, không log); bind loopback hoặc mạng nội bộ trường; image đặt sẵn biến offline, allowlist egress rỗng (NFR-PRIV-02); code SV chỉ chạy trong sandbox (FR-SEC-02). Giới hạn: body ≤ 256 KiB (413), `code` ≤ 65 536 ký tự, `compile_log` ≤ 2000, `test_report` ≤ 64 ca; một engine GPU nên xử lý tuần tự, hàng đợi đầy → 503 + `Retry-After`. Rate limit theo token (mặc định 30 req/phút), header `RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset` trên mọi 200 của POST (theo draft IETF, định nghĩa ở `components.headers`). Header `X-Request-Id` (vào/ra, cũng ghi vào `log.jsonl`), `X-Contract-Version` (ra), `Idempotency-Key` (vào, POST): cùng khoá + cùng sha256 body trong 24 giờ → trả lại response đã lưu kèm `Idempotent-Replayed: true`; khác body → 409.

---

## 2. Bảng file hợp đồng

Đường dẫn tương đối với `contracts/`. Trạng thái: **observed** = mô tả dữ liệu BTC đã thấy; **proposed** = nhóm đề xuất, chờ xác nhận; **internal** = artefact của nhóm.

| Hợp đồng | Schema | Ví dụ | FR / NFR / BR / C | Q | Trạng thái |
|---|---|---|---|---|---|
| File task T1 | `dataset/task1-grading.schema.json` | — (dữ liệu BTC, không sao chép) | FR-ING-01, FR-ING-04, BR-01, BR-02, BR-11, C-18 | Q-17 | observed |
| File task T2 | `dataset/task2-error-taxonomy.schema.json` | — | FR-ING-01, BR-12, BR-13 | Q-03, Q-18 | observed |
| File task T3 | `dataset/task3-feedback.schema.json` | — | FR-ING-01, FR-ING-04, BR-21, BR-22, BR-23 | Q-05, Q-14 | observed |
| `exams.json` | `dataset/exams.schema.json` | — | FR-POL-01, BR-07 | Q-08, Q-16 | observed |
| `label_space.json` | `dataset/label-space.schema.json` | — | FR-ING-01, BR-02, BR-12, BR-18 | Q-04, Q-18 | observed |
| Predictions T1 | `predictions/predictions-t1.schema.json` | `examples/predictions-t1.json` | FR-VAL-01, FR-VAL-04, FR-T1-01, BR-01, BR-02, C-09 | Q-01, Q-02, Q-11 | proposed (`x-status: proposed-pending-Q-01`) |
| Predictions T2 | `predictions/predictions-t2.schema.json` | `examples/predictions-t2.json` | FR-VAL-01, FR-VAL-04, FR-T2-01, BR-12, BR-13 | Q-01, Q-03 | proposed |
| Predictions T3 | `predictions/predictions-t3.schema.json` | `examples/predictions-t3.json` | FR-VAL-01, FR-T3-01, BR-24 | Q-01, Q-04 | proposed |
| BlindSample | `internal/blind-sample.schema.json` | dạng REST: `examples/api-grade-request.json` | FR-ING-05, FR-SEC-01, BR-21, C-07 | Q-05 | internal |
| CondSample | `internal/cond-sample.schema.json` | dạng REST: `examples/api-feedback-request.json` | FR-T3-01, BR-22, BR-23 | Q-14 | internal |
| T1Output, T1Prediction | `internal/t1-output.schema.json` | `examples/api-grade-response.json` | FR-T1-01, FR-T1-03, BR-01 | Q-26 | internal |
| T2Output, T2Prediction | `internal/t2-output.schema.json` | `examples/api-classify-response.json` | FR-T2-01, FR-T2-06, BR-14 | Q-03 | internal |
| T3Output, T3Prediction | `internal/t3-output.schema.json` | `examples/api-feedback-response.json` | FR-T3-01, FR-GRD-06 | Q-04 | internal |
| EvidenceCard v1.0 | `internal/evidence-card.schema.json` | `examples/evidence-card-ex02.json` (prompt + crash), `examples/evidence-card-ex01.json` | FR-EVD-01..07, FR-ING-03, FR-ING-06, FR-POL-02, FR-POL-04, BR-23 | Q-14, Q-16, Q-19 | internal |
| ProblemCard v1.0 | `internal/problem-card.schema.json` | `examples/problem-card-ex01.json` | FR-POL-01, C-05 | Q-08, Q-14 | internal |
| Run manifest | `internal/run-manifest.schema.json` | — | FR-REP-02, FR-REP-04, NFR-OBS-01, NFR-SEC-02 | Q-09, Q-25 | internal (tên trường đề xuất) |
| RunResult (dòng stdout cuối) | `internal/run-result.schema.json` | `examples/run-result-quota.json` | NFR-USAB-01, NFR-USAB-02, FR-VAL-03 | Q-11 | internal (đề xuất bổ sung 03 §4) |
| CLI/make | §1.4, §4 | — | NFR-USAB-01, NFR-USAB-02, FR-REP-04 | Q-22 | 03 §4 + đề xuất stdout |
| REST | `openapi.yaml` | `examples/api-*.json` (đủ 5 endpoint POST, `info`, `healthz`) | FR-SEC-02, FR-SEC-04, NFR-PRIV-02, NFR-PRIV-03, FR-RPT-05 | Q-06, Q-19, Q-22 | proposed, tuỳ chọn |

---

## 3. Tương thích, versioning, quy ước chung

### 3.1 Phiên bản

- Một số `contract_version` (semver) chung cho cả `contracts/`. Ghi ở `x-contract-version` của mọi schema, `info.version` của OpenAPI, `contract_version` của run manifest, header `X-Contract-Version`.
- `0.x.y` là bản nháp: thay đổi phá vỡ chỉ tăng MINOR. Đóng băng `1.0.0` khi Q-01 có trả lời và trước tag `v1.0` (gate private). Từ `1.0.0` áp đủ semver.
- `schema_version` của EvidenceCard và ProblemCard (`"1.0"`) là phiên bản cấu trúc artefact và nằm trong khoá cache (FR-REP-03). Đổi cấu trúc card thì tăng `schema_version` để cache cũ không bị đọc nhầm, đồng thời tăng `contract_version`.
- CI giữ bảng sha256 của từng file trong `contracts/`. File đổi mà `contract_version` không đổi thì CI đỏ (§6.2).

### 3.2 Khi nào tăng phiên bản

| Thay đổi | Ví dụ | Từ 1.0.0 | Ở 0.x |
|---|---|---|---|
| Sửa mô tả, ví dụ; kết quả validate không đổi | thêm `examples` | PATCH | PATCH |
| Thêm trường **tuỳ chọn** vào input hệ thống đọc | `t1_test_report` khi Q-14 cho phép | MINOR | MINOR |
| Thêm giá trị enum vào input hệ thống đọc; nới ràng buộc | `evidence_mode` mới; tăng `maxLength` của `code` | MINOR | MINOR |
| Thêm trường tuỳ chọn vào response REST | trường chẩn đoán mới | MINOR | MINOR |
| Thêm giá trị enum vào **output** bên khác đọc | `Verdict = sample_timeout` | MAJOR | MINOR |
| Đổi tên hoặc xoá trường; tuỳ chọn thành bắt buộc; siết kiểu, miền, enum; đổi nghĩa | đổi layout predictions theo Q-01 | MAJOR | MINOR |
| Đổi cấu trúc evidence card hoặc problem card | thêm stderr vào `TestRun` | MAJOR + `schema_version` | MINOR + `schema_version` |

- Schema quan sát (`dataset/*`) đổi theo dữ liệu BTC. BTC thêm trường thì đưa vào schema dạng tuỳ chọn (MINOR); trước đó audit chỉ cảnh báo.
- Đổi `SubmissionFormat` theo Q-01 chỉ đổi `Settings` (`val.layout`, `val.output_style`, `val.files`); writer và validator đọc cùng một `SubmissionFormat` (FR-VAL-01, TC-VAL-06).

### 3.3 Trường mới và deprecation

- Trường mới luôn **tuỳ chọn**, mặc định khớp hành vi cũ. Client REST bỏ qua trường lạ trong response. Writer không bao giờ ghi trường lạ vào file nộp (`additionalProperties: false`).
- Deprecation: đánh `deprecated: true` (JSON Schema 2020-12, OpenAPI 3.1), giữ ít nhất một MINOR, ghi changelog; REST trả thêm header `Deprecation` và `Sunset`. Xoá ở MAJOR kế tiếp. Không tái dùng tên trường đã xoá cho nghĩa khác.

### 3.4 Quy ước chung

| Chủ đề | Quy ước | Căn cứ |
|---|---|---|
| Mã hoá | UTF-8 không BOM; `json.dumps(ensure_ascii=False, separators=(",", ":"))` + `\n` | FR-VAL-04, NFR-REPRO-01 |
| Unicode | Mọi chuỗi ghi ra ở NFC; nhãn và level so khớp bytes NFC | BR-12, FR-ING-02 |
| Số nguyên | Điểm là `int` thật: không `7.0`, không `true`. JSON Schema `integer` chấp nhận `7.0`, nên validator đọc JSON thô và chặn số có phần thập phân | FR-VAL-04 |
| Số thực | Xác suất ∈ [0, 1]; điểm policy ∈ [0, 10]; ra số nguyên dùng `floor(z + 0.5)`, không banker's rounding | FR-T1-07 |
| Định danh | `sample_id` `^EX[0-9]{2}-S[0-9]{3}$` chỉ ở `dataset/*` (quan sát [GT], audit cảnh báo). Schema predictions chỉ đòi chuỗi không rỗng: id private có thể khác dạng (mã SV 4 chữ số, đề mới) mà private chỉ có một lần chạy; luật 4 ở §5 đã kiểm tập id. `exam_id` `^EX[0-9]{2}$`; `pid` `^P[0-9]+$`; `run_id` `<UTC %Y%m%dT%H%M%S>-<git_sha7>-<config_sha6>` | SRS §6.1; 03 §3.10 |
| Khoá cache evidence | (`code_sha`, `toolchain_id`, `tests_sha`, `evidence_mode`); `given_signals_only` dùng `toolchain_id = "000000000000"`; `tests_sha = null` khi `run = null` (§1.5) | FR-REP-03 |
| Hash | sha1 cho bytes code (`code_sha`, `raw_sha1`); sha256 cho file và artefact (predictions, adapter, config, data manifest); image `sha256:<64 hex>` | FR-ING-02, FR-REP-02 |
| Thứ tự | Phần tử predictions theo thứ tự input; nhãn theo thứ tự LS; khoá JSON theo thứ tự field của model | FR-VAL-04, FR-T2-01 |
| `null` và rỗng | `null` = không có hoặc không áp dụng (`test_report` của multi, `compile_rebuild` khi `given_signals_only`). Rỗng = có nhưng không có phần tử (`compile_log = ""`, `taxonomy_error = []`, `test_report = []` khi single không build) | 03 §2.1 |
| Khoá trong artefact nội bộ | Ghi đủ khoá, dùng `null` thay vì bỏ khoá (`exclude_none=False`) để bytes ổn định | NFR-REPRO-01 |
| Thời gian | ISO 8601 UTC hậu tố `Z`; ranh giới ngày của quota theo `val.timezone` | FR-VAL-03 |
| Dữ liệu nhạy cảm | Không ghi code, feedback, prompt, stdout SV vào log, message lỗi, Problem. Giá trị quan sát chỉ ghi kiểu, độ dài, sha1 | NFR-PRIV-03, FR-SEC-05 |

---

## 4. Bảng mã lỗi: exception ↔ exit code ↔ HTTP

`type` của Problem = `https://grader.invalid/problems/<slug>`. Exit code theo 03 §4: 0 thành công; 1 lỗi khác; 2 dữ liệu hoặc định dạng; 3 bảo mật; 4 upstream hoặc sandbox; 5 cache miss khi replay; 6 quota hoặc trần chi phí.

| Exception (03 §6.2) | Nơi bắt | Exit CLI | HTTP | Slug | Ghi chú |
|---|---|---|---|---|---|
| Lỗi ngoài `AppError` | `handle_errors` ở biên | 1 | 500 | `internal` | không trả stack trace |
| `ConfigError` | biên CLI | 2 | 500; lúc khởi động `/healthz` 503 | `config` | |
| `DataSchemaError` | biên CLI | 2 | 422 | `data-schema` | gồm khoá bị tường lửa chặn (`rule = forbidden-field`) |
| `DataIntegrityError` | biên CLI | 2 | 422 | `data-integrity` | REST: exam chưa đăng ký, `exam_type` lệch. CLI: sha manifest lệch, thiếu `code_file` khi train/audit |
| `SubmissionFormatError` | biên CLI | 2 | 500 | `output-format` | REST: response tự kiểm thất bại, là lỗi server |
| `FirewallViolation` | biên CLI; không stage nào bắt | 3 | 403 | `firewall` | prompt T1/T2 trúng shingle feedback |
| `EgressBlocked` | biên CLI | 3 | 500 | `egress-blocked` | service tự gọi ra ngoài là lỗi cấu hình |
| `TestIsolationError` | biên CLI | 3 | 403 | `test-isolation` | REST không đọc split; chỉ xảy ra khi cấu hình sai |
| `UpstreamError(status, host, endpoint)` | biên CLI | 4 | 502 | `upstream` | giữ status, host, endpoint đã thấy (PY-08) |
| `SandboxError` | EVD (hạ chế độ) hoặc biên CLI | 4 | 200 + lý do `sandbox_degraded`, hoặc 503 | `sandbox-unavailable` | 503 khi không được hạ chế độ |
| `CacheMissError` | biên CLI | 5 | 409 | `cache-miss` | REST chỉ khi service chạy `replay_only` |
| `QuotaExceeded` | biên CLI | 6 | 429 (tương đương) | `quota-exceeded` | chỉ CLI (quota nộp bài); REST không có quota nộp, `status = 429` chỉ để Problem trong `RunResult` hợp lệ |
| `CostCapExceeded` | biên CLI | 6 | 503 | `cost-cap` | chỉ khi bật egress |
| `PolicyExtractionError` | `pol/extract.py` | — | 200 + lý do `policy_extraction_failed` | — | không ra ngoài stage |
| `LLMOutputParseError` | stage | — | 200 + lý do `model_fallback` | — | |
| `GuardViolation` | `grd/guard.py` | — | 200, `guard.fallback ≠ none`, lý do `t3_template_fallback` | — | |

- Mã ở tầng HTTP, không có exception tương ứng: 400 `malformed-request` (JSON hoặc UTF-8 hỏng), 401 `unauthorized`, 409 `idempotency-conflict`, 413 `payload-too-large`, 415 `unsupported-media-type`, 429 `rate-limited`, 503 `service-busy` (hàng đợi đầy), 504 `request-timeout`.
- Body Problem (RFC 9457, `application/problem+json`): `type`, `title`, `status`, `detail`, `instance`, cùng phần mở rộng `code` (tên exception hoặc mã HTTP-level), `exit_code`, `request_id`, `errors[] {pointer, rule, observed}`. `detail` nêu điều đã thấy (trường, status, host, endpoint), không đoán nguyên nhân (NFR-USAB-02). `observed` chỉ có kiểu, độ dài, sha1. Ví dụ: `examples/api-problem-422.json`.
- CLI in Problem cùng cấu trúc trong trường `error` của dòng `RunResult` trên stdout (§1.4, `internal/run-result.schema.json`). Mọi exception có exit code đều có slug và `status`; lỗi chỉ có ở CLI dùng cột "HTTP" làm giá trị tương đương.

---

## 5. Luật validator cho predictions

Chạy trước **mọi** lượt nộp (`make submit`, FR-VAL-03) và ở `make predict`, `make predict-private`:
1. `write_predictions` chuẩn hoá (FR-VAL-04), đọc lại và so với object gốc.
2. Lớp schema: `predictions-t{1,2,3}.schema.json` (Draft 2020-12), chạy bằng `jsonschema` ở nhóm core (07 QĐ 9) để có cả khi cài `--no-dev`.
3. Lớp ngữ nghĩa: các luật "chỉ validator" (`x-validator-only` trong schema).
4. Script BTC (`val.official_check_cmd`) khi đã phát; kết luận khác lớp 2–3 → `SubmissionFormatError`, chặn nộp (FR-VAL-02).
5. File thuộc đúng một run hoàn chỉnh; quota; lý do nộp (FR-VAL-03, FR-VAL-05).

| # | Luật | Lớp | Ca TC-VAL-01 / TC khác | FR / BR |
|---|---|---|---|---|
| 1 | UTF-8 hợp lệ, không BOM, JSON parse được | ngữ nghĩa (bytes) | (12) | FR-VAL-01, FR-VAL-04 |
| 2 | Top-level đúng `val.layout` (mặc định mảng) | schema | (11) | FR-VAL-01 |
| 3 | Mỗi phần tử đúng 2 khoá `sample_id`, `output`; `output` đúng khoá của task | schema | — | FR-VAL-01 |
| 4 | Tập `sample_id` = tập id input của split (schema không ép pattern id, §3.4) | ngữ nghĩa | (1), (2); TC-VAL-02 | FR-VAL-01 |
| 5 | Không trùng `sample_id` (`uniqueItems` không đủ vì so cả object) | ngữ nghĩa | (3); TC-VAL-02 | FR-VAL-01 |
| 6 | Thứ tự phần tử = thứ tự input | ngữ nghĩa | — | FR-VAL-04 |
| 7 | Điểm là số nguyên: chặn chuỗi và `bool` (schema); chặn `7.0` (ngữ nghĩa) | cả hai | (4), (5) | FR-VAL-04 |
| 8 | Miền BR-02 cho 6 chiều và total | schema | (9); TC-VAL-04 | BR-02 |
| 9 | `total_score` = Σ 6 chiều | ngữ nghĩa | (7); TC-VAL-04 | BR-01 |
| 10 | Nhãn ∈ 10 chuỗi LS; nhãn NFD tự trượt `enum` (với điều kiện test "NFC trong contracts/" ở §6.2 xanh) | schema | (6), (10); TC-VAL-03 | BR-12 |
| 11 | Không trùng nhãn | schema | — | BR-12 |
| 12 | Nhãn theo thứ tự LS; `[]` hợp lệ | ngữ nghĩa | TC-T2-01 | FR-T2-01, BR-13 |
| 13 | Feedback có ký tự khác khoảng trắng; không ký tự điều khiển trừ `\n` | schema | (8) | FR-T3-01 |
| 14 | Feedback ở NFC và nhận diện là tiếng Việt | ngữ nghĩa | TC-T3-01 | FR-T3-01 |
| 15 | Không có khoá nội bộ (`gate_source`, xác suất, `run_id`) | schema | — | FR-VAL-04 |
| 16 | Script BTC đồng ý | ngoài | TC-VAL-06 | FR-VAL-02 |
| 17 | sha file khớp output của đúng một run hoàn chỉnh | ngoài | TC-VAL-08, TC-SEC-06 | FR-VAL-03, FR-SEC-05 |

`ValidationReport` ghi ra `runs/<run_id>/validation-<task>.json`: `{file, task, contract_version, ok, errors: [{sample_id, pointer, rule, observed}]}`. Báo **mọi** lỗi, không dừng ở lỗi đầu (FR-VAL-02, NFR-USAB-02). Trong `PrivateRun`, `observed` chỉ có kiểu, độ dài, sha1.

---

## 6. Ánh xạ hợp đồng ↔ test case

### 6.1 Hợp đồng → TC đã có

| Hợp đồng | TC | Ghi chú |
|---|---|---|
| `dataset/*` (input quan sát) | TC-ING-01, TC-ING-02 | Schema JSON là lớp thứ hai cạnh pydantic |
| `label-space` và chuỗi NFC | TC-ING-01, TC-T2-02, TC-VAL-03 | |
| Thiếu `code_file` ở predict, `missing_code` | TC-ING-06 | khớp `if/then` của evidence card |
| Tường lửa BlindSample, input T1/T2 | TC-ING-09, TC-T1-09, TC-T2-10, TC-SEC-01, TC-SEC-02 | |
| CondSample, không ghép `test_report` | TC-ING-10, TC-T3-04 | thêm test "Nguồn test của card T3" ở §6.2 cho đường evidence |
| Predictions + validator | TC-VAL-01..05, TC-T1-01, TC-T2-01, TC-T3-01 | §5 |
| Script BTC, biến thể định dạng | TC-VAL-06 | |
| Quota, run_id, upload | TC-VAL-07, TC-VAL-08 | |
| EvidenceCard (schema, tất định, không feedback) | TC-EVD-11, TC-EVD-12 | |
| `Verdict`, `fail_class`, cờ stdout | TC-EVD-02, TC-EVD-04, TC-EVD-05, TC-EVD-06, TC-EVD-07, TC-EVD-14 | xem lệch #4, #5 ở §7 |
| ProblemCard | TC-POL-01, TC-POL-02, TC-POL-03, TC-POL-04 | |
| Run manifest, live calls = 0 khi replay | TC-REP-02, TC-REP-06, TC-REP-07 | |
| Exit code và thông điệp lỗi | TC-ING-02, TC-ING-06, TC-REP-03, TC-SEC-06 | |
| Log không chứa code hay feedback | TC-SEC-08 | |
| Egress, chạy offline | TC-SEC-04, TC-SEC-05 | áp cho cả service REST |
| REST | chưa có | đề xuất ở §6.2 |

### 6.2 Đề xuất test hợp đồng mới (chưa đánh số TC; QA gán khi đưa vào 05)

| Test | Mô tả | Marker, môi trường |
|---|---|---|
| Schema quan sát trên bộ mẫu | Validate 3 file task, `exams.json`, `label_space.json` của 32 bài bằng `dataset/*`; 0 lỗi. Mọi vi phạm `additionalProperties` được audit ghi thành cảnh báo | `fixture32`, CI-local |
| Parity pydantic ↔ schema | `Model.model_json_schema()` của từng model §2 so với schema trong `contracts/` (tập field, `required`, enum, miền). Mỗi file `examples/` validate được bằng cả JSON Schema lẫn `Model.model_validate_json`. Chỉ xanh khi 03 nhận §7.1 #19–#21, #23; nếu 03 không sửa, test **bỏ qua có ghi lý do** (a) enum nhãn của `taxonomy_error` ở T2Output, T3Input, CondSample, T2Prediction, (b) `additionalProperties: false` của model lồng trong EvidenceCard/ProblemCard, (c) trường mới `tests_source`, `tests_sha`, `compile_given = null` | `unit`, CI-cloud |
| Ví dụ hợp lệ | Mọi `examples/*.json` validate theo schema tương ứng; `api-*` theo component OpenAPI (cả `/v1/assess`); `run-result-*` theo `run-result.schema.json` | `unit`, CI-cloud |
| NFC trong contracts/ | Mọi chuỗi trong `enum`, `const`, `required` và tên property của mọi file `contracts/` thoả `unicodedata.is_normalized("NFC", s)`; 10 nhãn và 4 level trùng bytes với `constants.py` | `unit`, CI-cloud |
| Nguồn test của card T3 | Card đưa vào `T3Generator.generate` (batch) và card của `/v1/feedback` có `tests_source ∈ {public_tests, none}`: không test nào lấy từ `test_report`, `text` không chứa số test của report. Bật `t3.use_t1_test_report` mới cho phép `given_report` | `compliance`, CI-cloud |
| Khoá cache evidence | Cùng `code_sha`: card `full` và `given_signals_only`, card `given_report` và `public_tests` cho 4 khoá khác nhau; không entry nào đè entry khác | `unit`, `llm_cache` |
| Tường lửa âm | Thêm lần lượt từng khoá cấm vào BlindSample, CondSample, BlindRequest: schema báo đúng khoá; pydantic raise | `compliance`, CI-cloud |
| Phân lớp lỗi predictions | 12 file hỏng của TC-VAL-01 được xếp "schema bắt" hoặc "chỉ validator bắt" đúng như §5; không file nào qua được cả hai lớp | `unit` |
| Số nguyên `7.0` | File có `7.0` qua JSON Schema nhưng bị validator chặn (chứng minh cần lớp ngữ nghĩa) | `unit` |
| Ma trận exit code | Mỗi exception của 03 §6.2 (gồm `QuotaExceeded`) → đúng exit code; dòng cuối stdout validate theo `run-result.schema.json`, `error` có slug và `status` của §4; stderr không chứa canary | `unit`, `privacy` |
| Ma trận Problem | Mỗi exception → đúng HTTP status và slug; body không chứa giá trị gốc (canary). `components.schemas.Problem` của OpenAPI và `$defs/Problem` của `run-result.schema.json` trùng tập field, `required`, enum `code` | `unit` (REST) |
| Lint OpenAPI | `openapi-spec-validator` pass; mọi `$ref` tương đối và `externalValue` tồn tại | `unit`, CI-cloud |
| Fuzz REST | Schemathesis sinh request từ `openapi.yaml` với code tổng hợp: không có 5xx ngoài danh sách; mọi response validate theo schema. Chỉ bật khi schemathesis qua `uv lock` cùng hypothesis, jsonschema đã pin (07 §2.6); trước đó test bị skip có lý do | CI-local, `sandbox` |
| Idempotency | Cùng khoá + cùng body → response giống byte + `Idempotent-Replayed`; khác body → 409 | `unit` (REST) |
| Khoá phiên bản | File trong `contracts/` đổi sha mà `contract_version` không đổi → CI đỏ | CI-cloud |
| REST offline, không log code | Service chạy trong container `--network none`: `/healthz` ok, `/v1/grade` 200 từ cache; gửi code chứa canary → 0 canary trong `log.jsonl` và stderr | `privacy`, CI-local |

---

## 7. Lệch so với 03 §2 và câu hỏi mở

### 7.1 Lệch và chỗ cần làm rõ

| # | Chỗ | 03 nói | Hợp đồng làm | Đề xuất sửa 03 |
|---|---|---|---|---|
| 1 | Sample, input, output của file task | pydantic mặc định (`ignore`) | schema JSON `additionalProperties: false`; audit hạ thành cảnh báo | ghi rõ `extra="ignore"` và vai trò của schema quan sát trong audit |
| 2 | Ràng buộc quan sát | `sample_id`, `exam_id`, `code_file`, `compile_log` là `str` | pattern, `maxLength 2000`, multi ⇒ `test_report = null`, single ⇒ mảng, single có 1 câu [GT] | giữ ở schema quan sát; chỉ đưa vào pydantic sau `make audit` trên train |
| 3 | Số nguyên | `strict=True` | JSON Schema `integer` chấp nhận `7.0` | validator đọc JSON thô (luật 7 ở §5) |
| 4 | `Verdict` | 5 giá trị | giữ 5 giá trị; nhưng 03 §3.2 dùng `sample_timeout`, TC-EVD-15 dùng `skipped: harness_disabled` | thêm `sample_timeout` vào `Verdict`; harness tắt thể hiện bằng `run = null` (multi) |
| 5 | `TestRun` | không có stderr | không thêm | FR-EVD-04 và TC-EVD-02 đòi lưu stderr: thêm `stderr_excerpt: str \| None` (cắt ở `sandbox.stdout_cap_bytes`) và `stderr_sha1`, tăng `schema_version` |
| 6 | `StaticEvidence.functions_present` | `dict[str, FunctionInfo]` | khoá là `pid` | ghi rõ khoá = pid, cùng khoá với `ProblemStatus.by_pid` |
| 7 | `ProblemStatus` | chỉ có comment | `{split_failed, by_pid: {pid: {correct, partial, wrong, missing}}}` là xác suất | định nghĩa class `ProblemStatus`, `StatusProbs` |
| 8 | `EvidenceCard.problem_status`, `policy` | `\| None` | card từ EVD ghi `null`; pipeline gắn kết quả POL rồi mới ghi cache (§1.5) | ghi thứ tự dựng card |
| 9 | Ràng buộc chéo trong card | văn xuôi ở 03 §3.2, §3.10 | `if/then`: `given_signals_only` ⇒ không rebuild, test chỉ `given_report`; `missing_code` ⇒ `given_signals_only`; multi ⇒ `run = null`; single ⇒ `functions_present = {}`; `strict_pass` ⇒ `fail_class = null`; thêm theo #19, #20, #23: `given_signals_only` ⇒ `toolchain_id` sentinel và `tests_source ∈ {given_report, none}`; `tests_source = none` ⇔ `run = null` ⇔ `tests_sha = null`; `public_tests` ⇒ mọi test `rerun`; `compile_given = null` ⇒ `full` + có rebuild | thêm `model_validator` tương ứng |
| 10 | `ProblemCard.exam_sha` | `str` | 40 hoặc 64 hex | chốt sha256 |
| 11 | `policy_failed` ⇔ `extraction_method = fallback_uniform` | ngầm định | `if/then` hai chiều | ghi rõ |
| 12 | Run manifest | không có model ở §2 | schema mới, tên trường đề xuất. Mọi mục TC-REP-06 kiểm (`lockfile_sha256`, `engine`, `docker`, `folds_sha256`, `gpu_hours`, `api_cost_usd`, `finished_at`) là khoá bắt buộc, `null` khi không áp dụng; `command` có `evd`; `exit_code = null` khi `status = running` | thêm `rep/manifest.py:RunManifest` vào §2 |
| 13 | `T1Prediction.gate_source` | metadata nội bộ, không ra file nộp | có ở `$defs/T1Prediction` và response REST; schema predictions không cho phép | — |
| 14 | `LabelSpace.rubric: RubricSpace` | `RubricSpace` chưa định nghĩa | `{dimensions[{name, range, vi}], total_range}`, ghim bằng `const` | định nghĩa `RubricSpace`, `Dimension` |
| 15 | Bất biến JSON Schema không diễn đạt được | `n_samples == len`, Σ = total, DAG, Σ trọng số = 10, tổng xác suất = 1, thứ tự LS | ghi ở `x-invariants`, `x-validator-only` | giữ ở pydantic validator và VAL |
| 16 | Tên trường REST | `CondSample.level`, `code_file` | REST dùng `target_feedback_level` (như file task T3) và `code` inline | — (ánh xạ ở §1.6) |
| 17 | CLI | 03 §4 không có `serve`, không quy ước stdout; `evd calibrate`, `evd calibrate-neardup` chỉ có ở 03 §3.2 | thêm `grader serve` (tuỳ chọn); dòng `RunResult` theo `internal/run-result.schema.json`; bảng §1.4 đủ lệnh `evd`; `QuotaExceeded` có slug `quota-exceeded`, HTTP tương đương 429 | bổ sung 03 §4 (bảng lệnh, stdout) và 03 §6.2 (slug cho exception chỉ có ở CLI) |
| 18 | Thư viện | 03 §1.4 chưa có jsonschema, FastAPI, schemathesis…; hypothesis chỉ có ở 05 §1.1 | liệt kê ở §8 | bổ sung 03 §1.4 (jsonschema ở core; hypothesis; nhóm `api` tuỳ chọn) nếu nhóm chấp nhận |
| 19 | Nguồn test của evidence card cho T3 | §1.2 vẽ `EVD2 → T3D` như một card dùng chung; §3.6 bước 1 "CondSample + evidence"; §2.3 không có trường nguồn test; SRS FR-EVD-01 "card dùng chung mọi task" | card có `tests_source`, `tests_sha`; T3 chỉ nhận card `public_tests` hoặc `none`, trừ khi `t3.use_t1_test_report` (§1.5) | sửa 03 §1.2 (card T3 riêng), §3.6 bước 1 (`tests_from="public_tests"`), §2.3 (thêm 2 trường); đề xuất SRS ghi FR-EVD-01 là "dùng chung cho mọi task có cùng nguồn test" |
| 20 | Khoá cache evidence | §2.3 ghi khoá (`code_sha`, `toolchain_id`); §3.2 đường dẫn `cache/evidence/<toolchain_id>/<code_sha>-<tests_sha>.json`; không chỗ nào có `evidence_mode` | `x-cache-key` = (`code_sha`, `toolchain_id`, `tests_sha`, `evidence_mode`); `given_signals_only` dùng `toolchain_id = "000000000000"` | sửa comment 03 §2.3; đường dẫn 03 §3.2 thành `cache/evidence/<evidence_mode>/<toolchain_id>/<code_sha>-<tests_sha \| none>.json`; thêm `NO_TOOLCHAIN_ID` vào `constants.py` |
| 21 | Kiểu nhãn và model lồng | `taxonomy_error: list[str]` (T2Output, T3Input), `tuple[str, ...]` (CondSample); model lồng của EvidenceCard/ProblemCard (CompileLogInfo, RebuildResult, TestRun, RunEvidence, FunctionInfo, StaticEvidence, ProblemSpec, ZeroClause, PolicyFeatures) và cả ProblemCard không có `extra="forbid"` | enum 10 nhãn + `uniqueItems` + `maxItems 10`; mọi object `additionalProperties: false` | thêm `TaxonomyLabel = Literal[<10 nhãn>]` vào `constants.py`, dùng cho T2Output, T3Input, CondSample, T2Prediction; đặt `ConfigDict(extra="forbid")` cho ProblemCard và mọi model lồng (bytes card phải ổn định). Chưa sửa thì test parity bỏ qua phần này (§6.2) |
| 22 | Tường lửa import | 03 §3.11 (a) cấm theo class `ing.views.CondSample`, `ing.gold.GoldT3View`; cây §1.5 chỉ có `views.py`, `gold.py` | cấm theo module `grader.t3`, `grader.ing.cond_views`, `grader.ing.gold_t3` (§1.5) | 03 §1.5, §3.11 tách `CondSample`, `GoldT3View` ra module riêng; `sec/firewall.py` nhận shingle qua injection từ `cli.py` (07 §7 #4) |
| 23 | Card khi thiếu dữ liệu | `compile_given: CompileLogInfo`; `RebuildResult.agrees_with_given: bool`; không quy ước khi `n_tests = 0` | `compile_given = null` khi client REST không gửi log, khi đó `agrees_with_given = null`; `n_tests = 0` ⇒ `strict_rate = lenient_rate = io_gap = 0.0`, `tests = []` | sửa kiểu ở 03 §2.3 (`CompileLogInfo \| None`, `bool \| None`); ghi quy ước `n_tests = 0` (single không build được, `test_report = []`) |

### 7.2 Câu hỏi mở

**Q đã có, ảnh hưởng trực tiếp tới hợp đồng:**

| Q | Ảnh hưởng | Mặc định đang chạy |
|---|---|---|
| Q-01 | Layout predictions (mảng hay object, lồng hay phẳng, 1 hay 3 file) | `predictions-t{1,2,3}` như §1.3; `x-status: proposed-pending-Q-01` |
| Q-11 | Đơn vị quota trong `submit` | 1 upload = 1 lượt |
| Q-14 | `CondSample.t1_test_report`; `FeedbackRequest` có nhận `test_report` không; card T3 được dùng `tests_source = given_report` không | không; card T3 chỉ `public_tests` hoặc `none` (§1.5); thêm sau thì là MINOR |
| Q-16 | `run` cho multi_problem khi bật harness EX01 | `run = null` với multi |
| Q-17 | Trường mới (`compiles_locally`) trong file task | audit cảnh báo; thêm vào schema dạng tuỳ chọn |
| Q-22 | Handler REST sync hay async | sync (`def`), theo đề xuất PY-00 |
| Q-26 | `gate_source = exam_type_default` cho exam mới | lý do duyệt `new_exam` |
| Q-05, Q-06 | Phạm vi tường lửa; service REST có được chạy, dữ liệu đi đâu | hiểu rộng nhất; chỉ mạng nội bộ, offline |

**Đề xuất Q mới (chưa đánh số):**
- **Có làm lớp REST (iv) không?** SRS chưa có FR cho web service, SRS §1.2 để triển khai production ngoài phạm vi, scenario §1 ghi "không có web service" (cùng câu hỏi ở 07 §7 #2). Tới khi có trả lời, §1.6, `openapi.yaml` và các ví dụ `api-*` là **không chuẩn tắc**: không test nào của chúng chặn merge hay gate private, và extra `api` không cài vào image.
- **Pre-commit và ví dụ tổng hợp:** pre-commit chặn JSON có khoá `samples` hoặc `feedback` (03 §3.11), nên sẽ chặn `examples/predictions-t3.json`, `api-feedback-*.json` và `api-assess-response.json`. Đề xuất miễn trừ đúng thư mục `docs/contracts/examples/`, với điều kiện mọi `sample_id` thuộc dải tổng hợp `S9xx` và linter tài liệu (TC-SEC-08) báo 0 hit.
- **Thuật toán hash:** `exam_sha` và `config_sha6` trong `run_id` dùng sha1 hay sha256?
- **Card thiếu trường:** thêm `sample_timeout` vào `Verdict`, thêm stderr vào `TestRun` (§7.1 #4, #5)?
- **`/v1/assess` cho hệ thống (2):** khi giảng viên không gửi nhãn, T3 có được dùng nhãn T2 dự đoán (`labels_source = predicted`) không?
- **`compile_log` dài hơn 2000 ký tự từ client REST:** từ chối (hiện tại, 422) hay cắt theo quy ước của BTC?
- **Mốc đóng băng `1.0.0`:** khi có trả lời Q-01, hay muộn nhất ở gate private?
- **Vận hành REST:** ai cấp và thu hồi token, service chạy trên máy nào (gắn với Q-06)?

---

## 8. Thư viện sử dụng cho lớp hợp đồng

Inventory, version và nhóm cài đặt chuẩn ở [07](07-libraries.md) và `deps/pyproject.draft.toml`; bảng dưới chỉ ghi vai trò ở lớp hợp đồng. Thư viện chỉ dùng cho REST nằm trong extra `api` (07 §2.4). Pipeline private không import chúng; fastapi và uvicorn vẫn có mặt gián tiếp trong image qua vllm (07 §2.8).

| Thư viện | Dùng cho | Lớp | Có ở 03 §1.4 | Ghi chú (nhóm theo 07) |
|---|---|---|---|---|
| pydantic v2 | Model §2; `model_json_schema()` xuất Draft 2020-12 sang `schemas/` | i, iii, iv | có | core; nguồn gốc của schema |
| pydantic-settings | `Settings`, biến `GRADER_*`, `SecretStr` | ii | có | core |
| Typer | CLI, exit code, dòng `RunResult` | ii | có | core |
| jsonschema (`Draft202012Validator`) | Lớp schema của predictions lúc chạy thật (§5 bước 2); validate schema quan sát và ví dụ trong test | i, iii | mới | **core** (07 QĐ 9); bật `FormatChecker` cho `date-time`, `uri` [KCK] |
| referencing | Registry cho `$ref` tương đối của `openapi.yaml` trong test hợp đồng | iv | mới | dev; runtime không import (schema predictions tự đứng) |
| stdlib `json`, `unicodedata`, `hashlib` | Writer byte-stable, NFC, sha1/sha256 | i | — | không dùng thư viện JSON khác để giữ bytes ổn định |
| pytest, hypothesis | Property test cho validator, hậu xử lý; test hợp đồng §6.2 | i–iv | pytest: có; hypothesis: chỉ ở 05 §1.1 | dev |
| import-linter | Tường lửa import T1/T2 ↛ T3 (theo module, §1.5) | iii | có | dev |
| httpx | Egress (sync); client test REST | iii, iv | có | core |
| FastAPI + uvicorn | Service REST tuỳ chọn, handler `def` (sync) | iv | mới | extra `api`; bind loopback |
| openapi-spec-validator | Lint `openapi.yaml` 3.1 trong CI | iv | mới | dev; [KCK] hỗ trợ 3.1 |
| schemathesis | Fuzz/contract test REST từ OpenAPI | iv | mới | dev, **chưa kiểm** version, license, tương thích với hypothesis/jsonschema; để dạng comment tới khi qua `uv lock` (07 §2.6) |
| PyYAML | Đọc `openapi.yaml` trong test | iv | — | đã có ở core (07 §2.1) |
