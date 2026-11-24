# Thư viện và phụ thuộc: hệ thống chấm rubric, phân loại lỗi, sinh phản hồi bài C++

| Mục | Giá trị |
|---|---|
| Phiên bản | 0.1, bản nháp để nhóm review |
| Ngày | 2026-10-04 |
| Vai trò soạn | Tech lead phụ trách dependency |
| Căn cứ | [Đặc tả kỹ thuật](03-technical-specification.md) §1.4, §1.5, §3, §5, §8; [Test case](05-test-cases.md) §1; dữ kiện resolve và metadata PyPI kiểm ngày 2026-10-04 (xem "Nguồn dữ kiện") |
| Sản phẩm kèm | [`deps/pyproject.draft.toml`](deps/pyproject.draft.toml): bản nháp `pyproject.toml`, chưa khoá |
| ID | **Không định nghĩa ID mới.** Chỉ trỏ C/BR/FR/NFR/Q (SRS), SC (scenario), TC (test). "QĐ n" chỉ là số thứ tự dòng trong bảng §1 |

**Nguồn dữ kiện (ghi ở cột "Version chốt"):**
- **lock**: `uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28` trên 5 nhóm core + infer + train + api + dev, ra 255 gói.
- **PyPI**: version mới nhất trên PyPI ngày 2026-10-04, dùng cho gói không có trong lock.
- **chưa kiểm**: không có trong dữ kiện. Tài liệu không ghi version hay license cho các gói này.
- Chưa có dữ kiện về: resolve cho macOS; version của `uv`, `gitleaks`, `make`; license của gói transitive.

**Ký hiệu:** [KCK] như đặc tả: cần kiểm chứng bằng `uv lock` hoặc `make smoke` (FR-REP-05) trước khi chốt.

**Kết quả kiểm thật (2026-10-05)** — đã chạy, thay cho các mục "[KCK] cần `uv lock`" bên dưới khi có xung đột thông tin:

| Kiểm | Kết quả |
|---|---|
| `uv lock` trên `deps/pyproject.draft.toml` (copy thành `pyproject.toml`, Python 3.12, 2 môi trường linux-x86_64 và darwin-arm64) | Đạt, 255 gói. vllm 0.30.0, torch 2.13.0, transformers 5.18.0, numpy 2.3.5, scipy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, peft 0.21.2, trl 1.14.1, bitsandbytes 0.50.2, pydantic 2.13.5, fastapi 0.136.3 (do vllm kéo theo), lightgbm 4.7.0, tree-sitter 0.26.0, tree-sitter-cpp 0.23.4, jsonschema 4.26.0, referencing 0.37.0 |
| macOS: `uv sync --dry-run --group dev` | 91 gói, không có vllm/torch/bitsandbytes → marker linux hoạt động |
| `ruff --show-settings` đọc `[tool.ruff]` | Hợp lệ |
| License bổ sung từ PyPI | referencing 0.37.0 MIT; schemathesis 4.29.1 MIT; rapidfuzz 3.14.6 MIT (đặc tả 03 §3.5 dùng, nhưng draft đang để comment → nhóm chọn: thêm `rapidfuzz>=3.14,<4` hoặc dùng `difflib` stdlib); pip-licenses 5.5.5 MIT (đề xuất cho quét license transitive). lightgbm: trường license trên PyPI trống, upstream repo dùng MIT → vẫn cần kiểm file LICENSE trong wheel trước W7 |

---

## 1. Tóm tắt quyết định

| QĐ | Quyết định | Lý do (dữ kiện) | Ảnh hưởng, trạng thái |
|---|---|---|---|
| 1 | **Python 3.12**, `requires-python = "==3.12.*"` | (a) Chỉ 3.12 đã resolve đủ 5 nhóm (255 gói). Với 3.11 mới resolve core + infer + train (+api), ra 227 gói; nhóm dev chưa thử. (b) scipy 1.18 và numpy 2.5 đòi Python ≥ 3.12, nên 3.11 bị giữ ở scipy 1.17.1. (c) Môi trường EDA trên macOS đang chạy 3.12, và các số golden (TC-T1-08, TC-EVL-02) sinh trên bản này | **Đề xuất đổi** so với đặc tả §1.4 (`==3.11.*`). Ảnh hưởng FR-REP-02 (pin lockfile) và NFR-REPRO-01/02 (cùng lock, cùng image). Phương án lùi: 3.11 vẫn resolve được với scipy 1.17.1, nhưng **bắt buộc** pin vllm (bẫy 1, §5.3) |
| 2 | **Một env Linux CUDA dùng chung cho infer và train**, `torch==2.13.0` | vllm 0.30.0 pin cứng `torch==2.13.0`. Tách env train (không có vllm) thì resolver chọn torch 2.14.1, tức là có hai bản torch | Adapter A3 train và serve trên cùng torch; image app chỉ có một digest (FR-REP-02) |
| 3 | **Env dev macOS chỉ cài core + dev** (thêm api nếu có lớp HTTP) | vllm, bitsandbytes, triton chỉ chạy trên Linux có GPU NVIDIA | Chạy được baseline CPU, chế độ `given_signals_only` (TC-REP-12), EDA, validator, guard lớp 1–2. Chưa chạy resolve cho macOS [KCK] |
| 4 | **Pin `vllm==0.30.0` là bắt buộc.** Pin chặt cả `torch==2.13.0` và `transformers==5.18.0` | Bẫy 1: không pin thì resolver lùi về vllm 0.5.0.post1 + torch 2.3.0 | `engine_version` nằm trong khoá cache (đặc tả §3.10), nên nâng vLLM làm mọi entry cache LLM thành miss. Khoá bản này sau smoke W2 |
| 5 | **Bỏ Unsloth**; train bằng PEFT + TRL + bitsandbytes | Bẫy 3: unsloth 2026.9.14 đòi `torch<2.13.0`, `transformers<=5.5.0`, `trl<=0.24.0`. Không pin thì resolver lùi về unsloth 2024.8 | Nếu thật cần tốc độ: env train riêng có lock riêng, ghi bản torch khác vào manifest. Không khuyến nghị |
| 6 | **Đưa `transformers` vào core**, chỉ dùng tokenizer và chat template trên CPU | Khoá cache có `prompt_token_ids` (đặc tả §3.10). Replay trên máy không GPU (Q-25) và đếm token của card (FR-EVD-01) cần đúng tokenizer đã pin mà không cần vLLM hay torch | Lệch nhỏ so với đặc tả §1.4, vốn chỉ dùng transformers khi huấn luyện. [KCK] token ids của transformers phải trùng của vLLM trên cùng revision; mọi import vllm/torch phải lười (PY-10) |
| 7 | **Không thêm index PyTorch riêng** | Lock chỉ resolve từ PyPI. Bản torch 2.13.0 trên PyPI kéo theo wheel CUDA 13 (`cuda-toolkit==13.0.3.0`, `nvidia-*-cu13`) | Máy GPU cần driver hỗ trợ CUDA 13 [KCK]; ghi phiên bản driver vào `env.json` |
| 8 | **Chia nhóm**: extras `infer`, `train`, `api`, `teacher` cho runtime; dependency-groups (PEP 735) `dev`, `experimental` | Nhóm dev không cần nằm trong metadata của gói; extras được chọn theo từng máy | Mọi gói infer/train mang marker `sys_platform == 'linux'` |
| 9 | **Sửa lại tập core so với lúc resolve**: bỏ `datasketch` và `charset-normalizer` (gói sau vẫn có mặt gián tiếp); **giữ `jsonschema` ở core**; chuyển `huggingface-hub` từ infer (lúc resolve được kéo vào qua `-r infer.in`) sang core; chuyển `lizard`, `iterative-stratification` sang experimental | Đặc tả §3.2 dùng Jaccard chính xác, không dùng MinHash. §3.1 decode theo chuỗi cố định utf-8 → cp1258 → latin-1. Lớp schema của validator predictions chạy trong `make predict`, `predict-private`, `submit` ([06](06-api-contract.md) §5), nên phải có khi cài `--no-dev` (máy sạch, `make reproduce`, TC-REP-02). `make setup` tải weights theo revision và replay trên máy sạch cần huggingface-hub mà không cài infer (Q-25) | Không làm đổi version nào trong lock. Resolve core trên macOS (có thêm jsonschema, huggingface-hub) chưa chạy [KCK] |

### 1.1 Môi trường

| Env | Nền tảng | Nhóm cài | Lệnh | Dùng cho (Test §1.2) |
|---|---|---|---|---|
| `gpu` | Linux x86_64, 1 GPU 24 GB | core + infer + train (+ dev ở CI-local) | `uv sync --frozen --extra infer --extra train` | GPU-24G, CI-local |
| image app | Như `gpu` | core + infer + train, không có dev | `uv sync --frozen --no-dev --extra infer --extra train` trong `docker/app.Dockerfile` | Dry-run và chạy private (TC-SEC-05, TC-VAL-09) |
| `dev-mac` | macOS arm64, không GPU | core + dev (+ api) | `uv sync --frozen` | Dev-CPU |
| `ci-cloud` | Linux x86_64, không GPU | core + dev | `uv sync --locked` | CI-cloud (`unit`, `compliance`, `privacy`) |
| Máy sạch | Linux hoặc macOS, không GPU, không Docker | core | `uv sync --frozen --no-dev` | Replay từ cache, TC-REP-02 (Q-25); điều kiện là QĐ 6 và QĐ 9 |
| sandbox | Image `gcc:13` pin digest | Không có env Python; `runner.py` chỉ dùng stdlib | — | Sandbox-g++13 |

---

## 2. Inventory theo nhóm

Cột "Ràng buộc" là giá trị đặt trong `pyproject.toml`; version cụ thể do `uv.lock` quyết định. Gói 0.x được chặn ở mức minor. Gói ảnh hưởng tới số liệu (pandas, scikit-learn, scipy) cũng chặn minor để OOF ổn định từng byte (TC-REP-01).

### 2.1 core

| Gói | Version chốt (nguồn) | Ràng buộc | License (PyPI) | Nhóm | Module (FR) | Vai trò | Thay thế | Rủi ro |
|---|---|---|---|---|---|---|---|---|
| pydantic | 2.13.5 (lock) | `>=2.13,<3` | MIT | core | ING, EVD, POL, VAL (FR-ING-01, FR-ING-05) | Hợp đồng dữ liệu (đặc tả §2); `BlindSample` frozen, `extra="forbid"` | attrs + cattrs | Thấp; dùng plugin `pydantic.mypy` |
| pydantic-settings | 2.15.0 (lock) | `>=2.15,<3` | MIT | core | `core/config.py` (FR-REP-02, NFR-SEC-02) | `Settings` đọc TOML + env `GRADER_`, `SecretStr` | dynaconf | Thứ tự nguồn TOML phải đúng đặc tả §5.1 |
| typer | 0.27.2 (lock) | `>=0.27,<0.28` | MIT | core | `cli.py` (NFR-USAB-01) | CLI sync, Makefile gọi vào | click (đã là phụ thuộc) | Bản 0.x: minor có thể phá API |
| pyyaml | 6.0.3 (lock) | `>=6.0.3,<7` | MIT | core | `configs/ablation.yaml`, `label_defs.yaml` (FR-RPT-01, FR-T2-03) | Đọc YAML | Chuyển sang TOML (stdlib) | Chỉ dùng `safe_load` |
| jinja2 | 3.1.6 (lock) | `>=3.1.6,<4` | BSD License | core | `llm/prompts/*.j2`, `t3/templates/` (FR-T3-03, FR-GRD-06) | Prompt; 44 template fallback | `string.Template` | Dùng `StrictUndefined` để slot thiếu thành lỗi |
| jsonschema | 4.26.0 (lock) | `>=4.26,<5` | MIT | core (QĐ 9) | `val/validator.py`: lớp schema của predictions ([06](06-api-contract.md) §5); test `schemas/*.json` (FR-VAL-01, FR-EVD-01) | `Draft202012Validator` | Chỉ pydantic + luật validator (bỏ lớp JSON Schema lúc chạy thật) | Cũng là transitive của vllm. Schema predictions tự đứng nên lúc chạy không cần import `referencing` |
| sqlite3 (stdlib) | Theo Python 3.12; libsqlite đi kèm bản build Python | — | stdlib | core | `llm/cache.py` (FR-REP-03) | Cache LLM chỉ thêm, chế độ WAL (đặc tả §1.4, §3.10) | — | Hành vi WAL phụ thuộc bản libsqlite: ghi `sqlite3.sqlite_version` vào `env.json` (đề xuất bổ sung FR-REP-04) |
| httpx | 0.28.1 (lock) | `>=0.28.1,<0.29` | BSD-3-Clause | core | **Chỉ** `sec/egress.py` (FR-SEC-04) | HTTP ra ngoài, mặc định tắt | `urllib` (stdlib) | Release cuối 2024-12. Lock có thêm `httpx2` 2.13.1 do SDK API kéo vào (§2.8) |
| numpy | 2.3.5 (lock; PyPI mới nhất 2.5.3) | `>=2.3.5,<2.4` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | core | EVL, T1, T2 | Mảng; `Generator` có seed (FR-REP-02) | — | Bẫy 2: stack vllm giữ numpy ở 2.3.5. Đặt cùng khoảng trên macOS để hai nền tảng giống nhau |
| pandas | 3.0.6 (lock) | `>=3.0.6,<3.1` | BSD 3-Clause License | core | OOF, `folds.csv`, `features/*.parquet` (FR-EVL-03) | Bảng dữ liệu | polars | Bản 3.x đổi một số mặc định; chặn minor |
| pyarrow | 25.0.1 (lock) | `>=25.0.1,<26` | Apache-2.0 | core | `oof/<task>/<run_id>.parquet` | Đọc ghi parquet | fastparquet | Wheel lớn |
| scipy | 1.18.1 (lock; với Python 3.11 là 1.17.1) | `>=1.18.1,<1.19` | Trường PyPI chỉ có dòng copyright (Enthought); cần kiểm file LICENSE | core | `rpt/bonus/soft_gating.py` (FR-POL-05), kiểm định (FR-EVL-02) | Tối ưu, thống kê | statsmodels | Bản 1.18 đòi Python ≥ 3.12 |
| scikit-learn | 1.9.1 (lock) | `>=1.9.1,<1.10` | BSD-3-Clause | core | `evl/metrics.py`, `evl/folds.py`, stacker, TF-IDF (FR-EVL-01, FR-EVL-03, FR-T1-09, FR-T2-04) | QWK, macro-F1 với `zero_division=np.nan`, `StratifiedGroupKFold` | Tự viết metric | Mặc định có thể đổi giữa các minor; golden TC-EVL-01..03 bắt được lệch |
| lightgbm | 4.7.0 (lock) | `>=4.7,<5` | Trường license trên PyPI rỗng; chưa kiểm (xem classifiers/LICENSE trước W7) | core | Nguồn G của T1/T2 (FR-T1-08, FR-T2-04) | GBDT có ràng buộc monotone | `HistGradientBoostingClassifier` (sklearn) | Cần `deterministic=True` và cố định `num_threads` để tái lập [KCK] |
| tree-sitter | 0.26.0 (lock) | `>=0.26,<0.27` | MIT License | core | `evd/static.py`, `grd/syntax.py`, `t3/grounding.py`, `t2/augment.py` (FR-EVD-05, FR-GRD-02, FR-T3-02, FR-T2-08) | Parser C++, chạy được cả khi bài lỗi biên dịch | Regex (yếu hơn) | Bản 0.x |
| tree-sitter-cpp | 0.23.4 (lock) | `>=0.23.4,<0.24` | MIT | core | Như dòng trên | Grammar C++ | — | Release 2024-11, trong khi binding là 0.26: tương thích ABI của grammar cần kiểm trong smoke [KCK] |
| transformers | 5.18.0 (lock) | `==5.18.0` | Apache 2.0 License | core (QĐ 6) | `llm/cache.py` (khoá theo token), `evd/card.py` (đếm token), F-arm (FR-REP-03, FR-EVD-01, FR-T1-05) | Tokenizer và chat template trên CPU; lớp model cần torch từ infer/train | `tokenizers` thuần + tự dựng chat template | Pin chặt, khớp ràng buộc `transformers>=5.10.4` của vllm. Chạy không có torch trên macOS [KCK] |
| huggingface-hub | 1.33.0 (lock; PyPI mới nhất 2.1.1) | `>=1.33,<2` | Apache-2.0 | core | `make setup` tải weights theo revision (FR-REP-02) | Tải một lần, sau đó chạy với `HF_HUB_OFFLINE=1` | Tải tay + checksum | Lock không lên 2.x: có trần từ transformers hoặc vllm, chưa xác nhận gói nào. Ruff cấm `upload_*`, `create_repo` (SC-39) |
| rank-bm25 | 0.2.2 (lock) | `>=0.2.2,<0.3` | Apache2.0 | core | Feature kNN, biến thể BM25 (đặc tả §5.2) | Truy hồi exemplar trên CPU | Tự viết BM25 (vài chục dòng) | Release cuối 2022-02; không khai báo `requires_python` |
| matplotlib | 3.11.2 (lock) | `>=3.11,<4` | Trường PyPI là đầu văn bản "License agreement for matplotlib…"; cần kiểm file LICENSE | core | `rpt/` (FR-RPT-06, FR-EVL-07) | Learning curve, đường Pareto | — | Dùng backend `Agg` |
| rapidfuzz | chưa kiểm | Chưa ghi | chưa kiểm | core (theo đặc tả §1.4) | `t2/vocab.py` (FR-T2-01) | Khớp mờ nhãn với `ratio ≥ 0.90` | `difflib.SequenceMatcher` (stdlib); chỉ có 10 nhãn | **Không có trong dữ kiện resolve**, nên draft để dạng comment. Quyết định ở §7 |

### 2.2 infer (chỉ Linux)

| Gói | Version chốt (nguồn) | Ràng buộc | License (PyPI) | Nhóm | Module (FR) | Vai trò | Thay thế | Rủi ro |
|---|---|---|---|---|---|---|---|---|
| vllm | 0.30.0 (lock) | `==0.30.0; sys_platform == 'linux'` | Apache-2.0 | infer | `llm/engine.py` (FR-T1-04, FR-T2-03, FR-T3-04, FR-GRD-04, FR-EVL-06) | `LLM.generate` offline, prefix caching, multi-LoRA, guided JSON, logprobs | transformers `generate` (chậm hơn nhiều) | Bẫy 1 và 2. Kéo theo SDK API, OpenTelemetry, mcp (§2.8). Đặt `VLLM_NO_USAGE_STATS=1` |
| torch | 2.13.0 (lock; PyPI mới nhất 2.14.1) | `==2.13.0; sys_platform == 'linux'` | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | infer, train | Engine, F-arm, huấn luyện | Tensor, CUDA | — | vllm pin bản này. Wheel CUDA 13 cần driver phù hợp [KCK] |
| xgrammar | 0.2.8 (lock, transitive của vllm) | Không khai báo (lock giữ) | Apache 2.0 | infer | Guided JSON: diagnose T3, trích problem card | Backend structured output | `llguidance`, `outlines-core` (đều có trong lock) | Forced-choice **không** đọc logprob dưới grammar mask (SC-28) |
| triton | 3.7.1 (lock, transitive) | Không khai báo | chưa kiểm | infer, train | Kernel GPU | — | — | Chỉ có trên Linux |

### 2.3 train (chỉ Linux)

| Gói | Version chốt (nguồn) | Ràng buộc | License (PyPI) | Nhóm | Module (FR) | Vai trò | Thay thế | Rủi ro |
|---|---|---|---|---|---|---|---|---|
| peft | 0.21.2 (lock) | `>=0.21.2,<0.22` + marker linux | Apache | train | `t1/f_arm.py`, `t2/f_arm.py`, adapter A3 (FR-T1-05, FR-T3-04) | LoRA, QLoRA | Unsloth (đã loại, QĐ 5) | Bản 0.x |
| trl | 1.14.1 (lock) | `>=1.14.1,<1.15` + marker linux | Apache-2.0 | train | SFT và DPO tuỳ chọn cho T3 (FR-T3-04) | `SFTTrainer`, `DPOTrainer` | Tự viết vòng train | Kéo theo `datasets` 5.0.1 (§2.8) |
| accelerate | 1.15.0 (lock) | `>=1.15,<1.16` + marker linux | Apache | train | Huấn luyện | Phân bổ thiết bị | — | — |
| bitsandbytes | 0.50.2 (lock) | `>=0.50.2,<0.51` + marker linux | MIT | train | QLoRA 4-bit (FR-T3-04, NFR-PERF-04) | Lượng tử hoá 4-bit | GPTQ, AWQ (chưa kiểm) | Chỉ chạy Linux + CUDA |
| safetensors | 0.8.0 (lock, transitive) | Không khai báo | Apache Software License | train, infer | Lưu adapter; sha256 vào manifest (FR-REP-02) | — | — | `*.safetensors` bị `.gitignore` chặn (FR-SEC-03) |
| tokenizers | 0.23.2 (lock, transitive của transformers) | Không khai báo | Apache Software License | core (gián tiếp) | Tokenizer nhanh cho transformers | — | — | — |

### 2.4 api (tuỳ chọn)

[06](06-api-contract.md) §1.6 và §8 đặc tả lớp REST (iv) là **tuỳ chọn**, không dùng trong `make predict-private`. Chỉ cài extra `api` khi nhóm giữ lớp này: quyết định ở §7 #2, gắn với đề xuất Q mới ở 06 §7.2. SRS chưa có FR nào cho web service, và scenario §1 ghi "không có web service". Lưu ý: pipeline private không import gói nào của nhóm này, nhưng fastapi 0.136.3 và uvicorn 0.54.0 vẫn có trong image app vì vllm kéo vào (§2.8).

| Gói | Version chốt (nguồn) | Ràng buộc | License (PyPI) | Nhóm | Module (FR) | Vai trò | Thay thế | Rủi ro |
|---|---|---|---|---|---|---|---|---|
| fastapi | 0.136.3 (lock; PyPI mới nhất 0.142.2) | `>=0.136.3,<0.137` | MIT | api | Lớp REST (iv), `grader serve` (06 §1.6; chưa có FR) | REST + OpenAPI | Dùng CLI Typer sẵn có, không cần HTTP | Lock giữ 0.136.3 dù đã có 0.142.2, tức là có trần từ stack vllm (chưa xác nhận). `fastapi-cli` kéo `sentry-sdk` (§2.8). Nếu Q-22 chọn sync thì viết endpoint bằng `def` |
| uvicorn | 0.54.0 (lock) | `>=0.54,<0.55` | BSD-3-Clause | api | ASGI server | — | hypercorn | Chỉ bind `127.0.0.1` (NFR-PRIV-02) |

### 2.5 teacher-optional

Chỉ dùng khi Q-06 cho phép. Trên máy GPU, hai gói này **đã có sẵn** vì vllm kéo vào; nhóm `teacher` chỉ để khai báo ý định và để cài trên macOS.

| Gói | Version chốt (nguồn) | Ràng buộc | License (PyPI) | Nhóm | Module (FR) | Vai trò | Thay thế | Rủi ro |
|---|---|---|---|---|---|---|---|---|
| anthropic | 1.11.0 (lock, qua vllm) | `>=1.11,<2` | MIT | teacher-optional | Level-swap T3 (FR-T3-06) | SDK API | Gọi HTTP trực tiếp qua `EgressClient` | Theo lock, SDK dùng `httpx2` chứ không dùng `httpx`, nên mặc định không đi qua `EgressClient` [KCK]. Ruff cấm import ngoài `sec/egress.py` |
| openai | 3.24.0 (lock, qua vllm) | `>=3.24,<4` | Apache-2.0 | teacher-optional | Như trên; hoặc endpoint OpenAI-compatible tự host | SDK API | Như trên | Như trên |

### 2.6 dev

| Gói | Version chốt (nguồn) | Ràng buộc | License (PyPI) | Nhóm | Module (FR/NFR) | Vai trò | Thay thế | Rủi ro |
|---|---|---|---|---|---|---|---|---|
| ruff | 0.16.10 (lock) | `>=0.16.10,<0.17` | MIT | dev | Toàn repo (NFR-MAINT-01) | Lint, format; `banned-api` thực thi PY-05 và FR-SEC-04 | flake8 + black | Minor mới thêm rule |
| mypy | 2.4.0 (lock) | `>=2.4,<2.5` | MIT | dev | Strict cho `core/`, `shared/`, `val/`, `evl/`, `sec/` (đặc tả §1.6) | Kiểm kiểu | pyright | `strict` là cờ toàn cục, không đặt theo module được → draft liệt kê từng cờ |
| pytest | 9.1.1 (lock) | `>=9.1.1,<10` | MIT | dev | `tests/` (NFR-MAINT-03) | Chạy test, marker theo Test §1.5 | — | Bật `--strict-markers` |
| pytest-cov | 7.1.0 (lock) | `>=7.1,<8` | MIT | dev | Coverage ≥ 80% cho module lõi (NFR-MAINT-03) | — | coverage CLI | Ngưỡng tính theo module; không đặt `fail_under` toàn cục |
| hypothesis | 6.168.3 (lock) | `>=6.168.3,<7` | **MPL-2.0** | dev | Property test (TC-T1-01, TC-T2-05) | — | Test tham số hoá viết tay | Copyleft mức file; chỉ ở dev, không vào image app |
| import-linter | 2.15 (lock) | `>=2.15,<3` | BSD 2-Clause License | dev | `[tool.importlinter]` (FR-SEC-01) | Cấm `t1`, `t2` import `t3` | Test AST | Chỉ kiểm theo **module**, không theo class (§7) |
| pre-commit | 4.6.2 (lock) | `>=4.6.2,<5` | MIT | dev | Hook chặn `*.cpp`, JSON dữ liệu, file > 1 MB, gitleaks (FR-SEC-03) | — | Git hook viết tay | Hook tải từ mạng lúc cài, nên pin `rev` |
| referencing | Có trong lock qua jsonschema; version chưa ghi | Chưa đặt cận (TODO sau `uv lock`) | chưa kiểm | dev | Test hợp đồng [06](06-api-contract.md) §6.2: resolve `$ref` tương đối của `openapi.yaml` sang `contracts/internal/*` | `referencing.Registry` | Gộp schema vào một file | Runtime không import trực tiếp (schema predictions tự đứng) |
| openapi-spec-validator | 0.9.0 (lock) | `>=0.9,<0.10` | Apache-2.0 | dev | Lint `contracts/openapi.yaml` (06 §6.2 "Lint OpenAPI") | — | — | Bỏ nếu không có lớp HTTP |
| schemathesis | chưa kiểm | Để dạng comment trong draft | chưa kiểm | dev (chờ) | Test "Fuzz REST" (06 §6.2) | Sinh request từ OpenAPI, kiểm response theo schema | Bỏ test fuzz; giữ test "Ví dụ hợp lệ" | Chưa kiểm tương thích với hypothesis 6.168.3 và jsonschema 4.26.0. Chỉ bỏ comment sau khi `uv lock` thành công |

### 2.7 experimental

| Gói | Version chốt (nguồn) | Ràng buộc | License (PyPI) | Nhóm | Module (FR) | Vai trò | Thay thế | Rủi ro |
|---|---|---|---|---|---|---|---|---|
| lizard | 1.24.0 (lock) | `>=1.24,<2` | MIT | experimental | Ứng viên feature cho chiều `complexity` (FR-T1-08) | Độ phức tạp cyclomatic | Tự tính trên AST tree-sitter | Đặc tả §1.4 chưa có gói này. Chỉ đưa vào core khi thắng ablation (FR-RPT-01) |
| iterative-stratification | 0.1.9 (lock) | `>=0.1.9,<0.2` | BSD 3 | experimental | Thử phân tầng đa nhãn cho T2 | `MultilabelStratifiedKFold` | `StratifiedGroupKFold` (đặc tả §3.8) | Không hỗ trợ group, mâu thuẫn với group CV theo cụm (FR-EVL-03). Release 2024-10, không khai báo `requires_python` |

### 2.8 Gói transitive cần lưu ý (có trong lock, không khai báo trực tiếp)

| Gói (version lock) | Bị kéo vào bởi | Lưu ý | Biện pháp |
|---|---|---|---|
| anthropic 1.11.0, openai 3.24.0, httpx2 2.13.1 | vllm | SDK của API thương mại có sẵn trong env GPU dù không bật `teacher` | Ruff `banned-api`; `sec.egress_allowlist = []`; chạy private trong `--network none` (TC-SEC-05) |
| opentelemetry-* 1.45.0, mcp 2.3.0 | vllm | Exporter OTLP chỉ gửi khi có endpoint cấu hình | Không đặt biến OTLP; `--network none` |
| fastapi-cli 0.0.32 → fastapi-cloud-cli 0.26.0 → sentry-sdk 2.71.0 | fastapi (do vllm hoặc nhóm api) | Trong env có client gửi lỗi ra ngoài | Không chạy lệnh cloud của fastapi CLI; `DO_NOT_TRACK=1` (đặc tả §3.11), hiệu lực cần kiểm [KCK] |
| datasets 5.0.1 | trl | Có thể tải dữ liệu từ Hub | Đề xuất thêm `HF_DATASETS_OFFLINE=1` vào image app |
| numba 0.65.0, llvmlite 0.47.0 | vllm | Nghi là nguồn trần numpy 2.3.5 (bẫy 2, chưa xác nhận) | Kiểm lại khi nâng vllm |
| cuda-toolkit 13.0.3.0, nvidia-*-cu13 | torch | Wheel runtime CUDA 13 | Driver hỗ trợ CUDA 13 [KCK] |
| opencv-python-headless, torchvision, torchaudio, torchcodec, pynvvideocodec | vllm | Phụ thuộc đa phương thức, hệ thống không dùng | Không cần làm gì; license chưa kiểm (§5.6) |

---

## 3. Đã cân nhắc nhưng không dùng

| Gói, dịch vụ | Tham chiếu (PyPI) | Lý do không dùng | Khi nào xem lại |
|---|---|---|---|
| W&B cloud (`wandb`) | Chưa kiểm | Đẩy prompt và metric (có thể chứa code SV) ra ngoài, vi phạm NFR-PRIV-02 (SC-38). `runs/<run_id>/manifest.json`, `summary.md`, `costs.jsonl` (FR-REP-04, NFR-OBS-02) đã đủ để theo dõi | TC-SEC-09 cho phép W&B offline hoặc self-host. Chỉ thêm khi nhóm cần dashboard và Q-06 cho phép |
| mlflow | 3.16.1 | YAGNI: manifest theo `run_id` đã thay cho tracking server. Trường license trên PyPI chỉ có dòng copyright | — |
| underthesea | 9.5.0, Apache-2.0 | Không cần tách từ tiếng Việt. `shared/vietnamese.py` chỉ nhận diện ngôn ngữ và văn phong bằng luật (FR-T3-07) | Khi guard lớp 4 (FR-GRD-04) cần gán nhãn từ loại |
| sentence-transformers | 6.1.0, Apache-2.0 | Qwen3-Embedding-0.6B chạy được qua vLLM hoặc transformers đã có; thêm gói là thêm một stack | Khi cần embedding ngoài vLLM (vẫn phải có torch) |
| unsloth (trong env chính) | 2026.9.14, Apache-2.0 | Bẫy 3 (§5.3) | Env train riêng, chỉ khi thật cần tốc độ |
| datasketch | 2.0.0, MIT | Đặc tả §3.2 tính Jaccard chính xác từng cặp trong exam (N nhỏ), không dùng MinHash | Khi số bài mỗi exam lên hàng nghìn |
| charset-normalizer (dùng trực tiếp) | 3.5.2, MIT; vẫn có trong lock qua `requests` | Đặc tả §3.1 decode theo chuỗi cố định để tất định | — |
| cppcheck, clang-tidy | Không phải gói PyPI | Đặc tả dùng tree-sitter + regex. Thêm vào image sandbox làm đổi `toolchain_id` và tăng bề mặt tấn công | — |
| pip-audit | Chưa kiểm | Chỉ đề xuất chạy ở CI (§5.5), không thêm vào core hay dev | — |
| Index PyTorch CUDA riêng | — | QĐ 7 | Khi driver máy GPU không hỗ trợ CUDA 13 |

---

## 4. Công cụ hệ thống, image, model weights

### 4.1 Công cụ và image

| Công cụ | Đã thấy | Cách pin | Vai trò | Trỏ |
|---|---|---|---|---|
| uv | Có trên máy dev, chưa ghi version | Ghi `uv --version` vào `env.json` (đề xuất bổ sung cho manifest của FR-REP-04); CI cài đúng bản đó | `uv lock`, `uv sync --frozen`, `uv sync --locked` | FR-REP-02, NFR-USAB-01 |
| Docker | 29.1.4 (máy dev) | Ghi version vào `env.json`. Cần cgroup để `--memory` và `--pids-limit` có hiệu lực [KCK] | Sandbox, mỗi bài một container | FR-SEC-02, NFR-SEC-01 |
| Image `gcc:13` | Chưa có cục bộ | `gcc:13@sha256:<digest>`; digest đi vào `toolchain_id` và manifest | Base của sandbox, thêm `python3-minimal` và `util-linux` (cho `setarch -R`) | FR-EVD-04, NFR-REPRO-04 |
| Image clang, `gcc:12` | — | Pin digest nếu dùng | Chỉ để calibrate profile `clang-O0-trap`, `gcc12-O2` (đặc tả §3.2) | FR-EVD-04 |
| gitleaks | Chưa ghi version | Pin `rev` trong `.pre-commit-config.yaml` và trong CI | Quét secret | FR-SEC-03, NFR-SEC-02 |
| git | Có | — | Tag `v1.0`, working tree sạch (TC-REP-07) | FR-REP-04 |
| GNU make | Chưa ghi version | — | Các target ở đặc tả §4 | NFR-USAB-01 |
| Driver NVIDIA | Chưa ghi | Ghi vào `env.json` | Phải hỗ trợ CUDA 13 [KCK]. Cần NVIDIA Container Toolkit nếu image app dùng GPU trong Docker [KCK] | NFR-PERF-04 |

### 4.2 Model weights

Revision luôn là commit SHA; `Settings` từ chối alias (TC-REP-11). Weights được tải ở `make setup`, sau đó chạy offline. Adapter được đối xử như dữ liệu và không bao giờ public (SC-39). License của base model và adapter phái sinh phải ghi vào model card (FR-RPT-05).

| HF id | License | Vai trò | Điều kiện dùng | Engine | Vào model card |
|---|---|---|---|---|---|
| `Qwen/Qwen3-8B` | Apache-2.0 | P-arm T1/T2, trạng thái câu (POL), diagnose và render T3, base của adapter A3 | Luôn dùng | vLLM bf16; QLoRA khi train | Có: base + adapter A3 phái sinh |
| `Qwen/Qwen3-1.7B` hoặc `Qwen/Qwen3-4B` | Chưa có trong dữ kiện; kiểm model card khi pin revision | F-arm T1/T2 và P-small đối chứng (FR-EVL-08) | Khi cổng D0 kích hoạt (đặc tả §5.3) | transformers (head) qua `llm/cache.py` | Có |
| `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 | Feature kNN | Luôn dùng; biến thể là BM25 | vLLM hoặc transformers [KCK] | Có |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | Chưa kiểm | Biến thể P-arm, pilot W2 | Tuỳ chọn | vLLM | Nếu dùng |
| ModernBERT-large | Chưa kiểm | Baseline classifier, không tính là F-arm (C-02) | Tuỳ chọn | transformers | Nếu dùng |
| Judge EVL và LLM-judge GRD: họ khác Qwen, 27–32B bản 4-bit | Chưa chọn | FR-EVL-06, FR-GRD-04 | Chọn ở W2 | vLLM, với định dạng lượng tử hoá vLLM hỗ trợ [KCK VRAM] | Có |
| Teacher local 27–32B bản 4-bit | Chưa chọn | Level-swap (FR-T3-06); chỉ dùng API khi Q-06 cho phép | Theo đặc tả §5.2 | Như dòng trên | Có, kèm chi phí (FR-RPT-03) |

---

## 5. Tương thích, khoá phiên bản, bảo mật, license

### 5.1 Ma trận Python

| Hạng mục | Python 3.11 | Python 3.12 |
|---|---|---|
| core + infer + train (+ api) | Resolve OK, 227 gói | Resolve OK |
| Thêm dev (đủ 5 nhóm) | Chưa resolve | Resolve OK, 255 gói |
| numpy khi có vllm | 2.3.5 (vllm đã pin) | 2.3.5; bản 2.5.x đòi Python ≥ 3.12 nhưng bị stack vllm giữ lại |
| scipy | 1.17.1 | 1.18.1 |
| vllm không pin | Lùi về 0.5.0.post1 + torch 2.3.0 (bẫy 1) | Chưa thử; vẫn pin |
| torch | 2.13.0 (vllm pin) | 2.13.0; env train riêng không có vllm sẽ lên 2.14.1 |
| unsloth chung env với vllm | Không được (bẫy 3) | Không được (bẫy 3) |
| Môi trường EDA hiện có | — | macOS, 3.12 |

### 5.2 Ma trận nền tảng × nhóm

| Nhóm | Linux x86_64 + NVIDIA | Linux x86_64, không GPU (CI-cloud) | macOS arm64 |
|---|---|---|---|
| core | ✓ | ✓ | ✓ [KCK resolve] |
| infer, train | ✓ | — | — (marker linux) |
| api | ✓ | ✓ | ✓ |
| teacher | Đã có qua vllm; chỉ dùng khi Q-06 cho phép | Tuỳ Q-06 | Tuỳ Q-06 |
| dev | ✓ (CI-local) | ✓ | ✓ |
| experimental | Tuỳ chọn | Tuỳ chọn | Tuỳ chọn |

### 5.3 Bẫy resolve

| Bẫy | Hiện tượng (dữ kiện) | Hậu quả nếu bỏ qua | Biện pháp trong draft |
|---|---|---|---|
| 1 | Python 3.11, core + infer, **không** pin vllm: resolver âm thầm chọn vllm 0.5.0.post1 + torch 2.3.0 (bản 2024). Pin `vllm>=0.30,<0.31` thì ra vllm 0.30.0 + torch 2.13.0 + numpy 2.3.5 | Mất `prompt_logprobs`, multi-LoRA, guided JSON của bản mới; smoke fail hoặc tệ hơn là lệch âm thầm (SC-28) | `vllm==0.30.0` |
| 2 | vllm 0.30.0 pin cứng `torch==2.13.0` và `transformers>=5.10.4`. Khi có vllm, numpy bị chốt ở 2.3.5 (trong lock có numba 0.65.0; nghi là nguồn trần, chưa xác nhận). Quan sát thêm từ lock: fastapi bị giữ ở 0.136.3 (đã có 0.142.2), huggingface-hub ở 1.33.0 (đã có 2.1.1) | macOS và Linux lệch phiên bản nếu để resolver tự chọn; số golden và OOF lệch giữa hai máy | `torch==2.13.0`, `transformers==5.18.0`; numpy `<2.4`, fastapi `<0.137`, huggingface-hub `<2` ở mọi nền tảng |
| 3 | unsloth 2026.9.14 đòi `torch<2.13.0`, `transformers<=5.5.0`, `trl<=0.24.0`; unsloth-zoo 2026.9.9 đòi `transformers<=5.17.0`, `trl<=1.13.0`. Thêm unsloth không pin vào cùng env thì resolver lùi về unsloth 2024.8 | Bản unsloth vô dụng, hoặc kéo lùi cả stack | Không có unsloth trong `pyproject` (QĐ 5) |

### 5.4 Quy trình khoá phiên bản

| Bước | Lệnh, cơ chế | Khi nào | Trỏ |
|---|---|---|---|
| 1. Sửa phụ thuộc | Sửa `pyproject.toml` → `uv lock` (lock universal, giới hạn bởi `tool.uv.environments`) → commit `uv.lock` trong cùng PR | Mỗi lần đổi | FR-REP-02 |
| 2. Gác lock ở CI | `uv lock --locked` (fail nếu lock cũ hơn `pyproject`), rồi `uv sync --locked` | Mỗi push, CI-cloud | NFR-REPRO-01 |
| 3. Cài trên máy | `make setup` = `uv sync --frozen` (đặc tả §4) | Máy mới, image app | NFR-USAB-01 |
| 4. Smoke trên GPU | `make smoke`: đúng revision; `logprobs` và `prompt_logprobs`; guided JSON; multi-LoRA; banned tokens; ứng viên forced-choice là token đơn; chạy 2 lần cho kết quả giống nhau; batch-invariant nếu bản pin hỗ trợ; sandbox (đặc tả §3.10, §8). **Đề xuất thêm:** token ids từ transformers (core) trùng với vLLM (QĐ 6); kiểm ABI tree-sitter-cpp; kiểm driver hỗ trợ CUDA 13 | Sau mỗi lần đổi lock; gate W2 | FR-REP-05, TC-REP-11, SC-28 |
| 5. Ghi manifest | `env.json`: python, sha `uv.lock`, torch, CUDA, driver, vLLM, GPU, VRAM (đặc tả §3.10). Đề xuất thêm: version `uv`, digest image app, `sqlite3.sqlite_version`. Sha `uv.lock` cũng ghi vào manifest (`lockfile_sha256`, 06 `run-manifest.schema.json`) | Mỗi run | FR-REP-04, TC-REP-06 |
| 6. Đóng băng | Sau smoke W2: không đổi vllm, torch, transformers (vì đổi là cache LLM miss toàn bộ). Muốn nâng thì làm PR riêng: `uv lock --upgrade-package <gói>` → smoke → golden → TC-REP-01. Không chạy `uv lock --upgrade` toàn bộ sau W2. Từ feature freeze W7: khoá hoàn toàn (Test §1.4) | W2, W7 | NFR-REPRO-01, NFR-REPRO-02 |
| 7. Image app | Build từ `uv.lock` với `--frozen --no-dev --extra infer --extra train`; ghi digest | Mỗi release ứng viên | FR-REP-02, NFR-REPRO-02 |

### 5.5 Cập nhật và bảo mật

- **Quét lỗ hổng (đề xuất, không thêm vào core):** chạy pip-audit trên kết quả `uv export` ở CI-cloud, hằng tuần và trước mỗi mốc. Việc quét chỉ gửi tên và version gói tới cơ sở dữ liệu lỗ hổng, không có dữ liệu SV, nhưng vẫn cần ghi vào văn bản "dữ liệu được đi đâu" (NFR-PRIV-02). Lỗ hổng mức cao trong gói runtime thì xử lý theo bước 6 của §5.4.
- **Chạy offline:** image app đặt `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1` (đặc tả §3.11). Đề xuất thêm `HF_DATASETS_OFFLINE=1` vì trl kéo `datasets` vào. Lệnh private chạy trong `--network none` (TC-SEC-05), nên mọi đường mạng gián tiếp ở §2.8 đều bị chặn.
- **Cưỡng chế bằng lint:** ruff `banned-api` cấm `os.environ` ngoài `core/config.py` (PY-05); cấm `httpx`, `requests`, `anthropic`, `openai` ngoài `sec/egress.py` (FR-SEC-04); cấm `huggingface_hub.upload_file`, `upload_folder`, `create_repo` (SC-39). Lint này bổ sung chứ không thay CI grep của đặc tả §3.11.
- **Secret:** gitleaks ở pre-commit và CI; API key chỉ đọc qua `SecretStr` (NFR-SEC-02, TC-SEC-03).
- **Chuỗi cung ứng:** `uv.lock` có hash; image dùng digest; pre-commit hook pin `rev`; không cài gói ngoài lock trong image.

### 5.6 Tuân thủ license

| Hạng mục | Kết luận | Căn cứ |
|---|---|---|
| Gói runtime trực tiếp (core, infer, train, api, teacher) | Theo trường license trên PyPI: MIT, BSD, Apache-2.0, hoặc tổ hợp permissive (numpy, torch). **Không có GPL** trong danh sách này | Dữ kiện PyPI §2 |
| Gói license chưa rõ | lightgbm: trường license trên PyPI rỗng; chưa kiểm (xem classifiers/LICENSE trước W7). referencing, schemathesis: chưa kiểm (§2.6). scipy, matplotlib: trường PyPI không phải mã SPDX → kiểm file LICENSE. triton: chưa kiểm | §2.1, §2.2 |
| hypothesis (MPL-2.0) | Chỉ ở nhóm dev; không vào image app (`--no-dev`); nhóm không phân phối lại | §2.6 |
| Gói transitive (255 gói trong lock) | **Chưa quét.** Đề xuất quét license toàn bộ env (ví dụ bằng pip-licenses) trước W7. Gói `nvidia-*`, `cuda-*` chưa kiểm license | §2.8 |
| Bối cảnh phân phối | Repo private, image không public, không phân phối lại thư viện. Nghĩa vụ chủ yếu là ghi nhận trong báo cáo | FR-SEC-03 |
| Model | License của base model và adapter phái sinh ghi vào model card; model "chưa kiểm" phải kiểm trước khi dùng | FR-RPT-05, §4.2 |

---

## 6. Ánh xạ thư viện ↔ FR/NFR/TC

| Thư viện | FR/NFR | TC |
|---|---|---|
| pydantic, pydantic-settings | FR-ING-01, FR-ING-05, FR-REP-02, NFR-SEC-02 | TC-ING-02, TC-ING-09, TC-REP-06 |
| typer, GNU make | NFR-USAB-01 | TC-REP-14 |
| jinja2 | FR-T3-03, FR-GRD-06 | TC-GRD-09 |
| httpx (`sec/egress.py`) | FR-SEC-04, NFR-COST-01 | TC-SEC-04 |
| numpy, scipy | FR-EVL-02, FR-POL-05, FR-REP-02 | TC-EVL-05, TC-POL-09 |
| scikit-learn | FR-EVL-01, FR-EVL-03, FR-T1-09, FR-T2-04 | TC-EVL-01, TC-EVL-03, TC-EVL-06 |
| lightgbm | FR-T1-08, FR-T2-04 | TC-T2-09 |
| pandas, pyarrow | FR-EVL-03, FR-REP-04 | TC-T2-09, TC-REP-01 |
| tree-sitter, tree-sitter-cpp | FR-EVD-05, FR-GRD-02, FR-T3-02, FR-T2-08 | TC-EVD-08, TC-GRD-02 |
| transformers (tokenizer, core) | FR-EVD-01, FR-REP-03 | TC-EVD-11, TC-REP-02 |
| rapidfuzz (hoặc difflib) | FR-T2-01 | TC-T2-02 |
| matplotlib | FR-RPT-06, FR-EVL-07 | TC-RPT-06, TC-EVL-11 |
| rank-bm25 | FR-EVL-03 (loại exemplar) | TC-EVL-07 |
| vllm, xgrammar | FR-T1-04, FR-T2-03, FR-REP-05, NFR-REPRO-04 | TC-T1-10, TC-T2-08, TC-REP-11 |
| torch, peft, trl, accelerate, bitsandbytes | FR-T1-05, FR-T3-04, NFR-PERF-04, NFR-REPRO-03 | TC-T3-10, TC-T1-12, TC-REP-05, TC-REP-13 |
| huggingface-hub | FR-REP-02, NFR-PRIV-02 | TC-SEC-05 |
| anthropic, openai (teacher) | FR-T3-06, FR-SEC-04, NFR-COST-01 | TC-SEC-04, TC-RPT-04 |
| fastapi, uvicorn, openapi-spec-validator, referencing, schemathesis | Lớp REST (iv) tuỳ chọn, [06](06-api-contract.md) §1.6 (chưa có FR) | Test hợp đồng đề xuất ở 06 §6.2 (chưa có TC) |
| sqlite3 (stdlib) | FR-REP-03 | TC-REP-02 |
| ruff, mypy | NFR-MAINT-01 | TC-REP-14 |
| pytest, pytest-cov, hypothesis | NFR-MAINT-03 | TC-T1-01, TC-T2-05, TC-REP-14 |
| import-linter | FR-SEC-01 | TC-ING-10 |
| pre-commit, gitleaks | FR-SEC-03, NFR-SEC-02, NFR-PRIV-01 | TC-SEC-03 |
| jsonschema | FR-VAL-01, FR-EVD-01 | TC-VAL-01, TC-EVD-11 |
| uv, `uv.lock` | FR-REP-02, NFR-REPRO-01, NFR-REPRO-02 | TC-REP-06, TC-REP-11 |
| Docker, image `gcc:13` | FR-SEC-02, FR-EVD-04, NFR-SEC-01 | TC-EVD-01, TC-EVD-13, TC-EVD-14 |

---

## 7. Câu hỏi mở và điểm cần nhóm quyết định

Không đánh số Q mới; các dòng "đề xuất Q mới" chờ nhóm đưa vào SRS §8 nếu đồng ý.

| # | Điểm cần quyết | Phương án | Khuyến nghị | Loại |
|---|---|---|---|---|
| 1 | Đổi `requires-python` từ `==3.11.*` (đặc tả §1.4) sang `==3.12.*` | (a) 3.12; (b) giữ 3.11, pin vllm, scipy 1.17.1, resolve lại cả nhóm dev | (a) | Đề xuất Q mới (nhóm tự chốt) |
| 2 | Có lớp HTTP không (nhóm `api`)? [06](06-api-contract.md) §1.6 đặt REST là lớp (iv) tuỳ chọn và nêu cùng câu hỏi ở 06 §7.2; SRS chưa có FR, scenario §1 ghi không có web service | (a) Không: bỏ fastapi, uvicorn, openapi-spec-validator; (b) Có: thêm FR/TC tương ứng, Q-22 quyết định endpoint sync hay async | Chốt cùng API contract | Đề xuất Q mới |
| 3 | `rapidfuzz` không có trong dữ kiện resolve | (a) Thêm vào core rồi `uv lock` lại và kiểm license; (b) thay bằng `difflib` (stdlib), đủ cho 10 nhãn | (b), vì ít phụ thuộc hơn; kiểm bằng TC-T2-02 | Quyết định kỹ thuật |
| 4 | import-linter không cấm được theo class (`CondSample`, `GoldT3View` ở đặc tả §3.11) | (a) Tách thành module riêng (draft đặt tên `grader.ing.cond_views`, `grader.ing.gold_t3`); (b) giữ nguyên và kiểm bằng test AST | (a); đồng thời `sec/firewall.py` nhận shingle qua injection từ `cli.py` để không vi phạm luật cấm import gián tiếp | Sửa đặc tả §1.5, §3.11 |
| 5 | Đưa `transformers` vào core (QĐ 6) để replay không cần GPU | (a) Core; (b) để ở infer và chấp nhận máy sạch phải cài infer (torch CUDA nặng nhiều GB) | (a), sau khi smoke xác nhận token ids trùng vLLM | Ảnh hưởng Q-25 |
| 6 | Máy GPU của nhóm có driver hỗ trợ CUDA 13 không? | (a) Có: giữ lock; (b) Không: thêm index PyTorch theo CUDA của máy và resolve lại (rủi ro lệch với wheel vllm) | Kiểm ở W1 | Đề xuất Q mới |
| 7 | Bỏ hẳn Unsloth | (a) Bỏ; (b) env train riêng, lock riêng, manifest ghi torch 2.14.1 | (a) | Quyết định kỹ thuật |
| 8 | Judge và teacher 27–32B bản 4-bit chạy bằng gì trên 1 GPU 24 GB | Định dạng lượng tử hoá vLLM hỗ trợ, hay bitsandbytes; nạp tuần tự với Qwen3-8B | Chốt ở smoke W2 (đặc tả §8) | [KCK] |
| 9 | Quét license gói transitive và quét lỗ hổng | pip-licenses, pip-audit chạy ở CI-cloud | Làm trước W7; không thêm vào core | Đề xuất |
