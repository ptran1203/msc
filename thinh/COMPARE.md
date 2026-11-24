# So sánh: track `thinh/` và `main` (`llm/`)

Mốc main: commit `504a129` (2026-11-23). Track thinh: POC tháng 10/2026, chỉ trên 32 bài `sample_dataset`.

## 1. Tổng quan

| Mặt | main (`llm/`) | thinh (`thinh/`) |
|---|---|---|
| Dữ liệu dùng | 32 sample + **328 public test** + 593 bài synth (bug injection) | Chỉ 32 sample |
| Leaderboard | Có nộp: T1 **0.7991**, T2 **0.1649**, T3 **0.30** | Chưa nộp |
| Model | Chỉ open-weight: Qwen2.5-Coder-7B (4-bit), Qwen3-8B thinking, Qwen 14B làm teacher | Claude Opus/Haiku (API, qua `claude` CLI) + Qwen3-8B, Qwen2.5-VL-7B local |
| Phần cứng | RTX 3060 12GB, Kaggle T4, Colab | Apple M5 32GB (MPS) |
| Fine-tune | Có: QLoRA 1.5B (xong), 7B (v5 hết giờ, v6 hết quota) | Chỉ thiết kế |
| T3 | Chạy đầy đủ: RAG cùng level, luật hậu xử lý, ensemble | Chỉ thiết kế (Diagnose → Render, guard 4 lớp) |
| EDA / audit nhãn | Gọn, đủ để chạy | Rất sâu: meta-audit, verify độc lập, phát hiện lỗi nhãn, bootstrap CI |
| Tài liệu | METHOD + EXPERIMENTS dạng nhật ký | Bộ BA/SRS/spec/scenario/test/contract (~3.700 dòng) + schema JSON |

## 2. Chỗ hai bên tự đi đến cùng kết luận (độc lập)

| Phát hiện | main | thinh |
|---|---|---|
| Không áp luật tiên quyết P1 ở multi_problem | `task1-multi-v3`: bỏ prerequisite (QA giảng viên) | `runs+prereq` làm EX01 tụt .605 → .395 |
| Lỗi link (file chỉ có hàm, thiếu `main`) không phải lỗi biên dịch | Link-based compile log làm multi về ~0.7/10; sửa thành compile-only → 0.69 → 0.73 | Cổng chỉ bắt `file:line:col: error:` ngay từ đầu |
| single_problem chấm theo test | `task1-v4-single-testrule`: pass hết→10, chỉ sai format→9, một phần→7, không pass→3, build lỗi→1 | Luật EX02 từ test, EX02 QWK .95 (fixed) |
| Chấm multi theo từng câu, code cộng điểm | Từng câu, trọng số đều | Từng câu × trọng số đề 1/4/2/3 |
| `total = sum(rubric)` | Bị leaderboard từ chối rồi mới sửa | Ép trong `Rubric` dataclass |
| Tín hiệu tất định > đổi model/prompt | T2 LinearSVC trên tín hiệu chạy thật thắng LLM | Predictor không LLM (.724) thắng mọi run prompt thuần |

## 3. Chỗ khác nhau thực sự

1. **Trọng số multi_problem.** main chia đều; thinh đọc 1/4/2/3 từ đề và thấy gold khớp "câu có làm" chứ không phải "câu chạy đúng".
   Ứng viên cải tiến cho main: thử trọng số đề + luật vòng lặp vô hạn (`attempt+loop`) trong `task1_multi.py`.
2. **Lỗi nhãn.** thinh tìm và chứng minh S214/S211/S216, S112, `max_score` 2.5 vs 1/4/2/3, `zero_division` —
   đây là phần **điểm thưởng +10%** mà main chưa khai thác.
3. **Quy mô và kết quả thật.** main có public test, synth data, fine-tune, 3 bảng leaderboard; thinh chưa chạm public test.
4. **Tuân thủ.** main giữ đúng "chỉ open-weight". thinh đã gửi code sinh viên thật qua Claude API — chấp nhận được
   cho phân tích nếu khai báo chi phí, nhưng **không được** dùng cho đường nộp bài; cần chuyển hết sang Qwen.
5. **T2.** main: Task 3 self-diagnosis + luật + SVC, local macro .30–.34, leaderboard .165. thinh: vote LLM .365/.405 local
   nhưng chỉ trên 32 bài; con số local của cả hai đều lạc quan (main ghi nhận CV .303 → leaderboard .15).

## 4. Đề xuất ghép hai track (chưa làm)

1. Đưa trọng số đề + `infinite_loop` vào `llm/src/task1_multi.py`, đo trên 15 bài EX01 rồi nộp.
2. Đưa phần audit nhãn của thinh vào báo cáo (mục bonus) và vào `llm/METHOD.md §1`.
3. Thay `call_llm` Claude trong `thinh/eda/poc` bằng `llm/src/llm_client.py` để mọi số đều từ open-weight.
