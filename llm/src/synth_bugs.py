"""Controlled bug taxonomy for synthetic data generation.

Each entry maps one of the official 10 taxonomy_error labels to a prompt instruction
(what bug to inject) and whether it's expected to break compilation. Labels are assigned
by CONSTRUCTION (we chose the bug), not by asking the LLM to self-diagnose -- the LLM is
only used to write the buggy code, never to grade it. Rubric scores come from the
objective oracle in synth_oracle.py.

Note: "Lỗi edge case" never appears in the real labeled data (see METHOD.md EDA finding) --
synthesizing it here directly fills that gap for training/few-shot purposes.
"""

BUG_TAXONOMY = {
    "compile_error": {
        "label": "Lỗi biên dịch",
        "instruction": "Sửa code để KHÔNG thể biên dịch được: xoá một dấu chấm phẩy, hoặc dùng một biến chưa khai báo, hoặc thiếu một dấu ngoặc đóng. Chỉ một lỗi cú pháp nhỏ, phần còn lại giữ nguyên.",
        "compile_breaking": True,
    },
    "io_format_error": {
        "label": "Lỗi nhập/xuất",
        "instruction": "Giữ logic đúng, nhưng làm SAI định dạng output: đổi cách viết hoa (ví dụ in 'TRUE'/'FALSE' thay vì 'True'/'False'), hoặc thêm khoảng trắng/newline dư thừa, hoặc đổi thứ tự các giá trị in ra. Logic tính toán bên trong vẫn đúng.",
        "compile_breaking": False,
    },
    "logic_error": {
        "label": "Lỗi logic",
        "instruction": "Làm sai một phép so sánh hoặc công thức tính toán cốt lõi (ví dụ đổi == thành !=, đổi dấu + thành -, hoặc sai công thức), khiến kết quả sai ở nhiều trường hợp thông thường, nhưng code vẫn biên dịch và chạy được.",
        "compile_breaking": False,
    },
    "loop_error": {
        "label": "Lỗi vòng lặp",
        "instruction": "Làm sai điều kiện hoặc biên của một vòng lặp (ví dụ dùng <= thay vì <, hoặc bắt đầu từ sai chỉ số, hoặc thiếu bước tăng khiến chạy sai số lần), nhưng không gây crash hay vòng lặp vô tận.",
        "compile_breaking": False,
    },
    "array_string_error": {
        "label": "Lỗi mảng/chuỗi",
        "instruction": "Làm sai việc truy cập mảng hoặc xử lý chuỗi (ví dụ lấy sai chỉ số substring, sai độ dài chuỗi con, hoặc off-by-one khi truy cập phần tử mảng), nhưng không gây crash ngay (vẫn chạy ra kết quả, chỉ sai giá trị).",
        "compile_breaking": False,
    },
    "function_error": {
        "label": "Lỗi hàm",
        "instruction": "Gọi một hàm với tham số sai, hoặc hàm trả về giá trị sai/thiếu return ở một nhánh, khiến kết quả cuối sai nhưng code vẫn biên dịch.",
        "compile_breaking": False,
    },
    "edge_case_error": {
        "label": "Lỗi edge case",
        "instruction": "Giữ logic đúng cho trường hợp thông thường, nhưng KHÔNG xử lý đúng một trường hợp biên cụ thể (ví dụ: input rỗng, n=0, chỉ 1 phần tử, tất cả phần tử giống nhau, chuỗi 1 ký tự). Code phải đúng với input thông thường và chỉ sai ở biên.",
        "compile_breaking": False,
    },
    "algorithm_error": {
        "label": "Lỗi thuật toán",
        "instruction": "Dùng một cách tiếp cận thuật toán SAI về bản chất (không phải lỗi cú pháp nhỏ) -- ví dụ chỉ xét một phần các khả năng cần xét, hoặc hiểu sai yêu cầu bài toán -- khiến kết quả sai một cách có hệ thống trên nhiều input khác nhau, không chỉ ở biên.",
        "compile_breaking": False,
    },
    "hardcode_error": {
        "label": "Lỗi hard-code",
        "instruction": "Viết code chỉ đúng cho đúng các ví dụ cụ thể trong đề bài (hard-code giá trị input/output của ví dụ, hoặc hard-code kích thước mảng theo ví dụ), sai hoàn toàn với input khác.",
        "compile_breaking": False,
    },
    "style_error": {
        "label": "Lỗi style",
        "instruction": "Giữ nguyên logic ĐÚNG HOÀN TOÀN, chỉ làm xấu code: đặt tên biến vô nghĩa (a, b, c, x1, x2...), không có cấu trúc rõ ràng, viết dồn nhiều lệnh trên 1 dòng. Không đổi hành vi chương trình.",
        "compile_breaking": False,
    },
}

# which bug types are meaningful for which reference problem (all apply to all 5 for now)
APPLICABLE_PROBLEMS = ["armstrong", "remove_duplicates", "earliest_login", "binary_search", "palindrome_check"]
