"""Pure text helpers for C++ student code: comments, tokens, functions, similarity.

No I/O and no compiler calls, so these helpers are safe to reuse in the grading
pipeline (feature extraction, plagiarism checks, harness generation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- comments


@dataclass
class StrippedCode:
    """Source split into code (comments removed) and the comments themselves.

    `code` keeps the same length/line layout as the original: comment characters
    become spaces (newlines kept), so offsets and line numbers stay valid.
    `blank` additionally replaces string/char literal contents with 'x', so brace
    matching never trips over a '{' inside a literal.
    """

    code: str
    blank: str
    comments: list[tuple[int, str]] = field(default_factory=list)  # (1-based line, text)
    comment_lines: set[int] = field(default_factory=set)


def strip_comments(src: str) -> StrippedCode:
    """Single-pass state machine for //, /* */, "..." and '...' (raw strings are rare in this corpus)."""
    code = list(src)
    blank = list(src)
    comments: list[tuple[int, str]] = []
    comment_lines: set[int] = set()
    i, n, line = 0, len(src), 1

    def wipe(start: int, end: int) -> None:
        for k in range(start, end):
            if src[k] != "\n":
                code[k] = " "
                blank[k] = " "

    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            end = src.find("\n", i)
            end = n if end == -1 else end
            comments.append((line, src[i + 2 : end]))
            comment_lines.add(line)
            wipe(i, end)
            i = end
        elif ch == "/" and nxt == "*":
            end = src.find("*/", i + 2)
            end = n if end == -1 else end + 2
            text = src[i + 2 : max(i + 2, end - 2)]
            comments.append((line, text))
            for k in range(text.count("\n") + 1):
                comment_lines.add(line + k)
            wipe(i, end)
            line += src.count("\n", i, end)
            i = end
        elif ch in "\"'":
            j = i + 1
            while j < n and src[j] != ch and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            for k in range(i + 1, min(j, n)):
                blank[k] = "x"
            i = min(j + 1, n)
        else:
            if ch == "\n":
                line += 1
            i += 1
    return StrippedCode("".join(code), "".join(blank), comments, comment_lines)


# ----------------------------------------------------------------------------- tokens

TOKEN_RE = re.compile(
    r"""
    (?P<str>"(?:\\.|[^"\\\n])*")
  | (?P<chr>'(?:\\.|[^'\\\n])*')
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[uUlLfF]*)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<op>::|->|<<=|>>=|<<|>>|\+\+|--|&&|\|\||[<>!=+\-*/%&|^]=|[{}()\[\];,.<>?:~!+\-*/%&|^=])
    """,
    re.X,
)

CPP_KEYWORDS = frozenset(
    """alignas alignof and asm auto bool break case catch char char16_t char32_t class const
    constexpr const_cast continue decltype default delete do double dynamic_cast else enum
    explicit extern false float for friend goto if inline int long mutable namespace new
    noexcept not nullptr operator or private protected public register reinterpret_cast return
    short signed sizeof static static_assert static_cast struct switch template this throw true
    try typedef typeid typename union unsigned using virtual void volatile wchar_t while NULL""".split()
)
# Library names keep their identity: they carry the solution's "shape" better than user names.
STD_NAMES = frozenset(
    """std cin cout cerr endl string getline ifstream ofstream fstream stringstream istringstream
    ostringstream stoi stof stod stol stoll to_string abs pow sqrt open close is_open eof clear
    seekg ios beg substr find length size push_back empty strlen strcpy strcmp strtok atoi
    printf scanf main vector""".split()
)
PREPROCESSOR_LINE_RE = re.compile(r"^\s*#.*$", re.M)


def tokenize(code_no_comments: str, drop_preprocessor: bool = True) -> list[tuple[str, str]]:
    """Return (kind, text) tokens; kind in {str, chr, num, id, op}."""
    text = PREPROCESSOR_LINE_RE.sub(" ", code_no_comments) if drop_preprocessor else code_no_comments
    return [(m.lastgroup or "op", m.group()) for m in TOKEN_RE.finditer(text)]


def normalize_tokens(tokens: list[tuple[str, str]]) -> list[str]:
    """Rename user identifiers/literals so renaming variables does not hide copying."""
    out: list[str] = []
    for kind, text in tokens:
        if kind == "id":
            out.append(text if text in CPP_KEYWORDS or text in STD_NAMES else "ID")
        elif kind == "num":
            out.append("N")
        elif kind == "str":
            out.append("S")
        elif kind == "chr":
            out.append("C")
        else:
            out.append(text)
    return out


def shingles(norm_tokens: list[str], k: int = 5) -> set[tuple[str, ...]]:
    return {tuple(norm_tokens[i : i + k]) for i in range(max(0, len(norm_tokens) - k + 1))}


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# -------------------------------------------------------------------------- functions

FUNC_HEADER_RE = re.compile(
    r"(?P<ret>(?:[A-Za-z_][\w:<>]*[\s\*&]+)+?)(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^(){};]*)\)\s*(?:const\s*)?\{"
)
NOT_FUNCTIONS = frozenset({"if", "while", "for", "switch", "catch", "return", "sizeof", "else", "do"})


@dataclass
class CppFunction:
    name: str
    ret: str
    params: str
    body: str  # comment-free text between the outer braces
    start_line: int


def match_brace(blank: str, open_idx: int) -> int:
    depth = 0
    for k in range(open_idx, len(blank)):
        if blank[k] == "{":
            depth += 1
        elif blank[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    return len(blank) - 1


def extract_functions(stripped: StrippedCode) -> list[CppFunction]:
    """Top-level-ish function definitions found by header regex + brace matching."""
    funcs: list[CppFunction] = []
    pos = 0
    blank = stripped.blank
    while True:
        m = FUNC_HEADER_RE.search(blank, pos)
        if not m:
            break
        name = m.group("name")
        ret = " ".join(m.group("ret").split())
        open_idx = m.end() - 1
        if name in NOT_FUNCTIONS or ret.split()[-1] in {"new", "return", "else"}:
            pos = m.end()
            continue
        close_idx = match_brace(blank, open_idx)
        funcs.append(
            CppFunction(
                name=name,
                ret=ret,
                params=" ".join(stripped.code[m.start("params") : m.end("params")].split()),
                body=stripped.code[open_idx + 1 : close_idx],
                start_line=blank.count("\n", 0, m.start("name")) + 1,
            )
        )
        pos = close_idx + 1
    return funcs


def normalize_type(text: str) -> str:
    text = re.sub(r"\bstd::", "", text)
    text = re.sub(r"\s*([\*&])\s*", r"\1", text)
    return " ".join(text.split())


def param_types(params: str) -> list[str]:
    """Drop parameter names: 'string filename, int& n' -> ['string', 'int&']."""
    if not params.strip() or params.strip() == "void":
        return []
    types = []
    for raw in params.split(","):
        part = normalize_type(raw.split("=")[0])
        part = re.sub(r"(?<=[\w\*&])\s*\b[A-Za-z_]\w*$", "", part) if re.search(r"[\s\*&]", part) else part
        types.append(part.strip())
    return types


def parse_prototype(proto: str) -> tuple[str, str, list[str]]:
    """'General* prob1(string filename, int& n);' -> ('General*', 'prob1', ['string', 'int&'])."""
    m = re.match(r"\s*(?P<ret>.+?)\b(?P<name>[A-Za-z_]\w*)\s*\((?P<params>.*)\)\s*;?\s*$", proto)
    if not m:
        raise ValueError(f"unparseable prototype: {proto!r}")
    return normalize_type(m.group("ret")), m.group("name"), param_types(m.group("params"))


# ------------------------------------------------------------------ text classifiers

VI_DIACRITIC_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I
)
CODE_LIKE_RE = re.compile(r"[;{}]|==|<<|>>|\b(?:int|return|cout|cin|for|while|if)\b")


def looks_like_code(text: str) -> bool:
    return bool(CODE_LIKE_RE.search(text))
