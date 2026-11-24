"""Compiler-log parsing, local re-compilation and sandboxed execution of student code.

Student code is untrusted: binaries are only run after a source-level deny-list
check, inside a fresh temp dir, with a wall-clock timeout, a CPU rlimit and a
zero file-size rlimit (so the program cannot write files even if the grep misses).
"""

from __future__ import annotations

import re
import resource
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# gcc/clang diagnostic: "file.cpp:12:5: error: msg"
DIAG_RE = re.compile(r"^(?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+): (?P<kind>fatal error|error|warning|note): (?P<msg>.*)$", re.M)
LINKER_ERROR_RE = re.compile(r"undefined reference to|Undefined symbols for architecture|collect2: error|ld returned|linker command failed")
MAIN_UNDEFINED_RE = re.compile(r"undefined reference to `main'|\"_main\", referenced from")
GNU_MARKERS = {
    "gnu_curly_quotes": re.compile(r"[‘’]"),
    "gnu_in_function": re.compile(r"In function ‘"),
    "gnu_libstdcxx_path": re.compile(r"/usr/include/c\+\+/\d+"),
    "gnu_collect2": re.compile(r"collect2"),
    "gnu_compilation_terminated": re.compile(r"compilation terminated\."),
    "gnu_fpermissive": re.compile(r"\[-fpermissive\]"),
}
CLANG_MARKERS = {"clang_n_generated": re.compile(r"\d+ (?:errors?|warnings?)(?: and \d+ \w+)? generated\."), "clang_name": re.compile(r"\bclang\b")}
MSVC_MARKERS = {"msvc_code": re.compile(r"\b(?:error|warning) C\d{4}\b")}

# Coarse message families, ordered: first match wins.
MESSAGE_FAMILIES: list[tuple[str, re.Pattern[str]]] = [
    ("missing_include_file", re.compile(r"No such file or directory|file not found")),
    ("undeclared_identifier", re.compile(r"was not declared|undeclared identifier|use of undeclared")),
    ("redeclaration", re.compile(r"redeclaration|redefinition|conflicting declaration")),
    ("missing_semicolon_or_token", re.compile(r"expected ‘?[;,)]|expected primary-expression|expected '[;,)]'|expected expression|expected ‘,’ or ‘;’")),
    ("return_without_value", re.compile(r"return-statement with no value|non-void function .* should return a value")),
    ("type_mismatch_operator", re.compile(r"no match for ‘operator|invalid operands|no matching function|no viable|cannot convert|invalid conversion")),
    ("vla_non_pod", re.compile(r"variable length array of non-POD")),
    ("missing_return_warning", re.compile(r"no return statement|control reaches end|non-void function does not return")),
    ("multichar_warning", re.compile(r"multi-character character constant")),
    ("other", re.compile(r".")),
]

# Deny-list from the task's safety rule, applied to the raw source before any binary runs.
UNSAFE_SOURCE_RE = re.compile(r"\bsystem\s*\(|\bpopen\b|\bfork\s*\(|\bexec\w*\s*\(|\bremove\s*\(|\brename\s*\(|\bofstream\b|\bfopen\w*\s*\(|\bsocket\s*\(")
TIMEOUT_MARK = "<TIMEOUT>"


@dataclass
class LogSummary:
    n_error: int = 0
    n_fatal: int = 0
    n_warning: int = 0
    n_note: int = 0
    linker_error: bool = False
    truncated: bool = False
    compiler_family: str = "none"
    first_error_family: str = ""
    families: list[str] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return self.n_error + self.n_fatal > 0 or self.linker_error


def message_family(msg: str) -> str:
    for name, pattern in MESSAGE_FAMILIES:
        if pattern.search(msg):
            return name
    return "other"


def summarize_log(log: str | None, truncation_len: int = 2000) -> LogSummary:
    log = log or ""
    s = LogSummary()
    for m in DIAG_RE.finditer(log):
        kind = m.group("kind")
        if kind == "note":
            s.n_note += 1
            continue
        fam = message_family(m.group("msg"))
        s.families.append(f"{kind}:{fam}")
        if kind == "warning":
            s.n_warning += 1
        else:
            s.n_fatal += kind == "fatal error"
            s.n_error += kind == "error"
            s.first_error_family = s.first_error_family or fam
    if LINKER_ERROR_RE.search(log):
        s.linker_error = True
        fam = "missing_main_link" if MAIN_UNDEFINED_RE.search(log) else "undefined_reference"
        s.families.append("link:" + fam)
        s.first_error_family = s.first_error_family or fam
    s.truncated = len(log) >= truncation_len
    if any(p.search(log) for p in MSVC_MARKERS.values()):
        s.compiler_family = "msvc"
    elif any(p.search(log) for p in CLANG_MARKERS.values()):
        s.compiler_family = "clang"
    elif any(p.search(log) for p in GNU_MARKERS.values()):
        s.compiler_family = "gnu"
    elif log.strip():
        s.compiler_family = "unknown"
    return s


def compiler_version(cxx: str) -> str:
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True, timeout=10)
        return out.stdout.splitlines()[0] if out.stdout else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


@dataclass
class CompileResult:
    returncode: int
    stderr: str
    binary: Path | None
    workdir: Path


def compile_source(
    code_path: Path, cxx: str, syntax_only: bool, workdir: Path, std: str = "c++11", extra_flags: list[str] | None = None
) -> CompileResult:
    """Copy the file as student.cpp (same name as the official log) and compile in `workdir`."""
    src = workdir / "student.cpp"
    shutil.copyfile(code_path, src)
    binary = workdir / "student"
    cmd = [cxx, f"-std={std}", *(extra_flags or []), "student.cpp"]
    cmd += ["-fsyntax-only"] if syntax_only else ["-O0", "-o", "student"]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=60)
    ok_binary = (not syntax_only) and proc.returncode == 0 and binary.exists()
    return CompileResult(proc.returncode, proc.stderr, binary if ok_binary else None, workdir)


def link_with_harness(code_path: Path, harness_src: str, cxx: str, workdir: Path, std: str = "c++11") -> tuple[bool, str]:
    """Compile student TU with its main renamed, compile harness TU, link both. Nothing is executed."""
    shutil.copyfile(code_path, workdir / "student.cpp")
    (workdir / "harness.cpp").write_text(harness_src, encoding="utf-8")
    steps = [
        [cxx, f"-std={std}", "-Dmain=student_main_renamed", "-c", "student.cpp", "-o", "student.o"],
        [cxx, f"-std={std}", "-c", "harness.cpp", "-o", "harness.o"],
        [cxx, "student.o", "harness.o", "-o", "harnessed"],
    ]
    for cmd in steps:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            return False, f"{cmd[-1]}: " + summarize_log(proc.stderr).first_error_family
    return True, ""


def build_harness(struct_and_prototypes: str, prototypes: list[str], weak_stubs: bool) -> str:
    """Harness that odr-uses every required function, optionally with weak default bodies."""
    head = "#include <iostream>\n#include <fstream>\n#include <sstream>\n#include <string>\n#include <cstring>\n#include <cmath>\nusing namespace std;\n"
    stubs = []
    if weak_stubs:
        for proto in prototypes:
            sig = proto.strip().rstrip(";")
            body = "{}" if re.match(r"\s*void\b", sig) else "{ return {}; }"
            stubs.append(f"__attribute__((weak)) {sig} {body}")
    names = [re.search(r"([A-Za-z_]\w*)\s*\(", p).group(1) for p in prototypes]  # type: ignore[union-attr]
    uses = "\n".join(f"    auto f{i} = &{n}; (void)f{i};" for i, n in enumerate(names))
    return head + struct_and_prototypes + "\n" + "\n".join(stubs) + f"\nint main() {{\n{uses}\n    return 0;\n}}\n"


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


@dataclass
class RunResult:
    stdout: str
    returncode: int | None
    timed_out: bool

    @property
    def as_report_actual(self) -> str:
        """Mimic the official report: stripped stdout, or <TIMEOUT>."""
        return TIMEOUT_MARK if self.timed_out else self.stdout.strip()


def run_binary(binary: Path, stdin_text: str, timeout_s: float = 2.0, max_out: int = 4096) -> RunResult:
    with tempfile.TemporaryDirectory(prefix="code_run_") as run_dir:
        try:
            proc = subprocess.run(
                [str(binary)],
                input=stdin_text.encode(),
                cwd=run_dir,
                capture_output=True,
                timeout=timeout_s,
                preexec_fn=_limit_child,
                env={"PATH": "/usr/bin:/bin"},
            )
        except subprocess.TimeoutExpired as exc:
            return RunResult((exc.stdout or b"")[:max_out].decode(errors="replace"), None, True)
    return RunResult(proc.stdout[:max_out].decode(errors="replace"), proc.returncode, False)


def is_safe_to_run(source: str) -> tuple[bool, list[str]]:
    hits = sorted({m.group().strip() for m in UNSAFE_SOURCE_RE.finditer(source)})
    return not hits, hits
