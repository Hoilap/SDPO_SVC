import __future__

import ast
import copy
import faulthandler
import io
import json
import multiprocessing
import os
import pickle
import re
import reprlib
import sys
import time
import traceback
from collections.abc import MutableMapping
from typing import Optional

import numpy as np

INCORRECT_FORMAT = "Incorrect format"
TIMEOUT = "Time out"
ERROR_PREFIX = "Error: "
OUTER_ERROR_PREFIX = "ERROR: "
MAX_ADDITIONAL_MEMORY_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_MAX_CONCURRENT_TEST_PROCESSES = 8
MAX_CONCURRENCY_ENV_VAR = "CODE_REWARD_MAX_CONCURRENCY"
DEFAULT_MAX_OUTPUT_CHARS = 4 * 1024 * 1024
MAX_FEEDBACK_CHARS = 4096
INFRASTRUCTURE_ERROR_KINDS = frozenset({
    "memory_limit_setup", "setup_memory", "setup_error", "ipc_memory", "ipc_error", "process_exit",
})
DEFAULT_TIMEOUT = 1
TIMEOUT_SCALER = 1.0
FORMAT_PENALTY = False

FILENAME = "Solution.py"
TESTS_FILENAME = "Tests.py"
CONTEXT_FILENAME = "Context.py"
DEBUG_BUFFER_NAME = "__debug_buffer__"
DEBUG_PRINT_NAME = "debug_print"

# Names in the execution namespace for which we prevent accidental overwriting by user code
# This does not prevent malicious code from overwriting these names.
PROTECTED_GLOBAL_NAMES = {DEBUG_PRINT_NAME, DEBUG_BUFFER_NAME}

# Best-effort import to support setting memory limits on POSIX systems.
try:
    import resource as _resource  # type: ignore
except Exception:  # pragma: no cover - platform may not provide resource
    _resource = None  # type: ignore


class MemoryLimitSetupError(RuntimeError):
    """The evaluator could not establish its address-space budget."""


class OutputLimitExceeded(RuntimeError):
    pass


def _current_vms_bytes():
    # Read in the child before reliability_guard disables file access.
    with open("/proc/self/statm") as stream:
        return int(stream.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")


def set_memory_limits(additional_memory_bytes: Optional[int]) -> None:
    """Limit virtual-address growth, not RSS, without relaxing inherited limits."""
    if additional_memory_bytes is None or additional_memory_bytes <= 0:
        return
    try:
        if _resource is None or not hasattr(_resource, "RLIMIT_AS"):
            raise MemoryLimitSetupError("RLIMIT_AS is unavailable")
        baseline = _current_vms_bytes()
        target = baseline + additional_memory_bytes
        soft, hard = _resource.getrlimit(_resource.RLIMIT_AS)
        inherited = [x for x in (soft, hard) if x != _resource.RLIM_INFINITY]
        if inherited and min(inherited) < target:
            raise MemoryLimitSetupError(
                f"Insufficient address-space budget: baseline={baseline}, extra={additional_memory_bytes}, "
                f"inherited_soft={soft}, inherited_hard={hard}"
            )
        _resource.setrlimit(_resource.RLIMIT_AS, (target, target))
    except MemoryLimitSetupError:
        raise
    except Exception as error:
        raise MemoryLimitSetupError("Failed to configure evaluation address-space limit") from error


class _BoundedOutput(io.StringIO):
    """Bound captured text at write time, including repeated writes and seeks."""

    def __init__(self):
        super().__init__()
        self.limit = int(os.environ.get("CODE_REWARD_MAX_OUTPUT_CHARS", DEFAULT_MAX_OUTPUT_CHARS))
        if self.limit <= 0:
            raise ValueError("CODE_REWARD_MAX_OUTPUT_CHARS must be positive")
        self.exceeded = False

    def write(self, text):
        if self.exceeded or self.tell() + len(text) > self.limit:
            self.exceeded = True
            raise OutputLimitExceeded("Captured output exceeds CODE_REWARD_MAX_OUTPUT_CHARS")
        return super().write(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def truncate(self, size=None):
        if (self.tell() if size is None else size) > self.limit:
            self.exceeded = True
            raise OutputLimitExceeded("Captured output exceeds CODE_REWARD_MAX_OUTPUT_CHARS")
        return super().truncate(size)


def _feedback_preview(value):
    if isinstance(value, str):
        return value[:MAX_FEEDBACK_CHARS]
    preview = reprlib.Repr()
    preview.maxstring = preview.maxother = MAX_FEEDBACK_CHARS
    return preview.repr(value)[:MAX_FEEDBACK_CHARS]


def _error_record(kind, message):
    return {"passed": False, "actual": message, "debug": "", "time": float("inf"), "error_kind": kind}


# Pre-serialize the fallback before applying child limits: never re-send a large
# record or construct a traceback while handling a failed allocation.
_IPC_FAILURES = {
    kind: pickle.dumps(_error_record(kind, message))
    for kind, message in (
        ("ipc_memory", "Evaluation infrastructure error: result serialization ran out of memory"),
        ("ipc_error", "Evaluation infrastructure error: result transport failed"),
    )
}


def _send_result(connection, record):
    try:
        connection.send(record)
    except BaseException as error:
        kind = "ipc_memory" if isinstance(error, MemoryError) else "ipc_error"
        try:
            connection.send_bytes(_IPC_FAILURES[kind])
        except BaseException:
            pass  # Parent records EOF/exit status as an infrastructure failure.


def _get_max_concurrent_test_processes() -> int:
    """Return the configured per-completion test-process concurrency limit."""
    raw_value = os.environ.get(MAX_CONCURRENCY_ENV_VAR)
    if raw_value is None:
        return DEFAULT_MAX_CONCURRENT_TEST_PROCESSES

    try:
        value = int(raw_value)
    except ValueError:
        print(
            f"Ignoring invalid {MAX_CONCURRENCY_ENV_VAR}={raw_value!r}; using {DEFAULT_MAX_CONCURRENT_TEST_PROCESSES}."
        )
        return DEFAULT_MAX_CONCURRENT_TEST_PROCESSES

    if value < 1:
        print(
            f"Ignoring invalid {MAX_CONCURRENCY_ENV_VAR}={raw_value!r}; using {DEFAULT_MAX_CONCURRENT_TEST_PROCESSES}."
        )
        return DEFAULT_MAX_CONCURRENT_TEST_PROCESSES
    return value


def _build_restricted_builtins():
    """
    Create a minimal, safer builtins set for executing user code.

    - Removes file system, process, and reflection primitives
    - Restricts imports to a small allowlist
    """
    import builtins as _builtins  # local import to avoid leaking into user ns

    allowed_names = {
        "abs": _builtins.abs,
        "all": _builtins.all,
        "any": _builtins.any,
        "bool": _builtins.bool,
        "bytes": _builtins.bytes,
        "callable": _builtins.callable,
        "chr": _builtins.chr,
        "dict": _builtins.dict,
        "enumerate": _builtins.enumerate,
        "filter": _builtins.filter,
        "float": _builtins.float,
        "format": _builtins.format,
        "frozenset": _builtins.frozenset,
        "hash": _builtins.hash,
        "hex": _builtins.hex,
        "int": _builtins.int,
        "isinstance": _builtins.isinstance,
        "issubclass": _builtins.issubclass,
        "iter": _builtins.iter,
        "len": _builtins.len,
        "list": _builtins.list,
        "map": _builtins.map,
        "max": _builtins.max,
        "min": _builtins.min,
        "next": _builtins.next,
        "object": _builtins.object,
        "ord": _builtins.ord,
        "pow": _builtins.pow,
        "print": _builtins.print,
        "range": _builtins.range,
        "repr": _builtins.repr,
        "reversed": _builtins.reversed,
        "round": _builtins.round,
        "set": _builtins.set,
        "slice": _builtins.slice,
        "sorted": _builtins.sorted,
        "str": _builtins.str,
        "sum": _builtins.sum,
        "tuple": _builtins.tuple,
        "zip": _builtins.zip,
        "input": _builtins.input,
    }

    allowed_modules = {
        "math",
        "cmath",
        "itertools",
        "functools",
        "operator",
        "statistics",
        "random",
        "collections",
        "heapq",
        "bisect",
        "array",
        "string",
        "re",
        "typing",
        "json",
        "io",
        "fractions",
        "decimal",
        "dataclasses",
        "datetime",
        "time",
        "sys",
        "sortedcontainers",
        "numpy",
    }

    real_import = _builtins.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
        root = name.split(".")[0]
        if root not in allowed_modules:
            raise ImportError(f"Import of module '{name}' is not allowed")
        return real_import(name, globals, locals, fromlist, level)

    # Shadow risky builtins
    allowed_names["open"] = None  # deny file I/O
    allowed_names["__import__"] = restricted_import

    # Allow class creation (required for 'class' statements)
    allowed_names["__build_class__"] = _builtins.__build_class__

    return allowed_names


def _capture_stderr(namespace):
    old_stderr = sys.stderr
    sys.stderr = namespace[DEBUG_BUFFER_NAME]
    return old_stderr


def _create_sandbox_namespace(extra_globals=None):
    """Return a fresh globals dict with restricted builtins for exec/eval."""
    ns = {"__builtins__": _build_restricted_builtins()}

    # Provide a dedicated debug print facility that is captured separately from stdout
    _debug_buffer = _BoundedOutput()

    def debug_print(*args, **kwargs):  # type: ignore[override]
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        try:
            text = sep.join(str(a) for a in args) + end
        except Exception:
            # Fallback stringify to be extra safe
            text = " ".join(["<unprintable>"] * len(args)) + end
        _debug_buffer.write(text)

    ns[DEBUG_PRINT_NAME] = debug_print
    ns[DEBUG_BUFFER_NAME] = _debug_buffer

    # Auto-import common typing aliases to reduce boilerplate in user code
    try:
        import typing as _typing
        _common_typing_names = (
            "Any",
            "Optional",
            "Union",
            "List",
            "Dict",
            "Set",
            "Tuple",
            "Callable",
            "Iterable",
            "Iterator",
            "Sequence",
            "Mapping",
            "MutableMapping",
            "MutableSequence",
            "MutableSet",
            "DefaultDict",
            "Deque",
            "FrozenSet",
            "Type",
            "TypeVar",
            "Generic",
            "Literal",
            "TypedDict",
            "NoReturn",
            "overload",
        )
        for _name in _common_typing_names:
            if hasattr(_typing, _name):
                ns[_name] = getattr(_typing, _name)
    except Exception:
        pass
    # Ensure __name__ exists to avoid NameError during module-level guards in functional tests
    if "__name__" not in ns:
        ns["__name__"] = "__not_main__"
    if extra_globals:
        ns.update(extra_globals)
    return ns


def _exec_with_isolated_locals(code_obj, globals_ns):
    """
    Execute code while preserving normal exec semantics (globals == locals)
    so that names defined during exec are visible to functions created and
    invoked within the same exec. Protected globals (e.g., debug hooks) are
    shielded from writes and served from preserved values within the block.
    """
    # Minimal locals proxy: forwards to globals, blocks writes/deletes to protected names
    class _GuardedLocals(MutableMapping):
        def __init__(self, backing):
            self._b = backing
        def __getitem__(self, key):
            return self._b[key]
        def __setitem__(self, key, value):
            if key in PROTECTED_GLOBAL_NAMES:
                return
            self._b[key] = value
        def __delitem__(self, key):
            if key in PROTECTED_GLOBAL_NAMES:
                return
            del self._b[key]
        def __iter__(self):
            return iter(self._b)
        def __len__(self):
            return len(self._b)

    exec(code_obj, globals_ns, _GuardedLocals(globals_ns))


def _to_safe_jsonable(value):
    """
    Convert a Python value into a JSON-serializable structure consisting only of
    primitives (None, bool, int, float, str) and containers (list, dict).
    Raises TypeError when encountering unsupported types to prevent equality
    spoofing via custom objects.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_safe_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_safe_jsonable(v) for k, v in value.items()}
    raise TypeError(f"Non-serializable result type: {type(value).__name__}")


def reliability_guard():
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    faulthandler.disable()

    # Suppress noisy SyntaxWarning (e.g., invalid escape sequences in user regex strings)
    try:
        import warnings as _warnings
        _warnings.filterwarnings("ignore", category=SyntaxWarning)
    except Exception:
        pass

    import builtins

    set_memory_limits(MAX_ADDITIONAL_MEMORY_BYTES)

    builtins.exit = None
    builtins.quit = None
    builtins.open = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
    sys.modules["inspect"] = None
    sys.modules["ctypes"] = None
    sys.modules["threading"] = None
    sys.modules["multiprocessing"] = None
    sys.modules["socket"] = None
    sys.modules["ssl"] = None
    sys.modules["urllib"] = None
    sys.modules["requests"] = None

    try:
        # Remove frame introspection which enables test data exfiltration
        sys._getframe = None  # type: ignore[attr-defined]
    except Exception:
        pass


def _short_trace(e, limit=3):
    """Return a compact traceback focused on the user's solution.

    We intentionally filter out frames that originate from this harness file
    so that the user only sees locations within their submitted code
    ("Solution.py"). If no solution frames are present (e.g., argument errors
    raised at the call site), we only show the error message.
    """
    frames = traceback.extract_tb(e.__traceback__)
    solution_frames = [
        f for f in frames
        if isinstance(getattr(f, "filename", None), str)
        and f.filename == FILENAME
    ]
    tail = solution_frames[-limit:] if solution_frames else []
    lines = [f"{type(e).__name__}: {e}"]
    for f in tail:
        if f.line:
            lines.append(f"  {f.line}")
        lines.append(f"Line {f.lineno} in {f.name} ({FILENAME})")
    return "\n".join(lines)


def run_test_func(completion, test_input, test_output, fn_name, namespace=None):
    namespace = _create_sandbox_namespace() if namespace is None else namespace
    # compile with postponed annotations (equivalent to: from __future__ import annotations)
    code_obj = compile(
        completion,
        FILENAME,
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    _exec_with_isolated_locals(code_obj, namespace)

    def _infer_func_name(src, explicit_name):
        try:
            tree = ast.parse(src)
            if explicit_name:
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == explicit_name:
                        return explicit_name
            for node in tree.body:  # fall back to first function definition
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception:
            pass
        return None

    if fn_name != "" and fn_name in namespace and callable(namespace[fn_name]):
        func_name = fn_name
    else:
        func_name = _infer_func_name(completion, fn_name)
        if func_name not in namespace or not callable(namespace.get(func_name, None)):
            func_name = completion.split("(")[0].split()[-1]

    output = _BoundedOutput()
    namespace.setdefault("_output_buffers", []).append(output)
    old_stdout = sys.stdout
    sys.stdout = output
    old_stderr = _capture_stderr(namespace)

    try:
        if isinstance(test_input, dict):
            result_output = namespace[func_name](**test_input)
        else:
            test_input_args = [json.loads(x) for x in test_input.split()]
            result_output = namespace[func_name](*test_input_args)

        # Enforce structural, JSON-like equality; reject custom objects
        try:
            lhs = _to_safe_jsonable(result_output)
            rhs = _to_safe_jsonable(json.loads(test_output))
            lhs_dump = json.dumps(lhs, sort_keys=True, separators=(",", ":"))
            rhs_dump = json.dumps(rhs, sort_keys=True, separators=(",", ":"))
            if lhs_dump != rhs_dump:
                return False, result_output
            return True, result_output
        except (MemoryError, OutputLimitExceeded):
            raise
        except Exception as ser_err:
            error_msg = f"{ERROR_PREFIX}{ser_err}"
            return False, error_msg

    except (MemoryError, OutputLimitExceeded):
        raise
    except BaseException as e:
        error_msg = f"{ERROR_PREFIX}{_short_trace(e)}"
        return False, error_msg

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def run_test_std(completion, test_input, test_output, namespace=None):
    namespace = _create_sandbox_namespace() if namespace is None else namespace
    output = _BoundedOutput()
    namespace.setdefault("_output_buffers", []).append(output)
    old_stdout, old_stdin = sys.stdout, sys.stdin
    old_stderr = _capture_stderr(namespace)
    try:
        sys.stdout = output
        sys.stdin = io.StringIO(test_input)
        code_obj = compile('__name__ = "__main__"\n' + completion, FILENAME, "exec")
        _exec_with_isolated_locals(code_obj, namespace)
        out = output.getvalue().strip().replace("\n", " ").replace("\r", "")
        expected = test_output.strip().replace("\n", " ").replace("\r", "")
        return out == expected, output.getvalue().strip()
    except (MemoryError, OutputLimitExceeded):
        raise
    except BaseException as e:
        return False, f"{ERROR_PREFIX}{_short_trace(e)}"
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin
        sys.stderr = old_stderr


def run_test_code(completion, test_input, namespace=None):
    namespace = _create_sandbox_namespace() if namespace is None else namespace
    namespace["__name__"] = "__main__"
    code_obj = compile(completion, FILENAME, "exec")
    _exec_with_isolated_locals(code_obj, namespace)

    old_stderr = _capture_stderr(namespace)
    try:
        test_code_obj = compile(test_input, TESTS_FILENAME, "exec")
        _exec_with_isolated_locals(test_code_obj, namespace)
        return True, "All tests pass"
    except (MemoryError, OutputLimitExceeded):
        raise
    except BaseException as e:
        return False, f"{ERROR_PREFIX}{_short_trace(e)}"
    finally:
        sys.stderr = old_stderr


def run_tests_for_one_example(test_cases, completion, send_conn, sparse_rewards, test_idx):
    """Execute one test and return bounded feedback, without echoing its inputs."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    namespace = None
    started = time.monotonic()
    stage = "setup"
    try:
        # Establish limits before executing context or generated code.
        reliability_guard()
        namespace = _create_sandbox_namespace()
        captured_stdout = _BoundedOutput()
        namespace["_output_buffers"] = [captured_stdout]
        sys.stdout = captured_stdout
        sys.stderr = namespace[DEBUG_BUFFER_NAME]
        stage = "execution"
        context = test_cases.get("context", "")
        if context.strip():
            _exec_with_isolated_locals(compile(context, CONTEXT_FILENAME, "exec"), namespace)

        test_input = test_cases["inputs"][test_idx]
        test_output = test_cases["outputs"][test_idx]
        test_type = test_cases["testtype"]
        if test_type == "functional":
            passed, actual = run_test_func(
                completion, copy.deepcopy(test_input), copy.deepcopy(test_output), test_cases["fn_name"], namespace
            )
        elif test_type == "stdin":
            test_output = test_output.strip()
            if test_output.endswith("-"):
                test_output = test_output[: test_output.rfind("-")].rstrip()
            passed, actual = run_test_std(completion, test_input, test_output, namespace)
        elif test_type == "code":
            passed, actual = run_test_code(completion, test_input, namespace)
        else:
            raise ValueError(f"Invalid test type: {test_type}")

        # A solution may catch output exceptions; reaching the cap still fails.
        buffers = namespace["_output_buffers"] + [namespace[DEBUG_BUFFER_NAME]]
        if any(buffer.exceeded for buffer in buffers):
            raise OutputLimitExceeded("Captured output exceeds CODE_REWARD_MAX_OUTPUT_CHARS")
        stage = "result"
        record = {
            "passed": passed,
            "actual": _feedback_preview(actual),
            "debug": namespace[DEBUG_BUFFER_NAME].getvalue()[:MAX_FEEDBACK_CHARS],
            "time": time.monotonic() - started,
            "error_kind": "execution_error" if not passed and isinstance(actual, str) and actual.startswith(ERROR_PREFIX) else "",
        }
    except MemoryLimitSetupError as error:
        record = _error_record("memory_limit_setup", f"Evaluation infrastructure error: {error}")
    except OutputLimitExceeded:
        record = _error_record("output_limit", "Error: Output limit exceeded")
    except MemoryError:
        if stage == "result":
            record = _error_record("ipc_memory", "Evaluation infrastructure error: preparing result ran out of memory")
        elif stage == "setup":
            record = _error_record("setup_memory", "Evaluation infrastructure error: setup ran out of memory")
        else:
            record = _error_record("execution_memory", "Error: MemoryError during test execution")
    except BaseException as error:
        kind = {"setup": "setup_error", "result": "ipc_error", "execution": "execution_error"}[stage]
        record = _error_record(kind, ERROR_PREFIX + _feedback_preview(_short_trace(error)))
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    try:
        _send_result(send_conn, record)
    finally:
        send_conn.close()


def extract_code(response):
    blocks = re.findall(r"```(\w*)\n(.*?)```", response, re.DOTALL)
    if not blocks:
        return None
    return max((code for _, code in blocks), key=len)


def format_test_feedback(
    records,
    was_truncated=False,
    max_tests_to_show=2,
    sort_test_cases_by_length=True,
    max_length=2000,
    max_input_chars=250,
    max_input_lines=8,
    max_expected_chars=250,
    max_actual_chars=250,
    max_debug_lines=10,
    max_debug_line_chars=300,
):
    """
    Render test feedback in a LeetCode-like style.

    Rules:
    - Only show failing cases.
    - For runtime errors/timeouts: show a concise error header and the last
      executed input.
    - For wrong answers: show Input, optional Stdout (debug_print), Output and
      Expected.
    - If all cases pass, return an empty string.
    """
    if not records:
        return "No test execution information available."

    def _truncate_str(value, max_chars):
        if not isinstance(value, str):
            value = str(value)
        if max_chars is not None and len(value) > max_chars:
            return value[:max_chars] + "..."
        return value

    # Filter to only failing cases to match LeetCode UI
    failing = [r for r in records if not r["passed"]]

    def _first(predicate):
        for rec in records:
            try:
                if predicate(rec):
                    return rec
            except Exception:
                continue
        return None
    selected = (
        _first(lambda rec: rec.get("error_kind") in INFRASTRUCTURE_ERROR_KINDS)
        or _first(lambda rec: isinstance(rec.get("actual"), str) and str(rec.get("actual")).startswith(ERROR_PREFIX))
        or _first(lambda rec: rec.get("actual") == TIMEOUT)
        or _first(lambda rec: rec.get("actual") == INCORRECT_FORMAT)
    )

    if selected is not None:
        failing = [selected]
    else:
        # Sort wrong-answer cases by length (input + output), shortest first
        if sort_test_cases_by_length:
            failing = sorted(failing, key=lambda x: len(str(x["input"])) + len(str(x["actual"])))

        if max_tests_to_show is not None:
            failing = failing[: int(max_tests_to_show)]

    if not failing:
        return ""

    parts = []

    def _render_input_block(title, inp):
        parts.append(title)
        if inp is None:
            return
        if isinstance(inp, dict):
            for k, v in inp.items():
                parts.append(f"{k} = {_truncate_str(v, max_input_chars)}")
        else:
            text = str(inp)
            lines = text.splitlines()
            shown = lines[:max_input_lines]
            for line in shown:
                parts.append(_truncate_str(line, max_input_chars))
            if len(lines) > max_input_lines:
                parts.append(f"... ({len(lines) - max_input_lines} more lines)")

    def _render_debug_block(dbg_text):
        dbg = (dbg_text or "").strip()
        if not dbg:
            return
        parts.append("")
        parts.append("Debug Output")
        dbg_lines = dbg.split("\n")
        limit = int(max_debug_lines) if max_debug_lines is not None else None
        for line in dbg_lines[:limit]:
            parts.append(_truncate_str(line, max_debug_line_chars))
        if max_debug_lines is not None and len(dbg_lines) > int(max_debug_lines):
            parts.append(f"... ({len(dbg_lines) - int(max_debug_lines)} more lines)")

    for r in failing:
        test_idx = r["test_idx"] + 1
        actual = r["actual"]
        expected = r["expected"]
        stdin = r["input"]
        debug_text = r["debug"] or ""

        is_error = isinstance(actual, str) and actual.startswith(ERROR_PREFIX)
        is_timeout = actual == TIMEOUT
        is_incorrect_format = actual == INCORRECT_FORMAT

        # Header similar to LeetCode per failing case
        if r.get("error_kind") in INFRASTRUCTURE_ERROR_KINDS:
            parts.append("Evaluation Infrastructure Error")
            parts.append(str(actual))
            parts.append(f"Category: {r['error_kind']}; exit code: {r.get('exit_code')}")
        elif is_error:
            parts.append("Runtime Error")
            parts.append(actual[len(ERROR_PREFIX):])
            parts.append("")
            _render_input_block("Last Executed Input", stdin)
            _render_debug_block(debug_text)
        elif is_timeout:
            parts.append("Time Limit Exceeded")
            parts.append("")
            _render_input_block("Last Executed Input", stdin)
            _render_debug_block(debug_text)
        elif is_incorrect_format:
            if was_truncated:
                parts.append("Truncated Attempt: Your previous response was too long and truncated because it reached the maximum response length. Try again with a shorter response.")
            else:
                parts.append("Incorrect Format: Put your code inside a ```python ... ``` block.")
        else:
            parts.append(f"Test Case {test_idx}: Wrong Answer")
            parts.append("")
            _render_input_block("Input", stdin)
            parts.append("")
            parts.append("Output")
            parts.append(_truncate_str(actual, max_actual_chars))
            if expected is not None:
                parts.append("")
                parts.append("Expected")
                parts.append(_truncate_str(expected, max_expected_chars))
            _render_debug_block(debug_text)

        parts.append("")  # blank line between cases

    result = "\n".join(parts).rstrip()
    if len(result) > max_length:
        result = result[:max_length]
    return result


def _cleanup_test_process(process, parent_conn) -> None:
    """Stop a test process and deterministically release its OS resources."""
    try:
        # Give a child that has closed its pipe time to publish its exit status.
        process.join(timeout=0.1)
        if process.is_alive():
            process.kill()
            process.join()
        return getattr(process, "exitcode", None)
    finally:
        parent_conn.close()
        try:
            process.close()
        except (AttributeError, ValueError):
            pass


def run_tests(test_cases: dict, solution, sparse_rewards, max_test_cases):
    completion = extract_code(solution)
    if completion is None:
        return [{
            "test_idx": 0,
            "input": None,
            "expected": None,
            "actual": INCORRECT_FORMAT,
            "passed": False,
            "debug": "",
            "time": float("inf"),
        }]

    num_test_cases = min(max_test_cases, len(test_cases["inputs"])) if max_test_cases else len(test_cases["inputs"])
    timeout_per_test_case = float(test_cases["time_limit"]) if test_cases["time_limit"] is not None else DEFAULT_TIMEOUT

    records = []
    max_concurrency = _get_max_concurrent_test_processes()

    # Starting every test at once can create hundreds of forked processes for
    # one completion. Each child has its own memory allowance, so an unbounded
    # fan-out can exhaust the node even when every child respects its limit.
    # Run bounded waves while retaining the existing per-test isolation and the
    # common deadline semantics within each wave.
    for wave_start in range(0, num_test_cases, max_concurrency):
        process_data = []
        wave_stop = min(wave_start + max_concurrency, num_test_cases)
        try:
            for test_idx in range(wave_start, wave_stop):
                parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
                process = None
                try:
                    process = multiprocessing.Process(
                        target=run_tests_for_one_example,
                        args=(test_cases, completion, child_conn, sparse_rewards, test_idx),
                    )
                    process.start()
                except BaseException:
                    child_conn.close()
                    if process is not None and process.pid is not None:
                        _cleanup_test_process(process, parent_conn)
                    else:
                        parent_conn.close()
                        try:
                            if process is not None:
                                process.close()
                        except (AttributeError, ValueError):
                            pass
                    raise
                child_conn.close()
                process_data.append(
                    {
                        "test_idx": test_idx,
                        "process": process,
                        "parent_conn": parent_conn,
                        "cleaned": False,
                    }
                )

            wave_started_at = time.time()
            for data in process_data:
                test_idx = data["test_idx"]
                process = data["process"]
                parent_conn = data["parent_conn"]

                timeout_this_test = max(
                    0,
                    timeout_per_test_case * TIMEOUT_SCALER + 1 - (time.time() - wave_started_at),
                )
                if parent_conn.poll(timeout_this_test):
                    try:
                        result = parent_conn.recv()
                        if not isinstance(result, dict) or not {"passed", "actual", "debug", "time"} <= result.keys():
                            raise ValueError("Invalid evaluator result")
                    except Exception as error:
                        result = _error_record(
                            "process_exit" if isinstance(error, EOFError) else "ipc_error",
                            f"Evaluation infrastructure error: {type(error).__name__} receiving test result",
                        )
                else:
                    result = _error_record("timeout", TIMEOUT) if process.is_alive() else _error_record(
                        "process_exit", "Evaluation infrastructure error: child exited without a result"
                    )

                # Inputs stay in the parent; the wire result contains only small
                # feedback fields. Preserve the public record layout.
                result["test_idx"] = test_idx
                result["input"] = test_cases["inputs"][test_idx]
                expected = test_cases["outputs"][test_idx]
                if test_cases["testtype"] == "stdin":
                    expected = expected.strip()
                    if expected.endswith("-"):
                        expected = expected[:expected.rfind("-")].rstrip()
                result["expected"] = expected
                result["exit_code"] = _cleanup_test_process(process, parent_conn)
                data["cleaned"] = True
                records.append(result)
        finally:
            for data in process_data:
                if not data["cleaned"]:
                    _cleanup_test_process(data["process"], data["parent_conn"])

    assert len(records) == num_test_cases
    return records


def compute_score(solution: str, ground_truth: str, extra_info = None, sparse_rewards=False, max_test_cases=None):
    split = extra_info["split"]
    was_truncated = extra_info.get("truncated", False)

    if split == "test":
        sparse_rewards = True

    try:
        test_cases = json.loads(ground_truth)
    except Exception:
        print("Error when reading tests: " + ground_truth[:1000])
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": "",
            "incorrect_format": 0,
            "error_in_test_cases": 1,
            "timed_out": 0,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 1 if was_truncated else 0,
            "feedback": "Failed to parse ground truth test cases.",
        }

    records = run_tests(test_cases=test_cases, solution=solution, sparse_rewards=sparse_rewards, max_test_cases=max_test_cases if split != "test" else None)

    correct_answers = [1.0 if r["passed"] else 0.0 for r in records]
    predictions = str([r["actual"] for r in records])[-5000:]
    accuracy = np.mean(correct_answers)

    if sparse_rewards:
        reward = 1.0 if accuracy == 1.0 else 0.0
    else:
        reward = accuracy

    incorrect_format = (len(records) == 1) and (not records[0]["passed"]) and (records[0]["actual"] == INCORRECT_FORMAT)
    error_in_test_cases = any([((not r["passed"]) and isinstance(r["actual"], str) and ERROR_PREFIX in r["actual"]) for r in records])
    timed_out = np.mean([1.0 if (not r["passed"]) and (r["actual"] == TIMEOUT) else 0.0 for r in records])
    if FORMAT_PENALTY and split == "train" and incorrect_format and (not was_truncated):
        reward -= 0.5

    return {
        "score": reward,
        "acc": accuracy,
        "pred": predictions,
        "incorrect_format": 1 if incorrect_format else 0,
        "error_in_test_cases": 1 if error_in_test_cases else 0,
        "timed_out": 1 if timed_out else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": format_test_feedback(records, was_truncated=was_truncated),
    }
