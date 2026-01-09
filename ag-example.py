from __future__ import annotations

import ast
import inspect
import json
import textwrap
import time
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal

from aethergraph import Service, NodeContext, graph_fn
from aethergraph.runtime import register_context_service

Stage = Literal["compile", "test_run", "apply", "test_validation"]


# ---------- low-level records ----------

@dataclass
class CallRecord:
    ok: bool
    duration_s: float
    args: tuple
    kwargs: dict
    result: Any = None
    exc_type: Optional[str] = None
    exc_msg: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "duration_s": self.duration_s,
            "args": self.args,
            "kwargs": self.kwargs,
            "result": self.result,
            "exc_type": self.exc_type,
            "exc_msg": self.exc_msg,
        }


@dataclass
class MyReport:
    stage: Stage
    passed: bool = True
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "failures": self.failures,
        }


# ---------- per-function state (in-process, no AG memory yet) ----------

@dataclass
class FunctionAgentState:
    fn: Callable
    name: str = field(init=False)
    title: str = field(init=False)
    docstring: str = field(init=False)
    fn_description: str = field(init=False)
    tests_description: str = field(init=False)
    signature: Optional[inspect.Signature] = field(init=False)
    source: str = field(init=False)

    # dynamic bits
    tests: Dict[str, str] = field(default_factory=dict)
    history: List[CallRecord] = field(default_factory=list)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = getattr(self.fn, "__name__", str(self.fn))

        # Best-effort source capture
        try:
            self.source = inspect.getsource(self.fn)
        except (OSError, TypeError):
            self.source = ""

        # Split title/docstring/body
        self.title, self.docstring, _body = _parse_single_function_source_str(self.source)

        # Extract <function> / <tests> from docstring, like the original
        self.fn_description, self.tests_description = _parse_docstring(self.docstring)

        # Signature
        try:
            self.signature = inspect.signature(self.fn)
        except (ValueError, TypeError):
            self.signature = None

    # ---- execution + bookkeeping ----

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            out = self.fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            rec = CallRecord(True, dt, args, kwargs, result=out)
            self.history.append(rec)
            return out
        except Exception as e:
            dt = time.perf_counter() - t0
            rec = CallRecord(
                False,
                dt,
                args,
                kwargs,
                exc_type=type(e).__name__,
                exc_msg=str(e),
            )
            self.history.append(rec)
            raise

    # ---- source / tests utilities (pure, no LLM) ----

    def validate_source(self, source_code: str) -> tuple[str, Optional[Callable], MyReport]:
        """
        Compile/validate candidate source. Returns (normalized_source, fn | None, report).
        """
        source = textwrap.dedent(source_code).strip()
        fn_title, _, _ = _parse_single_function_source_str(source)

        if fn_title and fn_title != self.title:
            return (
                source,
                None,
                MyReport(
                    stage="compile",
                    passed=False,
                    failures=[f"Function title error: should use '{self.title}'"],
                ),
            )

        try:
            candidate_fn = _compile_function_from_source(source, self.name)
            return source, candidate_fn, MyReport(stage="compile", passed=True, failures=[])
        except Exception:
            return (
                source,
                None,
                MyReport(
                    stage="compile",
                    passed=False,
                    failures=["Source code contains syntax error"],
                ),
            )

    def update_source(self, source_code: str) -> MyReport:
        """
        Promote new source iff it compiles and passes tests.
        """
        source, fn, compile_report = self.validate_source(source_code)
        if not compile_report.passed or fn is None:
            return MyReport(stage="apply", passed=False, failures=compile_report.failures)

        # run tests on candidate
        report = self.run_tests(fn)
        if report.stage == "test_run" and report.passed:
            self.fn = fn
            self.source = source
            return MyReport(stage="apply", passed=True)
        else:
            return MyReport(stage="apply", passed=False, failures=report.failures)

    def validate_tests(
        self,
        tests_code: str,
        *,
        allow_existing: bool = False,
    ) -> Tuple[List[Dict[str, str]], MyReport]:
        """
        Split tests code into individual tests, return good ones + validation report.
        """
        report = MyReport(stage="test_validation")
        good_tests: List[Dict[str, str]] = []

        src = textwrap.dedent(tests_code).strip()

        try:
            mod = ast.parse(src)
        except SyntaxError as e:
            report.passed = False
            report.failures.append(f"Syntax error in tests code: {e}")
            return [], report

        lines = src.splitlines()
        found_any = False

        for node in mod.body:
            if isinstance(node, ast.FunctionDef):
                found_any = True
                name = node.name

                if not name.startswith("test_"):
                    report.passed = False
                    report.failures.append(
                        f"Invalid test name '{name}' (must start with 'test_')."
                    )
                    continue

                if not allow_existing and name in self.tests:
                    report.passed = False
                    report.failures.append(
                        f"Invalid test name '{name}' (name already exists)."
                    )
                    continue

                start = node.lineno - 1
                end = node.end_lineno
                code_block = "\n".join(lines[start:end]).rstrip()

                good_tests.append({"name": name, "code": code_block})

        if not found_any:
            report.passed = False
            report.failures.append("No test functions found.")

        return good_tests, report

    def add_tests(self, tests_code: str) -> MyReport:
        tests, validation_report = self.validate_tests(tests_code, allow_existing=False)
        if not validation_report.passed:
            return validation_report

        for t in tests:
            self.tests[t["name"]] = textwrap.dedent(t["code"]).strip()
        return validation_report

    def update_test(self, name: str, test_code: str) -> MyReport:
        tests, validation_report = self.validate_tests(test_code, allow_existing=True)
        if not validation_report.passed:
            return validation_report

        if len(tests) != 1:
            validation_report.passed = False
            validation_report.failures.append(
                "More than one test function detected; expected exactly one."
            )
            return validation_report

        t = tests[0]
        if t["name"] != name:
            validation_report.passed = False
            validation_report.failures.append(
                "Test function name does not match the provided name."
            )
            return validation_report

        self.tests[name] = textwrap.dedent(t["code"]).strip()
        return validation_report

    def run_tests(self, fn: Optional[Callable] = None) -> MyReport:
        """
        Run all stored tests against the given callable (or current fn).
        """
        fn = fn or self.fn
        failures: List[str] = []

        ns: Dict[str, Any] = {self.name: fn}

        # load tests
        for tname, tcode in self.tests.items():
            try:
                exec(tcode, ns)
            except Exception as e:
                failures.append(f"[Load:{tname}] {type(e).__name__}: {e}")

        # run discovered test_* functions
        for obj_name, obj in list(ns.items()):
            if obj_name.startswith("test_") and callable(obj):
                try:
                    obj()
                except Exception as e:
                    failures.append(f"[Fail:{obj_name}] {type(e).__name__}: {e}")

        return MyReport(stage="test_run", passed=(len(failures) == 0), failures=failures)


# ---------- helpers shared by service ----------

def _parse_docstring(docstring: Optional[str]) -> Tuple[str, str]:
    """
    Expect XML-ish docstring:

        <function>...</function>
        <tests>...</tests>
    """
    if not docstring:
        return "", ""

    s = docstring.strip()
    s = re.sub(r"<\?xml.*?\?>", "", s, flags=re.DOTALL).strip()
    wrapped = f"<root>\n{s}\n</root>"

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(wrapped)
    except Exception:
        return "", ""

    def text_of(tag: str) -> str:
        node = root.find(tag)
        if node is None:
            return ""
        return "".join(node.itertext())

    fn_instruction = text_of("function")
    tst_instruction = text_of("tests")
    return fn_instruction, tst_instruction


def _parse_single_function_source_str(source: str) -> Tuple[str, str, str]:
    try:
        source = textwrap.dedent(source).strip()
        mod = ast.parse(source)
        fdefs = [n for n in mod.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(fdefs) != 1:
            return "", "", ""

        f = fdefs[0]
        lines = source.splitlines()

        title = lines[f.lineno - 1]

        if ast.get_docstring(f):
            doc_node = f.body[0]
            docstring = "\n".join(lines[doc_node.lineno - 1 : doc_node.end_lineno])
            body_nodes = f.body[1:]
        else:
            docstring = ""
            body_nodes = f.body

        if not body_nodes:
            body = ""
        else:
            body = "\n".join(
                lines[body_nodes[0].lineno - 1 : body_nodes[-1].end_lineno]
            )

        return title, docstring, body
    except Exception:
        return "", "", ""


def _compile_function_from_source(source: str, fn_name: str) -> Callable:
    ns: Dict[str, Any] = {}
    code = compile(source, "<agent_source>", "exec")
    exec(code, ns)
    fn = ns.get(fn_name)
    if not callable(fn):
        raise TypeError(f"Source did not define callable {fn_name}")
    return fn


# ---------- Service: AG-native function agent ----------

class FunctionAgentService(Service):
    """
    AG Service that lets you "talk to" Python functions and evolve them
    (update implementation, manage tests, run tests, and call the function)
    using the AG LLM client.

    State is in-process (lives as long as this worker runs); a future version
    can mirror it into ctx.memory()/artifacts.
    """

    def __init__(self):
        super().__init__()
        self._functions: Dict[str, FunctionAgentState] = {}

    # --- registration / lookup ---

    def register_function(self, fn: Callable, *, name: Optional[str] = None) -> None:
        """
        Register a Python function to be controlled by this agent.
        Idempotent: calling multiple times is cheap.
        """
        fn_name = name or getattr(fn, "__name__", str(fn))
        if fn_name in self._functions:
            return
        self._functions[fn_name] = FunctionAgentState(fn=fn)

    def _get_state(self, fn_name: str) -> FunctionAgentState:
        if fn_name not in self._functions:
            raise KeyError(f"Function '{fn_name}' is not registered in FunctionAgentService")
        return self._functions[fn_name]

    # --- LLM helpers (no tool-calls yet, JSON actions instead) ---

    @staticmethod
    def _action_schema() -> Dict[str, Any]:
        """
        Very loose schema: we just want canonical JSON back.
        The semantics are enforced in Python.
        """
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "What you want to do. "
                        "Allowed values: 'answer', 'run_function', 'update_source', "
                        "'add_tests', 'update_test', 'run_tests'."
                    ),
                },
                "answer": {
                    "type": "string",
                    "description": (
                        "User-facing natural language answer. "
                        "Required when action == 'answer'."
                    ),
                },
                "args": {
                    "type": "object",
                    "description": (
                        "Structured arguments for the chosen action. "
                        "Shapes:\n"
                        "- run_function: {\"a\": int, \"b\": int, ...}\n"
                        "- update_source: {\"source_code\": str}\n"
                        "- add_tests: {\"tests_code\": str}\n"
                        "- update_test: {\"name\": str, \"test_code\": str}\n"
                        "- run_tests: {}"
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def _call_llm_for_action(
        self,
        *,
        ctx: NodeContext,
        state: FunctionAgentState,
        user_message: str,
    ) -> Dict[str, Any]:
        """
        Single LLM call that returns a JSON action.
        """
        sys = (
            "You are an assistant responsible for maintaining the correctness of the "
            f"function `{state.name}`. You can either:\n"
            "1) Directly answer the user (action='answer'), OR\n"
            "2) Ask the runtime to manipulate code/tests by choosing one of:\n"
            "   - 'run_function': call the function with concrete arguments.\n"
            "   - 'update_source': propose a new implementation (full def line + body).\n"
            "   - 'add_tests': add new pytest-style tests.\n"
            "   - 'update_test': replace one existing test.\n"
            "   - 'run_tests': run all stored tests and report failures.\n\n"
            "You MUST respond with a single JSON object that matches the schema you were given. "
            "Do not include any extra commentary or markdown, only JSON."
        )

        # Short context of current function + tests
        fn_summary = f"{state.title or state.name}\n\nDescription:\n{state.fn_description or '(none)'}"
        if state.tests:
            tests_info = "Current tests: " + ", ".join(sorted(state.tests.keys()))
        else:
            tests_info = "No tests have been registered yet."

        messages: List[Dict[str, Any]] = []
        if not state.chat_history:
            messages.append({"role": "system", "content": sys})
            messages.append(
                {
                    "role": "system",
                    "content": f"Current function summary:\n{fn_summary}\n\n{tests_info}",
                }
            )
        else:
            messages.extend(state.chat_history)

        messages.append({"role": "user", "content": user_message})

        llm = ctx.llm()
        text, usage = await llm.chat(
            messages=messages,
            output_format="json_schema",
            json_schema=self._action_schema(),
            schema_name="FunctionAgentAction",
            strict_schema=False,   # allow extra keys if the model is chatty
            validate_json=True,
        )

        try:
            action = json.loads(text)
        except Exception as e:
            raise RuntimeError(f"Function agent got non-JSON LLM output: {text}") from e

        # Naive in-process history (no AG memory yet)
        state.chat_history = messages + [
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}
        ]

        # Optionally log usage later via ctx.logger or ctx.metrics
        _ = usage
        return action

    # --- public entrypoint used from @graph_fn ---

    async def chat_with_function(
        self,
        fn_name: str,
        *,
        message: str,
        ctx: Optional[NodeContext] = None,
    ) -> Dict[str, Any]:
        """
        Main "agent" entry: talk to a function by name.
        Returns a dict with at least {"action": ..., ...}.
        """
        if ctx is None:
            ctx = self.ctx()
        state = self._get_state(fn_name)

        action = await self._call_llm_for_action(ctx=ctx, state=state, user_message=message)
        act = action.get("action")
        args = action.get("args") or {}

        result_payload: Dict[str, Any] = {"action": act}

        result_payload: Dict[str, Any] = {"action": act}

        assistant_message = ""

        if act == "answer":
            answer = action.get("answer", "")
            result_payload["answer"] = answer
            assistant_message = answer or "(no answer text provided)"

        elif act == "run_function":
            try:
                out = state(**args)
                result_payload["ok"] = True
                result_payload["result"] = out
                assistant_message = (
                    f"I called `{state.name}({args})`.\n\n"
                    f"Result: **{out!r}**"
                )
            except Exception as e:
                result_payload["ok"] = False
                result_payload["error"] = f"{type(e).__name__}: {e}"
                assistant_message = (
                    f"I tried to call `{state.name}({args})` but it raised an error:\n\n"
                    f"`{type(e).__name__}: {e}`"
                )

        elif act == "update_source":
            src = args.get("source_code")
            if not isinstance(src, str):
                result_payload["ok"] = False
                msg = "update_source requires args.source_code (string)."
                result_payload["error"] = msg
                assistant_message = msg
            else:
                report = state.update_source(src)
                result_payload["ok"] = report.passed
                result_payload["report"] = report.to_dict()

                if report.passed:
                    assistant_message = (
                        "✅ I updated the implementation of "
                        f"`{state.name}` and **all tests passed**.\n\n"
                        "Here is the new function source:\n\n"
                        "```python\n"
                        f"{state.source}\n"
                        "```"
                    )
                else:
                    failures = "\n".join(f"- {f}" for f in report.failures)
                    assistant_message = (
                        "❌ I tried to update the implementation of "
                        f"`{state.name}`, but tests failed.\n\n"
                        "Failures:\n"
                        f"{failures or '(no details)'}"
                    )

        elif act == "add_tests":
            tests_code = args.get("tests_code")
            if not isinstance(tests_code, str):
                msg = "add_tests requires args.tests_code (string)."
                result_payload["ok"] = False
                result_payload["error"] = msg
                assistant_message = msg
            else:
                report = state.add_tests(tests_code)
                result_payload["ok"] = report.passed
                result_payload["report"] = report.to_dict()

                if report.passed:
                    assistant_message = (
                        "✅ I validated and added the new tests.\n\n"
                        "Current tests:\n"
                        + "\n".join(f"- `{name}`" for name in sorted(state.tests.keys()))
                    )
                else:
                    failures = "\n".join(f"- {f}" for f in report.failures)
                    assistant_message = (
                        "❌ I could not add the tests because validation failed:\n\n"
                        f"{failures or '(no details)'}"
                    )

        elif act == "update_test":
            name = args.get("name")
            test_code = args.get("test_code")
            if not isinstance(name, str) or not isinstance(test_code, str):
                msg = "update_test requires args.name and args.test_code (strings)."
                result_payload["ok"] = False
                result_payload["error"] = msg
                assistant_message = msg
            else:
                report = state.update_test(name=name, test_code=test_code)
                result_payload["ok"] = report.passed
                result_payload["report"] = report.to_dict()

                if report.passed:
                    assistant_message = f"✅ I updated test `{name}` successfully."
                else:
                    failures = "\n".join(f"- {f}" for f in report.failures)
                    assistant_message = (
                        f"❌ I could not update test `{name}`:\n\n"
                        f"{failures or '(no details)'}"
                    )

        elif act == "run_tests":
            report = state.run_tests()
            result_payload["ok"] = report.passed
            result_payload["report"] = report.to_dict()

            if report.passed:
                assistant_message = "✅ All tests **passed**."
            else:
                failures = "\n".join(f"- {f}" for f in report.failures)
                assistant_message = (
                    "❌ Some tests **failed**:\n\n"
                    f"{failures or '(no details)'}"
                )

        else:
            result_payload["ok"] = False
            result_payload["error"] = f"Unknown action: {act!r}"
            result_payload["raw_action"] = action
            assistant_message = (
                "⚠️ I produced an unknown action; this is likely a bug in the agent "
                "prompt or schema. Here is the raw action:\n\n"
                f"```json\n{json.dumps(action, indent=2, ensure_ascii=False)}\n```"
            )

        result_payload["assistant_message"] = assistant_message
        return result_payload


# ---------- Example usage in AG ----------

def add(a: int, b: int) -> int:
    """
    <function>
        Add up two integers and return the sum.
    </function>

    <tests>
        add(1, 2) == 3
        add(0, 0) == 0
        add(-1, 1) == 0
    </tests>
    """
    # Intentionally wrong to give the agent something to fix
    if a > 0:
        a = -a
    return 0


@graph_fn(
    name="add_chat_agent",
    inputs=["message", "files", "context_refs", "session_id", "user_meta"],
    outputs=["result"],
    as_agent={
        "id": "add_function_agent",
        "title": "Add Function Agent",
        "description": "An agent that can perform addition and improve its implementation over time.",
    },
)
async def add_chat_agent(message, files, context_refs, session_id, user_meta, *, context: NodeContext) -> Dict[str, Any]:
    channel = context.ui_session_channel()
    fnagent: FunctionAgentService = context.fnagent()  # type: ignore[attr-defined]

    fnagent.register_function(add)

    result = await fnagent.chat_with_function("add", message=message, ctx=context)

    # 1) Main assistant message
    assistant_msg = result.get("assistant_message") or "Function agent completed an action."
    await channel.send_text(assistant_msg)

    # 2) Optional: if you want raw debug info in a collapsible panel later
    # await channel.send_text(f"```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```")

    return {"result": result}


# --- register service globally ---
register_context_service("fnagent", FunctionAgentService())


""" Below is a simple local test you can run outside of AG server context without the UI. """

# @graph_fn(name="add_chat_function",
#           outputs=["action", "ok", "result"],)
# async def add_chat_function(*, context: NodeContext, message: str) -> Dict[str, Any]:
#     """
#     Thin wrapper graph that lets you talk to the `add` function.

#     In the UI this can be exposed as an app; each call lets the LLM either
#     answer directly or evolve the implementation/tests via FunctionAgentService.
#     """
#     fnagent: FunctionAgentService = context.fnagent()  # type: ignore[attr-defined]

#     # Idempotent – cheap if already registered
#     fnagent.register_function(add)

#     result = await fnagent.chat_with_function("add", message=message, ctx=context)
#     print("Function agent result:", result)
#     return result # {"action": ..., "ok": ..., "result": ...}

# if __name__ == "__main__":
#     from aethergraph.runner import run_async
#     from aethergraph import start_server
#     from asyncio import run

#     start_server() 
#     async def main():   
#         # Simple local test
#         await run_async(add_chat_function, inputs={"message": "What is 2 + 3?"})

#     run(main())