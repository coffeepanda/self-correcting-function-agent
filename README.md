Add Function Agent# Function-Self-Improving Agent (AetherGraph Example)

This example shows how to use **AetherGraph** to build an agent that can:

* Talk to a Python function (`add`).
* Call the function with arguments.
* Generate and update tests.
* Propose new implementations of the function.
* Apply those implementations **only if tests pass**.

All of this is done via a single AG agent exposed in the UI.

---

## 1. Install AetherGraph

```bash
pip install aethergraph
```

Make sure `ag-example.py` (or whatever you named the file with `FunctionAgentService` and `add_chat_agent`) is in your current directory.

---

## 2. Start the server with hot reload

From the folder containing your example:

```bash
aethergraph serve --load-path ./ag-example.py --reload
```

Notes:

* `--load-path` points to your Python module that defines `add_chat_agent` and registers the `FunctionAgentService`.
* `--reload` enables auto-reload on code changes (development mode).

  * When the backend reloads, **all in-process function state is reset** (tests, history, chat history, current implementation).
  * At the moment, we **do not persist this state into AG memory** yet.

If everything is correct, the terminal will show something like:

```text
AetherGraph server running at http://127.0.0.1:8745
UI: http://127.0.0.1:8745/ui
```

---

## 3. Open the UI in your browser

Open:

```text
http://127.0.0.1:8745/ui
```

This URL is printed in the server logs as well.

---

## 4. Run the “Add Function Agent” from the UI

1. In the left navigation bar, click **Agents**.

2. You should see an agent with the title:

   > **Add Function Agent**

   (this comes from `as_agent={ "title": "Add Function Agent", ... }` on `add_chat_agent`).

3. Click the agent to open its chat view.

4. Try a few messages, for example:

   * `What is 3 + 3?`
   * `Please run your tests and tell me if they pass.`
   * `Fix your implementation so that all tests pass, and show me the new code.`
   * `Add more edge-case tests for negative numbers.`

The agent will:

* Decide whether to **answer directly** (e.g., explain something),
* Or **perform an action** on the function (`run_function`, `run_tests`, `update_source`, `add_tests`, `update_test`),
* And respond with a friendly message (e.g. showing the new function source or test results).

---

## 5. Important behavior notes

### 5.1 Hot reload clears in-process function state

Because we run the server with:

```bash
aethergraph serve --load-path ./ag-example.py --reload
```

any code change triggers a backend reload. When that happens:

* The `FunctionAgentService` is recreated.
* All `FunctionAgentState` objects are recreated.
* You **lose**:

  * Stored tests,
  * Call history,
  * Chat history,
  * And any updated function implementations.

At this stage, we intentionally keep state **in-process only**. There is no AG memory / artifact wiring yet in this example.

In production you could:

* Mirror function definitions and tests into `ctx.artifacts()`.
* Log evolution events into `ctx.memory()` so you can search how a function changed over time.

---

### 5.2 Session vs function state

The function state is centralized in `FunctionAgentService` and **not** tied to a specific UI session.

* Starting a **new chat session** with the same agent **does not reset** the function state.
* All sessions share the same underlying `FunctionAgentState` (for this worker process).
* So if you:

  * Fix `add` in one chat,
  * Then open another session and ask `What is 3 + 3?`,
  * The second session will see the updated implementation.

Only restarting/reloading the backend (e.g., code change with `--reload`, or process restart) resets the state.

---

### 5.3 Differences from the original “@agentify” script

This example is adapted from a standalone script that used a custom `@agentify` decorator and OpenAI tool-calling inline.

Key differences:

1. **No custom decorator needed**

   * Instead of wrapping the function with `@agentify`, we:

     * Define a plain Python function (`add`).
     * Register it at runtime in `FunctionAgentService` via `register_function(add)`.

2. **AetherGraph-native service & agent**

   * All logic lives in an AG `Service` (`FunctionAgentService`) and a `@graph_fn` agent (`add_chat_agent`).
   * The AG server handles:

     * LLM calls via `context.llm()`.
     * UI sessions via `context.ui_session_channel()`.

3. **JSON-schema actions instead of tools**

   * Rather than using OpenAI `tools` and `tool_choice`, we ask the model for a single JSON `action` that looks like:

     ```json
     {
       "action": "run_function",
       "args": {"a": 3, "b": 3}
     }
     ```

   * The runtime (Python) interprets that action and executes the corresponding behavior.

   * This pattern works across multiple providers (OpenAI, Azure, OpenRouter, Gemini, etc.).

4. **UI integration for free**

   * Because the example is a `@graph_fn` with `as_agent={...}`, it automatically appears under **Agents** in the AG UI.
   * No custom UI or OpenAI client wiring is needed in the example itself.

---

### 5.4 Adding more function agents

You can extend this pattern without stopping the server:

* Define more functions (e.g., `minus`, `multiply`, `normalize_vector`, `simulate_ray_bundle`, etc.).

* Register them with the same service:

  ```python
  fnagent.register_function(minus)
  fnagent.register_function(multiply)
  ```

* Add additional `@graph_fn` wrappers (or one parameterized agent that selects the function by name).

As long as the process stays up (and you don’t trigger a reload), all these function agents share the same long-lived `FunctionAgentService` state.

This lets you build a small ecosystem of self-improving functions inside one AetherGraph project.
