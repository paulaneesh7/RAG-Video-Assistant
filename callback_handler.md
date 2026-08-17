`chain_config` is a small helper that builds the **extra options** you pass into LangChain’s `.invoke()`. It does two jobs: **turn tracing on**, and **tag that LLM call** (name + which Langfuse prompt).

---

### What LangChain callbacks are

When you run a chain, LangChain fires events like:

- chain started  
- LLM called  
- LLM replied  
- chain finished  

A **callback handler** is an object that **listens** to those events.

`CallbackHandler` from Langfuse is that listener. Every time GPT runs, it sends Langfuse:

- the prompt  
- the model (`gpt-4o-mini`)  
- tokens / cost  
- latency  
- the output  

Without it, LangChain still works — Langfuse just never hears about it. That’s why traces stay empty.

You attach it like this:

```python
chain.invoke(input, config={"callbacks": [get_langfuse_handler()]})
```

---

### What `chain_config` is doing, line by line

```python
def chain_config(system_prompt, *, run_name: str, **metadata) -> dict:
    return {
        "callbacks": [get_langfuse_handler()],   # 1. send this call to Langfuse
        "run_name": run_name,                    # 2. name in the UI, e.g. "map_summarize"
        "metadata": {
            "langfuse_prompt": system_prompt,    # 3. link this call to that prompt version
            **metadata,                          # 4. extra tags: chunk_index, etc.
        },
    }
```

So instead of repeating that dict on every invoke, you write:

```python
map_chain.invoke({"text": chunk}, config=chain_config(map_sys, run_name="map_summarize"))
```

That’s the same as passing callbacks + a name + prompt metadata.

---

### Simple picture

```
Your chain.invoke(...)
        ↓
LangChain runs GPT
        ↓
CallbackHandler notices
        ↓
Langfuse UI: a generation appears
        (named run_name, linked to the prompt you fetched)
```

---

### Why `langfuse_prompt` in metadata?

Fetching a prompt (`get_text_prompt(...)`) only **loads text**. It does **not** mean “this LLM call used that prompt.”

Putting `langfuse_prompt: system_prompt` tells Langfuse: **this generation used prompt X, version 1**. That’s how you later eval “v1 vs v2 of `summarize_reduce_system_prompt`.”

---

**Short version:**  
`CallbackHandler` = the wire from LangChain → Langfuse.  
`chain_config` = a reusable pack of “please trace this, call it X, and credit this prompt.”