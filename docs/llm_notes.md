# LLM Integration Notes

## Groq as the default provider

The sidebar now defaults to Groq's OpenAI-compatible endpoint. Set `GROQ_API_KEY` and
optionally `GROQ_MODEL` in a local `.env` file (or enter them directly in the sidebar)
and the app will call `https://api.groq.com/openai/v1/chat/completions`. You can switch
to Ollama or a custom router at any time via the **Provider** dropdown without losing
your saved profile inputs.

## Environment file support

Place a `.env` file alongside `app.py`—or copy `.env.example` to `.env`—to pre-populate
connection details. Supported keys include `GROQ_API_KEY`, `GROQ_MODEL`,
`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and the generic `LLM_BASE_URL`, `LLM_MODEL`, and
`LLM_API_KEY` fallbacks. Keys defined in the environment take precedence over values in
the `.env` file.

## DeepSeek via Ollama token limits

According to the [official DeepSeek model card in the Ollama library](https://ollama.com/library/deepseek-r1:32b), the maximum context window exposed by the `deepseek-r1` variants is 8192 tokens. Timeouts typically occur when responses approach this context limit and the serving hardware cannot stream the completion before the API's HTTP timeout (now set to 300 seconds in this app), so raising `max_tokens` beyond ~8000 can incur long latencies or dropped responses. Adjust the Streamlit sidebar timeout or Ollama server settings if longer generations are required.

## Base URL formatting

When configuring third-party routers (such as Hugging Face's `https://router.huggingface.co/v1`), you may enter the endpoint with or without the trailing `/v1`. The app now normalizes both forms automatically before calling `/chat/completions`, so either `https://router.huggingface.co` or `https://router.huggingface.co/v1` will resolve correctly. Ensure the model field matches an available deployment exposed by the selected provider.
