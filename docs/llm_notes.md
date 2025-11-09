# LLM Integration Notes

## DeepSeek via Ollama token limits

According to the [official DeepSeek model card in the Ollama library](https://ollama.com/library/deepseek-r1:32b), the maximum context window exposed by the `deepseek-r1` variants is 8192 tokens. Timeouts typically occur when responses approach this context limit and the serving hardware cannot stream the completion before the API's HTTP timeout (now set to 300 seconds in this app), so raising `max_tokens` beyond ~8000 can incur long latencies or dropped responses. Adjust the Streamlit sidebar timeout or Ollama server settings if longer generations are required.
