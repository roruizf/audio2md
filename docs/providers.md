# Hosted model providers

The provider boundary is `LlmAdapter`. Core evidence preparation and citation validation do not import a vendor SDK
or need network access.

## OpenRouter

The first adapter uses `https://openrouter.ai/api/v1/chat/completions` and defaults to
`deepseek/deepseek-chat-v3.1`. It reads `OPENROUTER_API_KEY` only when a request is executed, requests strict JSON
Schema when supplied, requires a route that supports those parameters, disables OpenRouter response caching, and
does not retry implicitly. Usage metadata is returned to the caller for an auditable run manifest.

Installing or importing the adapter sends nothing. A hosted request must be initiated explicitly for an approved
source bundle. Provider-side retention and the policies of the routed inference provider remain external concerns;
sensitive sources should use a local adapter instead.
