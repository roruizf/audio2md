# Hosted model providers

The provider boundary is `LlmAdapter`. Core evidence preparation and citation validation do not import a vendor SDK
or need network access.

## OpenRouter

The first adapter uses `https://openrouter.ai/api/v1/chat/completions` and defaults to
`deepseek/deepseek-chat-v3.1`. It reads `OPENROUTER_API_KEY` only when a request is executed, requests strict JSON
Schema when supplied, requires a route that supports those parameters, disables OpenRouter response caching, and
does not retry implicitly. Every structured request also requires zero data retention, denies data collection and
disables provider fallbacks. Usage metadata is returned to the caller for an auditable run manifest.

Installing or importing the adapter sends nothing. A hosted request must be initiated explicitly for an approved
source bundle. Provider-side retention and the policies of the routed inference provider remain external concerns;
sensitive sources should use a local adapter instead.

## OpenCode Go

`OpenCodeAdapter` uses an already authenticated OpenCode subscription and defaults to
`opencode-go/deepseek-v4.1-flash`. It invokes `opencode run --pure` from `/tmp`, never enables `--auto`, reads no
API key, and parses the JSON event stream for response, usage and cost. OpenCode still sends the supplied prompt to
its hosted service and stores a local session, so the same explicit source authorization rule applies.
