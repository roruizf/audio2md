# Synthesize knowledge artifacts

You receive a claim ledger whose entries already cite transcript segment IDs.

Return JSON only with the string fields `manual`, `summary`, and `checklist`.

Organize the manual by concepts rather than transcript order and target 2500-3500 words. Make the summary stand alone
in 500-800 words. Build a checklist of 20-30 concrete actions, decisions or verification points; do not turn every
claim into a separate bullet and do not use generic tasks such as "review and explain". Every factual paragraph or
list item must end with its supporting segment citations, for example `[S000042, S000043]`. Use only `source` claims
for factual prose. Keep uncertainties clearly labeled and do not silently correct them. Omit exclusions unless they
explain a material limitation. Do not add outside knowledge, cite claim IDs, or wrap Markdown in code fences.
Preserve attribution and temporal context for opinions, forecasts, recommendations and time-sensitive claims; do not
present consequential commentary as professional advice.
Checklist items must verify or analyze the source's claims. Never instruct the reader to buy, sell, hold, wait,
act, or make a financial, medical, legal or other consequential decision.
