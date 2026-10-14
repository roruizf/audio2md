# Verify grounding

Compare each factual statement in a derived artifact with the cited transcript segments.

Return JSON only with `status` and `issues`. `status` is `pass` only when every factual statement has relevant
evidence. Each issue contains the statement, citations, and one verdict: `missing_citation`, `unsupported`,
`partially_supported`, `contradictory`, or `citation_not_found`. Explain the smallest correction needed, but do not
rewrite the artifact or use outside knowledge.
