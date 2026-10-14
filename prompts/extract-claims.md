# Extract grounded claims

You receive an ordered packet of transcript segments. Each segment has an ID, timestamps and verbatim ASR text.

Return JSON only with these arrays: `claims`, `exclusions`, `uncertainties`, `glossary`.

For every claim include:

- `claim_id`: packet-local sequential ID.
- `text`: concise statement supported by the transcript.
- `evidence`: one or more exact segment IDs.
- `classification`: `source`, `inference`, or `uncertain`.

Rules:

- Use only the supplied segments. Do not add background knowledge.
- Preserve numbers, units, standards, versions and conditions exactly.
- Label inference explicitly; uncertainty is preferable to invented certainty.
- Put promotion, introductions, logistics and calls to action in `exclusions`.
- Record likely ASR errors in `uncertainties`; do not silently correct them.
