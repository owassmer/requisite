"""The market lens's adapters, read-only (spec §10, §13 WP13, §14: "Market adapters expose no write endpoints").

- kalshi: Kalshi's public trade API v2 (series, markets, the historical archive, filed documents), every request a GET
  through the cassettes; quotes are prices, never probabilities (I9).
- f1: F1's filed contracts (spec §9.3, §12): the draw over the filed record at a pinned instant, and the harvest of the
  drawn series' filings, rules text and settled outcomes into a case's store.
"""
