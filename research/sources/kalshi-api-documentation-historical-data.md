# Kalshi API documentation: Historical Data

## A market settled before Kalshi's archive cutoff is read from the historical endpoints; the latest settled market of a series is in the live set whenever any market of it settled after the cutoff.

> market_settled_ts Market settlement time Markets and their candlesticks that settled before this timestamp are only available via GET /historical/markets

https://docs.kalshi.com/getting_started/historical_data, read 2026-11-02
