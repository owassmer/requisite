# Kalshi Trade API OpenAPI specification (v3.35.0)

## The draw reads a window of a series' settled markets with min_settled_ts and max_settled_ts (Unix seconds); whether the bounds are inclusive is not stated, so the client asks a second wider on each side and decides each market by its recorded settlement_ts.

> MinSettledTsQuery: name: min_settled_ts in: query description: Filter items that settled after this Unix timestamp schema: type: integer format: int64 MaxSettledTsQuery: name: max_settled_ts in: query description: Filter items that settled before this Unix timestamp

https://docs.kalshi.com/openapi.yaml, read 2026-11-02

## The historical markets endpoint has no settlement-time filter (its parameters are limit, cursor, tickers, event_ticker, series_ticker, mve_filter, one filter at a time), so a series' archived markets can be read only whole.

> Endpoint for getting markets that have been archived to the historical database. Filters are mutually exclusive.

https://docs.kalshi.com/openapi.yaml, read 2026-11-02

## The Kalshi client reads markets 1,000 to a page, the API's largest.

> MarketLimitQuery: name: limit in: query description: Number of results per page. Defaults to 100. Maximum value is 1000.

https://docs.kalshi.com/openapi.yaml, read 2026-11-02
