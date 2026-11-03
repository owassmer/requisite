# Kalshi API documentation: Understanding Pagination

## The Kalshi client pages a listing by cursor until the response carries no cursor (null or empty, as the recorded KXCAGOAT response shows "cursor": "").

> 1. Make your initial request without a cursor 2. Check if the response includes a `cursor` field 3. If a cursor exists, make another request with `?cursor={cursor_value}` 4. Continue until the cursor is `null` (no more pages)

https://docs.kalshi.com/getting_started/pagination.md, read 2026-11-02
