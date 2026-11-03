# Kalshi API documentation: Rate Limits and Tiers

## The Kalshi client waits out a 429 by exponential backoff, since Kalshi states no Retry-After.

> 429 responses do not currently include `Retry-After` or `X-RateLimit-*` headers. There is no penalty or cooldown. The bucket keeps refilling, and your next request succeeds once the balance covers its cost. At a 1,200 tokens-per-second refill, an empty bucket replenishes the 10 tokens for an order in about 8.3 ms. Apply exponential backoff on 429.

https://docs.kalshi.com/getting_started/rate_limits.md, read 2026-11-02

## The Kalshi client paces live reads to the Basic tier's sustained read rate: 200 tokens per second at 10 tokens a request is 20 requests a second.

> Most requests cost the default of **10 tokens**. For endpoints that cost more or less, [`GET /account/endpoint_costs`](https://docs.kalshi.com/api-reference/account/list-non-default-endpoint-costs) is the authoritative list of non-default costs currently in effect.

https://docs.kalshi.com/getting_started/rate_limits.md, read 2026-11-02

## Pacing live Kalshi reads at a steady interval (the Basic tier's read budget of 200 tokens a second, 10 tokens a request, so 20 a second) stays within the bucket's refill; the tier table lists Basic with a Read budget of 200.

> Each budget is a token bucket. The bucket refills continuously at your per-second budget, up to its capacity, and a request is allowed whenever the bucket holds enough tokens to cover its cost. There are no fixed windows and no per-second resets.

https://docs.kalshi.com/getting_started/rate_limits.md, read 2026-11-02
