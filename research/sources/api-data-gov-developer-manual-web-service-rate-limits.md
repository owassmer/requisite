# api.data.gov Developer Manual: web service rate limits

## Pacing: the limit is a rolling hour, so a request slot frees one hour after the request that used it; a 429 means the key is over its limit and is waited out, not dropped.

> The hourly counters for your API key reset on a rolling basis. Example: If you made 500 requests at 10:15AM and 500 requests at 10:25AM, your API key would become temporarily blocked. This temporary block of your API key would cease at 11:15AM, at which point you could make 500 requests. At 11:25AM, you could then make another 500 requests. Rate Limit Error Response If your API key exceeds the rate limits, you will receive a response with an HTTP status code of 429 (Too Many Requests).

https://api.data.gov/docs/developer-manual/, read 2026-11-02
