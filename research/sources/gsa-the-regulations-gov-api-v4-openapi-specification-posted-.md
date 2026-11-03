# GSA: the Regulations.gov API v4 OpenAPI specification, posted-date filter

## The API's posted-date filter is by day, so a request can only bound by day and the as_of instant is settled by observable_at on each entry's postedDate.

> Filters results relative to the posted date. The value must be formatted as `yyyy-MM-dd`. Omission of a parameter modifier will match results to the exact date provided, otherwise, one of the parameter modifiers below may be used. `ge` - greater than or equal `le` - less than or equal

https://raw.githubusercontent.com/GSA/open-gsa-redesign/443dac706fca8556634307e43e364b8d4f1ed5a4/_apidocs/regulationsgov/v4/openapi.yaml, read 2026-11-02
