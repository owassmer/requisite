# GSA: the Regulations.gov API v4 OpenAPI specification (commit 443dac7)

## The listing limits the client pages by: at most 20 pages of at most 250 results per query, so a longer listing needs the lastModifiedDate continuation.

> Specifies the number for the page of results that will be returned from the query. Acceptable values are numerical between, and including, 1 and 20. required: false schema: type: integer - name: page[size] in: query description: >- Specifies the size per page of results that will be returned from the query. Acceptable values are numerical between, and including, 5 and 250.

https://raw.githubusercontent.com/GSA/open-gsa-redesign/443dac706fca8556634307e43e364b8d4f1ed5a4/_apidocs/regulationsgov/v4/openapi.yaml, read 2026-11-02
