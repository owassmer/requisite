"""The evidence layer's adapters (spec §5, §13 WP2): the public record retrieved and frozen.

- fetched: a response as a public source returned it, and Requisite's User-Agent, shared by every read-only adapter.
- regulations_gov: the regulations.gov v4 client, through the cassettes, paced by the key's hourly limit.
- artifacts: responses into content-addressed artifacts, a changed source linked to its predecessor (spec I2, I3).
- extract: each artifact's text by its format, versioned and recorded (OCR for pages without a text layer).
- items: a sample's comments read whole into core Items.
- frame: a wide run's frame from its criteria, frozen as a core Frame.
"""
