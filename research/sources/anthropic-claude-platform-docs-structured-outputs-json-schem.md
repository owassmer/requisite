# Anthropic, Claude Platform Docs: Structured outputs, JSON Schema limitations

## Structured outputs accept regex patterns with simple quantifiers only; with the next passage, why the decomposer's schema keeps simple patterns and leaves the 64-hex claim id to the definition's check in the answer's validator.

> Quantifiers: * , + , ? , simple {n,m} cases

https://platform.claude.com/docs/en/build-with-claude/structured-outputs, read 2026-11-02

## Listed under "NOT supported" for structured-output regex patterns: the reason the decomposer's answer schema does not carry the claim id's [0-9a-f]{64} pattern.

> Complex {n,m} quantifiers with large ranges

https://platform.claude.com/docs/en/build-with-claude/structured-outputs, read 2026-11-02
