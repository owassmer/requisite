# Notebook

What the factory has learned about its machine and tools, and the causes behind the coordinator's lapses with what
now answers them. Read whole. Handoff's notebook (~/dev/handoff/notebook.md) holds the history this factory starts from.

## The machine and its tools

- 2026-11-01 (coordinator) Three factories share this Mac (Handoff, Slope, Requisite): 8 cores, 8 GB of memory, about 5 GB of free disk at bootstrap. Each keeps its own worker slots (/private/tmp/<name>-locks), so their workers add up; Requisite runs at most three at once.
