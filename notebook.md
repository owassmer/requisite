# Notebook

What the factory has learned about its machine and tools, and the causes behind the coordinator's lapses with what
now answers them. Read whole. Handoff's notebook (~/dev/handoff/notebook.md) holds the history this factory starts from.

## The machine and its tools

- 2026-11-01 (coordinator) Three factories share this Mac (Handoff, Slope, Requisite): 8 cores, 8 GB of memory, about 5 GB of free disk at bootstrap. Each keeps its own worker slots (/private/tmp/<name>-locks), so their workers add up; Requisite runs at most three at once.
- 2026-11-01 (coordinator) The Claude subscription's session limit stopped the coordinator and three Opus xhigh workers at 18:33 ET, about 80 minutes into the first three dispatches; it reset at 18:40. Pi exits 0 on a model error, so scripts/work reads the run's last answer, calls it exit 5, and applies whatever the worker had changed: two runs left half-built code on the branch with no report. A run stopped this way is continued, not restarted: its task gains a line "Carries: <run>", and scripts/work copies the stopped run's .tmp/scratch and its whole record (tools/roles/transcript.py) into the new workspace's .tmp/carried/, since an hour of search is mostly what was ruled out, which no summary keeps.
