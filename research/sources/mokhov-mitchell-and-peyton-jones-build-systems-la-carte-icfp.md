# Mokhov, Mitchell and Peyton Jones, "Build Systems à la Carte", ICFP 2018 (Proc. ACM Program. Lang. 2, Article 79)

## Early cutoff as the runtime implements it: a recomputed node whose value hash is unchanged leaves its dependents unexecuted (spec §5 cites this paper).

> Shake also supports the early cutoff optimisation. When it executes a task and the result is unchanged from the previous build, it is unnecessary to execute the dependent tasks, and hence Shake can stop a build earlier, as illustrated in Fig. 3.

https://www.microsoft.com/en-us/research/uploads/prod/2018/03/build-systems.pdf, read 2026-11-02

## WP6.descendants: one changed input recomputes only its descendants, each at most once.

> Definition 2.1 (Minimality). A build system is minimal if it executes tasks at most once per build and only if they transitively depend on inputs that changed since the previous build.

https://www.microsoft.com/en-us/research/uploads/prod/2018/03/build-systems.pdf, read 2026-11-02

## The property test: an incremental recompute equals a from-scratch computation node for node.

> The build result is correct if the following two conditions hold: • result and store agree on inputs, that is, for all input keys k ∈ I : getValue k result == getValue k store. In other words, no inputs were corrupted during the build. • The result is consistent with the tasks, i.e. for all non-input keys k ∈ O, the result of recomputing the corresponding task matches the value stored in the result: getValue k result == compute task result.

https://www.microsoft.com/en-us/research/uploads/prod/2018/03/build-systems.pdf, read 2026-11-02
