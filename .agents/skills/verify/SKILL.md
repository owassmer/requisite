---
name: verify
description: Use the real product as a user would, from a clean start, and write what you found. The verifier's procedure.
---
The product is a clean copy of the committed code: no case exists yet, and every model, agent and source it calls
answers from its recordings, as it would for anyone who cloned it. `requisite` is how you use it, exactly as a user runs
it (`requisite --help`, and `--help` after any command). `requisite --ls var/<path>` lists what the product has written
and `requisite --file var/<path>` prints one file, such as a dossier page. You cannot read its code, and you do not need
to: judge it by what it does and shows.

You are the analyst who runs a case, and the human who decides at its gates (G1 to G4). Where the product asks a human
to decide, read what it shows and decide as that person would. Start each case in its own folder under `var/cases/`.

The analyst's work, in plain words. Businesses get hurt by events they can see coming but can't protect against: a new
overtime rule, a minimum wage increase, a tariff. Requisite does two things about that. It finds who is exposed, to
what and how badly, by asking small plain questions of several independent AI judges and trusting an answer only when
they agree even when the question is reworded; code does all the arithmetic, and a person signs off at fixed gates.
And it builds protection: a contract that pays when the business is hurt and in time for its cash need, whose wording
is tested before anyone prices it, by checking that independent readers settle it the way its precise version does.
A case runs as a sequence of steps, each a command; each step's output is what the next one reads, and a step that
needs a human decision waits for `requisite approve`. Read the Purpose in AGENTS.md before you start; it says what the
product may claim and what it never does.

Know what you tried before you judge it. Learn each step from `requisite --help` and the command's own `--help`, run
the steps in the order their help and their messages give, and read what each prints before the next. When something
fails or surprises you, first rule out your own use: re-read the help, check that the step it depends on ran and
passed, and try again as the help says. A behaviour is BROKEN only when the product did something its own help, its
own output elsewhere, the outcome or arithmetic says it should not, or when it gives a user no way to learn how to do
what the outcome says a user can do. Where you could not tell how to proceed, say exactly what the product told you
and what you tried, so the finding is about the product and not about how you used it.

To check an outcome, do what it says a user can do, the way a user would, using `feature-map.md` where it has a line.
Then try two or three features that share a layer with what changed. Check what the product tells you, not only that it
ran: a number shown should follow from what it shows it rests on; a claim should be no wider than what was tested; an
unknown should say why it is unknown. Write one line per behaviour you tried, starting with WORKS or BROKEN, then what
you did and what you saw. A BROKEN line is a finding: the command, what you saw, and what it contradicts (the outcome, the
product's own statement elsewhere, or arithmetic). It names no fix. For each feature you reached, write or update its
line in `feature-map.md`: the feature and the commands you ran.
Passing tests say the code is well formed. Only the running product says the work is done.
