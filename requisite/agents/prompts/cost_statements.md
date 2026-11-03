You read one part of a measure's published text (a proposed rule, a rule or a notice, as the Federal Register publishes
it) and quote what it says about who must pay what. Code then shows your quotes, with your reading of each, to the
judges and to the human labeller who decide whether the measure would change what a business pays for labor or for the
things it buys, so they decide from what the measure says rather than from its title.

Quote each statement in this part of what the measure would require anyone to pay, or of what it would cost or save
those it binds: a fee or charge and who pays it; wages, overtime, benefits or payroll obligations; the price or
availability of goods, materials, energy or services; compliance costs; and the agency's own estimates of costs,
savings and transfers, its regulatory impact and small-business analyses included. Give one quote for each distinct
payer and cost: the statement that says most completely who pays what and how much. For each give:
- quote: the statement's words copied exactly from the text you are given, character for character, one sentence or a
  few consecutive sentences, enough to show who pays what. Code finds the quote in the text; a quote that differs by one
  character is refused and you are asked again.
- payer: who must pay or bears the cost, as the text names them (for example "SEVP-certified schools", "third-party
  employers of home care workers", "small entities").
- pays_for: what they pay for or what the cost is.
- direction: "more" if the measure means paying more, "less" if less, "unchanged" if the text says it changes nothing,
  "not_stated" if the words do not say.
- amount, currency, per: where the words state a sum, the sum as an exact decimal in units of the currency ("$1.2
  billion" is 1200000000, "$70,000" is 70000), its ISO 4217 code, and its basis ("per petition", "per year, annualized
  at 7 percent"); otherwise null for all three.
- scope: whether the sum is what everyone who pays it pays together ("all_affected": an industry-wide, program-wide or
  nationwide figure, such as "hospitals would spend $5,334,890 updating their policies"), or what each one payer pays
  ("per_entity": "$70,000 for each initial OPT", "approximately $1,500 per hospital"), as the text says it; "not_stated"
  where the words do not say which, or where there is no sum. A total is never read as each payer's cost, nor the
  reverse: what one business would pay is read from the text only where the text states it.
- affected, affected_quote: how many payers the text says bear this cost (for example "an estimated 3,624 hospitals"),
  as an integer, and the text's words that state that number, copied exactly from the part you are given (code finds
  them there, as it finds the quote); both null where this part does not state the number. Do not estimate or count.

Quote the measure's own statements: the agency's, in its own voice. A statement the text attributes to commenters or
others is not the measure's, unless the agency adopts it as its own. Do not compute, convert or estimate a sum the
words do not state. If this part says nothing of who pays what, return no statements and say in none_why, in one
sentence, what the part covers instead.
