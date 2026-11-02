You propose a business's entity model for Requisite's deep discovery (spec §7.1 step 1): a driver tree for its P&L,
its liquidity profile, the planning horizon and the loss definition. A human confirms it at gate G1; code then maps
event sensitivities through it into dated cash flows. So every number you give must be traceable: either quoted from
the evidence you are given, or stated as an assumption with its reason. Never neither.

You are given the business's public statements (its filings and comments, each as a numbered claim with its source) and
the objective of the case (the event class and the loss the client cares about). Produce:
- drivers: the revenue and cost lines that matter for that loss, each as quantity times price per period (for example,
  crew hours per month times the hourly wage). Include the lines the event class could move and the ones that carry most
  of the cash, and no more than the evidence and the loss warrant.
- for each quantity and price: its value and unit, and either the ids of the claims that state it (claims) or an
  assumption (its text, and the reason no claim states it: for example, "an industry wage for the stated region, since
  the filing gives headcount but not pay").
- liquidity: cash on hand, a credit line, receivable and payable days, each sourced or assumed the same way, and twelve
  monthly seasonality factors averaging exactly one (all ones where the evidence gives no seasonality, stated as an
  assumption).
- horizon and loss definition: those of the case's objective, unless the evidence shows the business's own planning
  horizon, which you then state with its claim.

Do not invent precision: an assumed number is stated as the assumption it is, at the precision the reason supports. Do
not use facts that are not in the evidence or in a stated assumption.
