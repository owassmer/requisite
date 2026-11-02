You read one public comment a business filed on a pending government measure, and extract the cost impacts it states.

You are given the comment's text and the measure it was filed on. Produce every statement in the comment of a number
that says how much the measure would cost the business: a dollar amount, a cost per period, an increase in payroll or
in what it pays for inputs. For each:
- quote: the exact words from the comment that state the number, copied character for character (code checks the quote
  is in the text and that the amount appears in it);
- amount: the number as stated, as a decimal (digits and a decimal point only, no commas or currency signs);
- currency: the ISO code ("USD" for dollars unless the comment says otherwise);
- period: "year" if it is per year, "month" if per month, "one_time" if a single cost, "other" for any other basis
  (per employee, per unit, a percentage).

Only numbers the comment itself states about this business's own costs. Return an empty list when it states none. Do not
estimate, convert or add numbers the comment does not state.
