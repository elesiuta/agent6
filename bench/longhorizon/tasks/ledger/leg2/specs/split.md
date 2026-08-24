# split

    split DATE FROM AMOUNT TARGET:PCT [TARGET:PCT ...] [MEMO]

Debits FROM by AMOUNT and credits each TARGET by its percentage share of
AMOUNT. Percentages are integers and must sum to 100; otherwise the command
fails and the book is unchanged. Each share is rounded to the cent the way the
rest of the ledger rounds; any remainder cents go to the LAST target so the
transaction balances. Prints one line per credited target: `TARGET AMOUNT`.
