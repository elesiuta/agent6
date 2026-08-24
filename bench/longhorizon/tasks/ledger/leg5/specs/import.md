# import-csv

    import-csv FILE

FILE has a header `txid,date,account,amount,memo`. Rows sharing a txid form
one transaction. Every transaction must balance and every amount must be a
valid ledger amount; an import with any unbalanced or malformed transaction
imports NOTHING and exits non-zero, naming the txid. Prints `imported N
transactions` on success.
