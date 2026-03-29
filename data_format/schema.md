# Tick Data Schema

The study expects one CSV file per strike per trading day. Each file should contain irregular tick-level observations with the following minimum fields:

- `timestamp`: event timestamp with microsecond precision
- `last_traded_price`: option last traded price
- `last_traded_quantity`: quantity associated with the most recent trade
- `Bid_Price_1`: best bid price
- `Bid_Quantity_1`: best bid quantity
- `Ask_Price_1`: best ask price
- `Ask_Quantity_1`: best ask quantity
- `spot_ltp`: synchronized underlying spot price

## Notes
- Timestamps are treated in event time and must be convertible to absolute time.
- No fixed-interval resampling is required.
- Missing or malformed rows should be skipped conservatively.
- Filenames are expected to encode strike and option type, for example `MASTER_<id>_<strike>_<CE/PE>.csv`.
