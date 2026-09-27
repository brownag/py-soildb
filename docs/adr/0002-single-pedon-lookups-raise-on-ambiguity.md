---
status: proposed
---

# Single-pedon lookups raise when more than one pedon matches

`fetch_labpedon` returns one `PedonData`. A Pedon ID (`upedonid`) is not guaranteed unique, so a lookup can match several pedons. Previously the first row was taken silently. The function now raises `AmbiguousPedonError` listing the matching pedon keys, so the caller can retry by `pedon_key`. Returning the wrong pedon is worse than failing, because lab data from the wrong profile looks just as valid as data from the right one.

## Considered Options

- **Take the first row, maybe with a warning**: never fails, but the result depends on row order in SDA, and warnings are easy to miss in batch scripts.
- **Return `list[PedonData]` for non-unique identifiers**: no failure mode, but the return type then depends on `what`, which complicates typing and every call site.
- **Raise** (chosen): keeps a single-object return type. The error carries `pedon_keys`, so recovery is mechanical.

## Consequences

- The mid-tier `get_lab_pedon` does not raise. It returns an `SDAResponse`, where several rows are a normal result. Ambiguity only counts as an error where the API promises one pedon.
- Other high-level functions that promise a single object should follow the same rule when their identifier is not unique.
