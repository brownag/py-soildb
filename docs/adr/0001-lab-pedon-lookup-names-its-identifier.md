---
status: proposed
---

# Lab pedon lookups name the identifier they search

A lab pedon has several identifiers: pedon key (`pedon_key`, its unique record in lab data), NASIS pedon record ID (`peiid`, its unique record in NASIS, stored in lab data as `pedoniid`), Pedon ID (`upedonid`, assigned by the describer, not unique) and lab pedon number (`pedlabsampnum`). `get_lab_pedon_by_id` tried the value as a pedon key and fell back to Pedon ID. A value that was one pedon's key and another pedon's ID therefore returned the first pedon without any sign of the ambiguity. Lookups now take `what=` to name one identifier column, default to `"pedon_key"` (unique, and native to the lab data being queried) and never fall back.

## Considered Options

- **Key-then-ID fallback** (previous behavior): convenient, but the caller can't tell which identifier matched, and the result depends on which namespace is checked first.
- **One function per identifier** (`get_lab_pedon_by_key`, `..._by_id`, `..._by_labnum`, `..._by_peiid`): clear at the call site, but it multiplies the public surface at both the mid and high tier.
- **A `what=` parameter taking column names** (chosen): matches the existing `fetch_ldm(x, what=...)`, so callers learn one convention for the Lab Data Mart. It uses raw column names (`"upedonid"`, not `"pedon_id"`) for the same reason.

## Consequences

- `get_lab_pedon_by_id` and `fetch_labpedon_by_id` stay as deprecated wrappers with their old behavior, so existing callers keep working and see a `DeprecationWarning`.
- A caller who passes a Pedon ID without `what="upedonid"` gets no match rather than a guessed one, because `pedon_key` is the default.
