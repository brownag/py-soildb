---
status: proposed
---

# Identifiers and entity names stay scoped to their data source

SSURGO, NASIS, the KSSL Lab Data Mart, AWDB and Henry each have their own data model. py-soildb keeps each source's native identifiers (horizons are `chkey` in SSURGO, `chiid` in NASIS and `labsampnum` in lab data) and does not introduce a unified cross-source key. A pedon's Site and a monitoring Station are distinct concepts, even when a pedon was described at a station. The keys don't map one-to-one: one morphologic horizon can have several lab samples, and a Station has no required Site. A synthetic unified identifier would hide those differences and would have to be maintained against sources py-soildb doesn't control.

## Consequences

- Linking across sources (for example, pedons near a Station) is an explicit spatial or attribute join in the calling code or a high-level function, never an implicit identity.
- Column names returned to callers, and column names callers pass in (for example `what="pedoniid"`), are the physical names from the source schema, even where the name is awkward. Those schemas are stable, and the source's own documentation then applies directly. py-soildb adds no alias layer.
