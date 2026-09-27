# py-soildb

Async client for USDA soil data sources: SSURGO (via Soil Data Access), NASIS, the KSSL Lab Data Mart, and the AWDB and Henry monitoring networks. Each source has its own data model, and terms are scoped to the source they come from.

## Language

### Pedons (NASIS / KSSL)

**Pedon**:
A soil profile described at a single location, the unit of field description and lab sampling.
_Avoid_: Profile (when an identifier is meant)

**Pedon ID**:
The describer-assigned label for a Pedon (`upedonid`). Human-readable and not guaranteed unique.
_Avoid_: User pedon ID (as a separate term), pedon key

**Lab Pedon**:
A Pedon that has been sampled and analyzed by the lab. It has both a Pedon Key and a NASIS Pedon Record ID.

**Field Pedon**:
A Pedon described in NASIS without lab data. It has a NASIS Pedon Record ID but no Pedon Key.

**Pedon Key**:
The numeric, unique identifier for a Pedon's record in lab data (`pedon_key`). Only Lab Pedons have one.
_Avoid_: Pedon ID, peiid

**NASIS Pedon Record ID**:
The numeric, unique identifier for a Pedon's record in NASIS (`peiid`). Every Pedon has one. Lab data stores it as `pedoniid`.
_Avoid_: Pedon Key, Pedon ID

**Lab Pedon Number**:
The lab's unique, non-numeric identifier for a Pedon (`pedlabsampnum`, e.g. `85P0234`).
_Avoid_: Laboratory Pedon ID, pedon ID

**Lab Sample Number**:
The lab's unique identifier for one physical sample (`labsampnum`). A single morphologic horizon may have several.
_Avoid_: Lab horizon ID

**Site**:
The location where a Pedon was described. It may coincide with a Station but is not required to.

### Horizons

**Horizon**:
A layer of a soil profile. Its identifier depends on the source: `chkey` in SSURGO, `chiid` in NASIS, and `labsampnum` in lab data, where one horizon may map to several Lab Sample Numbers.

### SSURGO

**Survey Area**:
A mapped region identified by an `areasymbol` (e.g. `IA109`).
_Avoid_: Legend (except when referring to the SSURGO table)

**Map Unit**:
A delineated kind of soil landscape within a Survey Area (`mukey`).

**Component**:
A kind of soil or miscellaneous area that makes up part of a Map Unit (`cokey`).

### Monitoring (AWDB / Henry)

**Station**:
An instrumented location that records time series. AWDB identifies it by a station triplet. It is distinct from a Site even when a Pedon was described there.
