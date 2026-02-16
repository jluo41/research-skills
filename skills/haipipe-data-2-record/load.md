Subcommand: load
================

Purpose: Inspect existing RecordSet assets (read-only).

Use this when you need to understand record structure, verify temporal
alignment, check record counts, or debug data quality issues.

---

What To Read
============

RecordSets live under `_WorkSpace/2-RecStore/`. The directory structure
is **FLAT** (not hierarchical by patient):

```
_WorkSpace/2-RecStore/
+-- OhioT1DM_v0RecSet/
|   +-- Human-HmPtt/                  # FLAT: Human-{HumanName}
|   |   +-- df_Human.parquet
|   |   +-- schema.json
|   +-- Record-HmPtt.Ptt/            # FLAT: Record-{HumanName}.{RecordName}
|   |   +-- df_RecAttr.parquet
|   |   +-- df_RecIndex.parquet
|   +-- Record-HmPtt.CGM5Min/
|   |   +-- df_RecAttr.parquet
|   |   +-- df_RecIndex.parquet
|   +-- Record-HmPtt.Diet5Min/
|   |   +-- df_RecAttr.parquet
|   |   +-- df_RecIndex.parquet
|   +-- Record-HmPtt.Exercise5Min/
|   |   +-- df_RecAttr.parquet
|   |   +-- df_RecIndex.parquet
|   +-- Record-HmPtt.Med5Min/
|   |   +-- df_RecAttr.parquet
|   |   +-- df_RecIndex.parquet
|   +-- Extra-{name}/                 # Optional extra DataFrames
|   +-- manifest.json
+-- ...
```

**Directory naming:** `<CohortName>_v<N>RecSet/`

**Important:** It is NOT `HmPtt/<PatientID>/CGM5Min/`. It is flat.

---

Load Pattern (Correct API)
===========================

```python
source .venv/bin/activate
source env.sh

python -c "
from haipipe.record_base import RecordSet

# Load existing RecordSet -- path-based API
record_set = RecordSet.load_from_disk(
    path='_WorkSpace/2-RecStore/OhioT1DM_v0RecSet',
    SPACE=SPACE
)

# Alternative: load_asset (handles both local and remote paths)
record_set = RecordSet.load_asset(
    path='_WorkSpace/2-RecStore/OhioT1DM_v0RecSet',
    SPACE=SPACE
)

# Print summary info
record_set.info()
"
```

**Correct signature:** `RecordSet.load_from_disk(path=..., SPACE=...)`
**WRONG (does not exist):** `RecordSet.load_from_disk(set_name=..., store_key=...)`

---

Name_to_HRF Access Pattern
============================

The core data structure is `record_set.Name_to_HRF`, a flat dictionary
with string keys (Humans) and tuple keys (Records):

```python
# Access Human (string key)
human = record_set.Name_to_HRF['HmPtt']
# Human has: human.df_Human (DataFrame with patient demographics)

# Access Record (tuple key)
cgm_record = record_set.Name_to_HRF[('HmPtt', 'CGM5Min')]
# Record has: cgm_record.df_RecAttr (DataFrame with record attributes)
#             cgm_record.df_RecIndex (DataFrame with record index)

diet_record = record_set.Name_to_HRF[('HmPtt', 'Diet5Min')]
med_record  = record_set.Name_to_HRF[('HmPtt', 'Med5Min')]

# List all keys
for key in record_set.Name_to_HRF:
    if isinstance(key, str):
        print(f'Human: {key}')
    elif isinstance(key, tuple):
        print(f'Record: {key[0]}.{key[1]}')
```

**WRONG (these do NOT exist):**
- `record_set.items()` -- use `record_set.Name_to_HRF` directly
- `human.records` -- no such attribute
- `record.data` -- use `record.df_RecAttr` instead

---

Schema Inspection
=================

```python
# Get Human columns
human_cols = record_set.get_human_columns('HmPtt')
print('Human columns:', human_cols)

# Get Record columns
cgm_cols = record_set.get_record_columns('HmPtt', 'CGM5Min')
print('CGM columns:', cgm_cols)

# Full schema summary
print('Human_to_columns:', record_set.Human_to_columns)
print('Record_to_columns:', record_set.Record_to_columns)
```

---

Inspection Checklist
====================

1. **Patient count**: Check human.df_Human for number of patients
2. **Record types**: Verify expected Records exist as tuple keys in Name_to_HRF
3. **Temporal alignment**: Verify DT_s column has consistent intervals
   (5-min for CGM domain)
4. **attr_cols consistency**: Check attribute columns match the RecordFn definition
5. **Date ranges**: Verify temporal coverage per patient
6. **Data density**: Check CGM5Min record coverage (gaps indicate missing data)
7. **Manifest**: Check manifest.json for lineage back to SourceSet

---

Quick Inspection Commands (Bash)
=================================

```bash
# List available RecordSets
ls _WorkSpace/2-RecStore/

# List contents of a RecordSet (FLAT structure)
ls _WorkSpace/2-RecStore/OhioT1DM_v0RecSet/
# Should show: Human-HmPtt/  Record-HmPtt.Ptt/  Record-HmPtt.CGM5Min/  ...

# Check file sizes
du -sh _WorkSpace/2-RecStore/OhioT1DM_v0RecSet/

# Read manifest
cat _WorkSpace/2-RecStore/OhioT1DM_v0RecSet/manifest.json
```

---

MUST DO
=======

- **Activate** .venv and **load** env.sh before any Python operations
- **Use** path-based loading: `RecordSet.load_from_disk(path=..., SPACE=SPACE)`
- **Access** data through `record_set.Name_to_HRF` with correct key types
- **Check** manifest.json for lineage and metadata

---

MUST NOT
========

- **NEVER modify** loaded data or any files in _WorkSpace/2-RecStore/
- **NEVER assume** data exists -- **check** first with ls or try/except
- **NEVER use** `load_from_disk(set_name=..., store_key=...)` -- does not exist
- **NEVER use** `record_set.items()`, `human.records`, or `record.data` -- do not exist
- **NEVER assume** hierarchical directory layout (it is FLAT)
