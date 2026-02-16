Skill: haipipe-data-2-record
=============================

Layer 2 of the 6-layer data pipeline: Record Processing.

Converts a SourceSet into temporally-aligned structured records (RecordSet).
Each entity gets a Human wrapper with linked Record objects. In the CGM domain,
records are aligned to a 5-minute time grid; other domains may use different
alignment strategies.

Four subcommands:

  /haipipe-data-2-record load           Inspect existing RecordSet
  /haipipe-data-2-record cook           Run Record_Pipeline with config
  /haipipe-data-2-record design-chef    Create new HumanFn/RecordFn via builder
  /haipipe-data-2-record design-kitchen Upgrade Record_Pipeline infra

On invocation, read this file for shared rules, then read the matching
subcommand file (load.md, cook.md, design-chef.md, design-kitchen.md).

---

Architecture Position
=====================

```
Layer 1: Source           Raw files -> standardized tables (SourceSet)
    |
Layer 2: Record   <---   SourceSet -> temporally-aligned records (RecordSet)
    |
Layer 3: Case             RecordSet -> event-triggered feature extraction
    |
Layer 4: AIData           CaseSet -> ML-ready datasets (train/val/test)
    |
Layer 5: Model            DataSet -> trained model artifacts
    |
Layer 6: Endpoint         Model -> deployment packages
```

---

Cooking Metaphor
================

```
Kitchen  = Record_Pipeline class     (code/haipipe/record_base/)
Chef     = HumanFn + RecordFn        (code/haifn/fn_record/)        -- GENERATED
Recipe   = YAML config file          (config/ or tutorials/config/)
Dish     = RecordSet asset           (_WorkSpace/2-RecStore/)
Academy  = Builder scripts           (code-dev/1-PIPELINE/2-Record-WorkSpace/)
```

---

What Is a RecordSet?
====================

A RecordSet is a container that holds Human and Record objects in a flat
dictionary called **Name_to_HRF**. Keys are either strings (for Humans) or
tuples (for Records):

```python
record_set.Name_to_HRF = {
    'HmPtt':                  <Human object>,         # string key
    ('HmPtt', 'Ptt'):         <Record object>,        # tuple key
    ('HmPtt', 'CGM5Min'):     <Record object>,        # tuple key
    ('HmPtt', 'Diet5Min'):    <Record object>,        # tuple key
    ('HmPtt', 'Exercise5Min'):<Record object>,        # tuple key
    ('HmPtt', 'Med5Min'):     <Record object>,        # tuple key
}
```

**Important:** The HumanRecords mapping is domain-configurable. The example
above shows CGM-domain records. For an EHR domain you might have:

```python
# EHR example (hypothetical):
# HumanRecords: { HmPatient: [Encounters, Labs, Vitals, Medications] }
```

**Directory structure is FLAT** (not hierarchical by patient):

```
_WorkSpace/2-RecStore/{RecordSetName}/
+-- Human-HmPtt/                    # One per Human (FLAT naming)
|   +-- df_Human.parquet
|   +-- schema.json
+-- Record-HmPtt.Ptt/              # FLAT: Record-{HumanName}.{RecordName}
|   +-- df_RecAttr.parquet
|   +-- df_RecIndex.parquet
+-- Record-HmPtt.CGM5Min/
|   +-- df_RecAttr.parquet
|   +-- df_RecIndex.parquet
+-- Record-HmPtt.Diet5Min/
|   +-- df_RecAttr.parquet
|   +-- df_RecIndex.parquet
+-- Record-HmPtt.Exercise5Min/
|   +-- df_RecAttr.parquet
|   +-- df_RecIndex.parquet
+-- Record-HmPtt.Med5Min/
|   +-- df_RecAttr.parquet
|   +-- df_RecIndex.parquet
+-- Extra-{name}/                   # Optional extra DataFrames
+-- manifest.json
+-- _cache/
```

**It is NOT hierarchical** like HmPtt/<PatientID>/CGM5Min/. It is flat.

---

5-Minute Alignment (CGM Domain-Specific)
=========================================

In the CGM domain, all records are aligned to a 5-minute time grid:

- **DT_s**: Datetime column with 5-min intervals (00:00, 00:05, 00:10, ...)
- Continuous records (CGM): One row per 5-min interval
- Event records (Diet, Med, Exercise): Snapped to nearest 5-min boundary
- Missing intervals: Left as gaps (not filled with NaN)

**Note:** 5-minute alignment is specific to the CGM domain. Other domains
(e.g., EHR with daily encounters) may use different temporal alignment.

---

Concrete Code From the Repo
=============================

**Creating and running a pipeline** (code/haipipe/record_base/record_pipeline.py):

```python
from haipipe.record_base import Record_Pipeline

# Config keys: HumanRecords (dict), record_set_version (int, default 0)
config = {
    'HumanRecords': {'HmPtt': ['Ptt', 'CGM5Min', 'Diet5Min', 'Exercise5Min', 'Med5Min']},
    'record_set_version': 0
}
pipeline = Record_Pipeline(config, SPACE)

record_set = pipeline.run(
    source_set,                          # SourceSet (required)
    partition_index=None,                # 0-based partition index
    partition_number=None,               # Total partitions
    record_set_label=1,                  # Label for ID generation
    use_cache=True,                      # Load cached if available
    save_cache=True,                     # Save to cache
    profile=False                        # Return timing if True
)
# Name is auto-generated by _generate_record_set_name()
# e.g., "OhioT1DM_v0RecSet" from source_set_name "OhioT1DM/@..." + version 0
```

**Saving a RecordSet** (code/haipipe/assets.py):

```python
record_set.save_to_disk()  # auto-generates path from SPACE
```

**Loading a RecordSet** (code/haipipe/assets.py):

```python
record_set = RecordSet.load_from_disk(path='/full/path/to/record_set', SPACE=SPACE)
record_set = RecordSet.load_asset(path='/full/path/to/record_set', SPACE=SPACE)
```

**Accessing data via Name_to_HRF** (code/haipipe/record_base/record_set.py):

```python
# Access Human (string key)
human = record_set.Name_to_HRF['HmPtt']

# Access Record (tuple key)
cgm_record = record_set.Name_to_HRF[('HmPtt', 'CGM5Min')]
diet_record = record_set.Name_to_HRF[('HmPtt', 'Diet5Min')]

# RecordSet info summary
record_set.info()
```

**HumanFn module structure** (code/haifn/fn_record/human/HmPtt.py):

```python
# Module-level:
OneHuman_Args = {'HumanName': 'HmPtt', 'HumanID': 'PID', 'RawHumanID': 'PatientID', ...}
Excluded_RawNameList = ['CGM', 'Diet', 'Exercise', 'Medication', ...]

def get_RawHumanID_from_dfRawColumns(dfRawColumns):
    # Returns the raw human ID column name
    ...

MetaDict = {
    "OneHuman_Args": OneHuman_Args,
    "Excluded_RawNameList": Excluded_RawNameList,
    "get_RawHumanID_from_dfRawColumns": get_RawHumanID_from_dfRawColumns
}
```

**RecordFn module structure** (code/haifn/fn_record/record/CGM5Min.py):

```python
# Module-level:
OneRecord_Args = {'RecordName': 'CGM5Min', 'RecID': 'CGM5MinID', ...}
RawName_to_RawConfig = {'CGM': {'raw_columns': [...], ...}}
attr_cols = ['PID', 'PatientID', 'CGM5MinID', 'DT_s', 'BGValue']

def get_RawRecProc_for_HumanGroup(df_RawRec_for_HumanGroup, OneRecord_Args, df_Human):
    # Process raw data into aligned records
    ...
```

---

Currently Registered Fns
=========================

**HumanFn:**

```
HumanFn Name    | Builder File
----------------+-----------------------------------
HmPtt           | h1_build_human_HmPtt.py
```

**RecordFn:**

```
RecordFn Name   | Description              | Builder File
----------------+--------------------------+-----------------------------------
Ptt             | Patient time records     | r1_build_record_Ptt.py
CGM5Min         | CGM 5-min readings       | r2_build_record_CGM5Min.py
Diet5Min        | Diet 5-min alignment     | r3_build_record_Diet5Min.py
Exercise5Min    | Exercise 5-min alignment | r4_build_record_Exercise5Min.py
Med5Min         | Medication 5-min align   | r5_build_record_Med5Min.py
```

---

Prerequisites
=============

1. Python virtual environment activated: `source .venv/bin/activate`
2. Environment variables loaded: `source env.sh`
3. Input SourceSet must exist at `_WorkSpace/1-SourceStore/<source_set_name>/`
4. If SourceSet does not exist, run Source_Pipeline first (see haipipe-data-1-source cook)

---

MUST DO (All Subcommands)
=========================

1. **Activate** .venv first: `source .venv/bin/activate`
2. **Load** environment: `source env.sh`
3. **Verify** that input SourceSet exists before running Record_Pipeline
4. **Use** the correct load API: `RecordSet.load_from_disk(path=..., SPACE=SPACE)`
5. **Access** data via Name_to_HRF with string keys (Human) and tuple keys (Record)
6. **Present** plan to user and **get** approval before any code changes
7. **Remember** that output of Layer 2 = input of Layer 3 (Case)

---

MUST NOT (All Subcommands)
==========================

1. **NEVER edit** code/haifn/ directly (100% generated from builders)
2. **NEVER run** Python without .venv activated
3. **NEVER skip** `source env.sh` (store paths come from environment variables)
4. **NEVER use** fabricated APIs like `record_set.items()`, `human.records`, or `record.data`
5. **NEVER assume** hierarchical directory structure (it is FLAT: Human-X/, Record-X.Y/)
6. **NEVER pass** `cohort_name` to `pipeline.run()` (does not exist as a parameter)
7. **NEVER use** `load_from_disk(set_name=..., store_key=...)` (does not exist)
8. **NEVER change** attr_cols without updating all downstream CaseFn references

---

Key File Locations
==================

```
Pipeline framework:   code/haipipe/record_base/record_pipeline.py
                      code/haipipe/record_base/record_set.py
                      code/haipipe/assets.py
Fn loaders:           code/haipipe/record_base/builder/human.py
                      code/haipipe/record_base/builder/record.py

Generated HumanFn:    code/haifn/fn_record/human/HmPtt.py
Generated RecordFns:  code/haifn/fn_record/record/Ptt.py
                      code/haifn/fn_record/record/CGM5Min.py
                      code/haifn/fn_record/record/Diet5Min.py
                      code/haifn/fn_record/record/Exercise5Min.py
                      code/haifn/fn_record/record/Med5Min.py

Builders (edit these): code-dev/1-PIPELINE/2-Record-WorkSpace/h1_build_human_HmPtt.py
                       code-dev/1-PIPELINE/2-Record-WorkSpace/r1_build_record_Ptt.py
                       code-dev/1-PIPELINE/2-Record-WorkSpace/r2_build_record_CGM5Min.py
                       code-dev/1-PIPELINE/2-Record-WorkSpace/r3_build_record_Diet5Min.py
                       code-dev/1-PIPELINE/2-Record-WorkSpace/r4_build_record_Exercise5Min.py
                       code-dev/1-PIPELINE/2-Record-WorkSpace/r5_build_record_Med5Min.py

Test configs:         config/test-haistep-ohio/2_test_record.yaml
Store path:           _WorkSpace/2-RecStore/
```

---

See Also
========

- **haipipe-data-1-source**: How raw data becomes SourceSets (input to this layer)
- **haipipe-data-3-case**: How RecordSets become CaseSets (output of this layer)
- **haipipe-data-4-aidata**: How CaseSets become ML-ready datasets

---

File Layout
===========

```
haipipe-data-2-record/
+-- SKILL.md              This file (router + shared rules)
+-- README.md             Quick reference
+-- load.md               /haipipe-data-2-record load
+-- cook.md               /haipipe-data-2-record cook
+-- design-chef.md        /haipipe-data-2-record design-chef
+-- design-kitchen.md     /haipipe-data-2-record design-kitchen
+-- templates/
    +-- config.yaml       Config template for cook subcommand
```
