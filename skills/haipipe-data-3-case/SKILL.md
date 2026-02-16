Skill: haipipe-data-3-case
==========================

Layer 3 of the 6-layer data pipeline: Case Processing.

Converts a RecordSet into event-triggered cases (CaseSet). A TriggerFn
identifies time points of interest. CaseFn functions extract features
around each trigger point using Record Objects (RO).

Four subcommands:

  /haipipe-data-3-case load           Inspect existing CaseSet
  /haipipe-data-3-case cook           Run Case_Pipeline with config
  /haipipe-data-3-case design-chef    Create new TriggerFn/CaseFn via builder
  /haipipe-data-3-case design-kitchen Upgrade Case_Pipeline infra

On invocation, read this file for shared rules, then read the matching
subcommand file (load.md, cook.md, design-chef.md, design-kitchen.md).

---

Architecture Position
=====================

```
Layer 1: Source          Raw files -> standardized tables
    |
Layer 2: Record          SourceSet -> 5-min aligned patient records
    |
Layer 3: Case    <---    RecordSet -> event-triggered feature extraction
    |
Layer 4: AIData          CaseSet -> ML-ready datasets (train/val/test)
    |
Layer 5: Model           AIData -> trained model artifacts
    |
Layer 6: Endpoint        Model -> deployment packages
```

---

Cooking Metaphor
================

```
Kitchen  = Case_Pipeline class       (code/haipipe/case_base/)
Chef     = TriggerFn + CaseFn        (code/haifn/fn_case/)         -- GENERATED
Recipe   = YAML config file          (config/caseset/ or tutorials/config/)
Dish     = CaseSet asset             (_WorkSpace/3-CaseStore/)
Academy  = Builder scripts           (code-dev/1-PIPELINE/3-Case-WorkSpace/)
```

---

What Is a CaseSet?
==================

A CaseSet is the output of Layer 3. It contains one base table (df_case.parquet)
with trigger metadata, plus one @-prefixed parquet per CaseFn with extracted
features. All files live at the ROOT level of the CaseSet directory -- there
is NO CaseFn/ subdirectory.

**CaseSet directory structure:**

```
_WorkSpace/3-CaseStore/{RecSetName}/@v{N}CaseSet-{TriggerFolder}/
+-- df_case.parquet              Base case data (PID, ObsDT, trigger cols)
+-- df_lts.parquet               LTS segments (optional, for LTS triggers)
+-- df_Human_Info.parquet        Patient metadata (optional)
+-- @CGMValueBf24h.parquet       CaseFn features (at ROOT, NOT in subdirectory)
+-- @CGMValueAf24h.parquet       Another CaseFn (at ROOT)
+-- @DEMEventBf24h.parquet       Another CaseFn (at ROOT)
+-- @DEMEventAf24h.parquet       Another CaseFn (at ROOT)
+-- @PDemoBase.parquet           Another CaseFn (at ROOT)
+-- cf_to_cfvocab.json           Vocabulary per CaseFn
+-- manifest.json
```

**CRITICAL:** CaseFn files are `@{CaseFnName}.parquet` at ROOT level. The main
file is `df_case.parquet`, NOT `case_data.parquet`.

---

Feature Naming Convention
=========================

All CaseFn names follow `<Feature><Window>`:

**Window Suffixes:**

```
Suffix      | Meaning                  | Example
------------+--------------------------+--------------------------
Bf24h       | Before 24 hours          | CGMValueBf24h
Af24h       | After 24 hours           | CGMValueAf24h
Bf2h        | Before 2 hours           | CGMValueBf2h
Af2h        | After 2 hours            | CGMValueAf2h
Af2to8h     | After 2 to 8 hours       | CGMValueAf2to8h
CrntTime    | Current time features    | InvCrntTimeFixedLen
Base        | Baseline (no window)     | PDemoBase
Bf7d        | Before 7 days            | LabValueBf7d
Af30d       | After 30 days            | VitalSignAf30d
```

**Feature Prefixes:**

```
Prefix      | Domain                   | Example
------------+--------------------------+--------------------------
CGMValue    | CGM glucose values       | CGMValueBf24h
CGMInfo     | CGM statistics           | CGMInfoBf24h
DEMEvent    | Diet/Exercise/Med events | DEMEventBf24h
PDemo       | Patient demographics     | PDemoBase
PAge        | Patient age              | PAge5
Pgender     | Patient gender           | Pgender
Inv         | Invitation features      | InvCrntTimeFixedLen
DietBase    | Diet features            | DietBaseNutriN2CTknBf24h
```

This convention is domain-general: LabValueBf7d, VitalSignBf1h, etc. all follow
the same `<Feature><Window>` pattern.

---

ROName 3-Part Format
====================

CaseFn accesses patient data through Record Objects using a **3-part** ROName:

```
hHmPtt.rCGM5Min.cBf24h
|       |        +-- Checkpoint suffix (window)
|       +-- Record name
+-- Human name (h prefix)
```

The 3 parts map to ROName_to_RONameInfo:

```python
ROName_to_RONameInfo = {
    'hHmPtt.rCGM5Min.cBf24h': {
        'HumanName': 'HmPtt',
        'RecordName': 'CGM5Min',
        'CkpdName': 'Bf24h'
    }
}
```

**Examples:**

```
ROName                       | Human  | Record      | Checkpoint
-----------------------------+--------+-------------+-----------
hHmPtt.rCGM5Min.cBf24h      | HmPtt  | CGM5Min     | Bf24h
hHmPtt.rCGM5Min.cAf24h      | HmPtt  | CGM5Min     | Af24h
hHmPtt.rDiet5Min.cBf24h     | HmPtt  | Diet5Min    | Bf24h
hHmPtt.rPtt.cBase           | HmPtt  | Ptt         | Base
```

---

Trigger vs CaseFn Separation
=============================

```
+-------------------------------------------+
|  TriggerFn                                |
|  - Identifies WHEN cases happen           |
|  - Outputs: df_case, df_lts, df_Human_Info|
|  - Function: get_CaseTrigger_from_RecordBase()
|  - Examples: CGM5MinLTS, CGM5MinEntry     |
+---------------------+---------------------+
                      |
                      v
+-------------------------------------------+
|  CaseFn (one or more per trigger)         |
|  - Extracts WHAT features at trigger      |
|  - Accesses Record Objects (RO)           |
|  - Function: fn_CaseFn() with 6 params   |
|  - Returns: dict with SUFFIX-ONLY keys    |
|  - Examples: CGMValueBf24h, PDemoBase     |
+-------------------------------------------+
```

---

Concrete Code From the Repo
============================

**CaseFn module** (code/haifn/fn_case/case_casefn/CGMValueBf24h.py):

```python
CaseFnName = "CGMValueBf24h"

RO_to_ROName = {'RO': 'hHmPtt.rCGM5Min.cBf24h'}     # 3-PART format!

Ckpd_to_CkpdObsConfig = {'Bf24h': {
    'DistStartToPredDT': -1440,     # Minutes from trigger to window start
    'DistEndToPredDT': 5,           # Minutes from trigger to window end
    'TimeUnit': 'min',
    'StartIdx5Min': -288,           # Index offset (5-min units) for start
    'EndIdx5Min': 1                 # Index offset for end
}}

ROName_to_RONameInfo = {
    'hHmPtt.rCGM5Min.cBf24h': {
        'HumanName': 'HmPtt',
        'RecordName': 'CGM5Min',
        'CkpdName': 'Bf24h'
    }
}

HumanRecords = {'HmPtt': ['CGM5Min']}

COVocab = {'tid2tkn': ['<Pad>', '<UNK>'], 'tkn2tid': {'<Pad>': 0, '<UNK>': 1}}

# Main function -- 6 POSITIONAL PARAMETERS:
def fn_CaseFn(case_example, ROName_list, ROName_to_ROData,
              ROName_to_ROInfo, COVocab, context):
    """Extract raw CGM values from 24h before trigger."""
    SEQUENCE_LENGTH = 288
    ROName = ROName_list[0]
    ROData = ROName_to_ROData.get(ROName)

    if ROData is None or len(ROData) == 0:
        return {'--tid': [1] * SEQUENCE_LENGTH}         # SUFFIX-ONLY key!

    glucose_values = ROData['BGValue'].fillna(1).astype(int).values.tolist()
    if len(glucose_values) < SEQUENCE_LENGTH:
        glucose_values = [1] * (SEQUENCE_LENGTH - len(glucose_values)) + glucose_values
    elif len(glucose_values) > SEQUENCE_LENGTH:
        glucose_values = glucose_values[-SEQUENCE_LENGTH:]

    return {'--tid': glucose_values}                     # SUFFIX-ONLY key!
```

**CRITICAL:** Return keys are SUFFIX-ONLY (e.g., `'--tid'`, `'--wgt'`, `'--val'`).
The pipeline adds the CaseFnName prefix automatically:

```python
# Inside Case_Pipeline (case_pipeline.py, lines 438-453):
feature_result = fn_CaseFn(
    case, ROName_list, ROName_to_ROData, ROName_to_RONameInfo, COVocab, context
)
for feat_name, feat_value in feature_result.items():
    col_name = f"{cf_name}{feat_name}"    # e.g., "CGMValueBf24h--tid"
    case_features[col_name] = feat_value
```

**TriggerFn module** (code/haifn/fn_case/fn_trigger/CGM5MinLTS.py):

```python
Trigger = "CGM5MinLTS"

Trigger_Args = {
    'Trigger': 'CGM5MinLTS',
    'case_id_columns': ['PID', 'lts_id', 'ObsDT'],
    'case_raw_id_columns': ['PatientID', 'ObsDT'],
    'HumanID_list': ['PID'],
    'ObsDT': 'ObsDT',
    'ROName_to_RONameArgs': {
        'hHmPtt.rCGM5Min': {
            'attribute_columns': ['PID', 'PatientID', 'DT_s', 'BGValue'],
            'RecDT': 'DT_s'
        }
    },
    'min_segment_length': 36,
    'max_consecutive_missing': 3,
    'stride': 12,
    'buffer_start': 48,
    'buffer_end': 48,
}

# CRITICAL: Function name is get_CaseTrigger_from_RecordBase, NOT fn_TriggerFn!
def get_CaseTrigger_from_RecordBase(record_set, Trigger_Args, df_case_raw=None):
    """Returns dict: {'df_case': ..., 'df_lts': ..., 'df_Human_Info': ...}"""
    ...
```

**Case_Pipeline** (code/haipipe/case_base/case_pipeline.py):

```python
pipeline = Case_Pipeline(config, SPACE, context=None)

case_set = pipeline.run(
    df_case=None,           # Optional pre-generated case DataFrame
    df_case_raw=None,       # Optional raw trigger results
    record_set=record_set,  # RecordSet (required)
    use_cache=True,         # Load cached CaseFn features
    profile=False           # Return timing if True
)
```

---

Currently Registered Fns
=========================

**TriggerFn:**

```
TriggerFn Name     | Description              | Builder File
-------------------+--------------------------+-------------------------------------
CGM5MinLTS         | Long time series trigger | a1_build_trigger_cgm5minlts.py
CGM5MinDfCase      | DataFrame case trigger   | a2_build_trigger_cgm5mindfcase.py
CGM5MinEvent       | CGM event trigger        | a3_build_trigger_cgm5minevent.py
CGM5MinEntry       | Every CGM reading        | (built-in)
DfCaseInput        | DataFrame input trigger  | (built-in)
```

**CaseFn:**

```
CaseFn Name        | Description              | Builder File
-------------------+--------------------------+-------------------------------------
CGMValueBf24h      | CGM values before 24h    | c2_build_casefn_cgmvalue.py
CGMValueAf24h      | CGM values after 24h     | c2_build_casefn_cgmvalue.py
DEMEventBf24h      | Events before 24h        | c3_build_casefn_demevent.py
DEMEventAf24h      | Events after 24h         | c3_build_casefn_demevent.py
PDemoBase          | Patient demographics     | c4_build_casefn_pdemobase.py
```

---

Prerequisites
=============

Before working with Layer 3:

1. **Activate** .venv: `source .venv/bin/activate`
2. **Load environment**: `source env.sh`
3. **Input RecordSet** must exist at `_WorkSpace/2-RecStore/<record_set_name>/`
4. **Packages** installed: `pip install -e ".[dev]"`

---

MUST DO (All Subcommands)
=========================

1. **Activate** .venv first: `source .venv/bin/activate`
2. **Load** environment: `source env.sh`
3. **Remember** that output of Layer 3 = input of Layer 4 (AIData)
4. **Follow** feature naming `<Feature><Window>` convention
5. **Declare** ROName_to_RONameInfo with all data dependencies in CaseFn
6. **Use** 3-part ROName format: `hHmPtt.rCGM5Min.cBf24h`
7. **Return** suffix-only keys from fn_CaseFn (e.g., `'--tid'`, not `'CGMValueBf24h--tid'`)
8. **Name** trigger function `get_CaseTrigger_from_RecordBase` (not fn_TriggerFn)
9. **Present** plan to user and **get approval** before any code changes

---

MUST NOT (All Subcommands)
==========================

1. **NEVER edit** code/haifn/ directly (100% generated from builders)
2. **NEVER run** Python without .venv activated
3. **NEVER skip** `source env.sh` (store paths come from environment variables)
4. **NEVER invent** feature names outside `<Feature><Window>` convention
5. **NEVER access** Record data without going through ROName_to_ROData
6. **NEVER skip** the TriggerFn + CaseFn separation
7. **NEVER prefix** return keys with CaseFnName -- pipeline does that automatically
8. **NEVER assume** CaseFn parquets live in a CaseFn/ subdirectory -- they are at ROOT

---

Key File Locations
==================

```
Pipeline framework:   code/haipipe/case_base/case_pipeline.py
                      code/haipipe/case_base/case_set.py
                      code/haipipe/case_base/case_utils.py
Fn loaders:           code/haipipe/case_base/builder/triggerfn.py
                      code/haipipe/case_base/builder/casefn.py
                      code/haipipe/case_base/builder/rotools.py

Generated TriggerFns: code/haifn/fn_case/fn_trigger/CGM5MinLTS.py
                      code/haifn/fn_case/fn_trigger/CGM5MinDfCase.py
                      code/haifn/fn_case/fn_trigger/CGM5MinEvent.py
                      code/haifn/fn_case/fn_trigger/CGM5MinEntry.py
                      code/haifn/fn_case/fn_trigger/DfCaseInput.py
Generated CaseFns:    code/haifn/fn_case/case_casefn/CGMValueBf24h.py
                      code/haifn/fn_case/case_casefn/CGMValueAf24h.py
                      code/haifn/fn_case/case_casefn/DEMEventBf24h.py
                      code/haifn/fn_case/case_casefn/DEMEventAf24h.py
                      code/haifn/fn_case/case_casefn/PDemoBase.py

Builders (edit these): code-dev/1-PIPELINE/3-Case-WorkSpace/a1_build_trigger_cgm5minlts.py
                       code-dev/1-PIPELINE/3-Case-WorkSpace/a2_build_trigger_cgm5mindfcase.py
                       code-dev/1-PIPELINE/3-Case-WorkSpace/a3_build_trigger_cgm5minevent.py
                       code-dev/1-PIPELINE/3-Case-WorkSpace/c2_build_casefn_cgmvalue.py
                       code-dev/1-PIPELINE/3-Case-WorkSpace/c3_build_casefn_demevent.py
                       code-dev/1-PIPELINE/3-Case-WorkSpace/c4_build_casefn_pdemobase.py

Test configs:         config/test-haistep-ohio/3_test_case.yaml
Store path:           _WorkSpace/3-CaseStore/
```

---

See Also
========

- **haipipe-data-1-source**: How raw data becomes SourceSets
- **haipipe-data-2-record**: How SourceSets become RecordSets (input to this layer)
- **haipipe-data-4-aidata**: How CaseSets become ML-ready datasets (output of this layer)

---

File Layout
===========

```
haipipe-data-3-case/
+-- SKILL.md              This file (router + shared rules)
+-- README.md             Quick reference
+-- load.md               /haipipe-data-3-case load
+-- cook.md               /haipipe-data-3-case cook
+-- design-chef.md        /haipipe-data-3-case design-chef
+-- design-kitchen.md     /haipipe-data-3-case design-kitchen
+-- templates/
    +-- config.yaml       Config template for cook subcommand
```
