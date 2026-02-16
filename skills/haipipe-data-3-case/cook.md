Subcommand: cook
================

Purpose: Run Case_Pipeline with a YAML config (recipe).
Kitchen (Case_Pipeline) + Chef (TriggerFn/CaseFn) already exist.
You write the Recipe (config YAML).

---

How To Run
==========

**CLI command:**

```bash
source .venv/bin/activate
source env.sh
haistep-case --config config/test-haistep-ohio/3_test_case.yaml
```

**Test script:**

```bash
source .venv/bin/activate
source env.sh
python test/test_haistep/test_3_case/test_case.py \
    --config config/test-haistep-ohio/3_test_case.yaml
```

**Python API:**

```python
from haipipe.case_base import Case_Pipeline

pipeline = Case_Pipeline(config, SPACE, context=None)

case_set = pipeline.run(
    df_case=None,           # Optional pre-generated case DataFrame
    df_case_raw=None,       # Optional raw trigger results
    record_set=record_set,  # RecordSet (required)
    use_cache=True,         # Load cached CaseFn features from disk
    profile=False           # Return (case_set, timing) if True
)
```

---

Config Structure
================

The config uses `Case_Args` as a list of operations (trigger, filter, feature stages):

```yaml
record_set_name: "OhioT1DM_v0RecSet"

CaseArgs:
  Case_Args:
    # Stage 1: Trigger -- identifies WHEN cases happen
    - TriggerName: CGM5MinLTS
      TriggerArgs:
          TriggerFolderName: "CGM5MinLTS-Ohio"
          case_id_columns: ['PID', 'lts_id', 'ObsDT']
          case_raw_id_columns: ['PatientID', 'ObsDT']
          HumanID_list: ['PID']
          ObsDT: 'ObsDT'
          ROName_to_RONameArgs:
              'hHmPtt.rCGM5Min':
                  attribute_columns: ['PID', 'PatientID', 'DT_s', 'BGValue']
                  RecDT: 'DT_s'
          min_segment_length: 288
          stride: 12
          buffer_start: 240
          buffer_end: 240

    # Stage 2 (optional): Filter -- removes cases by condition
    # - FilterName: HighRisk
    #   FilterArgs: {...}

    # Stage 3: Feature extraction -- WHAT features to extract
    - CaseFnName: CGMFeatures
      CaseFnList:
          - CGMValueBf24h
          - CGMValueAf24h
          - DEMEventBf24h
          - DEMEventAf24h
          - PDemoBase
      CaseFnArgs: {}

  # Also accepted: CaseFn_list at CaseArgs level
  CaseFn_list: ['CGMValueBf24h', 'CGMValueAf24h', 'DEMEventBf24h', 'DEMEventAf24h', 'PDemoBase']

  case_set_version: 0          # Output: @v0CaseSet-{TriggerFolderName}/
  use_cache: false              # Set true to reuse cached @CaseFn.parquet files
```

**Required keys:** `record_set_name`, `CaseArgs.Case_Args`, `CaseArgs.case_set_version`

See `templates/config.yaml` for a fully annotated template.

---

Available Triggers
==================

```
TriggerFn Name     | Description                        | Key Config Params
-------------------+------------------------------------+----------------------------
CGM5MinLTS         | Long time series segments           | min_segment_length, stride
CGM5MinDfCase      | DataFrame-based case trigger        | case_id_columns
CGM5MinEvent       | CGM event detection (hypo/hyper)    | event thresholds
CGM5MinEntry       | Every 5-min CGM reading             | (none)
DfCaseInput        | DataFrame input trigger             | case_id_columns
```

---

Available CaseFns
=================

```
CaseFn Name        | Description                        | Window  | Returns
-------------------+------------------------------------+---------+--------
CGMValueBf24h      | CGM glucose values before 24h       | Bf24h   | --tid
CGMValueAf24h      | CGM glucose values after 24h        | Af24h   | --tid
DEMEventBf24h      | Diet/Exercise/Med events before 24h | Bf24h   | --str
DEMEventAf24h      | Diet/Exercise/Med events after 24h  | Af24h   | --str
PDemoBase          | Patient demographics baseline       | Base    | --val
```

---

LTS Trigger Configuration
==========================

For CGM5MinLTS trigger, these parameters control case generation:

```yaml
# Segment filtering
min_segment_length: 288       # Min points (288 = 24 hours at 5-min intervals)
max_consecutive_missing: 3    # Max gaps to interpolate (3 = 15 min)

# Sampling strategy
stride: 12                    # Sampling frequency (12 x 5min = hourly)
buffer_start: 240             # Skip first 240 points (20 hours)
buffer_end: 240               # Skip last 240 points (20 hours)

# Event records to attach to LTS segments
event_record_names: ['Diet5Min', 'Med5Min', 'Exercise5Min']
```

---

ROName_to_RONameArgs Format
============================

Record Objects tell Case_Pipeline which data to load for each case:

```yaml
ROName_to_RONameArgs:
  'hHmPtt.rCGM5Min':                # Human(HmPtt).Record(CGM5Min)
    attribute_columns:               # Columns to load
      - 'PID'
      - 'PatientID'
      - 'DT_s'
      - 'BGValue'
    RecDT: 'DT_s'                    # Datetime column for alignment
  'hHmPtt.rPtt':                     # Human(HmPtt).Record(Ptt)
    attribute_columns:
      - 'PID'
      - 'Gender'
      - 'YearOfBirth'
      - 'DiseaseType'
```

---

Naming Conventions
==================

**Output asset naming:** `@v{N}CaseSet-{TriggerFolderName}`
  Example: `@v0CaseSet-CGM5MinLTS-Ohio`

**Store path:** `_WorkSpace/3-CaseStore/{RecSetName}/@v{N}CaseSet-{TriggerFolder}/`

**Feature column naming:** Pipeline auto-prefixes CaseFn returns:
  `CGMValueBf24h` + `--tid` -> column `CGMValueBf24h--tid`

---

Verification
============

After running, verify the output:

1. **Check** output directory exists in _WorkSpace/3-CaseStore/
2. **Verify** df_case.parquet has expected row count
3. **Verify** @CaseFn.parquet files exist at ROOT (not in subdirectory)
4. **Confirm** all @CaseFn.parquet files have same row count as df_case
5. **Check** manifest.json was created
6. **Load** and inspect (see load.md for pattern)

---

Prerequisites
=============

Before running Case_Pipeline:

1. **Activate** .venv: `source .venv/bin/activate`
2. **Load** environment: `source env.sh`
3. Input RecordSet must **exist** at `_WorkSpace/2-RecStore/{record_set_name}/`
4. If patient_info_path is specified, that file must **exist**
5. If RecordSet does not exist, **run** Record_Pipeline first

---

MUST DO
=======

- **Use** only registered TriggerFn and CaseFn names (see Available sections above)
- **Ensure** record_set_name matches an existing RecordSet
- **Specify** ROName_to_RONameArgs with correct attribute_columns
- **Follow** config key format exactly
- **Activate** .venv and **source** env.sh before running

---

MUST NOT
========

- **NEVER invent** TriggerFn/CaseFn names that do not exist in code/haifn/fn_case/
- **NEVER skip** required config keys
- **NEVER create** files in code/haifn/ -- use design-chef if you need new functions
- **NEVER look** for case_data.parquet -- the file is df_case.parquet
- **NEVER expect** CaseFn files in a CaseFn/ subdirectory -- they are @-prefixed at ROOT
