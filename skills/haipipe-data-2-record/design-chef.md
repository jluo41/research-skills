Subcommand: design-chef
=======================

Purpose: Create a new HumanFn or RecordFn via the builder pattern.
Edit builders in code-dev/ -> run builder -> generates code/haifn/fn_record/.

---

Workflow
========

1. **Present** plan to user -> **get** approval
2. **Activate** environment:
   ```bash
   source .venv/bin/activate
   source env.sh
   ```
3. **Copy** an existing builder as starting point
4. **Edit** [CUSTOMIZE] sections only (keep [BOILERPLATE] as-is)
5. **Run** builder:
   ```bash
   python code-dev/1-PIPELINE/2-Record-WorkSpace/<your_builder>.py
   ```
6. **Verify** generated file in code/haifn/fn_record/human/ or fn_record/record/
7. **Register** in config YAML (HumanRecords mapping)
8. **Test** end-to-end with Record_Pipeline

---

Builder Location
================

```
code-dev/1-PIPELINE/2-Record-WorkSpace/
+-- h1_build_human_HmPtt.py          (Human builder -- patient demographics)
+-- r1_build_record_Ptt.py           (Patient time records)
+-- r2_build_record_CGM5Min.py       (CGM 5-min records)
+-- r3_build_record_Diet5Min.py      (Diet 5-min records)
+-- r4_build_record_Exercise5Min.py  (Exercise 5-min records)
+-- r5_build_record_Med5Min.py       (Medication 5-min records)
+-- examples/                         (Template builders for reference)
+-- notebooks/                        (Development notebooks)
+-- Old/                              (Legacy builders)
```

**Two types of builders:**

- **HumanFn builders (h* prefix):** Create Human wrapper objects with demographics
- **RecordFn builders (r* prefix):** Process individual record types from SourceSet

---

Naming Convention
=================

**HumanFn builder:** `h<N>_build_human_<HumanFnName>.py`
  Example: `h2_build_human_HmExtended.py`

**RecordFn builder:** `r<N>_build_record_<RecordFnName>.py`
  Example: `r6_build_record_InsulinPump5Min.py`

**Generated HumanFn:** `code/haifn/fn_record/human/<HumanFnName>.py`
**Generated RecordFn:** `code/haifn/fn_record/record/<RecordFnName>.py`

---

Builder Template: HumanFn
=========================

The generated module must expose these module-level variables and a MetaDict.

**What the generated file looks like** (code/haifn/fn_record/human/HmPtt.py):

```python
# Module-level variables:
OneHuman_Args = {
    'HumanName': 'HmPtt',
    'HumanID': 'PID',
    'RawHumanID': 'PatientID',
    'HumanIDLength': 10
}

Excluded_RawNameList = ['CGM', 'Diet', 'Exercise', 'Medication', 'Height', 'Weight']

def get_RawHumanID_from_dfRawColumns(dfRawColumns):
    """Identify the raw human ID column from available columns."""
    RawHumanID_selected = None
    if 'PatientID' in dfRawColumns:
        RawHumanID_selected = 'PatientID'
    return RawHumanID_selected

MetaDict = {
    "OneHuman_Args": OneHuman_Args,
    "Excluded_RawNameList": Excluded_RawNameList,
    "get_RawHumanID_from_dfRawColumns": get_RawHumanID_from_dfRawColumns
}
```

**Builder structure:**

```python
# ==========================================================
# [BOILERPLATE] Configuration
# ==========================================================
OUTPUT_DIR = 'fn_record/human'
HUMAN_FN_NAME = 'HmPtt'             # [CUSTOMIZE] Name
RUN_TEST = True

# ==========================================================
# [CUSTOMIZE] Define OneHuman_Args, Excluded_RawNameList,
#             get_RawHumanID_from_dfRawColumns
# ==========================================================
# ... your domain-specific logic here ...

# ==========================================================
# [BOILERPLATE] Code Generation + Test
# ==========================================================
# ... generates code/haifn/fn_record/human/<HumanFnName>.py ...
```

---

Builder Template: RecordFn
==========================

The generated module must expose OneRecord_Args, RawName_to_RawConfig,
attr_cols, and a processing function.

**What the generated file looks like** (code/haifn/fn_record/record/CGM5Min.py):

```python
# Module-level variables:
OneRecord_Args = {
    'RecordName': 'CGM5Min',
    'RecID': 'CGM5MinID',
    'RecIDChain': ['PID'],
    'RawHumanID': 'PatientID',
    'ParentRecName': 'Ptt',
    'RecDT': 'DT_s',
    'RawNameList': ['CGM'],
    'human_group_size': 50,
    'rec_chunk_size': 100000
}

RawName_to_RawConfig = {
    'CGM': {
        'raw_columns': ['PatientID', 'ObservationDateTime', 'BGValue', ...],
        'raw_base_columns': ['PatientID', 'ObservationDateTime', ...],
        'rec_chunk_size': 100000,
        'raw_datetime_column': 'ObservationDateTime'
    }
}

attr_cols = ['PID', 'PatientID', 'CGM5MinID', 'DT_s', 'BGValue']

def get_RawRecProc_for_HumanGroup(df_RawRec_for_HumanGroup, OneRecord_Args, df_Human):
    """Process raw data into aligned records."""
    # [CUSTOMIZE] Your domain-specific processing logic
    ...
```

**Builder structure:**

```python
# ==========================================================
# [BOILERPLATE] Configuration
# ==========================================================
OUTPUT_DIR = 'fn_record/record'
RECORD_FN_NAME = 'CGM5Min'          # [CUSTOMIZE] Name
RUN_TEST = True

# ==========================================================
# [CUSTOMIZE] Define OneRecord_Args, RawName_to_RawConfig,
#             attr_cols, get_RawRecProc_for_HumanGroup
# ==========================================================
# ... your domain-specific processing logic here ...

# ==========================================================
# [BOILERPLATE] Code Generation + Test
# ==========================================================
# ... generates code/haifn/fn_record/record/<RecordFnName>.py ...
```

---

attr_cols Consistency
=====================

Each RecordFn defines `attr_cols` that downstream CaseFn functions access.
These columns MUST remain consistent across all patients:

```python
# CGM5Min attr_cols:
['PID', 'PatientID', 'CGM5MinID', 'DT_s', 'BGValue']

# Diet5Min attr_cols:
['PID', 'PatientID', 'DT_s', 'Carbs', 'Calories', ...]

# Ptt attr_cols:
['PID', 'Gender', 'YearOfBirth', 'DiseaseType']
```

Changing attr_cols breaks ALL downstream CaseFn that reference them.

---

Domain Generality
=================

The RecordFn structure works for any temporal data type, not just CGM:

- **CGM domain:** CGM5Min with 5-minute alignment, Diet5Min, Med5Min, etc.
- **EHR domain:** Encounters, Labs, Vitals with appropriate time resolution
- **Wearable domain:** HeartRate, Steps, Sleep with device-specific alignment

The alignment strategy and attr_cols are defined per RecordFn, making the
framework domain-agnostic.

---

MUST DO
=======

- **Present** plan to user -> **get** approval -> **execute**
- **Activate** .venv and **load** env.sh before running builder
- **Follow** [BOILERPLATE] + [CUSTOMIZE] pattern
- **Set** RUN_TEST = True
- **Maintain** temporal alignment consistency for the domain
- **Keep** attr_cols consistent across patients
- **Include** MetaDict (for HumanFn) or all required module-level variables (for RecordFn)

---

MUST NOT
========

- **NEVER edit** code/haifn/fn_record/ directly
- **NEVER skip** builder -> generate -> test cycle
- **NEVER break** temporal alignment for the domain
- **NEVER change** attr_cols without updating all downstream CaseFn
- **NEVER skip** RUN_TEST
- **NEVER run** builder without `source .venv/bin/activate && source env.sh`
