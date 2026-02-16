Skill: haipipe-data-1-source
============================

Layer 1 of the 6-layer data pipeline: Source Processing.

Converts raw data files (CSV, XML, Parquet, JSON) into a standardized
SourceSet -- a dictionary of DataFrames keyed by table name.

Four subcommands:

  /haipipe-data-1-source load           Inspect existing SourceSet
  /haipipe-data-1-source cook           Run Source_Pipeline with config
  /haipipe-data-1-source design-chef    Create new SourceFn via builder
  /haipipe-data-1-source design-kitchen Upgrade Source_Pipeline infra

On invocation, read this file for shared rules, then read the matching
subcommand file (load.md, cook.md, design-chef.md, design-kitchen.md).

---

Architecture Position
=====================

```
Layer 6: Endpoint         Deployment package for inference
    |
Layer 5: ModelInstance    Trained model artifact
    |
Layer 4: AIData           ML-ready datasets (train/val/test splits)
    |
Layer 3: Case             Event-triggered feature extraction
    |
Layer 2: Record           5-min aligned patient time series
    |
Layer 1: Source  <---     Raw files -> standardized tables
```

---

Cooking Metaphor
================

```
Concept    Pipeline Term              Location
---------  -------------------------  ------------------------------------------
Kitchen    Source_Pipeline class       code/haipipe/source_base/
Chef       SourceFn functions         code/haifn/fn_source/           (GENERATED)
Recipe     YAML config file           config/ or tutorials/config/
Dish       SourceSet asset            _WorkSpace/1-SourceStore/
Academy    Builder scripts            code-dev/1-PIPELINE/1-Source-WorkSpace/
```

The Kitchen (Source_Pipeline) orchestrates execution. The Chef (SourceFn) does
the actual data transformation. The Recipe (YAML config) tells the Kitchen
which Chef to use and where to find raw data. The Dish (SourceSet) is the
output. The Academy (builder scripts) is where you train new Chefs.

---

What Is a SourceSet?
====================

A SourceSet is an Asset that holds **ProcName_to_ProcDf** -- a dictionary
mapping table names to pandas DataFrames.

```python
source_set.ProcName_to_ProcDf = {
    'CGM':        DataFrame(columns=['PatientID', 'ObservationDateTime', 'BGValue', ...]),
    'Medication': DataFrame(columns=['PatientID', 'AdministrationDate', 'Dose', ...]),
    'Diet':       DataFrame(columns=['PatientID', 'ObservationDateTime', 'Carbs', ...]),
    'Exercise':   DataFrame(columns=['PatientID', 'ObservationDateTime', ...]),
    'Ptt':        DataFrame(columns=['PatientID', 'Gender', 'DateOfBirth', ...]),
}
```

Each key is a **ProcName** (processed table name). Which ProcNames exist
depends on the domain and dataset:

```
Domain            Typical ProcNames
----------------  --------------------------------------------------
Diabetes / CGM    CGM, Medication, Diet, Exercise, Ptt, Height, Weight
EHR               Encounters, Labs, Vitals, Diagnoses, Procedures
Genomics          Variants, Samples, Annotations
Wearables         HeartRate, Steps, Sleep, SpO2
```

The SourceFn defines the full list in its **ProcName_List** attribute.

---

Schema Consistency
==================

Within a domain, all SourceFns MUST produce identical column sets for shared
table types. This is what makes Layer 2 (Record) work -- it expects the same
columns regardless of which dataset produced them.

**Example: Diabetes domain standard schemas**

Medication (11 columns):

```python
['PatientID', 'MedAdministrationID',
 'AdministrationDate', 'EntryDateTime', 'UserAdministrationDate',
 'AdministrationTimeZoneOffset', 'AdministrationTimeZone',
 'MedicationID', 'Dose',
 'medication', 'external_metadata']
```

Exercise (13 columns):

```python
['PatientID', 'ExerciseEntryID',
 'ObservationDateTime', 'ObservationEntryDateTime',
 'TimezoneOffset', 'Timezone',
 'ExerciseType', 'ExerciseIntensity',
 'ExerciseDuration', 'CaloriesBurned', 'DistanceInMeters',
 'exercise', 'external_metadata']
```

Diet (15 columns):

```python
['PatientID', 'CarbsEntryID',
 'ObservationDateTime', 'ObservationEntryDateTime',
 'TimezoneOffset', 'Timezone',
 'FoodName', 'ActivityType',
 'Carbs', 'Calories', 'Protein', 'Fat', 'Fiber',
 'nutrition', 'external_metadata']
```

**Core + Extended Fields Pattern:**

- Core fields: Kept as columns AND in JSON (e.g., Dose, Carbs, ExerciseDuration)
- Extended fields: Only in JSON (e.g., MedicationType, bwz_carb_input, HeartRate)
- Dataset-specific metadata: Only in the `external_metadata` JSON column

---

Concrete Code From the Repo
============================

**Loading an existing SourceSet:**

```python
from haipipe.source_base import SourceSet

# Load by full path
source_set = SourceSet.load_from_disk(
    path='/full/path/to/_WorkSpace/1-SourceStore/OhioT1DM/@OhioT1DMxmlv250302',
    SPACE=SPACE
)

# Equivalent approach
source_set = SourceSet.load_asset(
    path='/full/path/to/_WorkSpace/1-SourceStore/OhioT1DM/@OhioT1DMxmlv250302',
    SPACE=SPACE
)

# Inspect
for name, df in source_set.ProcName_to_ProcDf.items():
    print(f'{name}: {df.shape[0]} rows, {df.shape[1]} cols')
```

**Running the pipeline:**

```python
from haipipe.source_base import Source_Pipeline

pipeline = Source_Pipeline(config, SPACE)
source_set = pipeline.run(
    raw_data_name='OhioT1DM',         # Folder name in SourceStore
    raw_data_path=None,                 # None=check store, or S3 URL, or abs path
    payload_input=None,                 # Dict for inference mode (no caching)
    use_cache=True,                     # Load cached if available
    save_cache=True                     # Save results to cache
)
```

**Saving:**

```python
source_set.save_to_disk()                       # Auto path from SPACE
source_set.save_to_disk(path='/custom/path')     # Custom path
```

**SourceFn module structure** (code/haifn/fn_source/*.py):

```python
# Module-level attributes (required):
SourceFile_SuffixList = ['.xml']  # or ['.csv'], ['.parquet'], etc.
ProcName_List = ['CGM', 'Medication', 'Diet', 'Exercise', 'Ptt']
ProcName_to_columns = {
    'CGM': ['PatientID', 'ObservationDateTime', 'BGValue', ...],
    'Medication': ['PatientID', 'AdministrationDate', ...],
}

# Main function (required):
def process_Source_to_Processed(SourceFile_List, get_ProcName_from_SourceFile, SPACE=None):
    ProcName_to_ProcDf = {}
    # ... processing logic ...
    return ProcName_to_ProcDf
```

---

SourceSet On-Disk Layout
========================

```
_WorkSpace/1-SourceStore/{raw_data_name}/@{SourceFnName}/
    {ProcName1}.parquet
    {ProcName2}.parquet
    ...
    manifest.json
```

Example:

```
_WorkSpace/1-SourceStore/OhioT1DM/@OhioT1DMxmlv250302/
    CGM.parquet
    Medication.parquet
    Diet.parquet
    Exercise.parquet
    Ptt.parquet
    Height.parquet
    Weight.parquet
    manifest.json
```

---

Currently Registered SourceFns
==============================

```
SourceFn Name          Cohort        Input    Builder File
---------------------  ------------  -------  -----------------------------------------------
WellDocDataV251226     WellDoc       CSV/Pqt  c1_build_source_welldocdatav251226.py
OhioT1DMxmlv250302     OhioT1DM      XML      c2_build_source_Ohio251226.py
CGMacrosV251227        CGMacros      CSV      c3_build_source_CGMacrosV251227.py
dubossonV251227        dubosson      CSV      c4_build_source_dubossonV251227.py
AIREADIv2V251226       aireadi       Parquet  c5_build_source_aireadi251226.py
```

All builders live in: code-dev/1-PIPELINE/1-Source-WorkSpace/

---

Prerequisites (Before MUST DO)
==============================

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Load workspace paths
source env.sh

# 3. Verify installation
python -c "import haipipe, haifn; print('OK')"
```

---

MUST DO (All Subcommands)
=========================

1. **Activate .venv first**: `source .venv/bin/activate`
2. **Load environment**: `source env.sh`
3. **Respect the golden rule**: output of Layer 1 = input of Layer 2 (Record)
4. **Maintain schema consistency** per domain -- all SourceFns for a domain
   must produce identical column sets for shared table types
5. **Keep core fields as columns AND in JSON** metadata columns
6. **Store extended fields ONLY in JSON** metadata columns
7. **Present plan to user and get approval** before any code changes

---

MUST NOT (All Subcommands)
==========================

1. **NEVER edit** code/haifn/ directly -- 100% generated from builders
2. **NEVER run** Python without .venv activated
3. **NEVER skip** `source env.sh` -- store paths come from environment variables
4. **NEVER break** cross-dataset schema consistency within a domain
5. **NEVER invent** column names outside the domain's standard schema
6. **NEVER skip** RUN_TEST = True in builders

---

Key File Locations
==================

```
Pipeline framework:   code/haipipe/source_base/source_pipeline.py
                      code/haipipe/source_base/source_set.py
                      code/haipipe/source_base/source_utils.py
Fn loader:            code/haipipe/source_base/builder/sourcefn.py

Generated SourceFns:  code/haifn/fn_source/WellDocDataV251226.py
                      code/haifn/fn_source/OhioT1DMxmlv250302.py
                      code/haifn/fn_source/CGMacrosV251227.py
                      code/haifn/fn_source/dubossonV251227.py
                      code/haifn/fn_source/AIREADIv2V251226.py

Builders (edit here):  code-dev/1-PIPELINE/1-Source-WorkSpace/c1_build_source_welldocdatav251226.py
                       code-dev/1-PIPELINE/1-Source-WorkSpace/c2_build_source_Ohio251226.py
                       code-dev/1-PIPELINE/1-Source-WorkSpace/c3_build_source_CGMacrosV251227.py
                       code-dev/1-PIPELINE/1-Source-WorkSpace/c4_build_source_dubossonV251227.py
                       code-dev/1-PIPELINE/1-Source-WorkSpace/c5_build_source_aireadi251226.py

Test configs:          tutorials/config/test-haistep-ohio/1_test_source.yaml
Store path:            _WorkSpace/1-SourceStore/
```

---

File Layout
===========

```
haipipe-data-1-source/
    SKILL.md              This file (router + shared rules)
    README.md             Quick reference
    load.md               /haipipe-data-1-source load
    cook.md               /haipipe-data-1-source cook
    design-chef.md        /haipipe-data-1-source design-chef
    design-kitchen.md     /haipipe-data-1-source design-kitchen
    templates/
        config.yaml       Config template for cook subcommand
```

---

See Also
========

- **haipipe-data-0-overview**: Architecture map and decision guide
- **haipipe-data-2-record**: How SourceSets become RecordSets (Layer 2)
- **haipipe-data-3-case**: How RecordSets become CaseSets (Layer 3)
- **haipipe-data-4-aidata**: How CaseSets become ML-ready datasets (Layer 4)
