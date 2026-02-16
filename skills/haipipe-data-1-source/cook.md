Subcommand: cook
================

Purpose: Run Source_Pipeline with a YAML config (recipe).
Kitchen (Source_Pipeline) + Chef (SourceFn) already exist.
You write the Recipe (config YAML) and execute the pipeline.

---

Config Format
=============

The YAML config requires a **SourceArgs** block with at minimum
`raw_data_name` and `SourceFnName`:

```yaml
external_version: "@v1215"

SourceArgs:
  raw_data_name: "OhioT1DM"            # REQUIRED: Cohort directory name
  SourceFnName: "OhioT1DMxmlv250302"    # REQUIRED: Registered SourceFn name
  raw_data_path: null                    # OPTIONAL: Override path (local, S3)
  use_cache: true                        # OPTIONAL: Load cached if available
  save_cache: true                       # OPTIONAL: Save output to disk
```

**Required keys:** `SourceArgs.raw_data_name`, `SourceArgs.SourceFnName`

**Optional keys:** `SourceArgs.raw_data_path`, `use_cache`, `save_cache`

See `templates/config.yaml` for a fully annotated template.

---

How To Run
==========

**CLI command:**

```bash
source .venv/bin/activate
source env.sh
haistep-source --config tutorials/config/test-haistep-ohio/1_test_source.yaml
```

**Python API -- the run() method:**

```python
from haipipe.source_base import Source_Pipeline

pipeline = Source_Pipeline(config, SPACE)
source_set = pipeline.run(
    raw_data_name='OhioT1DM',         # Folder name in SourceStore
    raw_data_path=None,                 # None=auto-discover, or S3 URL, or abs path
    payload_input=None,                 # Dict for inference mode (no caching)
    use_cache=True,                     # Load cached SourceSet if available
    save_cache=True                     # Save processed output to disk
)
```

All five parameters of `run()`:

```
Parameter        Type            Default   Purpose
---------------  --------------  --------  ----------------------------------------
raw_data_name    str             required  Cohort directory name in SourceStore
raw_data_path    Optional[str]   None      Override path (None=check store, S3, abs)
payload_input    Optional[Dict]  None      JSON payload for inference mode
use_cache        bool            True      Whether to use cached SourceSet
save_cache       bool            True      Whether to save results to disk
```

---

Batch Mode vs Inference Mode
=============================

**Batch mode** (default): Processes files from a folder.

```python
source_set = pipeline.run(
    raw_data_name='OhioT1DM',
    raw_data_path='_WorkSpace/0-RawDataStore/OhioT1DM',
    use_cache=False,
    save_cache=True
)
```

- Reads raw files from raw_data_path (or auto-discovers from SourceStore)
- Supports caching: cached results stored at
  `_WorkSpace/1-SourceStore/{raw_data_name}/@{SourceFnName}/`
- Output persists to disk when save_cache=True

**Inference mode**: Processes a JSON payload directly.

```python
source_set = pipeline.run(
    raw_data_name='inference_request_001',
    payload_input={'patient_id': '123', 'prescription': {...}}
)
```

- No file I/O -- processes dict in memory
- No caching (real-time processing)
- Used by endpoint inference

---

Available Chefs (SourceFn)
==========================

```
SourceFn Name          Cohort         Input Format   Suffix
---------------------  -------------  -------------  --------
WellDocDataV251226     WellDoc        CSV/Parquet    .csv/.parquet
OhioT1DMxmlv250302     OhioT1DM       XML            .xml
CGMacrosV251227        CGMacros       CSV            .csv
dubossonV251227        dubosson       CSV            .csv
AIREADIv2V251226       aireadi        Parquet        .parquet
```

---

Naming Convention
=================

**Output asset name:** `{raw_data_name}/@{SourceFnName}`

  Example: `OhioT1DM/@OhioT1DMxmlv250302`

**Store path:** `_WorkSpace/1-SourceStore/{raw_data_name}/@{SourceFnName}/`

  Example: `_WorkSpace/1-SourceStore/OhioT1DM/@OhioT1DMxmlv250302/`

The `@` prefix distinguishes processed output from raw data directories.

---

Verification Steps
==================

After running, verify the output:

1. **Check output directory exists:**
   `ls _WorkSpace/1-SourceStore/{raw_data_name}/@{SourceFnName}/`

2. **Verify expected Parquet files** are present (one per ProcName)

3. **Check manifest.json** was created and contains valid metadata

4. **Load and inspect** the SourceSet (see load.md for the full pattern):
   ```python
   source_set = SourceSet.load_from_disk(
       path='_WorkSpace/1-SourceStore/OhioT1DM/@OhioT1DMxmlv250302',
       SPACE=SPACE
   )
   source_set.info()
   ```

---

Prerequisites
=============

Before running Source_Pipeline:

1. Raw data must exist at `_WorkSpace/0-RawDataStore/{raw_data_name}/`
   or at the path specified in `raw_data_path`

2. If raw data is on remote, pull it first:
   `hai-remote-sync --pull --rawdata --path 0-RawDataStore/{CohortName}`

3. The SourceFnName must match an existing file in `code/haifn/fn_source/`

---

MUST DO
=======

1. **Activate .venv and source env.sh** before running
2. **Use only registered SourceFn names** (see Available Chefs table)
3. **Verify raw data exists** before running the pipeline
4. **Follow the config key format** exactly as shown

---

MUST NOT
========

1. **NEVER invent** SourceFn names that do not exist in code/haifn/fn_source/
2. **NEVER skip** required config keys (raw_data_name, SourceFnName)
3. **NEVER create** files in code/haifn/ -- use design-chef if you need a new SourceFn
4. **NEVER run** without activating .venv and sourcing env.sh
