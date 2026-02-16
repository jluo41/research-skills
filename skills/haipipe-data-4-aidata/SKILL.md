Skill: haipipe-data-4-aidata
=============================

Layer 4 of the 6-layer data pipeline: AIData Processing.

Converts a CaseSet into ML-ready datasets (AIDataSet). Handles train/val/test
splitting, input transforms (tabular, text, tensor, token embeddings), and
output transforms (labels, forecasts, next-token targets).

Four subcommands:

  /haipipe-data-4-aidata load           Inspect existing AIDataSet
  /haipipe-data-4-aidata cook           Run AIData_Pipeline with config
  /haipipe-data-4-aidata design-chef    Create new TfmFn/SplitFn via builder
  /haipipe-data-4-aidata design-kitchen Upgrade AIData_Pipeline infra

On invocation, read this file for shared rules, then read the matching
subcommand file (load.md, cook.md, design-chef.md, design-kitchen.md).

---

Architecture Position
=====================

```
Layer 6: Endpoint           Deployment packaging
    |
Layer 5: Model              Model training reads AIDataSet splits
    |
Layer 4: AIData  <---       CaseSet -> ML-ready datasets (train/val/test)
    |
Layer 3: Case               RecordSet -> event-triggered feature extraction
    |
Layer 2: Record             SourceSet -> 5-min aligned patient records
    |
Layer 1: Source             Raw files -> standardized tables
```

---

Cooking Metaphor
================

```
Kitchen  = AIData_Pipeline class     (code/haipipe/aidata_base/)
Chef     = TfmFn + SplitFn          (code/haifn/fn_aidata/)       -- GENERATED
Recipe   = YAML config file          (config/aidata/ or tutorials/config/)
Dish     = AIDataSet asset           (_WorkSpace/4-AIDataStore/)
Academy  = Builder scripts           (code-dev/1-PIPELINE/4-AIData-WorkSpace/)
```

---

What Is an AIDataSet?
=====================

An AIDataSet is the output of Layer 4. It wraps a HuggingFace DatasetDict
with vocabulary files and metadata. The model training layer (Layer 5)
reads this directly.

**Core attributes:**

```python
aidata_set.dataset_dict       # DatasetDict with split keys:
                              #   'train', 'validation', 'test-id', 'test-od'
aidata_set.CF_to_CFVocab      # Per-CaseFn vocabulary (saved as cf_to_cfvocab.json)
aidata_set.feat_vocab          # Feature vocabulary from build_vocab_fn (saved as feat_vocab.json)
aidata_set.meta_info           # MetaArgs configuration
aidata_set.split_info          # Split configuration used
aidata_set.transform_info      # Transform configuration used
```

**On-disk layout:**

```
_WorkSpace/4-AIDataStore/{aidata_name}/@{aidata_version}/
+-- train/                    HuggingFace Dataset (Parquet format)
|   +-- data-00000-of-00001.parquet
+-- validation/
|   +-- data-00000-of-00001.parquet
+-- test-id/                  In-distribution test
|   +-- data-00000-of-00001.parquet
+-- test-od/                  Out-of-distribution test (optional)
|   +-- data-00000-of-00001.parquet
+-- cf_to_cfvocab.json        Per-CaseFn vocabulary
+-- feat_vocab.json           Feature vocabulary from build_vocab_fn
+-- manifest.json             Lineage and metadata
+-- _cache_*/                 Cached domain-format data (optional)
```

**CRITICAL:** Vocab files are **cf_to_cfvocab.json** and **feat_vocab.json**
at the root of the AIDataSet directory. There is NO vocab/ subdirectory.

---

Three-Part Config Pattern
=========================

An AIData config has up to three sections. SplitArgs is optional (omit to
skip splitting). OutputArgs is optional (omit for inference-only pipelines).

**1. SplitArgs** -- How to split cases into train/val/test:

```yaml
SplitArgs:
  SplitMethod: 'SplitByTimeBin'
  ColumnName: 'split_timebin'
  Split_to_Selection:
    train:
      Rules: [["split_ai", "==", "train"]]
      Op: "and"
    validation:
      Rules: [["split_ai", "==", "validation"]]
      Op: "and"
    test-id:
      Rules: [["split_ai", "==", "test-id"]]
      Op: "and"
```

**2. InputArgs** -- How to transform case features into model inputs:

```yaml
InputArgs:
  input_method: "InputTEToken"
  input_casefn_list:
    - 'CGMValueBf24h'
    - 'CGMValueAf24h'
  input_args:
    window_build: { ... }     # Stage 1: Build TEWindow
    tetoken: { ... }          # Stage 2: Convert to model format
```

**3. OutputArgs** -- How to extract labels/targets:

```yaml
OutputArgs:
  output_method: "OutputSingleLabel"
  output_casefn_list: []
  output_args:
    label_column: "clicked"
    label_rule: ["clicked", ">", 0]
```

---

Two-Stage TEToken Example
=========================

The TEToken transform (most common for time series) uses a two-stage
pipeline. This is a general temporal encoding -- applicable to any time
series domain, not just CGM.

```
Stage 1: window_build
  CaseFn fields -> TEWindow
  +-- time_series: concat CGMValueBf24h + CGMValueAf24h -> [576]
  +-- static_feat: PDemoBase -> key-value pairs
  +-- event_list: DEMEventBf24h + DEMEventAf24h -> event list

Stage 2: tetoken
  TEWindow -> 4 output types:
  +-- singlevalue_timestep:      [batch, 576, N_fields]  continuous
  +-- singletoken_timestep:      [batch, 576, N_tokens]  discrete
  +-- multitoken_sparsetimestep: event dicts with positions
  +-- multitoken_global:         static text or token IDs
```

---

Concrete Code From the Repo
============================

**Input TfmFn -- build_vocab_fn** (2 parameters):

```python
# code/haifn/fn_aidata/entryinput/InputTEToken.py

def build_vocab_fn(InputArgs, CF_to_CFVocab):
    """Build vocabulary from config and per-CaseFn vocabularies.

    Args:
        InputArgs:      Transform configuration dict
        CF_to_CFVocab:  Per-CaseFn vocabulary dict

    Returns:
        feat_vocab dict (saved as feat_vocab.json)
    """
    input_args = InputArgs.get('input_args', {})
    window_build = input_args.get('window_build', {})
    tetoken_config = input_args.get('tetoken', {})
    tetoken_vocabs = build_vocab(tetoken_config, CF_to_CFVocab=CF_to_CFVocab)
    feat_vocab = {'window_build': window_build, **tetoken_vocabs}
    return feat_vocab
```

**Input TfmFn -- tfm_fn** (4 parameters):

```python
# code/haifn/fn_aidata/entryinput/InputTEToken.py

def tfm_fn(case_features, InputArgs, CF_to_CFvocab, feat_vocab=None):
    """Transform one case's features into model-ready input.

    Args:
        case_features:  Dict with CaseFn data for one case
        InputArgs:      Transform configuration
        CF_to_CFvocab:  Per-CaseFn vocabulary
        feat_vocab:     Built vocabulary from build_vocab_fn (optional)

    Returns:
        dict with model-ready input features
    """
    ...
```

**Output TfmFn -- tfm_fn** (2 parameters):

```python
# code/haifn/fn_aidata/entryoutput/OutputSingleLabel.py

def tfm_fn(case, OutputArgs):
    """Extract single label from case.

    Args:
        case:        Single case dict with all features
        OutputArgs:  Config with output_args

    Returns:
        dict with output features (e.g., {'label': np.int64(0 or 1)})
    """
    output_args = OutputArgs.get('output_args', {})
    label_column = output_args['label_column']
    value = case[label_column]
    return {'label': np.int64(int(value))}
```

**SplitFn -- dataset_split_tagging_fn** (adds split_ai column):

```python
# code/haifn/fn_aidata/split/SplitByTimeBin.py

def dataset_split_tagging_fn(df_tag, SplitArgs):
    """Add 'split_ai' column to df_tag.

    Args:
        df_tag:     DataFrame with metadata columns
        SplitArgs:  Split configuration dict

    Returns:
        df_tag with 'split_ai' column added.
        Values: 'train', 'validation', 'test-id', 'test-od', or None
    """
    column_name = SplitArgs.get('ColumnName', 'split_timebin')
    # ... maps source values to target splits ...
    df_tag['split_ai'] = mapped_values
    return df_tag
```

**AIData_Pipeline -- initialization and run:**

```python
# code/haipipe/aidata_base/aidata_pipeline.py

pipeline = AIData_Pipeline(config, SPACE, cache_combined_case=True)

# Three processing paths:
# Path 1: CaseSet(s) or cached dataset -> AIDataSet
aidata_set = pipeline.run(
    case_set=my_case_set,
    aidata_name='ohiot1dm-cgm-tedata',
    aidata_version='v0'
)

# Path 2: DataFrame -> transformed DataFrame
df_transformed = pipeline.run(df_case=test_df, mode='inference')

# Path 3: Single case dict -> transformed dict
result = pipeline.run(case_example=single_case, mode='inference')
```

**Load an existing AIDataSet:**

```python
# code/haipipe/aidata_base/aidata_set.py

from haipipe.aidata_base.aidata_set import AIDataSet

aidata_set = AIDataSet.load_from_disk(path='/full/path/to/aidata', SPACE=SPACE)
# or
aidata_set = AIDataSet.load_asset(path='/full/path/to/aidata', SPACE=SPACE)
```

---

Registered Fns
==============

**Input Transforms (entryinput/):**

```
TfmFn Name                 Description                     Domain
---------------------------+-------------------------------+-----------------------
InputTEToken               Token embedding (general)       Any time series
InputMultiCGMSeqNumeric    Multi-sequence numeric          Numeric sequences
InputMultiCGMSeqConcat     Multi-sequence concatenated     Numeric sequences
InputMultiCF               Multi case-feature tabular      Tabular features
InputCGMEventText          Time series + events as text    Text models
InputCGMEventJSON          Time series + events as JSON    JSON-based models
InputCGMWithEventChannels  Time series with event channels Multi-channel sequences
InputMultiSeqConcat        Multi sequences concatenated    Numeric sequences
InputTemporalInterleaving  Temporally interleaved          Interleaved sequences
CatInputMultiCFSparse      Sparse categorical features     Sparse tensors
```

**Output Transforms (entryoutput/):**

```
TfmFn Name              Description                     Output Format
------------------------+-------------------------------+-----------------------
OutputNextToken         Next-token prediction labels     token ID sequence
OutputNumericForecast   Numeric forecasting labels       float array
OutputSingleLabel       Single classification label      int
```

**Split Functions (split/):**

```
SplitFn Name          Description                       Config Key
----------------------+----------------------------------+-----------------------
SplitByTimeBin        Temporal split by time bins        split_timebin
RandomByPatient       Random split respecting patients   PatientID
RandomByStratum       Stratified random split            stratum column
```

---

Generality Note
===============

These transforms and splits are domain-agnostic:

- Input transforms work for **any feature type**, not just CGM
- TEToken is a **general temporal encoding**, applicable to any time series
- Split strategies require **no domain knowledge**
- Output transforms handle **classification** (SingleLabel), **regression**
  (NumericForecast), and **generation** (NextToken)
- Everything is config-driven: change input_casefn_list and input_args
  for a different domain

---

Prerequisites
=============

Before working with AIData:

1. **Activate .venv**: `source .venv/bin/activate`
2. **Load environment**: `source env.sh`
3. **Input CaseSet must exist** at the path specified in case_set_name
4. **All CaseFn names** in input_casefn_list must be present in the CaseSet
5. **Layer 3 (Case) must be complete** before running Layer 4

---

MUST DO (All Subcommands)
=========================

1. **Activate** .venv first: `source .venv/bin/activate`
2. **Load** environment: `source env.sh`
3. **Remember** that output of Layer 4 = input of Model training (Layer 5)
4. **Ensure** input_casefn_list references CaseFn names that exist in the CaseSet
5. **Ensure** InputArgs.input_method matches a registered TfmFn name
6. **Present** plan to user and get approval before any code changes
7. **Use** correct API signatures (build_vocab_fn takes 2 params, not case_set)

---

MUST NOT (All Subcommands)
==========================

1. **NEVER edit** code/haifn/ directly (100% generated from builders)
2. **NEVER run** Python without .venv activated
3. **NEVER skip** `source env.sh` (store paths come from environment variables)
4. **NEVER invent** CaseFn names in input_casefn_list (must match actual CaseSet)
5. **NEVER assume** a vocab/ subdirectory exists (files are cf_to_cfvocab.json and feat_vocab.json at root)
6. **NEVER mix up** input tfm_fn (4 params) with output tfm_fn (2 params)
7. **NEVER assume** SplitFn returns a dict of split DataFrames (it adds a split_ai column to df_tag)

---

Key File Locations
==================

```
Pipeline framework:   code/haipipe/aidata_base/aidata_pipeline.py
                      code/haipipe/aidata_base/aidata_set.py
                      code/haipipe/aidata_base/aidata_utils.py
Fn loaders:           code/haipipe/aidata_base/builder/tfmfn.py
                      code/haipipe/aidata_base/builder/splitfn.py

Generated Input Tfms: code/haifn/fn_aidata/entryinput/InputTEToken.py
                      code/haifn/fn_aidata/entryinput/InputMultiCGMSeqNumeric.py
                      code/haifn/fn_aidata/entryinput/InputMultiCF.py
                      ... (10 total)
Generated Output Tfms: code/haifn/fn_aidata/entryoutput/OutputNextToken.py
                       code/haifn/fn_aidata/entryoutput/OutputNumericForecast.py
                       code/haifn/fn_aidata/entryoutput/OutputSingleLabel.py
Generated SplitFns:   code/haifn/fn_aidata/split/SplitByTimeBin.py
                      code/haifn/fn_aidata/split/RandomByPatient.py
                      code/haifn/fn_aidata/split/RandomByStratum.py

Builders (edit these): code-dev/1-PIPELINE/4-AIData-WorkSpace/c1_build_transforms_cgmntp.py
                       code-dev/1-PIPELINE/4-AIData-WorkSpace/c7_build_transforms_tetoken.py
                       code-dev/1-PIPELINE/4-AIData-WorkSpace/s1_build_splitfn_splitbytimebin.py

Test configs:         tutorials/config/test-haistep-ohio/4_test_aidata-cgm-tedata.yaml
Store path:           _WorkSpace/4-AIDataStore/
```

---

File Layout
===========

```
haipipe-data-4-aidata/
+-- SKILL.md              This file (router + shared rules)
+-- README.md             Quick reference
+-- load.md               /haipipe-data-4-aidata load
+-- cook.md               /haipipe-data-4-aidata cook
+-- design-chef.md        /haipipe-data-4-aidata design-chef
+-- design-kitchen.md     /haipipe-data-4-aidata design-kitchen
+-- templates/
    +-- config.yaml       Config template for cook subcommand
```

---

See Also
========

- **haipipe-data-1-source**: How raw data becomes SourceSets
- **haipipe-data-2-record**: How SourceSets become RecordSets
- **haipipe-data-3-case**: How RecordSets become CaseSets (input to this layer)
- **haipipe-nn-0-overview**: How AIDataSets feed into model training (Layer 5)
