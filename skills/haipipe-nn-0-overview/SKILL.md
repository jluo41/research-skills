Skill: haipipe-nn-0-overview
============================

Architecture map, decision guide, and model registry for the NN pipeline.
Read this FIRST before any other haipipe-nn skill.

**Scope of this skill:** Framework patterns only. It does not catalog
project-specific state (which datasets are available, which experiments have
been run). Model registry snapshots are labeled as such -- always discover
current state at runtime. This skill applies equally to any domain or model
family.

---

The 4-Layer Architecture
========================

Every model in this pipeline follows the same 4-layer separation:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: ModelSet / Pipeline                                   │
│  Packages everything into an Asset. Runs the training pipeline. │
│  Config-driven. Model-type agnostic.                            │
│  Files: code/haipipe/model_base/                                │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Instance                                              │
│  Thin orchestrator. Manages one or more Tuners.                 │
│  HuggingFace-style save/load. Config class. Registry.           │
│  Files: code/hainn/<model_family>/instance_*.py                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Tuner (Train / Optuna)                                │
│  Wraps ONE algorithm. Handles data conversion, training,        │
│  inference, serialization. The ONLY layer that imports           │
│  external libraries.                                            │
│  Files: code/hainn/<model_family>/models/modeling_*.py          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Algorithm                                             │
│  Raw external library. XGBoost, PyTorch, HuggingFace, etc.      │
│  We don't write this code -- we wrap it in a Tuner.             │
│  Installed via pip/conda.                                       │
└─────────────────────────────────────────────────────────────────┘

> JL: For Algorithm, we can also develop our own algorithm_xxxxx.py, the key is that we need to have a forward(x) to y, and that's the most essential part. 
```

**Layer numbering note:** Skills use bottom-up numbering (1=Algorithm,
2=Tuner, 3=Instance, 4=ModelSet). Source code docstrings use INVERTED
numbering: model_instance.py calls Instance "Layer 2" and model_tuner.py
calls Tuner "Layer 3". Both describe the same architecture stack -- the
numbering direction is simply reversed. Skills are the canonical reference.

---

Decision Tree: Integrating a New Algorithm
==========================================

```
START: I have an algorithm (e.g., diffusion model, LLM, tree model)
│
├─ Q1: What data format does the algorithm need?
│  ├─ Nixtla DataFrame (unique_id, ds, y)  →  domain_format = "nixtla"
│  ├─ Sparse matrix / flat features         →  domain_format = "sparse"
│  ├─ PyTorch tensors                       →  domain_format = "tensor"
│  ├─ HuggingFace Dataset (input_ids, ...)  →  domain_format = "hf_clm"
│  ├─ API call (JSON in, JSON out)          →  domain_format = "llmapi"
│  └─ Something else                        →  domain_format = "custom_name"
│
├─ Q2: How many Tuners does your model need?
│  ├─ ONE algorithm, one model
│  │  → Single Tuner pattern: model_base = {'MAIN': tuner}
│  ├─ ONE algorithm, but treatment/context encoded as input feature
│  │  → Single Tuner + Encoding pattern: model_base = {'MAIN': tuner}
│  │    (Instance handles the encoding before passing to Tuner)
│  └─ MULTIPLE algorithms (one per arm/variant) 
│     → Multi-Tuner pattern: model_base = {'arm1': tuner, 'arm2': tuner}
| > JL: how do deal with the bandit model here? we need the discussion again here. 
│
├─ Q3: Does your algorithm need Optuna hyperparameter tuning?
│  ├─ Yes → Implement objective() in Tuner.fit(), store best_params
│  └─ No  → Simple fit() in Tuner
│
└─ Q4: What serialization format?
   ├─ JSON (XGBoost, LightGBM)     →  model_{key}.json
   ├─ Pickle (sklearn)              →  model_{key}.pkl
   ├─ PyTorch state_dict            →  model_{key}/ directory
   ├─ HuggingFace save_pretrained   →  model_{key}/ directory
   └─ Custom                        →  your choice in save_model()
```

---

Adding a New Model: Step-by-Step Checklist
==========================================

```
Step 1: Algorithm (Layer 1)
   [ ] Identify the external library (e.g., xgboost, diffusers, transformers)
   [ ] Determine domain_format and serialization format
   [ ] This library is NEVER imported above the Tuner layer

Step 2: Tuner (Layer 2)
   [ ] Create file: code/hainn/<family>/models/modeling_<name>.py
   [ ] Define standalone transform_fn() at TOP of file
   [ ] Create class inheriting from ModelTuner
   [ ] Set domain_format class attribute
   [ ] Implement 5 abstract methods:
       - get_tfm_data(dataset)
       - fit(dataset, TrainingArgs)
       - infer(dataset, InferenceArgs)
       - save_model(key, model_dir)
       - load_model(key, model_dir)
   [ ] Import algorithm library ONLY in this file

Step 3: Instance (Layer 3)
   [ ] Create file: code/hainn/<family>/instance_<name>.py
   [ ] Create class inheriting from ModelInstance
   [ ] Set MODEL_TYPE class attribute (string for metadata.json)
   [ ] Create MODEL_TUNER_REGISTRY dict mapping name → module path
   [ ] Implement 5 abstract methods:
       - init()                  → create Tuner from registry
       - fit(Name_to_Data)       → delegate to Tuner(s)
       - infer(dataset)          → delegate to Tuner(s)
       - _save_model_base(dir)   → call tuner.save_model() for each
       - _load_model_base(dir)   → init() then tuner.load_model() for each
   [ ] Create @dataclass Config: code/hainn/<family>/configuration_<name>.py
       - Inherit from ModelInstanceConfig
       - Include: ModelArgs, TrainingArgs, InferenceArgs, EvaluationArgs (all Dict)
       - Implement from_aidata_set() factory classmethod
       - Optionally implement from_yaml() factory classmethod
   [ ] Register in code/hainn/model_registry.py:
       elif model_instance_type in ('MyModel', 'MyModelInstance'):
           from hainn.<family>.instance_<name> import MyModelInstance
           from hainn.<family>.configuration_<name> import MyModelConfig
           return MyModelInstance, MyModelConfig

Step 4: ModelSet / Pipeline (Layer 4)
   [ ] Write YAML config (see templates below)
   [ ] Run through ModelInstance_Pipeline -- no code changes needed at this layer
   [ ] Pipeline handles: config creation, init, fit, evaluate, PreFn, examples, packaging
```

> JL: [ ] Write YAML config (see templates below) This is not totally right because we also need the YAML for every four layer.At the algorithm part, which will need this as well.

---

Complete Model Registry
=======================

Snapshot (discover current registry at runtime):

```bash
cat code/hainn/model_registry.py
ls code/hainn/
```

Snapshot (as of 2026-02-21 -- always verify with commands above):
(file: code/hainn/model_registry.py)

```
Type String(s)                          Instance Class                  Config Class              Family
───────────────────────────────────     ─────────────────────────────   ────────────────────────  ──────────
'BanditV1'                              BanditInstance                  None (no config class)    bandit
'MLSLearnerPredictor'                   MLSLearnerPredictorInstance     MLSLearnerConfig          mlpredictor
  alias: 'MLSLearnerPredictorInstance'
'MLTLearnerPredictor'                   MLTLearnerPredictorInstance     MLTLearnerConfig          mlpredictor
  alias: 'MLTLearnerPredictorInstance'
'TSDecoderInstance'                     TSDecoderInstance               TSDecoderConfig           tsfm
  alias: 'TSDecoder'              *** NON-FUNCTIONAL: import targets do not exist on disk ***
'TEFMInstance'                          TEFMInstance                    TEFMConfig                tefm
  alias: 'TEFM'
'TEFMForecastInstance'                  TEFMForecastInstance            TEFMForecastConfig        tsforecast
  alias: 'TEFMForecast'           *** NON-FUNCTIONAL: import targets do not exist on disk ***
'TSForecastInstance'                    TSForecastInstance              TSForecastConfig          tsforecast
  aliases: 'DLForecastInstance',
           'TSForecast'
```

**When writing YAML configs:** Use any of the type strings listed above as
the value for ModelInstanceClass. The registry resolves aliases automatically.

---

YAML Config Templates
=====================

The YAML config drives ModelInstance_Pipeline. The required top-level keys:

```yaml
# Required
ModelInstanceClass: "<type string from registry>"
ModelArgs:
  model_tuner_name: "<tuner class name>"
  # ... tuner-specific args
TrainingArgs:
  # ... training hyperparameters

# Optional
EvaluationArgs:
  # ... evaluation settings
InferenceArgs:
  # ... inference settings
ExampleConfig:
  enabled: false
```

**modelinstance_set_name format WARNING:** Two different naming formats exist:
- ModelInstance_Pipeline uses: `f"{name}/{version}"` (e.g., "MyModel/v0001")
- ModelInstanceConfig.from_aidata_set uses: `f"{name}/@{version}"` (e.g., "MyModel/@v0001")
This is a known code inconsistency. The pipeline format is what gets used in practice.

**Template A: Time-Series Forecasting (TSForecast)**

```yaml
ModelInstanceClass: "TSForecast"       # or "DLForecastInstance"
ModelArgs:
  model_tuner_name: "NixtlaPatchTSTTuner"
  model_tuner_args:
    h: 12
    input_size: 288
  TfmArgs:
    target_col: "cgm_value"
    time_col: "date"
    id_col: "PatientID"
TrainingArgs:
  max_steps: 100
  learning_rate: 0.001
  early_stopping_patience: 10
  TrainSetNames: ["train"]
  ValSetNames: ["validation"]
InferenceArgs:
  InferSetNames: ["test"]
EvaluationArgs:
  metrics: ["mae", "rmse", "mape"]
```

**Template B: Treatment Effect (MLPredictor)**

```yaml
ModelInstanceClass: "MLSLearnerPredictor"
ModelArgs:
  model_tuner_name: "XGBoostTuner"
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
TrainingArgs:
  TrainSetNames: ["train"]
  OptimizeSetNames: ["validation"]
  optuna_n_trials: 50
EvaluationArgs:
  EvalSetNames: ["test"]
  metrics: ["auc", "accuracy"]
```

**Template C: Foundation Model (TEFM with Tuner)**

```yaml
ModelInstanceClass: "TEFM"
ModelArgs:
  model_tuner_name: "HFNTPTuner"
  model_tuner_args:
    model_name_or_path: "gpt2"
    max_seq_length: 576
  TfmArgs:
    tetoken_config:
      feature_channels: ["cgm", "temporal"]
TrainingArgs:
  num_train_epochs: 10
  per_device_train_batch_size: 32
  learning_rate: 0.0004
  weight_decay: 0.1
InferenceArgs:
  per_device_eval_batch_size: 64
```

**Template D: TEFM Direct Architecture (no Tuner)**

```yaml
ModelInstanceClass: "TEFM"
ModelArgs:
  model_type: "early_fusion"           # No model_tuner_name → direct mode
  d_model: 256
  n_heads: 8
  n_layers: 6
  task_config:                          # task_type lives inside task_config dict
    task_type: "forecasting"
    forecast_horizon: 12
TrainingArgs:
  learning_rate: 0.0001
  batch_size: 128
  num_epochs: 50
  early_stopping_patience: 10
```

---

Existing Model Families: Summary
=================================

Snapshot (discover current families at runtime):

```bash
ls code/hainn/
```

Snapshot (as of 2026-02-21 -- verify with ls above):

```
┌──────────────┬───────────────────────────────────────────────────────────┐
│ Family       │ Description                                             │
├──────────────┼───────────────────────────────────────────────────────────┤
│ tsforecast   │ Time-series forecasting. 15+ Tuners via dict registry.  │
│              │ Clean 4-layer separation. THE REFERENCE IMPLEMENTATION.  │
│              │ Models: PatchTST, NBEATS, NHITS, XGBoost, ARIMA, LLM... │
│              │ Files: code/hainn/tsforecast/                            │
├──────────────┼───────────────────────────────────────────────────────────┤
│ tefm         │ Time-Event Foundation Model. Multi-modal (timeseries +  │
│              │ events + static). Dual-mode: Tuner-based (HFNTPTuner)   │
│              │ or direct architecture (early_fusion, clip, diffusion).  │
│              │ Direct mode has torch imports in Instance (deviation).   │
│              │ Files: code/hainn/tefm/                                  │
├──────────────┼───────────────────────────────────────────────────────────┤
│ mlpredictor  │ Treatment effect estimation. S-Learner (single model,   │
│              │ treatment as feature) and T-Learner (separate models).   │
│              │ Non-canonical pattern: doesn't follow base class          │
│              │ conventions but still actively used in production.        │
│              │ Files: code/hainn/mlpredictor/                           │
├──────────────┼───────────────────────────────────────────────────────────┤
│ bandit       │ Multi-arm bandit. Thompson sampling. No config class.    │
│              │ Different __init__ signature (no config object).         │
│              │ Files: code/hainn/bandit/                                │
├──────────────┼───────────────────────────────────────────────────────────┤
│ tsfm         │ Time-Series Foundation Model (decoder). TSDecoder.      │
│              │ Files: code/hainn/tsfm/tsdecoder/                       │
└──────────────┴───────────────────────────────────────────────────────────┘
```

**Compliance with canonical pattern:**

The canonical interface requires: inherit base class, call super().__init__(),
use model_base dict, implement all 5 abstract methods (init, fit, infer,
_save_model_base, _load_model_base), use model_tuner_name in config.

```
                 Canonical?   Notes
tsforecast       YES          The reference implementation. Follow this.
tefm (tuner)     partial      Stubs for _save/_load_model_base. Has inference() not infer().
tefm (direct)    NO           Torch imports in Instance. Training loops in Instance.
mlpredictor      NO           No super().__init__(), uses model_tuner_base not model_base,
                              inference() not infer(), non-standard save/load method names.
bandit           NO           No super().__init__(), no config class, different __init__ signature.
tsfm/tsdecoder   N/A          Registry entry exists but import targets missing.
```

**For new models:** Follow the tsforecast pattern. Existing non-canonical code
should be migrated toward the standard over time.

---

Key File Locations
==================

```
Base classes:
  ModelTuner:           code/hainn/model_tuner.py
  ModelInstance:        code/hainn/model_instance.py
  ModelInstanceConfig:  code/hainn/model_configuration.py
  Model Registry:       code/hainn/model_registry.py

Pipeline (Layer 4):
  ModelInstance_Set:      code/haipipe/model_base/modelinstance_set.py
  ModelInstance_Pipeline: code/haipipe/model_base/modelinstance_pipeline.py
  PreFnPipeline:         code/hainn/prefn_pipeline.py
  Asset base class:      code/haipipe/assets.py

AutoModelInstance:
  code/hainn/model_instance.py (AutoModelInstance class)

Existing families:
  tsforecast:    code/hainn/tsforecast/
  tefm:          code/hainn/tefm/
  mlpredictor:   code/hainn/mlpredictor/
  bandit:        code/hainn/bandit/
  tsfm:          code/hainn/tsfm/
```

---

Test Notebook Conventions
=========================

Every model family has a `test-modeling-<name>/` directory with 4 test scripts
(one per layer) that double as reviewable notebooks:

```
test-modeling-te_clm/
├── config_te_clm_from_scratch.yaml
├── config_te_clm_pretrained.yaml
├── scripts/
│   ├── test_te_clm_1_algorithm.py   # Layer 1
│   ├── test_te_clm_2_tuner.py       # Layer 2
│   ├── test_te_clm_3_instance.py    # Layer 3
│   └── test_te_clm_4_modelset.py    # Layer 4
└── notebooks/                        # Auto-generated from scripts
    ├── test_te_clm_1_algorithm.ipynb
    ├── test_te_clm_2_tuner.ipynb
    ├── test_te_clm_3_instance.ipynb
    └── test_te_clm_4_modelset.ipynb
```

**Core display principle:** At every step, the reviewer should see the actual
data -- what goes IN and what comes OUT. Not just computed summaries.

Two display mechanisms used together:

1. **`print()`** -- Show actual data objects (datasets, samples, dicts)
2. **`display_df()`** -- Show key-value summary tables (config, metrics, status)

**Variable naming convention** (`type_qualifier`):

```
Single HF Dataset:
  ds_raw         raw dataset before transform_fn
  ds_tfm         dataset after transform_fn (has input_ids, attention_mask, labels)

Dict of datasets:
  data_tfm       output of get_tfm_data() -- {split_name: transformed_dataset}
                 (L2 tuner test only -- tests get_tfm_data directly)
  data_fit       raw datasets for fit -- {split_name: raw_dataset}
                 (passed to tuner.fit / instance.fit, which transforms internally)
  data_infer     raw datasets for inference -- {split_name: raw_dataset}
                 (L3/L4; passed to instance.inference)

Inference (L2 tuner, testing multiple input types):
  ds_infer_a     single raw dataset used to build data_infer_a
  data_infer_a   dict input for Type A: {split_name: dataset}
  ds_infer_b     single raw dataset for Type B (passed directly to tuner.infer)
```

**Before/after pattern** (the key convention):

```python
# BEFORE: show what goes into the operation
print("--- BEFORE transform_fn ---")
print(ds_raw)                  # the dataset object
print()
print("Sample [0]:")
print(ds_raw[0])               # one concrete sample

# Run the operation
ds_tfm = transform_fn(ds_raw, TfmArgs)

# AFTER: show what comes out
print("--- AFTER transform_fn ---")
print(ds_tfm)                  # the transformed dataset
print()
print("Sample [0]:")
print(ds_tfm[0])               # one concrete transformed sample
```

This pattern applies to every data operation: transform_fn, get_tfm_data,
fit, infer, save/load. Always print the input before calling, print the
output after.

**Step structure** (common to all 4 layers):

```python
# %% [markdown]
# ## Step N: Title
#
# Brief description of what this step does.

# %% Step N: Title

# ... code with print() and display_df() ...

display_df(pd.DataFrame([
    {'property': 'key1',   'value': val1},
    {'property': 'key2',   'value': val2},
    {'property': 'status', 'value': 'PASSED'},
]).set_index('property'))
```

**Summary table** (last cell in every notebook):

```python
results = pd.DataFrame([
    {'test': '1. Load Config',   'status': 'PASSED'},
    {'test': '2. Load AIData',   'status': 'PASSED'},
    ...
]).set_index('test')
display_df(results)
```

**Notebook regeneration** (after any script change):

```bash
python code/scripts/convert_to_notebooks.py \
    --dir <model>/test-modeling-<name>/scripts/ \
    -o <model>/test-modeling-<name>/notebooks/
```

**Reference implementation** (discover at runtime):

```bash
ls code/hainn/                                    # available model families
ls code/hainn/<family>/                           # family contents
ls code/hainn/<family>/test-modeling-<name>/      # test notebook dir
ls code/hainn/<family>/test-modeling-<name>/scripts/  # test scripts
```

# Example (illustrative -- actual paths from ls above):
# code/hainn/tefm/models/te_clm/test-modeling-te_clm/

---

See Also
========

- **haipipe-nn-1-algorithm**: Algorithm diversity and the "never import above Tuner" rule
- **haipipe-nn-2-tuner**: The universal Tuner contract (5 abstract methods)
- **haipipe-nn-3-instance**: The orchestrator contract (5 abstract methods, config, registry)
- **haipipe-nn-4-modelset**: Packaging, YAML config, pipeline flow, versioning
