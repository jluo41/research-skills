Skill: haipipe-nn-2-tuner
=========================

Layer 2 of the 4-layer NN pipeline: Model Tuner.

The universal wrapper contract. Every algorithm gets wrapped in a Tuner
that provides a uniform interface. This is the ONLY layer that imports
external algorithm libraries.

---

Architecture Position
=====================

```
Layer 4: ModelSet/Pipeline    Packages everything. Config-driven.
    |
Layer 3: Instance             Thin orchestrator. Manages Tuner dict.
    |
Layer 2: Tuner    <---        Wraps ONE algorithm. Owns data conversion,
    |                         training, inference, serialization.
    |                         ONLY layer that imports external libraries.
Layer 1: Algorithm            Raw external library.
```

---

The Contract
============

**File:** code/hainn/model_tuner.py

Every Tuner MUST provide these 5 methods:

```python
class ModelTuner(ABC):
    domain_format = "base"              # Override: "nixtla", "sparse", "tensor", etc.

    def __init__(self, TfmArgs=None, **kwargs):
        self.TfmArgs = TfmArgs or {}
        self.model = None               # Raw algorithm instance (set after fit)

        # AUTO-STORES extra kwargs as instance attributes:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        # e.g., MyTuner(TfmArgs=t, h=12, input_size=288)
        #   -> self.h = 12, self.input_size = 288

    @abstractmethod
    def get_tfm_data(self, dataset): ...   # Convert input -> domain format
    @abstractmethod
    def fit(self, dataset, TrainingArgs=None): ...
    @abstractmethod
    def infer(self, dataset, InferenceArgs=None): ...
    @abstractmethod
    def save_model(self, key: str, model_dir: str): ...
    @abstractmethod
    def load_model(self, key: str, model_dir: str): ...
```

**5 abstract methods, 1 class attribute, 2 instance attributes + auto-stored kwargs.**

Rules from the base class (model_tuner.py lines 18-32):
1. NO PRIVATE METHODS (self._xxx) in the **base class itself**
   (subclasses MAY use private helper methods, e.g., HFNTPTuner uses _ensure_model_loaded())
2. DATA CONVERSION via standalone transform_fn (not a method)
3. CACHING is handled by get_tfm_data() -- only AIData_Set gets cached

---

File Structure Template
=======================

Every Tuner file follows this layout:

```python
# modeling_my_algorithm.py

# ─── Layer 1 import (ONLY HERE) ───
import my_algorithm_lib

from hainn.model_tuner import ModelTuner

# ─── 1. Standalone transform_fn at TOP ───
def transform_fn(data, TfmArgs):
    """Convert HF Dataset / DataFrame / dict -> algorithm-native format."""
    # Your conversion logic
    return converted_data

# ─── 2. Optuna defaults (optional) ───
DEFAULT_OPTUNA_SPACE = {
    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
    ...
}

# ─── 3. Tuner class ───
class MyAlgorithmTuner(ModelTuner):
    domain_format = "my_domain"

    def __init__(self, TfmArgs=None, **model_specific_args):
        super().__init__(TfmArgs=TfmArgs, **model_specific_args)
        # NOTE: super().__init__ auto-stores all kwargs as attributes,
        # so self.h, self.input_size, etc. are already set.
        # You can still override or set defaults here if needed:
        self.h = model_specific_args.get('h', 24)
        ...

    def get_tfm_data(self, dataset):  # 5-case dispatch
    def fit(self, dataset, TrainingArgs=None):
    def infer(self, dataset, InferenceArgs=None):
    def save_model(self, key, model_dir):
    def load_model(self, key, model_dir):
```

---

The Standalone transform_fn Pattern
=====================================

**Why standalone?** Instance can call get_transform_fn(tuner_name) to
access it without instantiating the Tuner. It must be importable as a
top-level function.

**Input:** One of:
- HF Dataset (has .to_pandas())
- pd.DataFrame
- case_feature_dict (single case as dict)

**Output:** Algorithm-native format (Nixtla DataFrame, sparse matrix, tensor, etc.)

```python
# At TOP of tuner file, BEFORE the class:
def transform_fn(data, TfmArgs):
    target_col = TfmArgs.get('target_col', 'value')
    time_col = TfmArgs.get('time_col', 'ds')

    # Convert to pandas if HF Dataset
    if hasattr(data, 'to_pandas'):
        df = data.to_pandas()
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame([data])  # single case dict

    # Convert to domain format
    ...
    return domain_format_data
```

---

The 5-Case Data Dispatch (get_tfm_data)
========================================

get_tfm_data() is @abstractmethod -- you MUST implement it. The 5-case
dispatch below is the recommended structure (shown in base class docstring),
but the exact logic is not enforced -- each Tuner implements its own version.

```python
def get_tfm_data(self, dataset):
    # Case 1: AIData_Set (has .dataset_dict and .data_path)
    #   -> WITH caching (uses data_path for cache directory)
    if hasattr(dataset, 'dataset_dict') and hasattr(dataset, 'data_path'):
        cache_dir = os.path.join(dataset.data_path, f'_cache_{self.domain_format}')
        result = {}
        for split_name, hf_dataset in dataset.dataset_dict.items():
            cache_file = f'{self.domain_format}_{split_name}_{hash}.parquet'
            if os.path.exists(cache_file):
                result[split_name] = pd.read_parquet(cache_file)
            else:
                result[split_name] = transform_fn(hf_dataset, self.TfmArgs)
                result[split_name].to_parquet(cache_file)
        return result

    # Case 2: Dict[str, Dataset/DataFrame] -> transform each, no caching
    if isinstance(dataset, dict):
        return {split: transform_fn(ds, self.TfmArgs) for split, ds in dataset.items()}

    # Case 3: Single HF Dataset -> direct transform
    if hasattr(dataset, 'to_pandas'):
        return transform_fn(dataset, self.TfmArgs)

    # Case 4: pd.DataFrame -> direct transform
    if isinstance(dataset, pd.DataFrame):
        return transform_fn(dataset, self.TfmArgs)

    # Case 5: case_feature_dict (single case) -> direct transform
    return transform_fn(dataset, self.TfmArgs)
```

---

The Optuna Pattern
==================

Tuners optionally integrate Optuna for hyperparameter tuning:

```python
def fit(self, dataset, TrainingArgs=None):
    data = self.get_tfm_data(dataset)
    df_train, df_valid = ...  # extract train/valid

    n_trials = TrainingArgs.get('OptunaArgs', {}).get('n_trials', 0)

    if n_trials > 0:
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self._objective(trial, df_train, df_valid),
            n_trials=n_trials
        )
        self.best_params = study.best_params
    else:
        self.best_params = {}  # use defaults

    # Train final model with best params
    self.model = MyAlgorithm(**{**default_params, **self.best_params})
    self.model.fit(df_train)
```

---

The save_model / load_model Pattern
=====================================

**Naming convention:** `model_{key}` prefix inside model_dir. The exact
format varies by algorithm:

```
Convention A: Directory (PyTorch, NeuralForecast, HuggingFace)
  model_dir/model_MAIN/          # Directory with weights + config
    best_params.json
    tfm_args.json
    model_weights/

Convention B: Flat file (XGBoost, LightGBM)
  model_dir/MAIN_model.json      # XGBoost: {key}_model.json
  model_dir/MAIN_model.txt       # LightGBM: {key}_model.txt

Convention C: Pickle (sklearn)
  model_dir/MAIN_model.pkl       # Pickled model: {key}_model.pkl
```

**Example (directory convention -- most common for new models):**

```python
def save_model(self, key: str, model_dir: str):
    save_path = os.path.join(model_dir, f'model_{key}')  # e.g., model_MAIN/
    os.makedirs(save_path, exist_ok=True)

    # Save tuner metadata alongside weights:
    json.dump(self.best_params, open(f'{save_path}/best_params.json', 'w'))
    json.dump(self.TfmArgs, open(f'{save_path}/tfm_args.json', 'w'))

    # Algorithm-specific save (varies):
    self.model.save(f'{save_path}/model_weights')

def load_model(self, key: str, model_dir: str):
    load_path = os.path.join(model_dir, f'model_{key}')

    self.best_params = json.load(open(f'{load_path}/best_params.json'))
    self.TfmArgs = json.load(open(f'{load_path}/tfm_args.json'))
    self.model = MyAlgorithm.load(f'{load_path}/model_weights')
```

---

Known Deviations (existing code)
================================

- **MLPredictor Tuners**: Do NOT inherit from ModelTuner, different method
  signatures, no get_tfm_data(). Non-canonical.
- **HFNTPTuner**: Inherits ModelTuner but does not pass **kwargs to
  super().__init__(), only passes TfmArgs.

These are non-canonical. New Tuners MUST follow the standard contract.

---

MUST DO
=======

1. **Inherit from ModelTuner**
2. **Set domain_format class attribute** -- "nixtla", "sparse", "tensor", "hf_clm", etc.
3. **Define transform_fn() as STANDALONE function at TOP of file** -- not a class method
4. **Implement 5 abstract methods:**
   get_tfm_data(), fit(), infer(), save_model(), load_model()
5. **Call super().__init__(TfmArgs=TfmArgs, **kwargs)** --
   pass **kwargs through so the base class auto-stores them as attributes
6. **Store raw algorithm in self.model** -- base class initializes to None;
   this is the expected convention (set after fit or load)
7. **Save to model_{key}/ subfolder** inside model_dir
8. **Save tuner metadata alongside weights** (e.g., tfm_args.json, tuner_config.json,
   or best_params.json -- the exact filename varies, but tuner state must be persisted)
9. **Import the raw algorithm ONLY in this file**

---

MUST NOT
========

1. **NEVER know about ModelInstance, ModelInstance_Set, or ModelInstance_Pipeline**
2. **NEVER know about PreFnPipeline** -- that is above your scope
3. **NEVER know about modelinstance_dir** -- you receive model_dir only
4. **NEVER import other Tuners** -- each Tuner is self-contained
5. **NEVER put transform_fn as a class method** -- it must be importable standalone

**Open question:** The return type of infer() is not standardized across Tuners.
Some return DataFrames, some return dicts, some return HF Datasets. The base
class has a TODO note about this (model_tuner.py). For consistency, consider
returning a dict or DataFrame. Check existing Tuners in your model family
for the expected return format.

---

Key File Locations
==================

```
Base class:               code/hainn/model_tuner.py

TSForecast tuners:        code/hainn/tsforecast/models/
  neuralforecast/:        PatchTST, NBEATS, NHITS, Autoformer, DLinear, TFT
  mlforecast/:            XGBoost, LightGBM, CatBoost, Linear
  statsforecast/:         ARIMA, AutoETS, Naive
  api/:                   LLMAPITuner, NixtlaAPITuner (TimeGPT)

MLPredictor tuners:       code/hainn/mlpredictor/models/
  XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting, Logistic, DeepFM

TEFM tuners:              code/hainn/tefm/models/
  hfntp/:                 HFNTPTuner (HuggingFace causal language modeling)
```

---

Test Notebook: What Layer 2 Tests
==================================

The tuner test exercises the Tuner wrapper in isolation (NO Instance).
It verifies transform_fn, fit, infer, and save/load work correctly.

**Expected steps:**

```
Step 1: Load config (display_df with tuner_name, architecture)
Step 2: Load AIData (print(aidata), print(sample))
Step 3: Create Tuner (display_df with mode, params)
Step 4: Prepare data:
        4a: transform_fn -- BEFORE/AFTER pattern:
            print(raw_data), print(raw_data[0])
            -> run transform_fn ->
            print(ds_tfm), print(ds_tfm[0])
            + verify token range assertions
        4b: get_tfm_data -- BEFORE/AFTER pattern:
            print(aidata)
            -> run get_tfm_data ->
            print each split dataset
            + display_df with split summary
Step 5: Fit:
        print(data_fit) before calling
        -> tuner.fit(data_fit, TrainingArgs) ->
        display_df with training time, final loss
Step 6: Infer (test multiple input types):
        Type A: Dict[str, Dataset]
            print(input_dict) before
            -> tuner.infer(input) ->
            print(output keys, loss, shapes) after
        Type B: Single Dataset
            print(dataset) before
            -> tuner.infer(dataset) ->
            print(output) after
        display_df with both results
Step 7: Save/load roundtrip (display_df with weight match)
Step 8: YAML config validation (display_df comparing configs)
Summary: display_df with all steps PASSED/FAILED
```

**Key display rules for Layer 2:**

- Step 4 is the most important -- it tests BOTH transform_fn directly
  AND get_tfm_data via the Tuner (which calls transform_fn internally)
- Step 6 must test multiple input types (Dict vs single Dataset)
  because the Tuner's infer() dispatches based on input type
- Always print the actual data dict before calling fit/infer:
  ```python
  print("--- data_fit ---")
  for k, v in data_fit.items():
      print(f"  {k}: {v}")
  ```

**Reference:** `code/hainn/tefm/models/te_clm/test-modeling-te_clm/scripts/test_te_clm_2_tuner.py`

**Snapshot** (from te_clm, for quick reference -- canonical source is the file above):

```python
#!/usr/bin/env python3
"""
TE-CLM Tuner (Layer 2) Step-by-Step Test
==========================================

Test the TECLMTuner -- the Layer 2 Tuner that wraps the HuggingFace model.

Following the config -> data -> model pipeline pattern:

  Step 1: Setup workspace and load config
  Step 2: Load AIData (real OhioT1DM)
  Step 3: Create TECLMTuner from-scratch (architecture_config)
  Step 4: Prepare data (transform_fn on real data)
  Step 5: Fit on prepared data
  Step 6: Infer (multiple input types)
  Step 7: Save/load roundtrip (tuner_config.json + weights)
  Step 8: YAML config validation
  Step 9: Pretrained mode (optional, controlled by SKIP_PRETRAINED)

Usage:
    source .venv/bin/activate && source env.sh
    python code/hainn/tefm/models/te_clm/test-modeling-te_clm/scripts/test_te_clm_2_tuner.py
"""

# %% [markdown]
# # TE-CLM Tuner (Layer 2) Step-by-Step Test
#
# Tests the TECLMTuner in isolation (NO TEFMInstance wrapper),
# following the config -> data -> model pipeline pattern.
#
# Uses real OhioT1DM AIData.

# %% Step 1: Setup and Load Config
import os
import json
import yaml
import shutil

from pathlib import Path

import torch
import pandas as pd

from haipipe import setup_workspace

WORKSPACE_PATH, SPACE, logger = setup_workspace()

TEST_DIR = Path(WORKSPACE_PATH) / 'code' / 'hainn' / 'tefm' / 'models' / 'te_clm' / 'test-modeling-te_clm'

SKIP_PRETRAINED = True  # Set False to test pretrained mode (requires model download)


def load_yaml(filename):
    with open(TEST_DIR / filename) as f:
        return yaml.safe_load(f)


def display_df(df):
    try:
        from IPython.display import display, HTML
        display(HTML(df.to_html()))
    except Exception:
        print(df.to_string())


config = load_yaml('config_te_clm_from_scratch.yaml')
ModelArgs = config['ModelArgs']
tuner_args = ModelArgs['model_tuner_args']
arch = tuner_args['architecture_config']
TfmArgs = ModelArgs.get('TfmArgs', {})

# Build model_dir from env + config (same structure as real pipeline)
model_store = os.environ.get('LOCAL_MODELINSTANCE_STORE', '_WorkSpace/5-ModelInstanceStore')
model_name = config.get('modelinstance_name', 'Demo-TECLM')
model_version = config.get('modelinstance_version', '@v0001-demo-te_clm-from-scratch')
model_dir = os.path.join(WORKSPACE_PATH, model_store, model_name, model_version)

display_df(pd.DataFrame([
    {'key': 'WORKSPACE_PATH',    'value': WORKSPACE_PATH},
    {'key': 'TEST_DIR',          'value': str(TEST_DIR)},
    {'key': 'model_dir',         'value': model_dir},
    {'key': 'model_tuner_name',  'value': ModelArgs['model_tuner_name']},
    {'key': 'model_name_or_path','value': tuner_args['model_name_or_path']},
    {'key': 'max_seq_length',    'value': tuner_args['max_seq_length']},
    {'key': 'vocab_size',        'value': arch['vocab_size']},
    {'key': 'hidden_size',       'value': arch['hidden_size']},
    {'key': 'num_hidden_layers', 'value': arch['num_hidden_layers']},
    {'key': 'tetoken_config',    'value': str(TfmArgs.get('tetoken_config', {}))},
    {'key': 'special_tokens',    'value': f"PAD={TfmArgs.get('special_tokens', {}).get('pad_token_id', 0)}, num_special={TfmArgs.get('special_tokens', {}).get('num_special_tokens', 10)}"},
    {'key': 'SKIP_PRETRAINED',   'value': SKIP_PRETRAINED},
    {'key': 'CUDA available',    'value': torch.cuda.is_available()},
]).set_index('key'))


# %% [markdown]
# ## Step 2: Load AIData
#
# Load real OhioT1DM AIData from disk.

# %% Step 2: Load AIData

from haipipe.aidata_base import AIDataSet

aidata_name = config.get('aidata_name', 'OhioT1DM')
aidata_version = config.get('aidata_version', '@v0002_events_per1h_tewindow')
AIDATA_PATH = os.path.join(SPACE['LOCAL_AIDATA_STORE'], aidata_name, aidata_version)
assert os.path.exists(AIDATA_PATH), f"AIData not found at {AIDATA_PATH}"

aidata = AIDataSet.load_from_disk(AIDATA_PATH)

print(f"AIData: {aidata_name}/{aidata_version}")
print(f"  Splits: {list(aidata.dataset_dict.keys())}")
for split, ds in aidata.dataset_dict.items():
    print(f"    {split}: {len(ds)} cases")

display_df(pd.DataFrame([
    {'key': 'aidata_name',    'value': aidata_name},
    {'key': 'aidata_version', 'value': aidata_version},
    {'key': 'AIDATA_PATH',    'value': AIDATA_PATH},
]).set_index('key'))

# %% Display AIData structure

print(aidata)

# %% Check a sample from train split

if len(aidata.dataset_dict.get('train', [])) > 0:
    sample = aidata.dataset_dict['train'][0]
    print(sample)


# %% [markdown]
# ## Step 3: Create TECLMTuner (From-Scratch)
#
# Create tuner with architecture_config to build a tiny model from scratch.
# Verifies: lazy loading, architecture overrides, parameter count.

# %% Step 3: Create TECLMTuner

from hainn.tefm.models.te_clm import TECLMTuner

tuner = TECLMTuner(
    TfmArgs=dict(TfmArgs),
    **tuner_args,
)

# Before lazy load
assert tuner.model is None
assert tuner.architecture_config == arch

# Trigger lazy load
tuner._ensure_model_loaded()

# Verify architecture overrides
assert tuner.model is not None
assert tuner.tokenizer is not None
assert tuner.model.config.vocab_size == arch['vocab_size']
assert tuner.model.config.num_hidden_layers == arch['num_hidden_layers']
assert tuner.model.config.hidden_size == arch['hidden_size']

total_params = sum(p.numel() for p in tuner.model.parameters())
assert 100_000 < total_params < 5_000_000

display_df(pd.DataFrame([
    {'property': 'mode',              'value': 'from-scratch'},
    {'property': 'model_name_or_path','value': tuner.model_name_or_path},
    {'property': 'vocab_size',        'value': tuner.model.config.vocab_size},
    {'property': 'hidden_size',       'value': tuner.model.config.hidden_size},
    {'property': 'num_hidden_layers', 'value': tuner.model.config.num_hidden_layers},
    {'property': 'num_attention_heads','value': tuner.model.config.num_attention_heads},
    {'property': 'intermediate_size', 'value': tuner.model.config.intermediate_size},
    {'property': 'total_params',      'value': f'{total_params:,}'},
    {'property': 'params (M)',        'value': f'{total_params/1e6:.1f}M'},
    {'property': 'status',            'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 4: Prepare Data
#
# Test transform_fn directly, then test get_tfm_data via the tuner.
# Prepare real data subsets for fit/infer.

# %% Step 4: Prepare Data

from hainn.tefm.models.te_clm.modeling_te_clm import transform_fn

seq_len = tuner_args['max_seq_length']
vocab_size = arch['vocab_size']

# --- 4a: Test transform_fn directly ---
test_split_name = 'test-id' if 'test-id' in aidata.dataset_dict else list(aidata.dataset_dict.keys())[0]
test_ds_full = aidata.dataset_dict[test_split_name]
ds_raw = test_ds_full.select(range(min(20, len(test_ds_full))))

TfmArgs_full = {
    **TfmArgs,
    'model_name_or_path': tuner_args['model_name_or_path'],
    'max_seq_length': tuner_args['max_seq_length'],
}

# BEFORE: raw AIData
print("--- BEFORE transform_fn ---")
print(ds_raw)
print()
print("Sample [0]:")
print(ds_raw[0])

# %% Run transform_fn (4a)

ds_tfm = transform_fn(ds_raw, TfmArgs_full)

assert 'input_ids' in ds_tfm.column_names
assert 'attention_mask' in ds_tfm.column_names
assert 'labels' in ds_tfm.column_names
assert len(ds_tfm) == len(ds_raw)

# AFTER: transformed (input_ids, attention_mask, labels only)
print("--- AFTER transform_fn ---")
print(ds_tfm)
print()
print("Sample [0]:")
print(ds_tfm[0])

# %% Verify token ranges (4a continued)

for i in range(min(5, len(ds_tfm))):
    ids = ds_tfm[i]['input_ids']
    mask = ds_tfm[i]['attention_mask']
    labs = ds_tfm[i]['labels']
    assert len(ids) == seq_len, f"Sample {i}: len={len(ids)}, expected={seq_len}"
    # Real tokens (attention_mask==1) should be in [10, 400], pad tokens should be 0 (PAD)
    for j in range(seq_len):
        if mask[j] == 1:
            assert 10 <= ids[j] <= 400, \
                f"Sample {i}, pos {j}: real token {ids[j]} outside [10, 400]"
        else:
            assert ids[j] == 0, \
                f"Sample {i}, pos {j}: pad token should be 0, got {ids[j]}"
    for j in range(seq_len):
        if mask[j] == 0:
            assert labs[j] == -100

print("Token range assertions PASSED")

# --- 4b: Test get_tfm_data via tuner ---
print("--- BEFORE get_tfm_data ---")
print(aidata)

# %% Run get_tfm_data (4b)

data_tfm = tuner.get_tfm_data(aidata)

assert isinstance(data_tfm, dict)
for split_name in aidata.dataset_dict:
    assert split_name in data_tfm
for split_name, ds in data_tfm.items():
    expected_count = len(aidata.dataset_dict[split_name])
    assert len(ds) == expected_count
    if len(ds) > 0:
        assert 'input_ids' in ds.column_names

print("--- AFTER get_tfm_data ---")
for split_name, ds in data_tfm.items():
    print(f"  {split_name}: {ds}")

rows_tfm = []
for split_name, ds in data_tfm.items():
    cols = ', '.join(ds.column_names) if len(ds) > 0 else '(empty)'
    rows_tfm.append({'split': split_name, 'samples': len(ds), 'columns': cols})
display_df(pd.DataFrame(rows_tfm).set_index('split'))

# Prepare fit data: real subsets
train_ds_raw = aidata.dataset_dict['train'].select(range(min(100, len(aidata.dataset_dict['train']))))
test_ds_raw = aidata.dataset_dict[test_split_name].select(range(min(50, len(test_ds_full))))
data_fit = {'train': train_ds_raw, test_split_name: test_ds_raw}

display_df(pd.DataFrame([
    {'property': 'fit splits', 'value': ', '.join(f'{k}({len(v)})' for k, v in data_fit.items())},
    {'property': 'status',     'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 5: Fit
#
# Train the tuner on the prepared data (real or synthetic).
# TECLMTuner.fit() creates an HF Trainer and trains for 1 epoch.

# %% Step 5: Fit

import time

TrainingArgs = config.get('TrainingArgs', {}).copy()
TrainingArgs.update({
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 8,
    'logging_steps': 5,
    'output_dir': os.path.join(model_dir, 'train_checkpoints'),
    'save_strategy': 'no',
    'report_to': [],
    'gradient_checkpointing': False,
    'fp16': False,
})

print("--- data_fit ---")
for k, v in data_fit.items():
    print(f"  {k}: {v}")

t_start = time.time()
tuner.fit(data_fit, TrainingArgs)
t_elapsed = time.time() - t_start

assert tuner.model is not None
assert tuner.trainer is not None

final_log = tuner.trainer.state.log_history[-1] if tuner.trainer.state.log_history else {}
final_loss = final_log.get('train_loss', final_log.get('loss', 'N/A'))

display_df(pd.DataFrame([
    {'property': 'fit splits',    'value': ', '.join(f'{k}({len(v)})' for k, v in data_fit.items())},
    {'property': 'training time', 'value': f'{t_elapsed:.1f}s'},
    {'property': 'final loss',    'value': str(final_loss)},
    {'property': 'total params',  'value': f'{total_params:,}'},
    {'property': 'has model',     'value': tuner.model is not None},
    {'property': 'has trainer',   'value': tuner.trainer is not None},
    {'property': 'status',        'value': 'PASSED'},
]).set_index('property'))

shutil.rmtree(TrainingArgs['output_dir'], ignore_errors=True)  # cleanup checkpoints


# %% [markdown]
# ## Step 6: Infer
#
# Run inference with the trained tuner.
# Test two input types:
# - Type A: Dict[str, Dataset] (multiple splits)
# - Type B: Single HF Dataset

# %% Step 6: Infer

import math

hidden_size = arch['hidden_size']

# --- Type A: Dict[str, Dataset] ---
ds_infer_a = aidata.dataset_dict[test_split_name].select(
    range(min(50, len(aidata.dataset_dict[test_split_name]))))
data_infer_a = {test_split_name: ds_infer_a}
n_a = len(ds_infer_a)
infer_split_a = test_split_name

print("--- Type A input: Dict[str, Dataset] ---")
for k, v in data_infer_a.items():
    print(f"  {k}: {v}")

results_a = tuner.infer(data_infer_a)

assert isinstance(results_a, dict)
assert infer_split_a in results_a
r_a = results_a[infer_split_a]
assert 'loss' in r_a
assert 'embeddings' in r_a
assert 'predicted_tokens' in r_a
assert r_a['embeddings'].shape == (n_a, hidden_size), \
    f"Embeddings shape {r_a['embeddings'].shape} != expected ({n_a}, {hidden_size})"
assert r_a['predicted_tokens'].shape == (n_a, seq_len), \
    f"Predicted tokens shape {r_a['predicted_tokens'].shape} != expected ({n_a}, {seq_len})"

print("--- Type A output ---")
print(f"  keys: {list(results_a.keys())}")
print(f"  loss: {r_a['loss']:.4f}, perplexity: {math.exp(r_a['loss']):.2f}")
print(f"  embeddings: {r_a['embeddings'].shape}, predicted_tokens: {r_a['predicted_tokens'].shape}")

# --- Type B: Single HF Dataset ---
ds_infer_b = aidata.dataset_dict[test_split_name].select(
    range(min(10, len(aidata.dataset_dict[test_split_name]))))
n_b = len(ds_infer_b)

print("\n--- Type B input: Single HF Dataset ---")
print(ds_infer_b)

results_b = tuner.infer(ds_infer_b)

assert isinstance(results_b, dict)
assert 'loss' in results_b
assert 'embeddings' in results_b
assert 'predicted_tokens' in results_b
assert results_b['embeddings'].shape == (n_b, hidden_size), \
    f"Embeddings shape {results_b['embeddings'].shape} != expected ({n_b}, {hidden_size})"
assert results_b['predicted_tokens'].shape == (n_b, seq_len), \
    f"Predicted tokens shape {results_b['predicted_tokens'].shape} != expected ({n_b}, {seq_len})"

print("--- Type B output ---")
print(f"  keys: {list(results_b.keys())}")
print(f"  loss: {results_b['loss']:.4f}, perplexity: {math.exp(results_b['loss']):.2f}")
print(f"  embeddings: {results_b['embeddings'].shape}, predicted_tokens: {results_b['predicted_tokens'].shape}")

display_df(pd.DataFrame([
    {'property': 'Type A input',       'value': f'Dict ({infer_split_a}: {n_a} samples)'},
    {'property': 'Type A loss',        'value': f'{r_a["loss"]:.4f}'},
    {'property': 'Type A perplexity',  'value': f'{math.exp(r_a["loss"]):.2f}'},
    {'property': 'Type A embeddings',  'value': str(r_a['embeddings'].shape)},
    {'property': 'Type B input',       'value': f'Single Dataset ({n_b} samples)'},
    {'property': 'Type B loss',        'value': f'{results_b["loss"]:.4f}'},
    {'property': 'Type B perplexity',  'value': f'{math.exp(results_b["loss"]):.2f}'},
    {'property': 'Type B embeddings',  'value': str(results_b['embeddings'].shape)},
    {'property': 'status',             'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 7: Save/Load Roundtrip
#
# Save the trained tuner, load it back, verify:
# - tuner_config.json persists architecture_config and hyperparams
# - Loaded model has identical architecture
# - Weights match within tolerance

# %% Step 7: Save/Load Roundtrip

tmp_dir = os.path.join(model_dir, 'model')

# Save
tuner.save_model(key='MAIN', model_dir=tmp_dir)
save_path = os.path.join(tmp_dir, 'model_MAIN')
assert os.path.isdir(save_path)

tuner_config_path = os.path.join(save_path, 'tuner_config.json')
assert os.path.exists(tuner_config_path)

with open(tuner_config_path) as f:
    saved_config = json.load(f)

# Verify tuner_config.json has expected fields
assert saved_config['architecture_config'] == arch
assert saved_config['model_name_or_path'] == tuner_args['model_name_or_path']
assert saved_config['max_seq_length'] == tuner_args['max_seq_length']

expected_keys = {
    'model_name_or_path', 'max_seq_length', 'architecture_config',
    'learning_rate', 'per_device_train_batch_size', 'weight_decay', 'max_grad_norm',
}
assert expected_keys.issubset(set(saved_config.keys())), \
    f"Missing keys: {expected_keys - set(saved_config.keys())}"

# Load
loaded = TECLMTuner(
    model_name_or_path=tuner_args['model_name_or_path'],
    max_seq_length=tuner_args['max_seq_length'],
)
loaded.load_model(key='MAIN', model_dir=tmp_dir)

# Verify attributes restored
assert loaded.architecture_config == arch
assert loaded.model_name_or_path == tuner_args['model_name_or_path']
assert loaded.max_seq_length == tuner_args['max_seq_length']
assert loaded.model.config.vocab_size == arch['vocab_size']
assert loaded.model.config.num_hidden_layers == arch['num_hidden_layers']
assert loaded.model.config.hidden_size == arch['hidden_size']

# Compare weights (move to CPU for comparison)
orig_state = {k: v.cpu().float() for k, v in tuner.model.state_dict().items()}
loaded_state = {k: v.cpu().float() for k, v in loaded.model.state_dict().items()}
assert set(orig_state.keys()) == set(loaded_state.keys())

mismatched = [k for k in orig_state
              if not torch.allclose(orig_state[k].detach().cpu().float(),
                                    loaded_state[k].detach().cpu().float(), atol=1e-4)]
assert len(mismatched) == 0, f"Weight mismatch: {mismatched[:5]}"

saved_files = sorted(os.listdir(save_path))

display_df(pd.DataFrame([
    {'property': 'save_path',            'value': save_path},
    {'property': 'saved files',          'value': ', '.join(saved_files)},
    {'property': 'tuner_config keys',    'value': ', '.join(sorted(saved_config.keys()))},
    {'property': 'architecture restored','value': loaded.architecture_config == arch},
    {'property': 'state_dict keys',      'value': len(orig_state)},
    {'property': 'weight mismatches',    'value': len(mismatched)},
    {'property': 'status',               'value': 'PASSED'},
]).set_index('property'))

shutil.rmtree(tmp_dir)


# %% [markdown]
# ## Step 8: YAML Config Validation
#
# Verify both YAML configs match the pipeline format:
# - ModelInstanceClass, ModelArgs, TrainingArgs keys present
# - model_tuner_name is TECLMTuner
# - from-scratch has architecture_config

# %% Step 8: YAML Config Validation

fs_cfg = load_yaml('config_te_clm_from_scratch.yaml')
pt_cfg = load_yaml('config_te_clm_pretrained.yaml')

required_top_keys = {
    'aidata_name', 'aidata_version',
    'modelinstance_name', 'modelinstance_version',
    'ModelInstanceClass', 'ModelArgs', 'TrainingArgs', 'InferenceArgs', 'EvaluationArgs',
}

# From-scratch assertions
assert required_top_keys.issubset(set(fs_cfg.keys())), \
    f"Missing keys in from_scratch: {required_top_keys - set(fs_cfg.keys())}"
assert fs_cfg['ModelInstanceClass'] == 'TEFM'
assert fs_cfg['aidata_name'] == 'OhioT1DM'
assert fs_cfg['modelinstance_name'] == 'Demo-TECLM'
assert fs_cfg['ModelArgs']['model_tuner_name'] == 'TECLMTuner'
assert 'architecture_config' in fs_cfg['ModelArgs']['model_tuner_args']
assert fs_cfg['ModelArgs']['model_tuner_args']['architecture_config']['vocab_size'] == 401
assert 'TfmArgs' in fs_cfg['ModelArgs']
assert 'InferenceSetNames' in fs_cfg['InferenceArgs']
assert 'EvalSetNames' in fs_cfg['EvaluationArgs']

# Pretrained assertions
assert required_top_keys.issubset(set(pt_cfg.keys())), \
    f"Missing keys in pretrained: {required_top_keys - set(pt_cfg.keys())}"
assert pt_cfg['ModelInstanceClass'] == 'TEFM'
assert pt_cfg['ModelArgs']['model_tuner_name'] == 'TECLMTuner'
assert pt_cfg['pretrained_modelinstance_name'] == 'Demo-TECLM'
assert pt_cfg['pretrained_modelinstance_version'] == '@v0001-demo-te_clm-from-scratch'

# Architecture configs must match between from-scratch and pretrained
assert fs_cfg['ModelArgs']['model_tuner_args']['architecture_config'] == \
    pt_cfg['ModelArgs']['model_tuner_args']['architecture_config']

rows = []
for name, cfg in [('from_scratch', fs_cfg), ('pretrained', pt_cfg)]:
    rows.append({
        'config': name,
        'ModelInstanceClass': cfg['ModelInstanceClass'],
        'model_tuner_name': cfg['ModelArgs']['model_tuner_name'],
        'has architecture_config': 'architecture_config' in cfg['ModelArgs']['model_tuner_args'],
        'vocab_size': cfg['ModelArgs']['model_tuner_args']['architecture_config']['vocab_size'],
        'has TfmArgs': 'TfmArgs' in cfg['ModelArgs'],
    })

display_df(pd.DataFrame(rows).set_index('config'))
print("PASSED")


# %% [markdown]
# ## Step 9: Pretrained Mode (Optional)
#
# Load PRETRAINED model (no architecture_config).
# Controlled by SKIP_PRETRAINED flag.

# %% Step 9: Pretrained Mode

if SKIP_PRETRAINED:
    display_df(pd.DataFrame([
        {'property': 'mode', 'value': 'pretrained'},
        {'property': 'status', 'value': 'SKIPPED (SKIP_PRETRAINED=True)'},
    ]).set_index('property'))

else:
    config_pt = load_yaml('config_te_clm_pretrained.yaml')
    tuner_args_pt = config_pt['ModelArgs']['model_tuner_args']

    tuner_pt = TECLMTuner(
        model_name_or_path=tuner_args_pt['model_name_or_path'],
        max_seq_length=tuner_args_pt['max_seq_length'],
        learning_rate=tuner_args_pt['learning_rate'],
    )

    assert tuner_pt.architecture_config == {}

    tuner_pt._ensure_model_loaded()
    assert tuner_pt.model is not None
    assert tuner_pt.tokenizer is not None
    assert tuner_pt.model.config.vocab_size != 401

    total_params_pt = sum(p.numel() for p in tuner_pt.model.parameters())

    display_df(pd.DataFrame([
        {'property': 'mode',              'value': 'pretrained'},
        {'property': 'model_name_or_path','value': tuner_args_pt['model_name_or_path']},
        {'property': 'vocab_size',        'value': tuner_pt.model.config.vocab_size},
        {'property': 'hidden_size',       'value': tuner_pt.model.config.hidden_size},
        {'property': 'total_params',      'value': f'{total_params_pt:,}'},
        {'property': 'status',            'value': 'PASSED'},
    ]).set_index('property'))


# %% [markdown]
# ## Summary

# %% Summary

results = pd.DataFrame([
    {'test': '1. Load Config',            'status': 'PASSED'},
    {'test': '2. Load AIData',            'status': 'PASSED'},
    {'test': '3. Create TECLMTuner',      'status': 'PASSED'},
    {'test': '4. Prepare Data',           'status': 'PASSED'},
    {'test': '5. Fit',                    'status': 'PASSED'},
    {'test': '6. Infer',                  'status': 'PASSED'},
    {'test': '7. Save/Load Roundtrip',    'status': 'PASSED'},
    {'test': '8. YAML Config Validation', 'status': 'PASSED'},
    {'test': '9. Pretrained Mode',        'status': 'SKIPPED' if SKIP_PRETRAINED else 'PASSED'},
]).set_index('test')

display_df(results)

# Cleanup model_dir
shutil.rmtree(model_dir, ignore_errors=True)
```

---

See Also
========

- **haipipe-nn-0-overview**: Architecture map and decision guide
- **haipipe-nn-1-algorithm**: What algorithms are and their diversity
- **haipipe-nn-3-instance**: How Tuners are managed by Instances
- **haipipe-nn-4-modelset**: How Instances are packaged into assets
