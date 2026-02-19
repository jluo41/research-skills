Skill: haipipe-nn-1-algorithm
==============================

Layer 1 of the 4-layer NN pipeline: Algorithm.

An algorithm is any external ML library that does the actual computation.
We don't write algorithms -- we wrap them in Tuners (Layer 2).

---

Architecture Position
=====================

```
Layer 4: ModelSet/Pipeline    Packages everything. Config-driven.
    |
Layer 3: Instance             Thin orchestrator. Manages Tuner dict.
    |
Layer 2: Tuner                Wraps ONE algorithm. Owns data conversion.
    |
Layer 1: Algorithm  <---      Raw external library. Doesn't know the
                              pipeline exists.
```

---

What Is an Algorithm?
=====================

An algorithm is a third-party library installed via pip/conda that provides:
- A model class (e.g., xgb.XGBClassifier, PatchTST, AutoModelForCausalLM)
- A training method (e.g., .fit(), .train(), trainer.train())
- A prediction method (e.g., .predict(), .generate(), .forward())
- A serialization method (e.g., .save_model(), .save_pretrained(), pickle)

**You never write algorithm code.** You wrap it in a Tuner that provides
a unified interface to the rest of the pipeline.

---

Algorithm Diversity
===================

Algorithms vary across every dimension. The Tuner layer absorbs all
this diversity and presents a uniform interface upward.

```
Dimension           Examples                           Tuner Handles Via
──────────────────  ────────────────────────────────   ─────────────────────
Input format        DataFrame, sparse matrix, tensor,  transform_fn()
                    HF Dataset, JSON payload

Training paradigm   .fit(X,y), Trainer.train(),        fit(dataset, TrainingArgs)
                    custom loop, API call

Output format       array, DataFrame, dict, tensor     infer(dataset, InferenceArgs)

Serialization       .save_model(), state_dict,          save_model(key, dir)
                    pickle, save_pretrained

Hyperparameter      Optuna, grid search, none           TrainingArgs + objective()
tuning

Compute             CPU, GPU, multi-GPU, API            Handled inside Tuner
```

**Concrete examples across the spectrum:**

```
Algorithm             Library           Input Format      Save Format
────────────────────  ────────────────  ────────────────  ──────────────
XGBoost               xgboost           sparse matrix     .json
LightGBM              lightgbm          sparse matrix     .txt / .json
CatBoost              catboost          DataFrame         .cbm
PatchTST              neuralforecast    Nixtla DataFrame  checkpoint dir
NBEATS                neuralforecast    Nixtla DataFrame  checkpoint dir
ARIMA                 statsforecast     Nixtla DataFrame  pickle
XGBForecast           mlforecast        Nixtla DataFrame  .json
DeepFM                deepctr-torch     tensor            state_dict
GPT-2/LLaMA (CLM)    transformers      HF Dataset        save_pretrained
EarlyFusionTEFM       custom PyTorch    tensor            state_dict
DiffusionTEFM         custom PyTorch    tensor            state_dict
LLM API               openai/anthropic  JSON              N/A (stateless)
```

---

Domain Formats
==============

Each algorithm family expects data in a specific format. We call this
the "domain format." The Tuner's transform_fn() handles the conversion.

```
domain_format     What it means                   Used by
───────────────   ─────────────────────────────   ──────────────────────
"nixtla"          DataFrame: (unique_id, ds, y)   NeuralForecast,
                  + optional exogenous columns     MLForecast,
                                                   StatsForecast

"sparse"          scipy.sparse.csr_matrix for     XGBoost, LightGBM,
                  features + numpy array labels    DeepFM (via adapter)

"tensor"          PyTorch tensors, often           Custom PyTorch models,
                  (batch, seq_len, features)       TEFM architectures

"hf_clm"          HuggingFace Dataset with         GPT-2, LLaMA,
                  (input_ids, attention_mask,       HFNTPTuner
                  labels) for causal LM

"llmapi"          JSON payload for API call        OpenAI, Anthropic,
                                                   Nixtla TimeGPT API

"custom"          Any format your algorithm        Future models
                  needs -- you define it
```

When creating a new Tuner, pick the domain_format that matches your
algorithm. If none fit, define a new one.

---

Serialization Formats
=====================

```
Format              How to save                 How to load
──────────────────  ─────────────────────────   ─────────────────────────
JSON (tree models)  model.save_model(path)      model.load_model(path)
Pickle (sklearn)    pickle.dump(model, f)       pickle.load(f)
PyTorch state_dict  torch.save(state_dict, p)   model.load_state_dict(...)
HF save_pretrained  model.save_pretrained(dir)  AutoModel.from_pretrained(dir)
Checkpoint dir      model.save(dir)             Model.load(dir)
None (API models)   Save config only            Reconnect via API key
```

---

Concrete Code From the Repo
============================

These show actual algorithm calls inside Tuner files:

**xgboost** (mlpredictor/models/modeling_xgboost.py):

```python
model = xgb.XGBClassifier(**param)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
preds = model.predict_proba(X_valid)[:, 1]
```

**neuralforecast** (tsforecast/models/neuralforecast/modeling_nixtla_patchtst.py):

```python
nf = NeuralForecast(models=[PatchTST(h=24, input_size=288, ...)], freq='5min')
nf.fit(df=df_nixtla)  # DataFrame with [unique_id, ds, y] columns
forecasts = nf.predict()
```

**HuggingFace Transformers** (tefm/models/hfntp/modeling_hfntp.py):

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

**LLM API** (tsforecast/models/api/modeling_llmapi.py):

```python
client = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
response = client.invoke([SystemMessage(...), HumanMessage(prompt)])
predictions = json.loads(response.content)
```

---

How to Add a New Algorithm
==========================

You do NOT write code at Layer 1 -- you install the library and
write a Tuner (Layer 2) that wraps it.

```
1. Install the library:
   pip install new-algorithm-lib

2. Create a Tuner file:
   code/hainn/<family>/models/modeling_<name>.py

3. Import the algorithm ONLY in that file

4. Implement the Tuner interface (see haipipe-nn-2-tuner)

5. Register the Tuner in the Instance's MODEL_TUNER_REGISTRY
```

---

MUST DO
=======

1. **Identify the algorithm's native interface** --
   What does .fit() expect? What does .predict() return? What format?
2. **Determine domain_format** --
   Nixtla DataFrame? Sparse matrix? Tensor? HF Dataset? API payload?
3. **Determine serialization format** --
   JSON? Pickle? state_dict? save_pretrained? None?
4. **Create ONE Tuner per algorithm** --
   Each Tuner wraps exactly one algorithm

---

MUST NOT
========

1. **NEVER import algorithm libraries above the Tuner layer** --
   No xgboost, torch, neuralforecast, sklearn at Instance or Pipeline level
2. **NEVER modify algorithm internals** --
   Use the library's public API as-is; adaptation logic goes in the Tuner
3. **NEVER assume a specific algorithm in upper layers** --
   Pipeline must work with any algorithm that has a Tuner

---

Key File Locations
==================

```
Tuner base class:       code/hainn/model_tuner.py
TS Forecast tuners:     code/hainn/tsforecast/models/*/modeling_*.py
ML Predictor tuners:    code/hainn/mlpredictor/models/modeling_*.py
TEFM tuners:            code/hainn/tefm/models/*/modeling_*.py
```

---

Test Notebook: What Layer 1 Tests
==================================

The algorithm test exercises the raw library in isolation (NO Tuner wrapper).
It verifies the external API works before wrapping it.

**Expected steps:**

```
Step 1: Load config (display_df with architecture params)
Step 2: Load AIData (print(aidata), print(sample))
Step 3: Create model from config (display_df with model class, params)
Step 4: Tokenizer/preprocessor setup (display_df with pad_token, vocab)
Step 5a: Transform data -- BEFORE/AFTER pattern:
         print(raw_data), print(raw_data[0])
         -> run transform_fn ->
         print(transformed), print(transformed[0])
Step 5b: Verify token/feature ranges (assertions + display_df)
Step 5c: Convert to tensors (display_df with shapes)
Step 6: Forward pass (display_df with loss, logits shape)
Step 7: Gradient flow (display_df with grad coverage, params changed)
Step 8: Save/load roundtrip (display_df with weight match)
Summary: display_df with all steps PASSED/FAILED
```

**Key display rules for Layer 1:**

- Step 5a is the most important display -- it shows the full data
  transformation pipeline with actual values visible
- Use `print(dataset_object)` to show HF Dataset schema + row count
- Use `print(dataset_object[0])` to show one concrete sample
- Every step ends with a `display_df()` status table
- The before/after in Step 5a should clearly show:
  raw CGM values (e.g., 1.0, 126.0, 125.0) -> token IDs (10, 126, 125)

**Reference:** `code/hainn/tefm/models/te_clm/test-modeling-te_clm/scripts/test_te_clm_1_algorithm.py`

**Snapshot** (from te_clm, for quick reference -- canonical source is the file above):

```python
#!/usr/bin/env python3
"""
TE-CLM Algorithm (Layer 1) Step-by-Step Test
=============================================

Test the raw HuggingFace AutoModelForCausalLM -- the Layer 1 Algorithm
that TECLMTuner wraps.

Following the config -> data -> model pipeline pattern:

  Step 1: Setup workspace and load config
  Step 2: Load AIData (real OhioT1DM)
  Step 3: Create model from AutoConfig (from-scratch)
  Step 4: Tokenizer setup and pad token handling
  Step 5: Prepare data (transform_fn on real data)
  Step 6: Forward pass -- loss and logits
  Step 7: Gradient flow -- backward pass and parameter updates
  Step 8: HF native save/load roundtrip (save_pretrained / from_pretrained)

Usage:
    source .venv/bin/activate && source env.sh
    python code/hainn/tefm/models/te_clm/test-modeling-te_clm/scripts/test_te_clm_1_algorithm.py
"""

# %% [markdown]
# # TE-CLM Algorithm (Layer 1) Step-by-Step Test
#
# Tests the raw HuggingFace model in isolation (NO TECLMTuner wrapper),
# following the config -> data -> model pipeline pattern.
#
# Uses real OhioT1DM AIData.

# %% Step 1: Setup and Load Config
import os
import yaml
import shutil
import tempfile
from pathlib import Path

import torch
import pandas as pd

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from haipipe import setup_workspace

WORKSPACE_PATH, SPACE, logger = setup_workspace()

TEST_DIR = Path(WORKSPACE_PATH) / 'code' / 'hainn' / 'tefm' / 'models' / 'te_clm' / 'test-modeling-te_clm'


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

display_df(pd.DataFrame([
    {'key': 'WORKSPACE_PATH',    'value': WORKSPACE_PATH},
    {'key': 'TEST_DIR',          'value': str(TEST_DIR)},
    {'key': 'model_name_or_path','value': tuner_args['model_name_or_path']},
    {'key': 'max_seq_length',    'value': tuner_args['max_seq_length']},
    {'key': 'vocab_size',        'value': arch['vocab_size']},
    {'key': 'hidden_size',       'value': arch['hidden_size']},
    {'key': 'num_hidden_layers', 'value': arch['num_hidden_layers']},
    {'key': 'tetoken_config',    'value': str(TfmArgs.get('tetoken_config', {}))},
    {'key': 'special_tokens',    'value': f"PAD={TfmArgs.get('special_tokens', {}).get('pad_token_id', 0)}, num_special={TfmArgs.get('special_tokens', {}).get('num_special_tokens', 10)}"},
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


# %% [markdown]
# ## Step 3: Create Model from AutoConfig
#
# Load architecture template from HuggingFace, override with custom sizes,
# create model from config (random weights). This is what TECLMTuner does
# internally in from-scratch mode.

# %% Step 3: Create Model from AutoConfig

# Load base config template
base_config = AutoConfig.from_pretrained(
    tuner_args['model_name_or_path'],
    trust_remote_code=True,
)

# Override with custom architecture
for key, value in arch.items():
    setattr(base_config, key, value)

# Trim per-layer list attributes (e.g., Qwen2.5 has layer_types)
orig_num_layers = getattr(base_config, '_original_num_hidden_layers', None)
new_num_layers = arch['num_hidden_layers']
for attr_name in dir(base_config):
    if attr_name.startswith('_'):
        continue
    val = getattr(base_config, attr_name, None)
    if isinstance(val, list) and len(val) > new_num_layers:
        setattr(base_config, attr_name, val[:new_num_layers])

# Create model from config (random weights)
model = AutoModelForCausalLM.from_config(base_config, trust_remote_code=True)

assert model is not None
assert model.config.vocab_size == arch['vocab_size']
assert model.config.hidden_size == arch['hidden_size']
assert model.config.num_hidden_layers == arch['num_hidden_layers']

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

display_df(pd.DataFrame([
    {'property': 'model class',        'value': type(model).__name__},
    {'property': 'config class',       'value': type(base_config).__name__},
    {'property': 'vocab_size',         'value': model.config.vocab_size},
    {'property': 'hidden_size',        'value': model.config.hidden_size},
    {'property': 'num_hidden_layers',  'value': model.config.num_hidden_layers},
    {'property': 'num_attention_heads','value': model.config.num_attention_heads},
    {'property': 'intermediate_size',  'value': model.config.intermediate_size},
    {'property': 'total_params',       'value': f'{total_params:,}'},
    {'property': 'trainable_params',   'value': f'{trainable_params:,}'},
    {'property': 'all trainable',      'value': total_params == trainable_params},
    {'property': 'status',             'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 4: Tokenizer Setup
#
# Load tokenizer, verify pad token setup, verify vocab size compatibility
# with the from-scratch model.

# %% Step 4: Tokenizer Setup

from hainn.tefm.models.te_clm.modeling_te_clm import CLM_SPECIAL_TOKENS

tokenizer = AutoTokenizer.from_pretrained(tuner_args['model_name_or_path'])

# Override pad_token_id to match CLM vocabulary (same as TECLMTuner._ensure_model_loaded)
special_tokens = TfmArgs.get('special_tokens', CLM_SPECIAL_TOKENS)
tokenizer.pad_token_id = special_tokens['pad_token_id']
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or '[PAD]'

assert tokenizer.pad_token is not None
assert tokenizer.pad_token_id == special_tokens['pad_token_id']

# Verify vocab size is compatible
model_vocab = arch['vocab_size']
tokenizer_vocab = tokenizer.vocab_size

display_df(pd.DataFrame([
    {'property': 'tokenizer class',    'value': type(tokenizer).__name__},
    {'property': 'tokenizer vocab',    'value': tokenizer_vocab},
    {'property': 'model vocab',        'value': model_vocab},
    {'property': 'pad_token_id',       'value': tokenizer.pad_token_id},
    {'property': 'CLM PAD token',      'value': special_tokens['pad_token_id']},
    {'property': 'CLM num_special',    'value': special_tokens['num_special_tokens']},
    {'property': 'has pad_token',      'value': tokenizer.pad_token is not None},
    {'property': 'status',             'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 5a: Tokenize Real Data
#
# Run transform_fn on real AIData to produce transformed input_ids/labels.
# Show before/after to see how data flows through the pipeline.

# %% Step 5a: Tokenize Real Data

from hainn.tefm.models.te_clm.modeling_te_clm import transform_fn

seq_len = tuner_args['max_seq_length']
vocab_size = arch['vocab_size']

test_split_name = 'test-id' if 'test-id' in aidata.dataset_dict else list(aidata.dataset_dict.keys())[0]
test_ds_full = aidata.dataset_dict[test_split_name]
ds_raw = test_ds_full.select(range(min(20, len(test_ds_full))))

# BEFORE: raw AIData (TEWindow format with cgm, demographics, events, ...)
print("--- BEFORE transform_fn ---")
print(ds_raw)
print()
print("Sample [0]:")
print(ds_raw[0])

# %% Run transform_fn

TfmArgs_full = {**TfmArgs, 'max_seq_length': tuner_args['max_seq_length']}
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


# %% [markdown]
# ## Step 5b: Verify Custom Vocabulary Token Ranges
#
# With the custom CLM vocabulary, tokens are partitioned:
#
#     BEFORE (old):  glucose 0-400 -> token 0-400, pad = 151643 (Qwen's pad, OUT OF RANGE!)
#     AFTER  (new):  glucose 10-400 -> token 10-400, pad = 0 (custom PAD token)
#
#     Token 0-9:   special (PAD=0, UNK=1, BOS=2, EOS=3, MASK=4, 5-9 reserved)
#     Token 10-400: CGM glucose values (identity mapping)
#
# Verify: real tokens in [10, 400], pad tokens == 0, labels == -100 at pad positions.

# %% Step 5b: Verify Token Ranges

for i in range(min(5, len(ds_tfm))):
    ids = ds_tfm[i]['input_ids']
    mask = ds_tfm[i]['attention_mask']
    labs = ds_tfm[i]['labels']

    assert len(ids) == seq_len, f"Sample {i}: len={len(ids)}, expected={seq_len}"
    for j in range(seq_len):
        if mask[j] == 1:
            assert 10 <= ids[j] <= 400, \
                f"Sample {i}, pos {j}: real token {ids[j]} outside [10, 400]"
        else:
            assert ids[j] == 0, \
                f"Sample {i}, pos {j}: pad token should be 0, got {ids[j]}"
            assert labs[j] == -100, \
                f"Sample {i}, pos {j}: mask=0 but label={labs[j]}"

all_tokens = [v for row in ds_tfm['input_ids'] for v in row]
real_tokens = [v for row, m in zip(ds_tfm['input_ids'], ds_tfm['attention_mask'])
               for v, flag in zip(row, m) if flag == 1]
pad_tokens = [v for row, m in zip(ds_tfm['input_ids'], ds_tfm['attention_mask'])
              for v, flag in zip(row, m) if flag == 0]
active_counts = [sum(m) for m in ds_tfm['attention_mask']]

display_df(pd.DataFrame([
    {'property': 'real token range',   'value': f'[{min(real_tokens)}, {max(real_tokens)}]' if real_tokens else 'N/A'},
    {'property': 'real token mean',    'value': f'{sum(real_tokens)/len(real_tokens):.1f}' if real_tokens else 'N/A'},
    {'property': 'pad token value',    'value': f'{pad_tokens[0]}' if pad_tokens else 'N/A (no padding)'},
    {'property': 'active tokens (avg)','value': f'{sum(active_counts)/len(active_counts):.1f} / {seq_len}'},
    {'property': 'status',             'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 5c: Convert to Tensors
#
# Prepare batch tensors for the forward pass.

# %% Step 5c: Convert to Tensors

input_ids = torch.tensor(ds_tfm['input_ids'])
attention_mask = torch.tensor(ds_tfm['attention_mask'])
labels = torch.tensor(ds_tfm['labels'])
batch_size = len(ds_raw)

display_df(pd.DataFrame([
    {'property': 'input_ids shape',     'value': str(tuple(input_ids.shape))},
    {'property': 'attention_mask shape', 'value': str(tuple(attention_mask.shape))},
    {'property': 'labels shape',         'value': str(tuple(labels.shape))},
    {'property': 'status',               'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 6: Forward Pass
#
# Feed the prepared data through the model.
# Verify: loss is finite/positive, logits shape matches (batch, seq_len, vocab_size).

# %% Step 6: Forward Pass

model.eval()

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

assert outputs.loss is not None
assert torch.isfinite(outputs.loss), f"Loss is not finite: {outputs.loss.item()}"
assert outputs.loss.item() > 0, f"Loss should be > 0, got {outputs.loss.item()}"

expected_logits_shape = (batch_size, seq_len, vocab_size)
assert outputs.logits.shape == expected_logits_shape, \
    f"Logits shape {outputs.logits.shape} != expected {expected_logits_shape}"

perplexity = torch.exp(outputs.loss).item()

display_df(pd.DataFrame([
    {'property': 'input_ids shape',    'value': str(tuple(input_ids.shape))},
    {'property': 'logits shape',       'value': str(tuple(outputs.logits.shape))},
    {'property': 'expected shape',     'value': str(expected_logits_shape)},
    {'property': 'loss',               'value': f'{outputs.loss.item():.4f}'},
    {'property': 'perplexity',         'value': f'{perplexity:.2f}'},
    {'property': 'loss finite',        'value': torch.isfinite(outputs.loss).item()},
    {'property': 'loss positive',      'value': outputs.loss.item() > 0},
    {'property': 'shape correct',      'value': outputs.logits.shape == expected_logits_shape},
    {'property': 'status',             'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 7a: Forward Pass
#
# Put model in train mode, snapshot parameters, and compute loss.

# %% Step 7a: Forward Pass

model.train()

# Snapshot parameter values before update
param_snapshot = {
    name: p.data.clone()
    for name, p in model.named_parameters()
    if p.requires_grad
}

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Forward
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)
loss = outputs.loss

display_df(pd.DataFrame([
    {'property': 'loss (train mode)', 'value': f'{loss.item():.4f}'},
]).set_index('property'))


# %% [markdown]
# ## Step 7b: Backward + Gradient Check
#
# Run backward pass. Verify gradients are non-None and contain no NaNs.

# %% Step 7b: Backward + Gradient Check

optimizer.zero_grad()
loss.backward()

# Check gradients exist
params_with_grad = sum(
    1 for p in model.parameters()
    if p.requires_grad and p.grad is not None
)
total_trainable = sum(1 for p in model.parameters() if p.requires_grad)

assert params_with_grad > 0, "No parameters received gradients"

# Check no NaN gradients
nan_grads = sum(
    1 for p in model.parameters()
    if p.grad is not None and torch.isnan(p.grad).any()
)
assert nan_grads == 0, f"{nan_grads} parameters have NaN gradients"

display_df(pd.DataFrame([
    {'property': 'total trainable',  'value': total_trainable},
    {'property': 'params with grad', 'value': params_with_grad},
    {'property': 'grad coverage',    'value': f'{params_with_grad}/{total_trainable}'},
    {'property': 'NaN gradients',    'value': nan_grads},
]).set_index('property'))


# %% [markdown]
# ## Step 7c: Optimizer Step + Verify
#
# Step the optimizer and verify parameters actually changed.

# %% Step 7c: Optimizer Step + Verify

optimizer.step()

# Verify parameters changed
params_changed = 0
for name, p in model.named_parameters():
    if p.requires_grad and name in param_snapshot:
        if not torch.equal(p.data, param_snapshot[name]):
            params_changed += 1

assert params_changed > 0, "No parameters changed after optimizer step"

display_df(pd.DataFrame([
    {'property': 'params changed', 'value': params_changed},
    {'property': 'status',         'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 8: HF Native Save/Load Roundtrip
#
# Use `save_pretrained` / `from_pretrained` directly (no TECLMTuner).
# Verify: loaded model has identical config, weights, and forward output.

# %% Step 8: HF Native Save/Load Roundtrip

tmp_dir = tempfile.mkdtemp(prefix='test_te_clm_algo_')

save_path = os.path.join(tmp_dir, 'hf_model')

# Save
model.save_pretrained(save_path)
assert os.path.isdir(save_path)

saved_files = sorted(os.listdir(save_path))

# Load
loaded_model = AutoModelForCausalLM.from_pretrained(save_path, trust_remote_code=True)

# Verify config
assert loaded_model.config.vocab_size == arch['vocab_size']
assert loaded_model.config.hidden_size == arch['hidden_size']
assert loaded_model.config.num_hidden_layers == arch['num_hidden_layers']

# Verify weights
orig_state = model.state_dict()
loaded_state = loaded_model.state_dict()

assert set(orig_state.keys()) == set(loaded_state.keys())

mismatched = [
    k for k in orig_state
    if not torch.allclose(orig_state[k].float(), loaded_state[k].float(), atol=1e-4)
]
assert len(mismatched) == 0, f"Weight mismatch: {mismatched[:5]}"

# Verify forward pass produces same output
loaded_model.eval()
model.eval()

with torch.no_grad():
    out_orig = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out_loaded = loaded_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

loss_match = torch.allclose(out_orig.loss.float(), out_loaded.loss.float(), atol=1e-4)
logits_match = torch.allclose(out_orig.logits.float(), out_loaded.logits.float(), atol=1e-4)

display_df(pd.DataFrame([
    {'property': 'save_path',         'value': save_path},
    {'property': 'saved files',       'value': ', '.join(saved_files)},
    {'property': 'config match',      'value': True},
    {'property': 'state_dict keys',   'value': len(orig_state)},
    {'property': 'weight mismatches', 'value': len(mismatched)},
    {'property': 'loss match',        'value': loss_match},
    {'property': 'logits match',      'value': logits_match},
    {'property': 'status',            'value': 'PASSED'},
]).set_index('property'))

shutil.rmtree(tmp_dir)


# %% [markdown]
# ## Summary

# %% Summary

results = pd.DataFrame([
    {'test': '1. Load Config',            'status': 'PASSED'},
    {'test': '2. Load AIData',            'status': 'PASSED'},
    {'test': '3. Create from AutoConfig', 'status': 'PASSED'},
    {'test': '4. Tokenizer Setup',        'status': 'PASSED'},
    {'test': '5a. Tokenize Real Data',    'status': 'PASSED'},
    {'test': '5b. Verify Token Ranges',   'status': 'PASSED'},
    {'test': '5c. Convert to Tensors',    'status': 'PASSED'},
    {'test': '6. Forward Pass',           'status': 'PASSED'},
    {'test': '7. Gradient Flow',          'status': 'PASSED'},
    {'test': '8. HF Save/Load Roundtrip', 'status': 'PASSED'},
]).set_index('test')

display_df(results)
```

---

See Also
========

- **haipipe-nn-0-overview**: Architecture map and decision guide
- **haipipe-nn-2-tuner**: How to wrap an algorithm into a Tuner
- **haipipe-nn-3-instance**: How Tuners are managed by Instances
- **haipipe-nn-4-modelset**: How Instances are packaged into assets
