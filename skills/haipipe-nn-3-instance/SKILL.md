Skill: haipipe-nn-3-instance
=============================

Layer 3 of the 4-layer NN pipeline: Model Instance.

The orchestrator contract. Instance manages one or more Tuners and
provides HuggingFace-style save_pretrained/from_pretrained API.
It should be a THIN orchestrator: it knows WHAT to do, Tuners know HOW.

---

Architecture Position
=====================

```
Layer 4: ModelSet/Pipeline    Packages everything. Config-driven.
    |
Layer 3: Instance   <---      Thin orchestrator. Manages Tuner dict.
    |                         HuggingFace-style API. Delegates to Tuners.
Layer 2: Tuner                Wraps ONE algorithm.
    |
Layer 1: Algorithm            Raw external library.
```

---

The Contract
============

**File:** code/hainn/model_instance.py

Every Instance MUST provide these 5 methods:

```python
class ModelInstance(ABC):
    MODEL_TYPE = "ModelInstance"         # Override in subclass

    def __init__(self, config=None, SPACE=None):
        self.config = config
        self.SPACE = SPACE or {}
        self.model_base = {}            # Dictionary of Tuner instances
        self.prefn_pipeline = None      # Set by pipeline before saving
        self.model_path = None          # Set after save/load

    @abstractmethod
    def init(self): ...                  # Create Tuner instance(s)
    # NOTE: Subclasses commonly add optional params, e.g., init(self, ModelArgs=None).
    # This is allowed since Python permits adding optional params when overriding.

    @abstractmethod
    def fit(self, Name_to_Data, TrainingArgs=None): ...
    @abstractmethod
    def infer(self, dataset, InferenceArgs=None): ...
    @abstractmethod
    def _save_model_base(self, model_dir): ...
    @abstractmethod
    def _load_model_base(self, model_dir): ...

    # PROVIDED by base class (do NOT override):
    def save_pretrained(self, model_dir): ...    # Saves config + metadata + model + prefn
    @classmethod
    def from_pretrained(cls, model_dir, SPACE): ...  # Loads via metadata auto-detection
```

**5 abstract methods, 1 class attribute (MODEL_TYPE), 3 instance attributes
set in __init__ (model_base, prefn_pipeline, model_path) + 1 set by
from_pretrained (modelinstance_dir = parent of model_dir).**

---

Three Composition Patterns
==========================

**Pattern A: Single Tuner**

One algorithm, one model. The simplest and most common pattern.

```python
# Used by: TSForecast, TEFM (Tuner mode)
model_base = {'MAIN': tuner}

def init(self):
    TunerClass = get_model_tuner_class(ModelArgs['model_tuner_name'])
    self.model_base['MAIN'] = TunerClass(TfmArgs=..., **model_args)

def fit(self, Name_to_Data, TrainingArgs=None):
    self.model_base['MAIN'].fit(Name_to_Data, TrainingArgs)

def infer(self, dataset, InferenceArgs=None):
    return self.model_base['MAIN'].infer(dataset, InferenceArgs)
```

**Pattern B: Single Tuner + Input Encoding**

One algorithm, but the Instance encodes treatment/context as an
additional input feature before passing to the Tuner.

```python
# Used by: S-Learner (treatment as feature)
model_base = {'MAIN': tuner}

def fit(self, Name_to_Data, TrainingArgs=None):
    # Instance adds one-hot treatment encoding to features
    X_combined = combine(X_features, onehot_treatment)
    self.model_base['MAIN'].fit((X_combined, y), TrainingArgs)

def infer(self, dataset, InferenceArgs=None):
    # Counterfactual: predict for EVERY action per sample
    for action in action_to_id:
        X_action = combine(X_features, onehot(action))
        pred = self.model_base['MAIN'].infer(X_action)
```

**Pattern C: Multiple Tuners**

Separate model per variant (treatment arm, task, etc.).

```python
# Used by: T-Learner (one model per treatment arm)
model_base = {'default': tuner1, 'authority': tuner2, 'commitmentPrompt': tuner3}

def init(self):
    for action_name in action_to_id:
        TunerClass = get_tuner_class(ModelArgs['model_tuner_name'])
        self.model_base[action_name] = TunerClass(**model_args)

def fit(self, Name_to_Data, TrainingArgs=None):
    for action_name, tuner in self.model_base.items():
        data_for_action = filter_by_action(Name_to_Data, action_name)
        tuner.fit(data_for_action, TrainingArgs)
```

---

Tuner Registry
==============

Instances use a registry to lazily load Tuner classes by name string.
This avoids importing algorithm libraries until they're needed.

**Style A: Dictionary Registry (preferred for many tuners)**

```python
# At top of instance file
MODEL_TUNER_REGISTRY = {
    'NixtlaPatchTSTTuner':        'hainn.tsforecast.models.neuralforecast.modeling_nixtla_patchtst',
    'NixtlaXGBoostForecastTuner': 'hainn.tsforecast.models.mlforecast.modeling_nixtla_xgboost',
    'NixtlaARIMATuner':           'hainn.tsforecast.models.statsforecast.modeling_nixtla_arima',
    'HFNTPTuner':                 'hainn.tefm.models.hfntp.modeling_hfntp',
    # ... add new tuners here
}

def get_model_tuner_class(model_tuner_name):
    module_path = MODEL_TUNER_REGISTRY[model_tuner_name]
    module = importlib.import_module(module_path)
    return getattr(module, model_tuner_name)
```

**Style B: if/elif Chain (simpler for fewer tuners)**

```python
def init(self, ModelArgs=None):
    model_tuner_name = ModelArgs['model_tuner_name']
    if model_tuner_name == 'XGBoostTuner':
        from .models.modeling_xgboost import XGBoostTuner
        tuner = XGBoostTuner(**args)
    elif model_tuner_name == 'LightGBMTuner':
        from .models.modeling_lightgbm import LightGBMTuner
        tuner = LightGBMTuner(**args)
```

For new models, prefer Style A (dictionary registry).

---

Config Class Contract
=====================

Every Instance needs a matching Config class:

```python
@dataclass
class MyModelConfig(ModelInstanceConfig):     # MUST inherit base
    # Required identification
    aidata_name: str = None

    # Model identification
    modelinstance_set_name: Optional[str] = None
    model_path: Optional[str] = None

    # Auto-extracted by from_aidata_set() -- your Config MUST accept these:
    action_to_id: Optional[Dict[str, int]] = field(default_factory=lambda: {'default': 0})
    num_actions: int = 1
    action_column: str = 'experiment_config'
    InputArgs: Dict[str, Any] = field(default_factory=dict)
    OutputArgs: Dict[str, Any] = field(default_factory=dict)
    SplitArgs: Dict[str, Any] = field(default_factory=dict)

    # MUST have these 4 Dict fields:
    ModelArgs: Dict[str, Any] = field(default_factory=dict)
    TrainingArgs: Dict[str, Any] = field(default_factory=dict)
    InferenceArgs: Dict[str, Any] = field(default_factory=dict)
    EvaluationArgs: Dict[str, Any] = field(default_factory=dict)

    # Non-serializable (excluded from to_dict() automatically):
    aidata_set: Optional[Any] = None
    SPACE: Optional[Dict[str, str]] = None
```

**Config base class** (code/hainn/model_configuration.py) provides:
- to_dict() / from_dict()
- to_json() / from_json()
- from_pretrained(model_dir) -- loads config.json
- from_yaml(yaml_source, SPACE) -- abstract, implement in subclass
- from_aidata_set(aidata_set, modelinstance_name, ...) -- factory method

**from_aidata_set() auto-extracts metadata:**

```python
@classmethod
def from_aidata_set(cls, aidata_set, modelinstance_name,
                    modelinstance_version='v0001',
                    ModelArgs=None, TrainingArgs=None,
                    EvaluationArgs=None, InferenceArgs=None,
                    SPACE=None, **kwargs):
    aidata_name = getattr(aidata_set, 'aidata_name', 'unknown_aidata')
    modelinstance_set_name = f"{modelinstance_name}/@{modelinstance_version}"

    meta_info = getattr(aidata_set, 'meta_info', {})
    action_to_id = meta_info.get('action_to_id', {'default': 0})
    num_actions = len(action_to_id)
    action_column = meta_info.get('action_column', 'experiment_config')

    transform_info = getattr(aidata_set, 'transform_info', {})
    InputArgs = transform_info.get('InputArgs', {})
    OutputArgs = transform_info.get('OutputArgs', {})
    SplitArgs = getattr(aidata_set, 'split_info', {})

    return cls(
        aidata_name=aidata_name,
        action_to_id=action_to_id,
        num_actions=num_actions,
        action_column=action_column,
        modelinstance_set_name=modelinstance_set_name,
        InputArgs=InputArgs,
        OutputArgs=OutputArgs,
        SplitArgs=SplitArgs,
        ModelArgs=ModelArgs or {},
        TrainingArgs=TrainingArgs or {},
        EvaluationArgs=EvaluationArgs or {},
        InferenceArgs=InferenceArgs or {},
        SPACE=SPACE,
        **kwargs
    )
```

**IMPORTANT:** If your Config uses the BASE from_aidata_set(), your @dataclass
MUST accept ALL the fields above as constructor arguments, otherwise it will
crash with unexpected keyword arguments. However, subclasses MAY override
from_aidata_set() entirely (e.g., TSForecastConfig extracts time-series
specific parameters instead of treatment-effect metadata).

**modelinstance_set_name format WARNING:** from_aidata_set uses `/@` prefix
(e.g., "MyModel/@v0001"), but ModelInstance_Pipeline uses `/` without `@`
(e.g., "MyModel/v0001"). This is a known code inconsistency.

---

Model Registry
==============

Every Instance type must be registered in code/hainn/model_registry.py:

```python
def load_model_instance_class(model_instance_type):
    # Pattern: map type string(s) to (InstanceClass, ConfigClass)
    if model_instance_type in ('TSForecast', 'TSForecastInstance', 'DLForecastInstance'):
        from hainn.tsforecast.instance_tsforecast import TSForecastInstance
        from hainn.tsforecast.configuration_tsforecast import TSForecastConfig
        return TSForecastInstance, TSForecastConfig
    # ... more types ...
    else:
        raise ValueError(f"Model type {model_instance_type} not found")
```

**To register a new model:** Add an elif block mapping your MODEL_TYPE
string(s) to (InstanceClass, ConfigClass). See haipipe-nn-0-overview
for the complete registry listing.

---

HuggingFace-Style API (provided by base class)
===============================================

**save_pretrained(model_dir)** saves:

```
model_dir/
  config.json       # config.to_json()
  metadata.json     # {"model_type": "TSForecast", "created_at": "..."}
  model_MAIN/       # _save_model_base() -> tuner.save_model('MAIN', model_dir)
  prefn_*.json      # prefn_pipeline.save_pretrained() if attached
```

**from_pretrained(model_dir, SPACE)** loads (classmethod):

```python
@classmethod
def from_pretrained(cls, model_dir, SPACE):
    # 1. Read metadata.json -> get model_type
    # 2. Get ConfigClass via cls._get_config_class()
    # 3. Load config from config.json
    # 4. Create instance: cls(config=config, SPACE=SPACE)
    # 5. Set instance.modelinstance_dir = parent of model_dir
    # 6. Call instance._load_model_base(model_dir)
    return instance
```

**_get_config_class() resolution logic:**

from_pretrained() calls `cls._get_config_class()` to find the Config class.
Resolution has 3 phases:

Phase 1 -- Generate config_name from class name:
  - If class name contains `Instance`: replace with `Config`
  - Elif contains `Predictor`: replace with `Config`
  - Else: append `Config`

Phase 2 -- Look in the same module as the Instance class:
  - If the module has an attribute matching config_name, return it

Phase 3 -- Hardcoded fallback map:
  - Maps known class names to registry type strings
  - Calls load_model_instance_class() to get ConfigClass
  - If all fail: raises NotImplementedError

**For new models:** Follow the `*Instance` -> `*Config` naming convention.
Phase 1 + 2 will find it automatically. No override needed.

**AutoModelInstance** (like HuggingFace's AutoModel):

```python
from hainn.model_instance import AutoModelInstance
model = AutoModelInstance.from_pretrained('path/to/model', SPACE)
# Reads metadata.json, resolves MODEL_TYPE via registry, loads correct class.
# IMPORTANT: Returns a CONCRETE class instance (e.g., TSForecastInstance),
# NOT an AutoModelInstance. Internally calls:
#   ModelClass, _ = load_model_instance_class(model_type)
#   return ModelClass.from_pretrained(model_dir, SPACE)
```

---

infer() vs inference()
======================

The abstract contract defines `infer()`. The pipeline currently calls
`inference()`. This is a known inconsistency to be unified.

**The standard is `infer()`.** New models implement `infer()`. If the
pipeline calls `inference()`, add a one-line alias:

```python
def inference(self, dataset, InferenceArgs=None):
    return self.infer(dataset, InferenceArgs)
```

**Pipeline also passes NO TrainingArgs to fit():**

```python
# modelinstance_pipeline.py:
model_instance.fit(Name_to_Data)   # No TrainingArgs parameter
```

Models read TrainingArgs from `self.config` when the parameter is None.

---

Concrete Example: Clean Pattern (TSForecastInstance)
====================================================

**File:** code/hainn/tsforecast/instance_tsforecast.py

```python
class TSForecastInstance(ModelInstance):
    MODEL_TYPE = 'TSForecast'

    def __init__(self, config=None, SPACE=None):
        if isinstance(config, dict):
            config = TSForecastConfig.from_dict(config)
        super().__init__(config=config, SPACE=SPACE)
        self.modelinstance_dir = None    # Also set by from_pretrained() in base class

    def init(self, ModelArgs=None):
        ModelArgs = ModelArgs or self.config.ModelArgs
        model_tuner_name = ModelArgs.get('model_tuner_name')
        TunerClass = get_model_tuner_class(model_tuner_name)  # Dict registry lookup
        self.model_base['MAIN'] = TunerClass(TfmArgs=..., **model_tuner_args)
        self.model_tuner = self.model_base['MAIN']  # Convenience reference

    def fit(self, Name_to_Data, TrainingArgs=None):
        # NOTE: TSForecast names this 'dataset' in actual code, but
        # the contract uses 'Name_to_Data'. Either works -- the key is
        # that the pipeline passes a dict of {split_name: dataset}.
        TrainingArgs = TrainingArgs or self.config.TrainingArgs
        self.model_tuner.fit(Name_to_Data, TrainingArgs)  # Delegate to Tuner

    def infer(self, dataset, InferenceArgs=None):
        InferenceArgs = InferenceArgs or self.config.InferenceArgs
        return self.model_tuner.infer(dataset, InferenceArgs)

    def _save_model_base(self, model_dir):
        for key, tuner in self.model_base.items():
            tuner.save_model(key, model_dir)

    def _load_model_base(self, model_dir):
        self.init()                          # Create Tuner shells first
        for key, tuner in self.model_base.items():
            tuner.load_model(key, model_dir)  # Then load weights
```

**Why this is the reference:** Calls super().__init__(), uses model_base dict,
delegates everything to Tuner, no algorithm imports, clean registry.

---

Known Deviations (existing code)
================================

Existing families that don't follow the canonical interface:

- **mlpredictor**: No super().__init__(), uses model_tuner_base instead of
  model_base, inference() instead of infer(), non-standard save/load names.
- **bandit**: No super().__init__(), no config class, different __init__ signature.
- **tefm (direct mode)**: Torch imports in Instance, training loops in Instance,
  _save/_load_model_base are empty stubs.
- **tefm (tuner mode)**: Follows canonical pattern except inference() not infer()
  and _save/_load_model_base are stubs.

These are non-canonical. **New models MUST follow the TSForecast pattern.**
Existing code should be migrated toward the standard over time.

---

MUST DO
=======

1. **Inherit from ModelInstance**
2. **Set MODEL_TYPE class attribute** -- string that goes into metadata.json
3. **Call super().__init__(config=config, SPACE=SPACE)**
4. **Implement 5 abstract methods:**
   init(), fit(), infer(), _save_model_base(), _load_model_base()
5. **Store Tuners in self.model_base dict** --
   keys: 'MAIN' (single) or action/variant names (multi-Tuner)
6. **Be a THIN orchestrator** -- delegate to Tuners, never touch algorithms
7. **Use a Tuner Registry** (dict or if/elif) for lazy-loading by name string
8. **Register in model_registry.py** --
   add mapping: MODEL_TYPE -> (InstanceClass, ConfigClass)
9. **Create matching @dataclass Config class** inheriting ModelInstanceConfig
10. **Config MUST include:** ModelArgs, TrainingArgs, InferenceArgs,
    EvaluationArgs as Dict fields
11. **_load_model_base() must call self.init() first** --
    recreate Tuner shells, then load weights into them
12. **Use `model_tuner_name` as the config key** for selecting the Tuner class
13. **Add `inference()` as alias for `infer()`** until pipeline is unified

---

MUST NOT
========

1. **NEVER import algorithm libraries** (xgboost, torch, sklearn, neuralforecast)
   -- those belong ONLY in Tuner files (Layer 2)
2. **NEVER do data conversion** (sparse matrices, tensors, DataFrames)
   -- Tuner's transform_fn handles this
3. **NEVER implement training loops** (backward pass, optimizer steps, loss computation)
   -- Tuner's fit() handles this
4. **NEVER know about ModelInstance_Set or ModelInstance_Pipeline**
   -- those are Layer 4
5. **NEVER hardcode algorithm-specific parameters** (learning rate, batch size)
   -- pass through via TrainingArgs to the Tuner

---

Key File Locations
==================

```
Base class:             code/hainn/model_instance.py
Config base class:      code/hainn/model_configuration.py
Model registry:         code/hainn/model_registry.py
AutoModelInstance:      code/hainn/model_instance.py (AutoModelInstance class)

TSForecast:
  Instance:             code/hainn/tsforecast/instance_tsforecast.py
  Config:               code/hainn/tsforecast/configuration_tsforecast.py

MLPredictor (S-Learner):
  Instance:             code/hainn/mlpredictor/instance_slearner.py
  Config:               code/hainn/mlpredictor/configuration_slearner.py

MLPredictor (T-Learner):
  Instance:             code/hainn/mlpredictor/instance_tlearner.py
  Config:               code/hainn/mlpredictor/configuration_tlearner.py

TEFM:
  Instance:             code/hainn/tefm/instance_tefm.py
  Config:               code/hainn/tefm/configuration_tefm.py

Bandit:
  Instance:             code/hainn/bandit/instance_bandit.py

TSDecoder:  *** NON-FUNCTIONAL: files do not exist on disk ***
  Instance:             code/hainn/tsfm/tsdecoder/instance_decoder.py
  Config:               code/hainn/tsfm/tsdecoder/configuration_decoder.py
```

---

Test Notebook: What Layer 3 Tests
==================================

The instance test exercises the Instance orchestrator with a real Tuner.
It verifies config -> init -> fit -> inference -> save/load roundtrip.

**Expected steps:**

```
Step 1: Load config (display_df with instance identity, tuner_name)
Step 2: Load AIData (print(aidata), print(sample))
Step 3: Create Instance + init():
        Create config, instantiate, call init()
        display_df with instance class, MODEL_TYPE, tuner class, model_base keys
Step 4: Prepare data:
        print(data_fit) and print(data_infer) -- show each split's dataset:
            print("--- data_fit ---")
            for k, v in data_fit.items():
                print(f"  {k}: {v}")
        display_df with split summary
Step 5: Fit:
        print(data_fit) before calling
        -> instance.fit(data_fit, TrainingArgs) ->
        display_df with training time, loss, params
Step 6: Inference:
        print(data_infer) before calling
        -> instance.inference(data_infer) ->
        print(results) after -- show keys, loss, shapes:
            for k, v in results.items():
                print(f"  {k}: keys={list(v.keys())}, loss={v['loss']:.4f}")
        display_df with loss, perplexity, shapes
Step 7: save_pretrained / from_pretrained roundtrip:
        Verify config.json, metadata.json, model_MAIN/, weights
        display_df with saved items, loaded identity, weight match
Summary: display_df with all steps PASSED/FAILED
```

**Key display rules for Layer 3:**

- Step 3 is unique to Instance: it tests the init() -> registry -> Tuner
  creation chain. Show the model_base dict keys and tuner class name.
- Steps 5-6 delegate to Tuner, so the display is similar to Layer 2
  but through the Instance API (instance.fit, instance.inference)
- Step 7 tests HuggingFace-style save/load. Print the saved directory
  structure and verify config/metadata/weights roundtrip.

**Reference:** `code/hainn/tefm/models/te_clm/test-modeling-te_clm/scripts/test_te_clm_3_instance.py`

**Snapshot** (from te_clm, for quick reference -- canonical source is the file above):

```python
#!/usr/bin/env python3
"""
TE-CLM Instance (Layer 3) Step-by-Step Test
============================================

Test TEFMInstance with TECLMTuner -- the Layer 3 Instance that manages
the Layer 2 Tuner.

Following the config -> data -> model pipeline pattern:

  Step 1: Setup workspace and load config
  Step 2: Load AIData (real OhioT1DM)
  Step 3: Create TEFMInstance + init() (creates TECLMTuner via registry)
  Step 4: Prepare data (real subsets)
  Step 5: Fit (instance.fit delegates to model_tuner.fit)
  Step 6: Inference (instance.inference delegates to model_tuner.infer)
  Step 7: save_pretrained / from_pretrained roundtrip

Usage:
    source .venv/bin/activate && source env.sh
    python code/hainn/tefm/models/te_clm/test-modeling-te_clm/scripts/test_te_clm_3_instance.py
"""

# %% [markdown]
# # TE-CLM Instance (Layer 3) Step-by-Step Test
#
# Tests TEFMInstance orchestration with TECLMTuner,
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
    {'key': 'WORKSPACE_PATH',     'value': WORKSPACE_PATH},
    {'key': 'TEST_DIR',           'value': str(TEST_DIR)},
    {'key': 'model_dir',          'value': model_dir},
    {'key': 'aidata_name',        'value': config.get('aidata_name')},
    {'key': 'aidata_version',     'value': config.get('aidata_version')},
    {'key': 'modelinstance_name', 'value': config.get('modelinstance_name')},
    {'key': 'modelinstance_version','value': config.get('modelinstance_version')},
    {'key': 'ModelInstanceClass',  'value': config['ModelInstanceClass']},
    {'key': 'model_tuner_name',   'value': ModelArgs['model_tuner_name']},
    {'key': 'vocab_size',         'value': arch['vocab_size']},
    {'key': 'max_seq_length',     'value': tuner_args['max_seq_length']},
    {'key': 'special_tokens',     'value': f"PAD={TfmArgs.get('special_tokens', {}).get('pad_token_id', 0)}, num_special={TfmArgs.get('special_tokens', {}).get('num_special_tokens', 10)}"},
    {'key': 'CUDA available',     'value': torch.cuda.is_available()},
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
# ## Step 3: Create TEFMInstance + init()
#
# Create a TEFMConfig from the YAML, instantiate TEFMInstance,
# call init() which routes to MODEL_TUNER_REGISTRY and creates
# a TECLMTuner stored in model_base['MAIN'].

# %% Step 3: Create TEFMInstance + init()

from hainn.tefm.instance_tefm import TEFMInstance
from hainn.tefm.configuration_tefm import TEFMConfig

tefm_config = TEFMConfig(
    aidata_name=config.get('aidata_name'),
    aidata_version=config.get('aidata_version'),
    modelinstance_name=config.get('modelinstance_name'),
    modelinstance_version=config.get('modelinstance_version'),
    ModelArgs=config['ModelArgs'],
    TrainingArgs=config['TrainingArgs'],
    InferenceArgs=config.get('InferenceArgs', {}),
    EvaluationArgs=config.get('EvaluationArgs', {}),
)

# Verify config fields propagated
assert tefm_config.aidata_name == config['aidata_name']
assert tefm_config.modelinstance_name == config['modelinstance_name']
assert tefm_config.ModelArgs['model_tuner_name'] == 'TECLMTuner'

instance = TEFMInstance(config=tefm_config, SPACE=SPACE)

assert instance is not None
assert instance.model_base == {}

# init() should create TECLMTuner via registry
instance.init()

assert 'MAIN' in instance.model_base
assert hasattr(instance, 'model_tuner')
assert instance.model_tuner is not None

tuner = instance.model_base['MAIN']
tuner_class_name = type(tuner).__name__

display_df(pd.DataFrame([
    {'property': 'instance class',    'value': type(instance).__name__},
    {'property': 'MODEL_TYPE',        'value': instance.MODEL_TYPE},
    {'property': 'config class',      'value': type(tefm_config).__name__},
    {'property': 'aidata_name',       'value': tefm_config.aidata_name},
    {'property': 'modelinstance_name','value': tefm_config.modelinstance_name},
    {'property': 'model_base keys',   'value': ', '.join(instance.model_base.keys())},
    {'property': 'tuner class',       'value': tuner_class_name},
    {'property': 'tuner domain',      'value': tuner.domain_format},
    {'property': 'tuner model (pre-fit)', 'value': tuner.model is None},
    {'property': 'status',            'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 4: Prepare Data
#
# Use real data subsets for fit and inference.

# %% Step 4: Prepare Data

seq_len = tuner_args['max_seq_length']
vocab_size = arch['vocab_size']

train_ds = aidata.dataset_dict['train'].select(
    range(min(100, len(aidata.dataset_dict['train']))))
test_split_name = 'test-id' if 'test-id' in aidata.dataset_dict else list(aidata.dataset_dict.keys())[-1]
test_ds = aidata.dataset_dict[test_split_name].select(
    range(min(50, len(aidata.dataset_dict[test_split_name]))))

# TEFMInstance.fit() expects data dict: {split_name: dataset}
data_fit = {'train': train_ds, test_split_name: test_ds}
# Use 'test' as key for inference -- must match InferenceArgs.InferenceSetNames
data_infer = {'test': test_ds}

print("--- data_fit ---")
for k, v in data_fit.items():
    print(f"  {k}: {v}")
print()
print("--- data_infer ---")
for k, v in data_infer.items():
    print(f"  {k}: {v}")

display_df(pd.DataFrame([
    {'property': 'fit splits',  'value': ', '.join(f'{k}({len(v)})' for k, v in data_fit.items())},
    {'property': 'infer splits','value': ', '.join(f'{k}({len(v)})' for k, v in data_infer.items())},
    {'property': 'status',      'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 5: Fit
#
# Call instance.fit() which delegates to model_tuner.fit().
# TECLMTuner creates an HF Trainer and trains for 1 epoch.

# %% Step 5: Fit

import time

training_override = {
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 8,
    'save_strategy': 'no',
    'logging_steps': 5,
    'report_to': [],
    'output_dir': os.path.join(model_dir, 'train_checkpoints'),
    'gradient_checkpointing': False,
    'fp16': False,
}

print("--- fit input ---")
for k, v in data_fit.items():
    print(f"  {k}: {v}")

t_start = time.time()
instance.fit(data_fit, TrainingArgs=training_override)
t_elapsed = time.time() - t_start

# After fit, the tuner should have model and trainer
assert instance.model_tuner.model is not None
assert instance.model_tuner.trainer is not None

total_params = sum(p.numel() for p in instance.model_tuner.model.parameters())
final_log = instance.model_tuner.trainer.state.log_history[-1] if instance.model_tuner.trainer.state.log_history else {}
final_loss = final_log.get('train_loss', final_log.get('loss', 'N/A'))

display_df(pd.DataFrame([
    {'property': 'fit splits',      'value': ', '.join(f'{k}({len(v)})' for k, v in data_fit.items())},
    {'property': 'training time',   'value': f'{t_elapsed:.1f}s'},
    {'property': 'final loss',      'value': str(final_loss)},
    {'property': 'tuner has model', 'value': instance.model_tuner.model is not None},
    {'property': 'tuner has trainer','value': instance.model_tuner.trainer is not None},
    {'property': 'total params',    'value': f'{total_params:,}'},
    {'property': 'status',          'value': 'PASSED'},
]).set_index('property'))

shutil.rmtree(training_override['output_dir'], ignore_errors=True)  # cleanup checkpoints


# %% [markdown]
# ## Step 6: Inference
#
# Call instance.inference() which routes to model_tuner.infer().
# Returns per-split loss, embeddings, and predicted_tokens.

# %% Step 6: Inference

import math

print("--- inference input ---")
for k, v in data_infer.items():
    print(f"  {k}: {v}")

results = instance.inference(data_infer)

print()
print("--- inference output ---")
for k, v in results.items():
    print(f"  {k}: keys={list(v.keys())}, loss={v['loss']:.4f}, embeddings={v['embeddings'].shape}")

assert results is not None
assert isinstance(results, dict)

# Results should have the test split key with loss, embeddings, predicted_tokens
# Inference uses InferenceSetNames from config (default: ['test'])
infer_key = 'test'
assert infer_key in results, f"Missing '{infer_key}' in results, got: {list(results.keys())}"

test_results = results[infer_key]
assert 'loss' in test_results
assert 'embeddings' in test_results
assert 'predicted_tokens' in test_results
assert test_results['loss'] > 0

n_test = len(data_infer[infer_key])
assert test_results['embeddings'].shape[0] == n_test
assert test_results['predicted_tokens'].shape[0] == n_test
assert test_results['predicted_tokens'].shape[1] == seq_len

perplexity = math.exp(test_results['loss'])
hidden_size = test_results['embeddings'].shape[1]

display_df(pd.DataFrame([
    {'property': 'inference splits',       'value': ', '.join(results.keys())},
    {'property': 'test loss',              'value': f'{test_results["loss"]:.4f}'},
    {'property': 'test perplexity',        'value': f'{perplexity:.2f}'},
    {'property': 'embeddings shape',       'value': str(test_results['embeddings'].shape)},
    {'property': 'predicted_tokens shape', 'value': str(test_results['predicted_tokens'].shape)},
    {'property': 'hidden_size',            'value': hidden_size},
    {'property': 'status',                 'value': 'PASSED'},
]).set_index('property'))


# %% [markdown]
# ## Step 7: save_pretrained / from_pretrained Roundtrip
#
# Save the TEFMInstance via save_pretrained (HF-style API).
# Load it back via TEFMInstance.from_pretrained.
# Verify: config, model weights, and architecture match.

# %% Step 7: save_pretrained / from_pretrained Roundtrip

save_path = os.path.join(model_dir, 'model')

# Save
instance.save_pretrained(save_path)
assert os.path.isdir(save_path)
assert os.path.exists(os.path.join(save_path, 'config.json'))
assert os.path.exists(os.path.join(save_path, 'metadata.json'))

saved_items = sorted(os.listdir(save_path))

# -- Verify metadata.json (Layer 3 identity) --
with open(os.path.join(save_path, 'metadata.json')) as f:
    metadata = json.load(f)
assert metadata['model_type'] == 'TEFM'
assert 'created_at' in metadata

# -- Verify config.json (Layer 3 full config roundtrip) --
with open(os.path.join(save_path, 'config.json')) as f:
    saved_cfg = json.load(f)

# Config must preserve aidata and modelinstance identification
assert saved_cfg.get('aidata_name') == config['aidata_name']
assert saved_cfg.get('aidata_version') == config['aidata_version']
assert saved_cfg.get('modelinstance_name') == config['modelinstance_name']
assert saved_cfg.get('modelinstance_version') == config['modelinstance_version']

# Config must preserve ModelArgs (Layer 2+1 config)
assert 'ModelArgs' in saved_cfg
assert saved_cfg['ModelArgs']['model_tuner_name'] == 'TECLMTuner'
assert 'model_tuner_args' in saved_cfg['ModelArgs']
assert saved_cfg['ModelArgs']['model_tuner_args']['model_name_or_path'] == 'Qwen/Qwen2.5-0.5B-Instruct'
assert saved_cfg['ModelArgs']['model_tuner_args']['architecture_config']['vocab_size'] == 401
assert 'TfmArgs' in saved_cfg['ModelArgs']

# Config must preserve TrainingArgs, InferenceArgs, EvaluationArgs
assert 'TrainingArgs' in saved_cfg
assert 'InferenceArgs' in saved_cfg
assert 'EvaluationArgs' in saved_cfg

# -- Verify model_MAIN/ subfolder (Layer 2 tuner save) --
model_main_dir = os.path.join(save_path, 'model_MAIN')
assert os.path.isdir(model_main_dir)
assert os.path.exists(os.path.join(model_main_dir, 'tuner_config.json'))

with open(os.path.join(model_main_dir, 'tuner_config.json')) as f:
    tuner_cfg = json.load(f)
assert tuner_cfg['architecture_config']['vocab_size'] == 401
assert tuner_cfg['model_name_or_path'] == 'Qwen/Qwen2.5-0.5B-Instruct'

# -- Load and verify --
loaded_instance = TEFMInstance.from_pretrained(save_path, SPACE)
assert loaded_instance is not None
assert loaded_instance.MODEL_TYPE == 'TEFM'

# Verify loaded config preserved identification fields
assert loaded_instance.config.aidata_name == config['aidata_name']
assert loaded_instance.config.modelinstance_name == config['modelinstance_name']
assert loaded_instance.config.ModelArgs['model_tuner_name'] == 'TECLMTuner'

# Verify the loaded instance has TECLMTuner
assert hasattr(loaded_instance, 'model_tuner')
assert type(loaded_instance.model_tuner).__name__ == 'TECLMTuner'

# Compare weights (move to CPU for comparison)
orig_state = {k: v.cpu().float() for k, v in instance.model_tuner.model.state_dict().items()}
loaded_state = {k: v.cpu().float() for k, v in loaded_instance.model_tuner.model.state_dict().items()}
assert set(orig_state.keys()) == set(loaded_state.keys())

mismatched = [
    k for k in orig_state
    if not torch.allclose(orig_state[k], loaded_state[k], atol=1e-4)
]
assert len(mismatched) == 0, f"Weight mismatch: {mismatched[:5]}"

display_df(pd.DataFrame([
    {'property': 'save_path',              'value': save_path},
    {'property': 'saved items',            'value': ', '.join(saved_items)},
    {'property': 'metadata model_type',    'value': metadata['model_type']},
    {'property': 'config aidata_name',     'value': saved_cfg.get('aidata_name')},
    {'property': 'config modelinstance',   'value': saved_cfg.get('modelinstance_name')},
    {'property': 'config model_tuner',     'value': saved_cfg['ModelArgs']['model_tuner_name']},
    {'property': 'tuner_config arch',      'value': str(tuner_cfg['architecture_config']['vocab_size'])},
    {'property': 'loaded aidata_name',     'value': loaded_instance.config.aidata_name},
    {'property': 'loaded modelinstance',   'value': loaded_instance.config.modelinstance_name},
    {'property': 'loaded tuner class',     'value': type(loaded_instance.model_tuner).__name__},
    {'property': 'state_dict keys',        'value': len(orig_state)},
    {'property': 'weight mismatches',      'value': len(mismatched)},
    {'property': 'status',                 'value': 'PASSED'},
]).set_index('property'))

shutil.rmtree(save_path)


# %% [markdown]
# ## Summary

# %% Summary

results_summary = pd.DataFrame([
    {'test': '1. Load Config',             'status': 'PASSED'},
    {'test': '2. Load AIData',             'status': 'PASSED'},
    {'test': '3. Create TEFMInstance',     'status': 'PASSED'},
    {'test': '4. Prepare Data',            'status': 'PASSED'},
    {'test': '5. Fit',                     'status': 'PASSED'},
    {'test': '6. Inference',               'status': 'PASSED'},
    {'test': '7. save/load Roundtrip',     'status': 'PASSED'},
]).set_index('test')

display_df(results_summary)

# Cleanup model_dir
shutil.rmtree(model_dir, ignore_errors=True)
```

---

See Also
========

- **haipipe-nn-0-overview**: Architecture map, decision guide, YAML templates
- **haipipe-nn-1-algorithm**: What algorithms are and their diversity
- **haipipe-nn-2-tuner**: The universal Tuner contract (5 abstract methods)
- **haipipe-nn-4-modelset**: Packaging, pipeline flow, versioning
