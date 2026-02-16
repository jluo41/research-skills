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

See Also
========

- **haipipe-nn-0-overview**: Architecture map, decision guide, YAML templates
- **haipipe-nn-1-algorithm**: What algorithms are and their diversity
- **haipipe-nn-2-tuner**: The universal Tuner contract (5 abstract methods)
- **haipipe-nn-4-modelset**: Packaging, pipeline flow, versioning
