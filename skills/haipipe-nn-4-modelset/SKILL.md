Skill: haipipe-nn-4-modelset
=============================

Layer 4 of the 4-layer NN pipeline: ModelSet & Pipeline.

Packages a trained Instance into a complete asset with training results,
evaluation, examples, and lineage tracking. Provides run versioning,
remote sync, and YAML-driven configuration. Model-type agnostic.

---

Architecture Position
=====================

```
Layer 4: ModelSet/Pipeline  <---  Packages everything. Asset with
    |                             save/load/remote. Run versioning.
Layer 3: Instance                 Config-driven. Model-type agnostic.
    |
Layer 2: Tuner                    Wraps ONE algorithm.
    |
Layer 1: Algorithm                Raw external library.
```

---

Two Classes at Layer 4
======================

**ModelInstance_Pipeline** -- Orchestrator (creates the asset)

```
__init__:
  - Training mode: Resolve ModelInstanceClass + ConfigClass from registry
  - Inference mode: ModelInstance_Set.load_asset(path, SPACE) → load existing

Training flow (_run_training):
  1. ConfigClass.from_aidata_set() → create config with auto-extracted metadata
  2. ModelInstanceClass(config, SPACE) → create instance
  3. model_instance.init() → create Tuner shells
  4. model_instance.fit(Name_to_Data) → train
  5. model_instance.evaluate(Name_to_Data, EvaluationArgs) → evaluate (if available)
  6. Collect training results (timestamps, model stats)
  7. PreFnPipeline.from_aidata_set() → create and attach feature pipeline
  8. ExampleFn → generate test examples (optional)
  9. Package into ModelInstance_Set → RETURN (caller saves to disk)

Inference flow (_run_inference):
  1. model_instance = self.modelinstance_set.get_model_instance()
  2. model_instance.inference(test_data, InferenceArgs) → predictions
     NOTE: Pipeline currently calls inference(), not infer(). The standard
     is infer(). New models should implement infer() and alias inference().

Evaluate flow (_run_evaluation):
  1. Load existing model from ModelInstance_Set
  2. Pipeline wraps evaluate in _evaluate_model() which:
     a. Converts aidata_set.dataset_dict → Name_to_Data dict
     b. Calls model_instance.evaluate(Name_to_Data, EvaluationArgs)
     c. Returns evaluation_results
```

**IMPORTANT:** _run_training() RETURNS the ModelInstance_Set without
calling save_to_disk(). The caller must save explicitly:

```python
modelinstance_set = pipeline.run(aidata_set, mode='fit', ...)
modelinstance_set.save_to_disk()  # Caller's responsibility
```

**ModelInstance_Set** -- Asset (the saved package)

```
Inherits from Asset to get the framework:
  - save_to_disk() / load_from_disk() / load_asset()
  - push_to_remote() with S3/GCS/Databricks/GDrive support
  - Manifest handling with lineage tracking
  - Cache management

ModelInstance_Set implements 5 Asset abstract methods:
  - _get_set_name()              → returns modelinstance_set_name
  - _get_default_store_key()     → returns 'LOCAL_MODELINSTANCE_STORE'
  - _save_data_to_disk(path)     → writes model/ + training/ + evaluation/ + examples/
  - _load_data_from_disk(path, manifest) → reads them back
  - _generate_custom_manifest()  → adds lineage chain to manifest.json
Also overrides save_to_disk() for run versioning (@run-v000X).
```

---

YAML Configuration
==================

The YAML config drives ModelInstance_Pipeline. This is the primary
user-facing interface for training models.

**Required top-level keys:**

```yaml
ModelInstanceClass: "<type string from model registry>"
ModelArgs:
  model_tuner_name: "<tuner class name>"
  # ... tuner-specific args
TrainingArgs:
  TrainSetNames: ["train"]
  # ... training hyperparameters
```

**Optional keys:**

```yaml
EvaluationArgs:
  EvalSetNames: ["test"]
  metrics: ["mae", "rmse"]
InferenceArgs:
  InferSetNames: ["test"]
ExampleConfig:
  enabled: false
  ExampleFnName: "DiverseActionCoverage"
  num_examples: 10
```

**See haipipe-nn-0-overview for complete YAML templates per model type.**

---

ModelInstance_Pipeline
=====================

**File:** code/haipipe/model_base/modelinstance_pipeline.py

```python
class ModelInstance_Pipeline:
    def __init__(self, config: Dict, SPACE: Dict):
        # Case 1: Inference mode (load existing)
        if config.get('modelinstance_set_path'):
            self.modelinstance_set = ModelInstance_Set.load_asset(path, SPACE)
            self.mode = 'inference'
            return

        # Case 2: Training mode (create new)
        # Registry resolution happens HERE in __init__, not in _run_training.
        ModelInstanceClass = config['ModelInstanceClass']
        if isinstance(ModelInstanceClass, str):
            self.ModelInstanceClass, self.ConfigClass = load_model_instance_class(
                ModelInstanceClass
            )
        else:
            self.ModelInstanceClass = ModelInstanceClass
            self.ConfigClass = ModelInstanceClass._get_config_class()
        self.model_args = config.get('ModelArgs', {})
        self.training_args = config.get('TrainingArgs', {})
        self.evaluation_args = config.get('EvaluationArgs', {})
        self.inference_args = config.get('InferenceArgs', {})
        self.example_config = config.get('ExampleConfig', {})

        # Load ExampleFn if enabled
        if self.example_config.get('enabled', False):
            from haipipe.model_base.builder.examplefn import ExampleFn
            example_fn_name = self.example_config.get('ExampleFnName', 'DiverseActionCoverage')
            self.example_fn = ExampleFn(example_fn_name, SPACE)
        else:
            self.example_fn = None

        self.mode = 'training'

    def run(self, aidata_set, mode='fit',
            modelinstance_name=None, modelinstance_version='v0001'):
        if mode in ['fit', 'train']:
            return self._run_training(aidata_set, modelinstance_name, modelinstance_version)
        elif mode == 'inference':
            return self._run_inference(aidata_set)
        elif mode == 'evaluate':
            return self._run_evaluation(aidata_set)
```

**Training flow (_run_training) -- actual code:**

```python
def _run_training(self, aidata_set, modelinstance_name, modelinstance_version):
    # NOTE: self.ModelInstanceClass and self.ConfigClass were already resolved
    # in __init__ via load_model_instance_class(). _run_training uses them directly.

    # 1. Create config from aidata_set (metadata auto-extracted)
    config = self.ConfigClass.from_aidata_set(
        aidata_set=aidata_set,
        modelinstance_name=modelinstance_name,
        modelinstance_version=modelinstance_version,
        ModelArgs=self.model_args,
        TrainingArgs=self.training_args,
        EvaluationArgs=self.evaluation_args,
        InferenceArgs=self.inference_args,
        SPACE=self.SPACE
    )

    # 2. Create model instance
    model_instance = self.ModelInstanceClass(config=config, SPACE=self.SPACE)

    # 3. Initialize model architecture
    model_instance.init()

    # 4. Train -- NOTE: Pipeline passes NO TrainingArgs parameter.
    # Models read TrainingArgs from self.config instead.
    Name_to_Data = {split: ds for split, ds in aidata_set.dataset_dict.items()}
    model_instance.fit(Name_to_Data)

    # 5. Evaluate (if method exists and EvaluationArgs provided)
    # Uses _evaluate_model() wrapper which converts dataset_dict → Name_to_Data
    evaluation_results = None
    if hasattr(model_instance, 'evaluate') and config.EvaluationArgs:
        evaluation_results = self._evaluate_model(model_instance, aidata_set)
        # Internally calls: model_instance.evaluate(Name_to_Data, config.EvaluationArgs)

    # 6. Collect training results
    training_results = {
        'trained_at': datetime.now().isoformat(),
        'model_type': model_instance.__class__.__name__,
    }

    # 7. Create + attach PreFnPipeline
    prefn_pipeline = PreFnPipeline.from_aidata_set(aidata_set, SPACE=self.SPACE)
    model_instance.prefn_pipeline = prefn_pipeline

    # 8. Generate examples (optional)
    examples_data = None
    if self.example_fn:
        examples_data = self._generate_examples(model_instance, aidata_set, ...)

    # 9. Package into ModelInstance_Set
    # NOTE: Pipeline uses "/" (no @), while ConfigClass.from_aidata_set uses "/@".
    # This is a known code inconsistency -- the pipeline format is what gets saved.
    modelinstance_set_name = f"{modelinstance_name}/{modelinstance_version}"
    self.modelinstance_set = ModelInstance_Set(
        modelinstance_set_name=modelinstance_set_name,
        model_instance=model_instance,
        aidata_set_manifest=aidata_set.manifest,
        training_results=training_results,
        evaluation_results=evaluation_results,
        examples_data=examples_data,
        SPACE=self.SPACE
    )
    return self.modelinstance_set  # Caller calls save_to_disk()
```

---

ModelInstance_Set
=================

**File:** code/haipipe/model_base/modelinstance_set.py

```python
class ModelInstance_Set(Asset):
    asset_store_type = 'MODELINSTANCE'

    def __init__(self, modelinstance_set_name=None, model_instance=None,
                 aidata_set_manifest=None, training_results=None,
                 evaluation_results=None, inference_results=None,
                 examples_data=None, SPACE=None):
        super().__init__(SPACE)
        self.modelinstance_set_name = modelinstance_set_name
        self.model_instance = model_instance
        self.aidata_set_manifest = aidata_set_manifest or {}
        self.training_results = training_results or {}
        self.evaluation_results = evaluation_results or {}
        self.inference_results = inference_results or {}
        self.examples_data = examples_data or []

    # Asset abstract methods:
    def _get_set_name(self): ...
    def _get_default_store_key(self):
        return 'LOCAL_MODELINSTANCE_STORE'
    def _save_data_to_disk(self, path): ...
    def _load_data_from_disk(self, path, manifest): ...
    def _generate_custom_manifest(self): ...

    # ModelInstance_Set specific:
    def get_model_instance(self): ...
```

**Two API levels:**

```
ModelInstance_Set API (our Asset):
  save_to_disk()          → saves model/ + training/ + evaluation/ + manifest.json
  load_asset(name, SPACE) → loads everything via manifest auto-detection
  push_to_remote()        → syncs to S3/GCS/GDrive

model_instance API (HuggingFace-style, nested inside):
  save_pretrained(model_dir)        → saves config.json + metadata.json + weights + prefn
  from_pretrained(model_dir, SPACE) → loads via metadata.json auto-detection
```

---

Directory Structure
===================

```
Demo_AIData_SLearner_20251217/              # ModelInstance_Set root
├── @run-v0001/                             # Archived run 1
├── @run-v0002/                             # Archived run 2
├── model/                                  # Latest model (HuggingFace-style)
│   ├── config.json                         # Model config (architecture + settings)
│   ├── metadata.json                       # {"model_type": "...", "created_at": "..."}
│   ├── model_MAIN/                         # Tuner weights (subfolder per key)
│   ├── prefn_config.json                   # InputArgs, OutputArgs, TriggerArgs
│   ├── prefn_cf_to_cfvocab.json           # Per-CaseFn vocabulary
│   ├── prefn_feat_vocab.json              # Combined vocabulary with offsets
│   ├── checkpoint_best/                    # Best checkpoint during training
│   └── checkpoint_final/                   # Final checkpoint
├── training/                               # Training results
│   └── results.json
├── evaluation/                             # Evaluation results
│   ├── df_case_eval.pkl
│   └── df_report_*.pkl
├── inference_results.pkl                   # Inference output (when run)
├── examples/                               # Test examples (optional)
│   ├── example_000/
│   │   ├── example_info.json
│   │   ├── prediction_results.json
│   │   ├── df_case_example.json
│   │   └── ProcName_to_ProcDf/
│   └── example_001/
└── manifest.json                           # Lineage chain
```

---

Run Versioning
==============

```
First run:   Files saved directly to root (no @run- folder)
Second run:  Current root → @run-v0001, new files → @run-v0002, copied to root
Third run:   New files → @run-v0003, copied to root
...
Root contains the latest run (copied, not symlinked).
NOTE: Only model/, training/, evaluation/, manifest.json, and inference_results.pkl
are copied to root. examples/ is NOT copied (only in @run-vXXXX).
```

Two modes controlled by overwrite parameter:

- **overwrite=True** (default): Deletes existing, saves fresh
- **overwrite=False**: Creates @run-v000X subdirectories for version history

---

PreFnPipeline (Feature Pipeline Attachment)
============================================

Created from AIDataSet and attached to model_instance before saving.
Ensures the feature engineering pipeline travels with the model.

```python
# In ModelInstance_Pipeline._run_training():
prefn_pipeline = PreFnPipeline.from_aidata_set(aidata_set, SPACE=SPACE)
model_instance.prefn_pipeline = prefn_pipeline
```

When model_instance.save_pretrained() is called, it saves:

```
model/
  prefn_config.json           # InputArgs, OutputArgs, TriggerArgs, MetaArgs,
                              # ProcName_to_columns, and other pipeline metadata
  prefn_cf_to_cfvocab.json   # Per-CaseFn vocabulary
  prefn_feat_vocab.json      # Combined vocabulary with offsets
```

---

Manifest and Lineage
=====================

Every ModelInstance_Set saves manifest.json with full lineage chain:

```json
{
    "modelinstance_set_name": "MyModel/v0001",
    "model_type": "TSForecast",
    "aidata_set_manifest": {
        "aidata_set_name": "Demo_AIData",
        "case_set_manifest_list": [
            {
                "case_set_name": "Demo_CaseSet",
                "record_set_manifest": {
                    "record_set_name": "Demo_RecordSet",
                    "source_set_manifest": {
                        "source_set_name": "WellDoc2022CGM",
                        "source_fn": "SMSParquetV250211"
                    }
                }
            }
        ]
    },
    "created_at": "2025-12-17T14:30:00"
}
```

This allows tracing: model -> aidata -> caseset -> recordset -> source.

---

Auto-Detection on Load
======================

Loading uses metadata.json to auto-detect the correct class:

```python
# Via ModelInstance_Set (Asset API)
modelinstance_set = ModelInstance_Set.load_asset('MyModel/v0001', SPACE)
model = modelinstance_set.get_model_instance()

# Via AutoModelInstance (HuggingFace API)
from hainn.model_instance import AutoModelInstance
model = AutoModelInstance.from_pretrained('path/to/model', SPACE)

# Internal flow:
# 1. Read metadata.json -> {"model_type": "TSForecast"}
# 2. load_model_instance_class("TSForecast")
#    -> (TSForecastInstance, TSForecastConfig)
# 3. TSForecastInstance.from_pretrained(model_dir, SPACE)
```

---

Example Generation
==================

Optional step that creates test examples for PreFn validation:

```yaml
ExampleConfig:
  enabled: true
  ExampleFnName: 'DiverseActionCoverage'
  num_examples: 10
  lookback_days: 90
  lookforward_days: 30
  validate: true
```

**Flow:**

1. ExampleFn selects representative test cases from evaluation results
2. Traces lineage: AIDataSet -> CaseSet -> RecordSet -> SourceSet
3. Extracts raw ProcName_to_ProcDf from SourceSet
4. Validates PreFn pipeline: raw data -> features -> predictions
5. Saves to examples/ directory in ModelInstance_Set

---

Usage Patterns
==============

**Training a new model:**

```python
pipeline = ModelInstance_Pipeline(
    config={
        'ModelInstanceClass': 'TSForecast',
        'ModelArgs': {'model_tuner_name': 'NixtlaPatchTSTTuner', ...},
        'TrainingArgs': {'max_steps': 100, 'TrainSetNames': ['train']},
        'EvaluationArgs': {'metrics': ['mae', 'rmse']},
    },
    SPACE=SPACE
)
modelinstance_set = pipeline.run(
    aidata_set, mode='fit',
    modelinstance_name='MyModel', modelinstance_version='v0001'
)
modelinstance_set.save_to_disk()  # Explicit save -- pipeline does NOT auto-save
```

**Loading for inference:**

```python
pipeline = ModelInstance_Pipeline(
    config={'modelinstance_set_path': 'MyModel/v0001'},
    SPACE=SPACE
)
results = pipeline.run(aidata_set, mode='inference')
```

**Direct Asset loading:**

```python
modelinstance_set = ModelInstance_Set.load_asset('MyModel/v0001', SPACE)
model = modelinstance_set.get_model_instance()
predictions = model.infer(test_data)
# Standard is infer(). Pipeline currently calls inference() -- to be unified.
```

**Remote sync:**

```python
# push_to_remote() auto-resolves paths from SPACE if remote_path not given.
modelinstance_set.push_to_remote()

# Or specify a custom remote path:
modelinstance_set.push_to_remote(remote_path='gdrive:models/MyModel/v0001')
```

---

MUST DO
=======

1. **Inherit ModelInstance_Set from Asset** --
   gets save_to_disk/load_from_disk/push_to_remote for free
2. **Set asset_store_type = 'MODELINSTANCE'** --
   used by Asset for default store key resolution
3. **Implement Asset abstract methods:**
   _get_set_name(), _get_default_store_key(),
   _save_data_to_disk(), _load_data_from_disk(), _generate_custom_manifest()
4. **Save manifest.json with full lineage chain** --
   aidata_set_manifest containing case/record/source chain
5. **Use model_instance.save_pretrained() for the model/ folder** --
   delegate HuggingFace-style saving to the Instance (Layer 3)
6. **Attach PreFnPipeline before saving** --
   model_instance.prefn_pipeline = PreFnPipeline.from_aidata_set(...)
7. **Use load_model_instance_class() from model_registry** --
   string -> (InstanceClass, ConfigClass) resolution
8. **Use ConfigClass.from_aidata_set()** --
   auto-extracts metadata (action_to_id, num_actions, InputArgs, etc.)
9. **Support both overwrite and versioned save modes** --
   overwrite=True deletes existing; overwrite=False creates @run-v000X
10. **Pipeline returns ModelInstance_Set without saving** --
    caller must call save_to_disk() explicitly

---

MUST NOT
========

1. **NEVER import algorithm libraries** --
   No xgboost, torch, sklearn, neuralforecast at Layer 4
2. **NEVER call tuner methods directly** --
   Always go through model_instance.init() / .fit() / .infer()
3. **NEVER create model weights directly** --
   Always through pipeline flow: init() -> fit()
4. **NEVER hardcode model types** --
   Use registry lookup via load_model_instance_class()
5. **NEVER skip PreFnPipeline attachment** --
   Model without prefn_pipeline cannot do inference from raw data
6. **NEVER break the lineage chain** --
   Always include aidata_set_manifest in manifest.json

---

Key File Locations
==================

```
ModelInstance_Set:       code/haipipe/model_base/modelinstance_set.py
ModelInstance_Pipeline:  code/haipipe/model_base/modelinstance_pipeline.py
Asset base class:       code/haipipe/assets.py
PreFnPipeline:          code/hainn/prefn_pipeline.py
Model registry:         code/hainn/model_registry.py
```

---

See Also
========

- **haipipe-nn-0-overview**: Architecture map, decision guide, YAML templates
- **haipipe-nn-1-algorithm**: What algorithms are and their diversity
- **haipipe-nn-2-tuner**: The universal Tuner contract (5 abstract methods)
- **haipipe-nn-3-instance**: The orchestrator contract (5 abstract methods)
