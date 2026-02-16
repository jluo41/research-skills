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

See Also
========

- **haipipe-nn-0-overview**: Architecture map and decision guide
- **haipipe-nn-2-tuner**: How to wrap an algorithm into a Tuner
- **haipipe-nn-3-instance**: How Tuners are managed by Instances
- **haipipe-nn-4-modelset**: How Instances are packaged into assets
