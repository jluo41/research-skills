Skill: haipipe-nn-1-algorithm
==============================

Layer 1 of the 4-layer NN pipeline: Algorithm.

An algorithm is any external ML library that does the actual computation.
We don't write algorithms -- we wrap them in Tuners (Layer 2).

**Scope of this skill:** Framework patterns only. Does not catalog
project-specific state (which algorithms are installed, which models have
been trained). Concrete code examples are illustrative. This skill applies
equally to any domain or algorithm type.

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

These show actual algorithm calls inside Tuner files.
(Illustrative -- discover actual Tuner files at runtime):

```bash
ls code/hainn/<family>/models/    # e.g., ls code/hainn/tsforecast/models/
```

**xgboost** (illustrative -- from mlpredictor/models/modeling_xgboost.py):

```python
model = xgb.XGBClassifier(**param)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
preds = model.predict_proba(X_valid)[:, 1]
```

**neuralforecast** (illustrative -- from tsforecast/models/neuralforecast/modeling_nixtla_patchtst.py):

```python
nf = NeuralForecast(models=[PatchTST(h=24, input_size=288, ...)], freq='5min')
nf.fit(df=df_nixtla)  # DataFrame with [unique_id, ds, y] columns
forecasts = nf.predict()
```

**HuggingFace Transformers** (illustrative -- from tefm/models/hfntp/modeling_hfntp.py):

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

**LLM API** (illustrative -- from tsforecast/models/api/modeling_llmapi.py):

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

Discover at runtime:

```bash
ls code/hainn/                          # all model families
ls code/hainn/<family>/models/          # tuner files for a family
cat code/hainn/model_tuner.py           # base class contract
```

Fixed locations:

```
Tuner base class:       code/hainn/model_tuner.py
Tuners (discover):      code/hainn/<family>/models/    <- ls to find
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
  raw feature values -> transformed/tokenized values

**Reference** (discover at runtime):

```bash
ls code/hainn/<family>/test-modeling-<name>/scripts/
# e.g.:
# ls code/hainn/tefm/models/te_clm/test-modeling-te_clm/scripts/
```

See the actual test file (discover with ls above) for a concrete example.
The "Expected steps" and display rules above are the framework pattern.

---

See Also
========

- **haipipe-nn-0-overview**: Architecture map and decision guide
- **haipipe-nn-2-tuner**: How to wrap an algorithm into a Tuner
- **haipipe-nn-3-instance**: How Tuners are managed by Instances
- **haipipe-nn-4-modelset**: How Instances are packaged into assets
