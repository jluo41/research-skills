haipipe-data-2-record
=====================

Layer 2: Record Processing -- converts SourceSets into temporally-aligned
structured records. In the CGM domain, records are aligned to a 5-minute
time grid. The HumanRecords mapping is domain-configurable.

---

Quick Reference
---------------

```
/haipipe-data-2-record load           Inspect existing RecordSet (read-only)
/haipipe-data-2-record cook           Run Record_Pipeline with a YAML config
/haipipe-data-2-record design-chef    Create new HumanFn/RecordFn via builder
/haipipe-data-2-record design-kitchen Upgrade Record_Pipeline infrastructure
```

---

Common Commands
---------------

```bash
# 1. Activate environment
source .venv/bin/activate && source env.sh

# 2. Run Record_Pipeline via CLI
haistep-record --config config/test-haistep-ohio/2_test_record.yaml

# 3. Load and inspect (Python)
from haipipe.record_base import RecordSet
record_set = RecordSet.load_from_disk(path='/full/path/to/record_set', SPACE=SPACE)
record_set.info()

# 4. Access data
human = record_set.Name_to_HRF['HmPtt']                    # string key
cgm   = record_set.Name_to_HRF[('HmPtt', 'CGM5Min')]       # tuple key
```

---

Key Paths
---------

```
Pipeline:    code/haipipe/record_base/record_pipeline.py
Asset:       code/haipipe/record_base/record_set.py
Generated:   code/haifn/fn_record/human/*.py               (DO NOT EDIT)
             code/haifn/fn_record/record/*.py               (DO NOT EDIT)
Builders:    code-dev/1-PIPELINE/2-Record-WorkSpace/*.py    (edit these)
Store:       _WorkSpace/2-RecStore/
Config:      config/test-haistep-ohio/2_test_record.yaml
```

---

Files
-----

```
SKILL.md           Full rules, Name_to_HRF pattern, MUST DO / MUST NOT
README.md          This file (quick reference)
load.md            Rules for inspecting RecordSets
cook.md            Rules for running Record_Pipeline
design-chef.md     Rules for creating new HumanFn/RecordFn builders
design-kitchen.md  Rules for modifying Record_Pipeline framework
templates/
  config.yaml      Annotated config template
```

---

See Also
--------

- **haipipe-data-1-source**: How raw data becomes SourceSets (input to this layer)
- **haipipe-data-3-case**: How RecordSets become CaseSets (output of this layer)
- **haipipe-data-4-aidata**: How CaseSets become ML-ready datasets
