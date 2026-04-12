# Step 04a — Reset event wrappers (OmniReset on Isaac Lab)

**Estou a caminho!**

## What changed
- `source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/events.py`

Isaac Lab may invoke **reset-mode** event terms before timeline **PLAY** has converted `ManagerTermBase` **classes** into instances. Calling the class like a function breaks (`__init__` kwargs mismatch).

This step adds **lazy singleton wrappers** (explicit function signatures for static validation) for:
- `reset_end_effector_round_fixed_asset`
- `reset_end_effector_from_grasp_dataset`
- `reset_insertive_object_from_partial_assembly_dataset`

Implementation pattern matches earlier fixes for partial assemblies / grasp sampling: store the instantiated term in `env.extras` under a private key.

## Why it matters
Unblocks `scripts_v2/tools/record_reset_states.py` for peg/peghole tasks that chain dataset-driven reset events.

## Validation notes
Validated on `tai` while recording `resets_ObjectPartiallyAssembledEEAnywhere.pt` (smoke run; datasets resolved via `UWLAB_CLOUD_ASSETS_DIR` + symlinked `Datasets/OmniReset`).
