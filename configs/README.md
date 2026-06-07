# configs

_YAML configs. Read by Hydra in experiments, by yaml-cpp on-device._

| File | What |
|---|---|
| `rfdetr_base.yaml` | RF-DETR base: model geometry, runtime backend + precision, export + benchmark knobs |

One config selects detector + runtime + precision — that is the model-swap lever
made declarative. `experiments/run_pipeline.py` reads it (plain YAML now,
Hydra-ready); the C++ apps take the same values via CLI args (yaml-cpp later).
