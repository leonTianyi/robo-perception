# Walkthrough — learn the RF-DETR slice end to end

This folder is a **guided tour** of the system that was built when the standalone
`rfdetr-poc` was integrated into `robo-perception`. It teaches both the *system
design* (how the pieces fit and why) and the *implementation* (the actual files,
in an order that builds understanding).

If `docs/00..06` answer "*why does TensorRT work this way?*", this folder answers
"*how is the whole repo wired, and why those choices?*".

## How to read this

Read in order — each chapter assumes the previous one.

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 1 | [System design](01_system_design.md) | the problem, the architecture, the **seam**, and every design decision + trade-off |
| 2 | [C++ code tour](02_cpp_tour.md) | bottom-up: `core` → `io/decode` → `runtime/tensorrt` → `detection/rfdetr` → `apps` |
| 3 | [The parity seam](03_parity_seam.md) | the crux: pybind11 + how Python calls the *same* C++ pre/post |
| 4 | [The Python edge](04_python_edge.md) | the `roboperc` package: runtimes, export, eval, experiments |
| 5 | [Build, tools & workflow](05_build_tools_and_workflow.md) | CMake, the critical tools, and the exact end-to-end run (with the gotchas hit) |

## The 60-second mental model

The repo has **two homes for code, meeting at one seam**:

```
        C++  (canonical, deployment)                Python (the soft edge)
   ┌───────────────────────────────┐         ┌──────────────────────────────┐
   │ core      types + Detector     │         │ poc / experiments            │
   │ io/decode image decode  ───────┼──┐      │ runtime: pytorch · onnx · trt │
   │ detection/rfdetr pre/post ─────┼─ pybind11 ─► roboperc._native ◄────────┤
   │ runtime/tensorrt  engine exec  │  │      │ export · eval · report        │
   │ apps      run_inference, bench │  │      └──────────────────────────────┘
   └───────────────────────────────┘  │
                                   "one implementation of decode + pre/post,
                                    called from BOTH languages"
```

- **C++** is the deployment path: fast, on-robot, no Python.
- **Python** is for prototyping, export, experiments, and scoring.
- They **share one copy** of decode + pre/post (the C++ one, exposed via pybind11).
  That's what makes a PyTorch-vs-TensorRT comparison a comparison of *models*, not
  of two slightly-different pipelines.

Everything else in the design falls out of protecting that property.

## If you only open three files

1. `core/include/roboperc/core/detector.hpp` — the contract everything speaks.
2. `bindings/src/module.cpp` — the seam in code.
3. `python/roboperc/experiments/run_pipeline.py` — the whole thing in motion.
