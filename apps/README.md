# apps

_C++ executables: run_inference, benchmark._

| Executable | What |
|---|---|
| `run_inference` | `run_inference <engine> <image> [out.jpg] [detector=rfdetr] [conf]` — builds a detector via the registry, draws + prints detections |
| `benchmark` | `benchmark <engine> <image> <out.json> [warmup] [iters]` — device-scope (CUDA events) + e2e latency; JSON for `eval/report.py` |

`run_inference` is the deployment entry point and the C++ side of the eval parity
check: its decode (`io/decode`) and pre/post (`detection/image/rfdetr`) are the
same code the Python paths call through `roboperc._native`.

`benchmark` measures the **runtime**, so it drives `TensorRTRunner` directly rather
than going through the `Detector` (timing the engine, not the pre/post).

`ros2/` holds `rclcpp` node wrappers later — kept ROS-agnostic until then.
