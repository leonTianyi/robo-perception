# Environment Notes (as inspected)

Real findings from inspecting *this* Jetson, including a couple of gotchas that
cost time. Recorded so you (and future-you) don't rediscover them.

---

## What's pre-installed vs what setup.sh adds

| Component | State at start | Notes |
|---|---|---|
| Python | 3.10.12 | system |
| CUDA | 12.6 (V12.6.68) | `/usr/local/cuda` |
| TensorRT (C++) | 10.3.0.30 | headers `/usr/include/aarch64-linux-gnu`, libs `/usr/lib/aarch64-linux-gnu` |
| TensorRT (Python) | 10.3.0 ✓ | `/usr/lib/python3.10/dist-packages/tensorrt` — already there |
| `trtexec` | present | `/usr/src/tensorrt/bin` (on PATH) |
| OpenCV (C++ & Python) | 4.8.0 ✓ | NVIDIA build |
| g++ / make | 11.4 / 4.3 ✓ | |
| cmake | **missing** | we use a hand-written Makefile instead |
| numpy / Pillow / psutil | ✓ | |
| jetson-stats (jtop) | 4.3.2 ✓ | for Python resource sampling |
| **PyTorch** | **missing** | setup.sh installs the JetPack 6 wheel |
| **onnx / onnxruntime / onnxsim** | **missing** | setup.sh |
| **rfdetr / supervision** | **missing** | setup.sh |
| **cuda-python** | **missing** | setup.sh — needed by the Python TRT runner |

---

## Gotcha 1: the OpenCV `pkg-config` file was broken (now FIXED)

**Original problem:** `pkg-config --cflags opencv4` reported
`-I/usr/local/include/opencv4`, but that directory is **empty**. The real OpenCV
4.8 install is:

- headers: `/usr/include/opencv4`
- dev `.so` links: `/usr/lib/libopencv_*.so` → `libopencv_*.so.4.8.0`

The `.pc` file (`/usr/lib/pkgconfig/opencv4.pc`) had `prefix=/usr/local`, which is
simply wrong for this image. There is *also* a second, older OpenCV (Ubuntu's
4.5.4d) sitting in `/usr/lib/aarch64-linux-gnu/libopencv_*.so.4.5.4d` — so trusting
the broken pkg-config could mix 4.8 headers with 4.5 libs.

**Fix applied (2026-06-02):** corrected the one wrong line in the `.pc`:

```
prefix=/usr/local   →   prefix=/usr
```

`exec_prefix`, `libdir`, and `includedir` all derive from `prefix`, so that single
change fixes everything. Backup kept at `/usr/lib/pkgconfig/opencv4.pc.bak`.

Verified:

```
$ pkg-config --cflags opencv4        # -I/usr/include/opencv4   (real headers)
$ ldd 04_cpp/infer | grep opencv_core
  libopencv_core.so.408 => /lib/libopencv_core.so.408   (the 4.8 build)
```

`04_cpp/Makefile` now uses standard `pkg-config --cflags --libs opencv4` again, so
it's portable to a clean image.

---

## Gotcha 2: TensorRT package version string

`dpkg` shows `libnvinfer-dev 10.3.0.30-1+cuda12.5` — note the **cuda12.5** suffix
even though the system CUDA is 12.6. This is normal: NVIDIA builds the TRT debs
against a minor-version CUDA and they run fine on 12.6 (CUDA minor versions are
forward-compatible within 12.x). Don't let the `cuda12.5` tag alarm you.

---

## Gotcha 3: `nvidia-smi` is mostly blank on Jetson

`nvidia-smi` exists but reports `N/A` for utilisation, memory, power — the Tegra
iGPU isn't exposed through the NVML path that discrete GPUs use. Use **jtop** or
**tegrastats** for real numbers. Our benchmark code does exactly this.

---

## Memory architecture reminder

61 GB RAM is **shared** between CPU and GPU (no discrete VRAM). Implications:
- `cudaMalloc` comes out of the same pool as host RAM
- H2D / D2H copies are in-DRAM moves, not PCIe transfers — cheap
- "GPU memory" in the benchmark report overlaps with system RAM; we report RAM
  used during the run as the proxy

---

## Reproducing the clock lock

```bash
sudo nvpmodel -m 0     # MAXN
sudo jetson_clocks     # pin max clocks
sudo jetson_clocks --show | head   # verify
```

All benchmark numbers in `results/` assume this state.
