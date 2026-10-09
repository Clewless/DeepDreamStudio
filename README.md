# DeepDream Studio

[![CI](https://github.com/Clewless/DeepDreamStudio/actions/workflows/ci.yml/badge.svg)](https://github.com/Clewless/DeepDreamStudio/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
![Python 3.11 | 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

A PyTorch based local backend and GUI for creating "DeepDream" images. DeepDream images are known for their intricate, psychedelic-like patterns, produced by amplifying the features a neural network has learned to detect. Inspired by Alexander Mordvintsev et al. (Google) 2015 "Inceptionism: Going Deeper into Neural Networks".

Runs on Windows, macOS, and Linux, with CUDA, Apple Silicon (MPS), or CPU.

![Original photo vs DeepDream result](docs/images/example-before-after.jpg)

![Finished DeepDream run](docs/images/gui-result.png)

## How to run

```bat
LaunchDeepDreamStudio.bat     REM Windows launcher script
```

```sh
./LaunchDeepDreamStudio.sh   # macOS / Linux launcher script
```

- **Windows:** Double-click `LaunchDeepDreamStudio.bat`.
- **Linux:** Run `./LaunchDeepDreamStudio.sh` from a terminal (see [Troubleshooting](#troubleshooting)).
- **macOS:** Run `./LaunchDeepDreamStudio.sh` from a terminal (see [Troubleshooting](#troubleshooting)).

The first launch sets everything up automatically (see [First launch](#first-launch)).

Select an image from your disk by either dragging and dropping it in the window or browsing your files using your default file browser. Press the "Dream" button, and that's it!

You can inspect your images using the built-in photo viewer, and find them on your disk in the "out" folder of this program's root. Generated images are saved automatically. For zoom or video runs, click **Save Video** in the GUI to export the generated frames as an MP4; source audio is copied when available for video-mode runs.

## Troubleshooting

> - **macOS "unidentified developer" warning:** right-click the `.sh` file, click Open, and confirm once. Always run the script from a terminal; double-clicking opens it in an editor instead of running it.
> - **Linux won't run the script:** double-clicking opens the file in an editor. Run `chmod +x LaunchDeepDreamStudio.sh LaunchDeepDreamCLI.sh` once, then run the script from a terminal.
> - **First run downloads a lot:** Python packages are about 200 MB (CPU) or 2.5 GB (NVIDIA GPU), plus 50–550 MB of model weights. Keep the window open and keep about 5 GB of free disk space.
> - **Don't move a working folder between computers:** `.venv` is tied to its machine and path. Copy only the source files, then run the launcher on the new machine to create a fresh environment.

## Repository layout

```text
.
├── LaunchDeepDreamStudio.bat    # GUI launcher (Windows), start here
├── LaunchDeepDreamStudio.sh     # GUI launcher (Linux/macOS), start here
├── LaunchDeepDreamCLI.bat       # Optional one-shot CLI presets (Windows)
├── LaunchDeepDreamCLI.sh        # Optional one-shot CLI presets (Linux/macOS)
├── app/                         # Application package (engine + GUI)
│   ├── gui.py                   # GUI entry point
│   ├── deepdream.py             # Engine and CLI entry point
│   ├── protocol.py              # Structured progress events (CLI -> GUI)
│   ├── bootstrap.py             # First-launch setup helpers
│   ├── check_deps.py            # First-launch setup helpers
│   └── ...
├── tests/                       # Test suite
├── docs/images/                 # README screenshots and example results
├── requirements*.txt            # Dependencies
├── pyproject.toml               # Wheel build metadata and dev tool config
├── README.md
├── LICENSE                      # Project license (BSD-3-Clause)
└── THIRD_PARTY_NOTICES.md       # Third-party licenses
```

### Running from source

The launchers live at the repository root, while the application modules live in `app/`. From the root, run:

```bash
python -m app.gui         # GUI
python -m app.deepdream   # CLI
```

The project also builds an installable wheel, which provides `deepdream` and `deepdream-studio` commands:

```sh
python -m pip install .
deepdream photo.jpg
deepdream-studio
```

The installed version and the repository can coexist; the launchers always use the local `app/` package.

## Install

Requires Python 3.11, 3.12, 3.13, or 3.14*.

*Python 3.14 is currently not supported on macOS.

On Linux, install Tkinter if needed — without it the GUI launcher stops with an error telling you the exact command:

```sh
# Debian/Ubuntu
sudo apt install python3-tk
# Fedora
sudo dnf install python3-tkinter
# Arch
sudo pacman -S tk
# macOS (Homebrew ships Tkinter separately — match your Python version)
brew install python-tk@3.13  # e.g. python-tk@3.11 for Python 3.11, python-tk@3.12 for 3.12
```

The first time you run one, it creates `.venv`, installs the right requirements (CUDA or CPU), and verifies the install. See [First launch](#first-launch). The manual steps below are only for setting up an environment yourself to troubleshoot, or when working without the launcher scripts.

For NVIDIA GPUs use the CUDA requirements; otherwise use the standard requirements (CPU on Linux, CPU/MPS on macOS).

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
# NVIDIA GPU, CUDA 12.8:
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
# CPU:
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS (Python 3.11–3.13):**

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# CPU / Apple MPS:
.venv/bin/python -m pip install -r requirements.txt
```

**Linux:**

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# NVIDIA GPU on Linux, CUDA 12.8:
.venv/bin/python -m pip install -r requirements-cuda.txt
# Otherwise:
.venv/bin/python -m pip install -r requirements.txt
```

### First launch

On first run, each launcher will:

1. Find a supported Python interpreter (3.11–3.14 on Windows and Linux; 3.11–3.13 on macOS).
2. Create `.venv` if it is missing, or repair it if the interpreter is broken or too old — the old folder is *renamed* to `.venv.bad-<timestamp>` after confirmation, never deleted.
3. Install dependencies into `.venv` via `app/check_deps.py` (choosing `requirements-cuda.txt` on NVIDIA machines, otherwise `requirements.txt`), then confirm the stack imports before starting.

If the CUDA packages cannot be installed — a blocked or throttled connection to the PyTorch wheel index is the usual cause — the launcher retries with the standard CPU packages so you still get a working app, just without GPU acceleration.

If a GPU is detected but cannot actually start (a driver too old for the pinned CUDA build, or an unsupported GPU architecture), the run automatically continues on the CPU instead of stopping.

The launcher creates `.venv` here and does not install or alter system Python. Model weights are downloaded to PyTorch's cache on first use. If no supported Python is found, the launcher prints an install command for your system and exits:

| System | Command |
|---|---|
| Windows | `winget install Python.Python.3.14` (or python.org; tick "Add python.exe to PATH") |
| Debian/Ubuntu | `sudo apt update && sudo apt install python3 python3-venv python3-tk` |
| Fedora/RHEL | `sudo dnf install python3 python3-tkinter` |
| Arch Linux | `sudo pacman -S python tk` |
| macOS | `brew install python@3.13 python-tk@3.13` (or install Python 3.13 from python.org, which bundles Tkinter) |

The `python3-tk` / `python3-tkinter` / `tk` / `python-tk@X.Y` packages are only needed for the GUI; CLI-only use can skip them.

The import check is cached in `.deps_ok` and re-runs automatically if the requirements files or Python version change.

On Windows the GUI launcher writes startup output to `deepdream-launch.log` next to the launchers: if the Studio window never appears, that file has the error.

## Command line

Results go in `out/`. Model weights download from PyTorch on first use.

`LaunchDeepDreamCLI.bat`/`LaunchDeepDreamCLI.sh` are optional one-shot presets: they dream a single image with the settings shown near the top of the script (VGG16, 100 iterations, 10 octaves, 1600 px). Use them for drag-and-drop or quick single-image runs; the GUI is the main application. For batches or custom options, call the CLI directly:

```bat
.venv\Scripts\python.exe -m app.deepdream photo.jpg
```

```sh
.venv/bin/python -m app.deepdream photo.jpg
```

## CLI options

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `vgg16` | `googlenet`, `inceptionv3`, `vgg16`, or `vgg19`. |
| `--layer` | Model's recommended layer | Layer name; multiple layers may be comma-separated. |
| `--guide` | None | Reference image for feature-guided dreaming. |
| `--class` | None | ImageNet category name or class index (0–999). |
| `--iterations` | Model-specific | Gradient-ascent steps per octave. Defaults: GoogLeNet/VGG16 20, VGG19 25, InceptionV3 30. |
| `--octaves` | `0` | Scale levels; `0` chooses a count automatically. |
| `--octave-scale` | Model-specific | Scale factor between octaves. Defaults: GoogLeNet 1.4, InceptionV3 1.35, VGG 1.32. |
| `--lr` | Model-specific | Gradient-ascent step size. Defaults: GoogLeNet 0.03, InceptionV3 0.025, VGG 0.018. |
| `--size` | `512` | Maximum image dimension; `0` keeps native resolution. |
| `--jitter` | `32` | Random pixel shift per step, which reduces grid artifacts. |
| `--timelapse` | Off | Save an iteration timelapse as MP4. |
| `--timelapse-fps` | `30` | Timelapse frame rate. |
| `--zoom` | Off | Generate a recursive zoom animation. |
| `--zoom-frames` | `60` | Number of zoom frames. |
| `--zoom-scale` | `1.03` | Scale per zoom frame; must be at least 1.0. |
| `--zoom-rot` | `0.0` | Rotation per zoom frame, in degrees. |
| `--fps` / `--zoom-fps` | `30` | Video frame rate. |
| `--zoom-x` / `--zoom-y` | `0.5` / `0.5` | Normalized zoom focus coordinates (0–1). |
| `--video` | Off | Process an input video, preserving its audio when possible. |
| `--video-blend` | `0.5` | Previous-dream weight for video blending; higher values retain more of the previous frame. |
| `--frames` | Off | Save iteration frames as PNGs. Existing matching frames require `--clear-frames`. |
| `--clear-frames` | Off | With `--frames` or `--timelapse`, explicitly remove existing iteration frames in `--frames-dir` before writing replacements. Other files in the directory make the run fail safely. |
| `--frames-dir` | None | Dedicated frame output directory; `--frames` defaults to `out/frames`, while batch runs use per-run directories. |
| `--frames-only` | Off | Generate zoom/video frames without encoding an MP4. |
| `--encode-frames DIR` | None | Encode an existing frame sequence without running the dream again. |
| `--encode-audio-from` | Sequence source | Audio source used with `--encode-frames`. |
| `--resume` | Off | With `--zoom` or `--video` and `--frames-dir`, continue after the last valid saved frame. |
| `--no-cuda` / `--cpu` | Off | Force CPU execution. |
| `--out` | `out` | Output directory; `--encode-frames` also accepts an explicit MP4 path. |
| `--name` | Input filename | Output filename stem. |
| `--video-quality` | `balanced` | H.264 preset: `small` (CRF 28), `balanced` (17), or `lossless` (0). |
| `--crf` | Preset value | Explicit H.264 CRF from 0 to 51; overrides the quality preset. |
| `--seed` | None | Seed Python and PyTorch RNGs; in a batch each image uses `--seed` plus its position, so a result does not depend on how many other images are in the run. Some GPU operations may still vary. |
| `--preview-file` | None | Write intermediate previews for the GUI. |

For zoom and video runs, `--frames-only` separates generation from encoding. Keep the frame directory and export it repeatedly at different quality settings:

```bat
python -m app.deepdream photo.jpg --zoom --zoom-frames 60 --frames-only --frames-dir out\zoom_seq
python -m app.deepdream --encode-frames out\zoom_seq --out out\preview.mp4 --video-quality small
python -m app.deepdream --encode-frames out\zoom_seq --out out\master.mp4 --video-quality lossless
```

Use an empty, dedicated `--frames-dir` for still-image iteration frames. `--clear-frames` replaces matching iteration PNGs only after checking that the directory contains no unexpected files.

## Models and layers

| Model (weights) | Default layer | Notes |
|---|---|---|
| `googlenet` (0.05 GB) | `inception4c` | The original 2015 DeepDream architecture; fastest option. |
| `inceptionv3` (0.10 GB) | `Mixed_6d` | Slightly slower than InceptionV1; choose it for richer textures and more recognizable, object-like patterns. |
| `vgg16` (0.53 GB) | `features.22` | Default model; slower but easier to get good result. |
| `vgg19` (0.55 GB) | `features.26` | Deeper VGG variant. Even slower and hard to get consistent results. |

Earlier layers tend to produce edges and fine textures; deeper layers produce larger, more recognizable patterns, like eyes and dogs:

| Architecture | Early | Middle | Deep |
|---|---|---|---|
| GoogLeNet | `inception3a`–`inception3b`: fine textures | `inception4a`–`inception4e`: edges, shapes, and object motifs (`inception4c` is the classic choice) | `inception5a`–`inception5b`: large, abstract structures |
| InceptionV3 | `Mixed_5b`–`Mixed_5d`: textures and edges | `Mixed_6a`–`Mixed_6e`: shapes and objects (`Mixed_6d` default) | `Mixed_7a`–`Mixed_7c`: high-level patterns |
| VGG16 / VGG19 | `features.3` / `.8`: fine textures and soft patterns | `.15` / `.17`: shapes and edges; `.22` / `.26`: objects forming | `.29` / `.35`: deeper patterns |

Combining layers mixes their objectives. Experimental. GoogLeNet accepts both `inception*` and legacy `mixed*` layer names.

## Video and animation

```bat
REM Zoom loop
python -m app.deepdream photo.jpg --zoom --zoom-frames 90 --zoom-scale 1.04 --zoom-rot 0.5

REM Dream an input video; blend previous frames to reduce flicker
python -m app.deepdream clip.mp4 --video --video-blend 0.6 --layer inception4c

REM Record the octave-by-octave process
python -m app.deepdream photo.jpg --timelapse --iterations 25 --octaves 4
```

Video modes reduce the GUI's iteration count to limit feedback buildup. Still-image iteration frames use `o*_i*.png` names; zoom and video sequences use `frame_*.png` names and include a `meta.json` for later export.

## Tuning and troubleshooting

- More iterations, octaves, or a deeper layer generally produce stronger patterns. Lower `--lr` if results become noisy.
- Each model has a minimum input size: GoogLeNet 15 px, InceptionV3 75 px, and VGG 32 px on the short edge. An image that is natively too small cannot be fixed by setting `--size 0`.
- `--size` caps the input resolution. Larger images use more memory; lower the cap or use `--cpu` if the GPU runs out of memory.
- CPU works but is slower. CUDA and Apple MPS are detected automatically unless CPU mode is forced.
- The first run downloads the selected weights into PyTorch's cache: about 50 MB for GoogLeNet, 104 MB for InceptionV3, 528 MB for VGG16, and 548 MB for VGG19.
- `ModuleNotFoundError: torch` usually means the system Python was used instead of `.venv`. Run the matching command from [Command line](#command-line) above.
- If a zoom or video run is interrupted, keep its frame directory and use `--resume` (or **Resume** in the GUI) to continue. Resuming stops with an error if the saved frames were made with different settings (resolution, model, layers, zoom or blend parameters), rather than producing an inconsistent sequence.
- An out-of-memory failure reports the resolution that failed and a smaller `--size` to try. It does not retry automatically, since that would silently change the output size and filename.
- Video preview decoding happens on a background thread, so scrubbing and playback stay responsive on long clips. If a frame is slow to decode, the previous frame stays on screen until the new one is ready.

`LaunchDeepDreamCLI.bat` and `LaunchDeepDreamCLI.sh` use VGG16 `features.22`, 100 iterations, 10 octaves, and native resolution capped at 1600 px. Edit the variables near the top of the script to change the preset.

## Credits and licenses

This project is an independent, modern PyTorch implementation of the 2015 DeepDream technique, inspired by the original work and code by Alexander Mordvintsev and colleagues at Google. Pretrained model weights and ImageNet labels come from `torchvision`. Video encoding uses FFmpeg through `imageio-ffmpeg`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE) for license details.

DeepDream Studio aims to be a modern, self-contained DeepDream tool: a single launcher, fully local execution, and no cloud dependency.

Much of the application code in this repository was generated with the assistance of AI coding agents. See [LICENSE](LICENSE) for the project's licensing terms.
