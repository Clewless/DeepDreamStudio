# DeepDream Studio — Command-Line Interface

The `CLI/` folder contains optional one-shot preset launchers and full CLI documentation. The **GUI** (`Launch.bat` / `Launch.sh` in the repo root) is the primary way to use DeepDream Studio; the CLI is intended for AI agents, headless Linux setups, and power users who prefer scripting over the GUI.

## One-shot preset launchers

`LaunchCLI.bat` (Windows) and `LaunchCLI.sh` (Linux/macOS) are drag-and-drop preset wrappers: they dream a single image using the settings defined near the top of each script (VGG16, `features.22`, 100 iterations, 10 octaves, 1600 px max resolution).

```bat
REM Windows — drag an image onto LaunchCLI.bat, or run from a terminal:
CLI\LaunchCLI.bat photo.jpg
```

```sh
# Linux/macOS — make executable once, then run:
chmod +x CLI/LaunchCLI.sh
./CLI/LaunchCLI.sh photo.jpg
```

Edit the variables near the top of the script to change the preset (model, layer, iteration count, etc.). For batches or custom options, call the CLI directly (see below).

> **Note:** Results go in `out/` in the repo root (parent folder), next to `Launch.bat` / `Launch.sh`, because the CLI launchers change to the repo root before running.

## Invoking the CLI directly

Once the environment is bootstrapped (run `Launch.bat` / `Launch.sh` at least once, or run `python app/bootstrap.py`), you can call the CLI without the preset wrapper:

```bat
.venv\Scripts\python.exe -m app.deepdream photo.jpg
```

```sh
.venv/bin/python -m app.deepdream photo.jpg
```

Model weights download from PyTorch on first use.

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
| `--video` | Off | Process an input video, carrying its audio over as AAC when possible. |
| `--video-blend` | `0.5` | Previous-dream weight for video blending; higher values retain more of the previous frame. |
| `--frames` | Off | Save iteration frames as PNGs. Existing matching frames require `--clear-frames`. |
| `--clear-frames` | Off | With `--frames` or `--timelapse`, explicitly remove existing iteration frames in `--frames-dir` before writing replacements; with `--zoom` or `--video`, replace a saved frame sequence in the frame folder, which requires `--frames-dir` or `--frames-only`; only the first video frame is validated before the old sequence is removed. Other files in the directory make the run fail safely. Cannot be combined with `--resume`. |
| `--frames-dir` | None | Dedicated frame output directory; `--frames` defaults to `out/frames`, while batch runs use per-run directories. |
| `--frames-only` | Off | Generate zoom/video frames without encoding an MP4. |
| `--encode-frames DIR` | None | Encode an existing frame sequence without running the dream again. |
| `--encode-audio-from` | Sequence source | Audio source used with `--encode-frames`. |
| `--resume` | Off | With `--zoom` or `--video` and `--frames-dir`, continue after the last valid saved frame. |
| `--no-cuda` / `--cpu` | Off | Force CPU execution. |
| `--out` | `out` | Output directory; `--encode-frames` also accepts an explicit MP4 path. |
| `--name` | Input filename | Output filename stem. |
| `--video-quality` | `balanced` | H.264 preset: `small` (CRF 28), `balanced` (17), or `lossless` (visually lossless; 12, 2× resolution / 4× pixels). |
| `--crf` | Preset value | Explicit H.264 CRF from 0 to 51; overrides the quality preset's CRF (geometry stays at the preset's scale). |
| `--seed` | None | Seed Python and PyTorch RNGs; in a batch each image uses `--seed` plus its position, so a result does not depend on how many other images are in the run. Some GPU operations may still vary. |
| `--preview-file` | None | Write intermediate previews for the GUI. |

## Video and animation

```bat
REM Zoom loop
python -m app.deepdream photo.jpg --zoom --zoom-frames 90 --zoom-scale 1.04 --zoom-rot 0.5

REM Dream an input video; blend previous frames to reduce flicker
python -m app.deepdream clip.mp4 --video --video-blend 0.6 --model googlenet --layer inception4c

REM Record the octave-by-octave process
python -m app.deepdream photo.jpg --timelapse --iterations 25 --octaves 4
```

Video modes reduce the GUI's iteration count to limit feedback buildup. Still-image iteration frames use `o*_i*.png` names; zoom and video sequences use `frame_*.png` names and include a `meta.json` for later export.

### Separating generation from encoding

For zoom and video runs, `--frames-only` lets you generate frames once and encode at different quality settings without re-dreaming:

```bat
python -m app.deepdream photo.jpg --zoom --zoom-frames 60 --frames-only --frames-dir out\zoom_seq
python -m app.deepdream --encode-frames out\zoom_seq --out out\preview.mp4 --video-quality small
python -m app.deepdream --encode-frames out\zoom_seq --out out\master.mp4 --video-quality lossless
```

Use an empty, dedicated `--frames-dir` for still-image iteration frames. `--clear-frames` replaces matching iteration PNGs only after checking that the directory contains no unexpected files. A completed iteration temp kept for recovery (`o*_i*.png.p*.tmp` plus its `.complete` sidecar, left when a locked destination blocked publishing) is never removed by `--clear-frames` or the automatic sweeps; delete it manually once recovered.

Fresh zoom/video runs also require an empty frame folder, so a new sequence can
never be mixed with saved frames. Interrupted atomic-save temps
(`.frame_*.tmp.png`, `meta.json.tmp`) are swept automatically and never block a
run. To continue an interrupted sequence, use `--resume` with its original
settings; to deliberately start over in the same folder, add `--clear-frames`,
which removes the saved frames and `meta.json` and refuses to touch anything
else in the directory.

The `lossless` preset uses CRF 12 at 2× width and height (4× pixels) in
standard YUV 4:2:0, so chroma density after subsampling matches native-resolution
full-chroma detail. Saved PNG frames stay at the dreamed native size; only the
encoded MP4 grows. Custom `--crf` values never change geometry — they use the
named preset's scale (1× for `small`/`balanced`, 2× for `lossless`). Because all
output is 4:2:0, even CRF 0 is not bit-exact.

**Frame-size limit.** H.264 Level 5.1 caps a frame at 36,864 macroblocks (about
9.4 megapixels). Hardware decoders are often rated only up to that level and may
refuse larger frames. `lossless` doubles each side, so its MP4 exceeds the limit
when the dreamed frame is larger than about 2.36 megapixels: a square frame above
1536 px, or a 3:2 frame at Max Size 2048 (doubled to 4096×2730). The default size
of 512 is unaffected. Software players such as ffmpeg should still play these
files, but this has not been tested. The 1× presets stay under the limit at any
Max Size up to 2048. A large `--size 0` (native resolution) can exceed it with any
preset. For a file that must play on hardware-limited devices, use `balanced` and
keep the dreamed frame under the limit.

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

## Tuning tips

- More iterations, octaves, or a deeper layer generally produce stronger patterns. Lower `--lr` if results become noisy.
- Each model has a minimum input size: GoogLeNet 15 px, InceptionV3 75 px, and VGG 32 px on the short edge. An image that is natively too small cannot be fixed by setting `--size 0`.
- `--size` caps the input resolution. Larger images use more memory; lower the cap or use `--cpu` if the GPU runs out of memory.
- CPU works but is slower. CUDA and Apple MPS are detected automatically unless CPU mode is forced.
- The first run downloads the selected weights into PyTorch's cache: about 50 MB for GoogLeNet, 104 MB for InceptionV3, 528 MB for VGG16, and 548 MB for VGG19.
- `ModuleNotFoundError: torch` usually means the system Python was used instead of `.venv`. Run `.venv/bin/python -m app.deepdream` (or the Windows equivalent) instead of `python`.
- If a zoom or video run is interrupted, keep its frame directory and use `--resume` (or **Resume** in the GUI) to continue. Resuming stops with an error if the saved frames were made with different settings (resolution, model, layers, zoom or blend parameters), rather than producing an inconsistent sequence.
- An out-of-memory failure reports the resolution that failed and a smaller `--size` to try. It does not retry automatically, since that would silently change the output size and filename.
- Video preview decoding happens on a background thread, so scrubbing and playback stay responsive on long clips. If a frame is slow to decode, the previous frame stays on screen until the new one is ready.
