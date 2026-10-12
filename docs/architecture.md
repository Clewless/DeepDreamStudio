# Architecture

## Modules

| Module | Responsibility |
|---|---|
| `app/deepdream.py` | CLI validation and generation orchestration, progress logging, frame/preview output, video generation, and encoding. |
| `app/dream_engine.py` | Numerical ascent, guided/class objectives, octave planning, and CPU-backed pyramid processing with synchronous progress callbacks. |
| `app/models.py` | Typed runtime model registry, PyTorch model construction, layer resolution, truncated forwards, and activation hooks. |
| `app/media_io.py` | EXIF-aware image decoding, normalized tensor conversion, and atomic image, preview, frame, and completed-video publication. |
| `app/gui.py` | Tk widgets, user interaction, generation queue orchestration, and applying worker results. |
| `app/gui_classifier.py` | Target-class picker: search, filters, dropdown, and listbox UI. |
| `app/gui_job.py` | Generation settings snapshots, CLI command construction, and progress tracking. |
| `app/gui_layers.py` | Dream-layer panel: single-layer menu, multi-layer chips, and tier grouping. |
| `app/gui_playback.py` | In-window video bar: timeline scrubbing, play/pause, and async frame loading. |
| `app/gui_process.py` | Consistent CLI process creation and streaming output/completion to queues. |
| `app/gui_video.py` | Video export: quality controls, frame-sequence encoding, and encode drain. |
| `app/gui_export.py` | Immutable video export requests and output-name planning. |
| `app/gui_viewer.py` | Viewport coordinate mapping, cursor-centered zoom, and reveal-divider geometry. |
| `app/gui_helpers.py` | Media decoding worker, frame cache, rendering utilities, paths, and process termination. |
| `app/gui_widgets.py` | Reusable Tk widget styling and behavior. |
| `app/model_data.py` | Torch-free model metadata, quality presets, filenames, and sequence metadata. |
| `app/protocol.py` | Prefixed JSON progress events shared by the engine and GUI. |
| `app/bootstrap.py`, `app/check_deps.py` | Standard-library first-launch setup and dependency verification. |

## Generation lifecycle

The numerical engine does not print logs or write media. The CLI adapter supplies
typed synchronous callbacks for startup, octave/iteration progress, and previews.
Startup runs after numerical/input validation, so invalid settings cannot clear
existing frames. Callbacks must not mutate tensors or consume the jitter RNG.
The CPU pyramid and carried detail keep only the current octave on the device.

1. The GUI snapshots controls into a `GenerationJob`, including its own frame
   directory and preview path.
2. The GUI starts `python -m app.deepdream` on the Tk thread. The process uses a
   dedicated process group on POSIX so cancellation can terminate its children.
3. A reader worker streams stdout into a queue. The engine emits both ordinary
   log text and structured events.
4. The Tk thread drains a bounded number of messages per tick, updates progress,
   and applies the newest preview. It identifies completion by process and job
   identity so a late completion cannot finish a newer run.
5. Queue orchestration decides whether to start the next job. Cancellation and
   window close terminate active child processes and clean their temporary files.

## Video export and preview

Zoom/video generation can retain frames and metadata independently of MP4 export.
`VideoExportRequest` snapshots the encoding settings; export uses the same
subprocess transport as generation. Completion is applied on the Tk thread.
The active export retains its immutable request so completion reports the CRF
used for encoding even if the controls change. Stop terminates both generation
and export when they are active concurrently.

Video preview decoding is performed by `VideoFrameLoader`. Requests carry tokens
and generations so stale results can be ignored. A bounded `FrameCache` supports
scrubbing without decoding on the Tk thread.

`ViewerGeometry` models the source image, rendered viewport, and enclosing label.
It handles both label centering and image letterboxing. Focus selection, zoom
anchoring, and slide-reveal interactions share that mapping.

## Files and recovery

Final images and videos are published using sibling temporary files and atomic
replacement. Locked destinations preserve the completed temporary output.
Sequence metadata records the settings needed to validate a resumed run.
Cleanup must remain scoped to a job's own files and must not remove unexpected
files from a user-selected directory.

## Distribution

The wheel installs the `app` package and two console commands. The source
distribution and launcher ZIP also include setup scripts and their supporting
files through `MANIFEST.in`.

CI exercises source behavior, verifies the extracted ZIP, and smoke-tests the
installed wheel outside the checkout. Releases reuse those verified artifacts.
