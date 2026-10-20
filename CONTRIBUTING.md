# Contributing to DeepDream Studio

DeepDream Studio is a beta desktop application. Changes should keep first launch
approachable and protect completed images and frame sequences when a run fails.

## Development setup

Use Python 3.11–3.14 on Windows/Linux or 3.11–3.13 on macOS. The GUI also needs
Tkinter; see the [installation guide](README.md#install) for OS packages.

Create the environment with the launcher, or follow the manual installation
instructions in the README, then install development tools:

```powershell
# Windows, from the repository root
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\pre-commit.exe install
```

```sh
# Linux/macOS, from the repository root
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/pre-commit install
```

Activate the environment for the commands below:

```powershell
.venv\Scripts\Activate.ps1
```

```sh
source .venv/bin/activate
```

The Markdown pre-commit hook requires Node.js. CI runs it with Node 20.

## Checks

```sh
python -m ruff check app tests scripts
python -m ruff format --check app tests scripts
python scripts/run_mypy.py app tests scripts
python -m pytest -q
npx --yes markdownlint-cli2@0.18.1 "**/*.md"
```

Tests include real model forward/backward passes, CLI integration runs, video
encoding, and property-based input checks. Some integration tests download model
weights on first use. GUI tests create real Tk windows and automatically skip if
there is no usable display. On headless Linux, install Tk and Xvfb, then use:

```sh
xvfb-run -a python -m pytest -q
```

For quicker feedback, use `python -m pytest -q -m "not integration and not gui"`.
Before submitting changes, run the full suite with a display. CI checks all four
supported Python versions across Windows, Linux, and macOS, excluding macOS 3.14.

### Coverage

Each CI test job reports statement and branch coverage in its log and uploads
XML plus browsable HTML as `coverage-<os>-py<version>` artifacts, retained for
14 days. Reports are uploaded after test failures when available. Coverage is
reported without a minimum-percentage gate while we establish the baseline.

Generate the same reports locally with a usable Tk display:

```sh
python -m pytest -q --cov --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=html:htmlcov
```

On headless Linux, prefix that command with `xvfb-run -a`. Open
`htmlcov/index.html` to inspect missed lines and branch paths. Compare reports
from the same platform and Python version: OS-specific branches, GUI availability,
and accelerator availability affect the result. Reports measure the test process;
CLI and encoding subprocesses are not instrumented, and scripts are outside the
configured `app` source scope. Prioritize uncovered cancellation, stale-worker
handling, recovery, and output preservation over the overall percentage.

## Architecture

See [docs/architecture.md](docs/architecture.md) for module responsibilities and
the generation/export lifecycle. The important boundaries are:

- The Tk GUI never imports PyTorch. Generation and encoding run in subprocesses.
- Workers put results onto queues; only the Tk thread updates widgets.
- Shared model and filename metadata stays in the torch-free `model_data.py`.
- PyTorch model loading and activation capture belong in `models.py`; its typed
  registry keeps runtime implementation separate from presentation metadata.
- Image decoding, tensor conversion, and atomic media publication belong in
  `media_io.py`; generation orchestration and frame recovery remain in `deepdream.py`.
- Numerical ascent and octave processing belong in `dream_engine.py`. CLI
  logging, progress events, and file output are attached through synchronous
  callbacks in `deepdream.py`. Preserve operation order and seeded jitter draws
  when changing the numerical code; the offline numerical baselines cover all
  four objective modes across multiple octaves.
- Structured progress messages belong in `protocol.py`, with human-readable logs
  retained for CLI users.
- Coordinate mapping and export planning are widget-free, independently testable
  components rather than additional responsibilities of the GUI class.

## Dependencies

`requirements.txt` and the wheel metadata describe minimum compatible runtime
versions. Source/launcher installs additionally use `constraints-runtime.txt`,
which pins the direct dependencies to the tested baseline. It is **not** a full
transitive lockfile. CUDA requirements select the matching `+cu128` wheels.

To update the baseline:

1. Update `constraints-runtime.txt`; update torch and torchvision as a pair.
2. Keep the CUDA pins in `requirements-cuda.txt` aligned with that pair.
3. Install the requirements in a fresh environment and run the full checks.
4. Run the CI platform/Python matrix and distribution smoke checks before release.
5. For accelerator changes, also exercise a real CUDA or MPS run; CPU CI cannot
   verify driver and hardware compatibility.

The `.deps_ok` fingerprint includes the constraints file, so changing the baseline
invalidates the launcher's cached import verification. Existing working installs
are not automatically upgraded; reinstall requirements to adopt the new baseline.

Wheel consumers can select the baseline with `pip install -c constraints-runtime.txt .`
from the source tree. Standard wheel installation keeps
the declared compatibility ranges.

## Packaging and releases

Use a clean output directory so smoke checks cannot accidentally inspect an old
artifact:

```sh
python -m pip install build
python -m build --outdir dist-review
python scripts/release_bundle.py dist-review
```

`MANIFEST.in` includes the launchers, requirements, constraints, CLI docs,
screenshots, contributor docs, and test fixtures in the source distribution.
`release_bundle.py` converts it to a ZIP, verifies required files, parses Python
sources, and runs the lightweight setup entry points from an extracted temporary
directory.

To check the wheel, install it into a separate environment with the runtime
baseline, then run `python scripts/smoke_installed.py` (under Xvfb on headless
Linux). It checks installed package discovery, console-entry metadata, CLI help,
and real GUI startup outside the checkout. Isolated Python prevents the local
`app/` directory from masking a broken installed package.

The release workflow runs the reusable CI workflow first and uploads only its
verified ZIP, wheel, and source distribution. Bump `app/__init__.py` for a release;
it is the single source of truth for the version. Describe changes and known beta
limitations in the GitHub release notes.

### Release validation

CI runs the test suite and coverage reports on every supported platform/Python
combination, always on CPU. Distribution builds, launcher-bundle verification,
and the installed-wheel smoke check run on Ubuntu with Python 3.12. Before a
release, complete this manual pass on at least one machine running the graphical
app, and record the result (OS, Python, GPU/driver, dependency set) in the
release notes.

- [ ] Launch the extracted release ZIP from a fresh folder; first-launch setup
  completes and the window appears.
- [ ] Dream a still image on the GPU (CUDA or MPS) and again with `--no-cuda`.
- [ ] Cancel an active generation, then start another job successfully.
- [ ] For a zoom/video run: generate frames, press Save Video, and change the
  quality control during encoding; the finished export reports the CRF it
  started with, and the same frames can be exported again without re-dreaming.
- [ ] Play and scrub the in-window preview, switch between outputs, and close
  the window while a job is running (no orphaned processes or temp files).
- [ ] Interrupt or kill a run mid-generation; completed outputs remain intact
  and `--resume` continues the saved frame sequence.
- [ ] Confirm that when the accelerator cannot start, the run continues on CPU
  with a clear message.

## Proposing changes and reporting bugs

Keep pull requests focused and explain the user-visible behavior or failure mode
they address. Add regression tests for substantive behavior changes, especially
file publication, cancellation, resume, and cross-thread lifecycle handling.

For bug reports, include the OS, Python version, model/mode, relevant settings,
reproduction steps, and the error or console log. On Windows, startup failures
are recorded in `deepdream-launch.log`. Include GPU/driver information when the
problem depends on acceleration.
