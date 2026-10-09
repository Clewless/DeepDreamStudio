# Third-Party Notices

This project's own code is released under the BSD 3-Clause License (see [LICENSE](LICENSE)).
The following third-party software, models, and data are used by this project.
They are **not** part of the BSD-licensed source code and remain under their
own licenses.

## Direct runtime dependencies (Python packages)

| Package | License | Notes |
|---|---|---|
| `torch` | BSD 3-Clause | Downloaded from PyTorch. CUDA builds are selected by `requirements-cuda.txt`. |
| `torchvision` | BSD 3-Clause | Source of model architectures, ImageNet category names, and the pretrained-weight URLs. The weights themselves may carry separate terms (see Models & weights). |
| `numpy` | BSD 3-Clause and additional component licenses | Package metadata also lists 0BSD, MIT, Zlib, and CC0-1.0; see licenses shipped with the installed package. |
| `pillow` | MIT-CMU | Pillow includes notices for bundled components in its installed license file. |
| `imageio` | BSD 2-Clause | |
| `imageio-ffmpeg` | BSD 2-Clause | Bundles an FFmpeg binary (see FFmpeg below). |
| `tkinterdnd2` | MIT (Python wrapper); TkDND terms (bundled extension) | Drag-and-drop is optional. The TkDND notice is reproduced below. |

Transitive dependencies are installed by `pip` and retain their own licenses. Their versions can vary by platform and dependency resolution; consult the license metadata distributed with each installed package.

## Models & weights

Pretrained **GoogLeNet, InceptionV3, VGG16, and VGG19** weights are fetched from
the PyTorch `torchvision` model zoo on first use and cached by PyTorch. The
weight files are not included in this repository. The models were trained on
ImageNet (ILSVRC) data; this project does not include ImageNet images or grant
access to that dataset. ImageNet's terms apply to the dataset and its use.

`torchvision` distributes these weights but does not itself relicense them. Its
documentation states that its pre-trained models "may have their own licenses or
terms and conditions derived from the dataset used for training," and that it is
the user's responsibility to determine whether they have permission to use a
model for a given purpose. The VGG16 and VGG19 weights were originally ported
from the authors' Caffe models. Anyone redistributing the weights, or a cache
that contains them, must satisfy those terms in addition to `torchvision`'s
license.

## ImageNet category labels

The file `app/imagenet_categories.py` contains the 1,000 ImageNet category names
copied from `torchvision`'s model metadata. The names identify ImageNet classes;
they are not the ImageNet images or dataset. The BSD-3-Clause license for
torchvision is at <https://github.com/pytorch/vision/blob/main/LICENSE>.

## DeepDream technique

The original **DeepDream** technique and code were created by Alexander
Mordvintsev and colleagues at Google in 2015 (originally Apache-2.0-licensed).
This project is an independent PyTorch reimplementation inspired by that work.

## TkDND

`tkinterdnd2` packages George Petasis' TkDND extension binaries along with its
MIT-licensed Python wrapper. TkDND's `license.terms` from the v2.9.2 release
is reproduced here as required by its distribution terms:

```text
This software is copyrighted by:
   Georgios Petasis, Athens, Greece.
   e-mail: petasisg@yahoo.gr, petasis@iit.demokritos.gr

   Mac portions (c) 2009-2014 Kevin Walzer/WordTech Communications LLC,
   kw@codebykevin.com

The following terms apply to all files associated with the
software unless explicitly disclaimed in individual files.

The authors hereby grant permission to use, copy, modify, distribute,
and license this software and its documentation for any purpose, provided
that existing copyright notices are retained in all copies and that this
notice is included verbatim in any distributions. No written agreement,
license, or royalty fee is required for any of the authorized uses.
Modifications to this software may be copyrighted by their authors
and need not follow the licensing terms described here, provided that
the new terms are clearly indicated on the first page of each file where
they apply.

IN NO EVENT SHALL THE AUTHORS OR DISTRIBUTORS BE LIABLE TO ANY PARTY
FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
ARISING OUT OF THE USE OF THIS SOFTWARE, ITS DOCUMENTATION, OR ANY
DERIVATIVES THEREOF, EVEN IF THE AUTHORS HAVE BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

THE AUTHORS AND DISTRIBUTORS SPECIFICALLY DISCLAIM ANY WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.  THIS SOFTWARE
IS PROVIDED ON AN "AS IS" BASIS, AND THE AUTHORS AND DISTRIBUTORS HAVE
NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.

GOVERNMENT USE: If you are acquiring this software on behalf of the
U.S. government, the Government shall have only "Restricted Rights"
in the software and related documentation as defined in the Federal
Acquisition Regulations (FARs) in Clause 52.227.19 (c) (2).  If you
are acquiring the software on behalf of the Department of Defense, the
software shall be classified as "Commercial Computer Software" and the
Government shall have only "Restricted Rights" as defined in Clause
252.227-7013 (c) (1) of DFARs.  Notwithstanding the foregoing, the
authors grant the U.S. Government and others acting in its behalf
permission to use and distribute the software in accordance with the
terms specified in this license.
```

Source and license: <https://github.com/petasis/tkdnd/tree/tkdnd-release-test-v2.9.2>.

## FFmpeg

Video encoding and audio remuxing use **FFmpeg** through `imageio-ffmpeg`.
The Python wrapper is BSD-2-Clause (<https://github.com/imageio/imageio-ffmpeg/blob/v0.6.0/LICENSE>).
Its platform wheels include FFmpeg executables, whose licenses depend on the
build. The Windows x86-64 executable installed here by `imageio-ffmpeg` 0.6.0
identifies itself as FFmpeg 7.1 `essentials_build-www.gyan.dev`. Its `-L`
output reports GPL version 3 or later; `-buildconf` enables `--enable-gpl`,
`--enable-version3`, `--enable-libx264`, and `--enable-libx265`. Gyan's build
page identifies its static Windows builds as GPLv3:
<https://www.gyan.dev/ffmpeg/builds/>. The build's exact options can be
inspected with `ffmpeg -buildconf`; license information is printed by
`ffmpeg -L`. Other platforms or externally selected executables may use
different builds and licenses. FFmpeg's license overview and source are at
<https://ffmpeg.org/legal.html> and <https://ffmpeg.org/download.html>.

This project invokes FFmpeg as a separate process and does not link against
it. The GPL status of a separately executed FFmpeg binary does not, by itself,
change this project's BSD-3-Clause license. Anyone redistributing the binary
or an application environment that contains it must meet the obligations for
that exact binary and its enabled libraries, including applicable source-code
and license-notice requirements. See the GPLv3 text at
<https://www.gnu.org/licenses/gpl-3.0.html>.

## Trademarks

"DeepDream" is associated with Google. This project is an independent
reimplementation and is not affiliated with, endorsed by, or sponsored by
Google.
