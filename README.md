# pyfastnoise2

[FastNoise2](https://github.com/Auburn/FastNoise2), a SIMD noise library, as Python wheels
with a thin `ctypes` binding. A wheel contains the FastNoise2 1.1.1 shared library and a
binding that builds node trees by name and fills NumPy arrays with 2D and 3D noise.

## Install

The wheels are attached to the
[releases](https://github.com/ZettaAI/pyfastnoise2/releases); the package is not on PyPI.
Install the wheel of your platform by its URL:

```sh
pip install https://github.com/ZettaAI/pyfastnoise2/releases/download/v1.1.1/<wheel>
```

| Platform | Wheel name ends in |
| --- | --- |
| Linux x86_64, glibc 2.28 or newer | `manylinux_2_28_x86_64.whl` |
| Linux aarch64, glibc 2.28 or newer | `manylinux_2_28_aarch64.whl` |
| macOS 11 or newer, Apple silicon | `macosx_11_0_arm64.whl` |

The binding does not use the Python C API, so each wheel is tagged `py3-none` and serves
every Python version from 3.10 on. NumPy 1.21 or newer is the only dependency.

## Example

fBm over Simplex noise at arbitrary positions:

```python
import numpy as np
import pyfastnoise2 as fn2

octaves, gain = 5, 0.5

source = fn2.Node("Simplex")
source.set("Feature Scale", 1.0)  # size of a lattice cell; FastNoise2's default is 100
# FractalFBm sums its octaves without normalizing them. Scaling the source by the sum
# of the octave amplitudes keeps the result within [-1, 1]. Set the minimum first:
# FastNoise2 derives the range when the maximum is set.
peak = 1.0 / sum(gain**octave for octave in range(octaves))
source.set("Output Min", -peak)
source.set("Output Max", peak)

fbm = fn2.Node("FractalFBm")
fbm.set("Source", source)
fbm.set("Octaves", octaves)
fbm.set("Gain", gain)
fbm.set("Lacunarity", 2.0)

x, y = np.random.default_rng(0).uniform(0, 10, (2, 1000)).astype(np.float32)
values = fbm.gen_position_array_2d(x, y, seed=1337)  # float32, the shape of x
```

`fn2.node_types()` lists the node types and `fn2.members("FractalFBm")` the members of
one, under FastNoise2's own names. `Node.from_encoded` builds a tree from a string of the
FastNoise2 Node Editor. Besides `gen_position_array_2d` there are `gen_uniform_grid_2d`,
`gen_single_2d` and the same three for 3D. The array methods take C-contiguous float32
arrays and an optional `out`, and raise on anything else. A tree can be up to 128 nodes
deep: generating recurses through a chain of nodes on the native stack and needs up to
about 1 MiB of it at that depth.

Member values are not range-checked, because FastNoise2 does not limit them: a billion
`Octaves` are accepted and generate practically forever. `Node.from_encoded` refuses a
damaged string, but a string from an untrusted source can describe such a tree.

Generating is thread-safe and releases the GIL: any number of threads can generate from
one tree. Changing a tree is not safe while another thread generates from it: FastNoise2
reads the members of a node without a lock, and a node that `set` replaces may be freed
under that thread. Build a tree first, then share it. A node cannot be pickled or copied,
so build it in the process that uses it; a forked child can keep using the nodes of its
parent.

## Identical results on every CPU

FastNoise2 picks the best SIMD level of the CPU at run time: SSE2, SSE4.1, AVX2 or AVX-512
on x86-64, and the one level of arm64. The library in the wheels is built with
`FASTNOISE2_STRICT_FP` and without floating-point contraction, which keeps fast-math and
fused multiply-add out of it, so that every level, compiler and platform returns the same
float32 values. The tests check this exactly, with golden values of an fBm tree and a
digest of the values of every node type, on every platform the wheels are built for and
at every level the test machine has.

FastNoise2 1.1.1 has three exceptions. `AddDimension` at 3D positions returns other values
at AVX2 than at SSE2, SSE4.1 and on arm64. `Terrace` returns 0.0 at SSE2 where the other
levels return -0.0. `DomainRotate` takes the sine and cosine of its angles from the math
library of the system.

`fn2.simd_levels()` lists the levels of the machine, `node.simd_level` is the one a node
runs at, and `fn2.Node(..., max_simd_level="SSE2")` runs a node at a lower one.

## Building from source

```sh
CXX=clang++ pip install .
```

needs Clang or AppleClang with C++17, git, CMake 3.28 or newer and Ninja or Make (pip
installs CMake and Ninja if they are missing), and network access: the build fetches
FastNoise2 and its dependency FastSIMD from GitHub at pinned commits. This also holds for
the source distribution, which does not contain them. To build offline, pass local
checkouts with `-C cmake.define.FETCHCONTENT_SOURCE_DIR_FASTNOISE2=<dir>` and
`-C cmake.define.CPM_FastSIMD_SOURCE=<dir>`.

The build refuses GCC, because FastNoise2 built with GCC 11 ran 1.8 times slower than
built with Clang 14. `-C cmake.define.PYFASTNOISE2_ALLOW_GCC=ON` builds with it anyway.

If Clang reports `cannot find -lstdc++`, it uses the directories of the newest GCC on the
machine, which lacks its C++ library: install the `libstdc++-N-dev` package of that GCC
version, `libstdc++-12-dev` next to `gcc-12`.

Run the tests against the installed package with `pip install pytest && pytest`.

The version is the version of FastNoise2; a release that only changes the packaging is
`1.1.1.postN`. Pushing a tag `v<version>` builds the wheels and creates the release.

## Licenses

pyfastnoise2 is MIT licensed, see [LICENSE](LICENSE). The wheels contain FastNoise2 and
FastSIMD, both MIT licensed and copyright Jordan Peck; their license texts are in
[LICENSES](LICENSES) and in every wheel.
