"""Tests of the binding. They run against the installed distribution."""

from __future__ import annotations

import base64
import copy
import gc
import hashlib
import importlib.metadata
import itertools
import numbers
import os
import pickle
import platform
import re
import signal
import sys
import threading
import time
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

import pyfastnoise2 as fn2
from pyfastnoise2 import _binding

SEED = 1337
# Five octaves with gain 0.5 sum to at most 1 + 0.5 + ... + 0.5**4 times the source.
PEAK = 1.0 / 1.9375

# Positions that are exact in float32, in and far outside the first lattice cells.
X = np.array(
    [0.0, 0.25, -0.75, 1.5, 3.125, -7.5, 12.875, 0.015625, 100.0625, -250.5, 1000.25, -4096.75],
    dtype=np.float32,
)
Y = np.array(
    [0.0, -0.5, 2.25, -1.125, 0.0625, 9.75, -31.5, 0.984375, -64.125, 500.25, -2000.5, 8191.375],
    dtype=np.float32,
)
Z = np.array(
    [0.0, 0.75, -1.25, 2.5, -0.03125, 17.25, 5.625, -0.4375, 33.5, -120.75, 640.125, -1024.5],
    dtype=np.float32,
)

# Bit patterns of simplex_fbm() at (X, Y) and (X, Y, Z) with SEED. FastNoise2 is built with
# strict floating point, so every platform and SIMD level must reproduce them exactly.
GOLDEN_2D = np.array(
    [
        0xB3F80000, 0xBF1CEC9A, 0xBD1826E3, 0xBD88C16C, 0x3ECB83B6, 0xBE80B06E,
        0x3E124F20, 0xBF0D6303, 0xBEE0A88D, 0xBE10D16A, 0x3DE782B0, 0xBEBE631E,
    ],
    dtype=np.uint32,
).view(np.float32)  # fmt: skip
GOLDEN_3D = np.array(
    [
        0x00000000, 0x3E461D1A, 0xBDA90386, 0x3DC08810, 0x3E95DC82, 0xBF033020,
        0xBE92E8C1, 0xBC3C3E54, 0x3DAA585E, 0x3EB4F6A6, 0x3C3AB7A8, 0xBEED33AC,
    ],
    dtype=np.uint32,
).view(np.float32)  # fmt: skip

NEGATIVE_ZERO = 0x80000000
# The deepest tree, as the Node docstring and the README state it.
MAX_DEPTH = 128

# Members with which a node of these types is neither constant nor a copy of its source.
# DomainRotate takes the sines and cosines of its angles from the math library of the
# system. Those of these three float32 angles are each within 0.005 ULP of a float32, so
# every library that is accurate to 0.99 ULP returns the same six values. Each fused
# multiply-add a compiler makes of the rotation matrix changes the matrix for them.
LONE_NODE_MEMBERS: dict[str, dict[str, float]] = {
    "Gradient": {"X Multiplier": 1.37, "Y Multiplier": -0.73, "Z Multiplier": 0.41},
    "DomainScale": {"Scaling": 1.37},
    "DomainOffset": {"X Offset": 0.31, "Y Offset": -0.57, "Z Offset": 0.83},
    "DomainRotate": {
        "Yaw": 0.8332262635231018,
        "Pitch": 0.9995752573013306,
        "Roll": 0.9233737587928772,
    },
    "DomainAxisScale": {"X Scaling": 1.37, "Y Scaling": 0.73, "Z Scaling": 1.19},
    "Fade": {"Fade": 0.37},
}

# lattice_digest(lone_node(node_type)) of every node type, the same on every platform and
# at every SIMD level.
GOLDEN_DIGESTS = {
    "Constant": "fef951e6c76ad6a0",
    "White": "32de67f7b31ca1b3",
    "Checkerboard": "4badd1e8307ad842",
    "SineWave": "ceb40b0a84ac4658",
    "Gradient": "f73fb343d4c680e7",
    "DistanceToPoint": "0113762fdb558d2a",
    "Simplex": "9ddce4109cea39b0",
    "SuperSimplex": "472d232338bf68de",
    "Perlin": "c49418e1cf6bd1ab",
    "Value": "dd11910beea13065",
    "CellularValue": "fbe23635b38767eb",
    "CellularDistance": "8405c65ca496f475",
    "CellularLookup": "12adbe4fbf02371d",
    "FractalFBm": "7792a499b5ce192e",
    "PingPong": "83e88efce9ef1d91",
    "FractalRidged": "177c0021ecbeb8d9",
    "DomainWarpSimplex": "e5c0b54421b8c8f4",
    "DomainWarpSuperSimplex": "befca694ed43f511",
    "DomainWarpGradient": "388f84cebf6445e1",
    "DomainWarpFractalProgressive": "e843d6db7d1c202b",
    "DomainWarpFractalIndependent": "2942453b1cc918e7",
    "Add": "3525f76be90d952a",
    "Subtract": "c372d050b52cc612",
    "Multiply": "3b8dd613f5b4709b",
    "Divide": "1a19304354597ed4",
    "Abs": "4cb753076fc9cee7",
    "Min": "6e0cc3b2ed28f108",
    "Max": "806767dc76c2a74d",
    "MinSmooth": "e5d7ad57ed121585",
    "MaxSmooth": "c968e53e8ef74ed6",
    "SignedSquareRoot": "f733a9d3c3a23f26",
    "PowFloat": "b09352a6f628a9b5",
    "PowInt": "60bb5e62f58b40d4",
    "DomainScale": "4b3359f28644a01c",
    "DomainOffset": "4bde93a02c86b415",
    "DomainRotate": "d659c0a382dd2591",
    "DomainAxisScale": "38d7b9593dfdb287",
    "SeedOffset": "2aadd9127526a013",
    "ConvertRGBA8": "502f677b198142c7",
    "GeneratorCache": "9ddce4109cea39b0",
    "Fade": "70a012ce17aaaee0",
    "Remap": "87639f7fbdf4339a",
    "Terrace": "cadf6d7ad0075962",
    "AddDimension": "162eb45ad62fd7ea",
    "RemoveDimension": "78e659d180f72ebc",
    "Modulus": "bd13a26b9e9722c2",
    "DomainRotatePlane": "0483e68f168c0788",
}

# A 4 x 3 grid and a 4 x 3 x 2 grid.
GRID_2D: dict[str, Any] = {"x_count": 4, "y_count": 3, "x_step": 0.25, "y_step": 0.5, "seed": SEED}
GRID_3D: dict[str, Any] = {**GRID_2D, "z_count": 2, "z_step": 1.0}

# simplex_fbm() and mixed_tree() serialized by FastNoise::Metadata::SerialiseNodeData of
# FastNoise2 1.1.1.
FBM_ENCODED = "DQAF@BCQY@ACAPxAIIQS/GAghBD8M"
MIXED_ENCODED = "IgkNAAQ@BI@AgQAkG@BwD8IBw@BQCCw@BBAIAI@BwAg@ADgB@BCwAAQD8ECwAAgD4EAw@AD8K/wAABA=="


def simplex_fbm(max_simd_level: str | None = None) -> fn2.Node:
    """fBm of five octaves over Simplex noise, scaled into [-1, 1]."""
    source = fn2.Node("Simplex", max_simd_level=max_simd_level)
    source.set("Feature Scale", 1.0)
    source.set("Output Min", -PEAK)
    source.set("Output Max", PEAK)
    node = fn2.Node("FractalFBm", max_simd_level=max_simd_level)
    node.set("Source", source)
    node.set("Octaves", 5)
    return node


def mixed_tree() -> fn2.Node:
    """A tree with every kind of member, a per-dimension member and a node used twice."""
    cellular = fn2.Node("CellularDistance")
    cellular.set("Feature Scale", 2.0)
    cellular.set("Distance Function", "Manhattan")
    cellular.set("Return Type", 1)
    cellular.set("Distance Index 1", 2)
    cellular.set("Grid Jitter", 0.75)
    simplex = fn2.Node("Simplex")
    simplex.set("Feature Scale", 1.5)
    simplex.set("Seed Offset", 7)
    fbm = fn2.Node("FractalFBm")
    fbm.set("Source", simplex)
    fbm.set("Octaves", 4)
    fbm.set("Lacunarity", 2.5)
    fbm.set("Gain", cellular)
    fbm.set("Weighted Strength", 0.25)
    offset = fn2.Node("DomainOffset")
    offset.set("Source", fbm)
    offset.set("X Offset", 0.5)
    offset.set("Y Offset", simplex)
    return offset


def grid_positions(
    offsets: tuple[float, ...], counts: tuple[int, ...], steps: tuple[float, ...]
) -> list[npt.NDArray[np.float32]]:
    """The positions of a uniform grid, x first, computed in float32 like the library's."""
    axes = [
        np.arange(count, dtype=np.float32) * np.float32(step) + np.float32(offset)
        for offset, count, step in zip(offsets, counts, steps, strict=True)
    ]
    grids = np.meshgrid(*reversed(axes), indexing="ij")
    return [np.ascontiguousarray(grid) for grid in reversed(grids)]


def test_node_types() -> None:
    names = fn2.node_types()
    assert {"Simplex", "FractalFBm", "DomainOffset"} <= set(names)
    assert len(names) == len(set(names))
    for name in names:
        assert fn2.Node(name).node_type == name


def test_members() -> None:
    assert fn2.members("FractalFBm") == {
        "Octaves": fn2.Member("int", 3),
        "Lacunarity": fn2.Member("float", 2.0),
        "Source": fn2.Member("node", None),
        "Gain": fn2.Member("hybrid", 0.5),
        "Weighted Strength": fn2.Member("hybrid", 0.0),
    }
    assert fn2.members("Fade")["Interpolation"] == fn2.Member(
        "enum", "Linear", ("Linear", "Hermite", "Quintic")
    )
    offsets = [name for name in fn2.members("DomainOffset") if name.endswith("Offset")]
    assert offsets == ["X Offset", "Y Offset", "Z Offset", "W Offset"]
    with pytest.raises(ValueError, match="unknown node type 'Nope'"):
        fn2.members("Nope")
    with pytest.raises(ValueError, match="unknown node type"):
        fn2.Node("Nope")


@pytest.mark.parametrize(
    ("node_type", "name", "value", "error"),
    [
        ("FractalFBm", "Octave", 3, ValueError),
        ("FractalFBm", "Octaves", 3.0, TypeError),
        ("FractalFBm", "Octaves", 2**31, ValueError),
        ("FractalFBm", "Lacunarity", "2", TypeError),
        ("FractalFBm", "Lacunarity", 10**400, ValueError),
        ("FractalFBm", "Gain", 10**400, ValueError),
        ("FractalFBm", "Lacunarity", fn2.Node("Simplex"), TypeError),
        ("FractalFBm", "Gain", "0.5", TypeError),
        ("FractalFBm", "Source", 1.0, TypeError),
        ("Fade", "Interpolation", "Cubic", ValueError),
        ("Fade", "Interpolation", 3, ValueError),
        ("Fade", "Interpolation", -1, ValueError),
        ("Fade", "Interpolation", 1.0, TypeError),
        # Only a domain warp node fits here.
        ("DomainWarpFractalProgressive", "Domain Warp Source", fn2.Node("Simplex"), TypeError),
    ],
)
def test_set_rejects(node_type: str, name: str, value: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        fn2.Node(node_type).set(name, value)


def test_set_enum_by_name_and_index() -> None:
    def values(function: str | int) -> npt.NDArray[np.float32]:
        node = fn2.Node("CellularDistance")
        node.set("Feature Scale", 1.0)
        node.set("Distance Function", function)
        return node.gen_position_array_2d(X, Y, SEED)

    assert np.array_equal(values("Manhattan"), values(2))
    assert not np.array_equal(values("Manhattan"), values("Euclidean"))


def test_member_value_conversion_may_use_nodes() -> None:
    class Number:
        """A number whose conversion creates and changes a node."""

        def __float__(self) -> float:
            fn2.Node("Simplex").set("Feature Scale", 2.0)
            return 0.25

        def __index__(self) -> int:
            fn2.Node("Simplex").set("Seed Offset", 2)
            return 2

    numbers.Real.register(Number)
    number: Any = Number()
    node, expected = simplex_fbm(), simplex_fbm()
    for name, value in {"Lacunarity": 0.25, "Gain": 0.25, "Octaves": 2}.items():
        node.set(name, number)
        expected.set(name, value)
    assert np.array_equal(
        node.gen_position_array_2d(X, Y, SEED), expected.gen_position_array_2d(X, Y, SEED)
    )
    node, expected = fn2.Node("CellularDistance"), fn2.Node("CellularDistance")
    node.set("Distance Function", number)
    expected.set("Distance Function", "Manhattan")
    assert np.array_equal(
        node.gen_position_array_2d(X, Y, SEED), expected.gen_position_array_2d(X, Y, SEED)
    )


def test_hybrid_takes_node_or_float() -> None:
    expected = simplex_fbm().gen_position_array_2d(X, Y, SEED)
    gain = fn2.Node("Constant")
    gain.set("Value", 0.25)
    node = simplex_fbm()
    node.set("Gain", gain)
    assert not np.array_equal(node.gen_position_array_2d(X, Y, SEED), expected)
    node.set("Gain", 0.5)
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), expected)


def test_hybrid_set_to_a_float_lets_go_of_its_node() -> None:
    node = simplex_fbm()
    gain = fn2.Node("Add")
    gain.set("LHS", fn2.Node("Simplex"))
    node.set("Gain", gain)
    with pytest.raises(ValueError, match="depends on this node"):
        gain.set("RHS", node)
    node.set("Gain", 0.5)
    gain.set("RHS", node)
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), GOLDEN_2D)
    gain.gen_single_2d(0.0, 0.0, SEED)


def test_extreme_integers() -> None:
    node = fn2.Node("Simplex")
    for value in (-(2**31), 2**31 - 1):
        node.set("Seed Offset", value)
        assert np.isfinite(node.gen_single_2d(0.5, 0.5, value))
        assert np.isfinite(node.gen_position_array_3d(X, Y, Z, value)).all()


def test_golden_values() -> None:
    node = simplex_fbm()
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), GOLDEN_2D)
    assert np.array_equal(node.gen_position_array_3d(X, Y, Z, SEED), GOLDEN_3D)


def test_simd_levels() -> None:
    levels = fn2.simd_levels()
    assert levels
    assert fn2.Node("Simplex").simd_level == levels[-1]
    for level in levels:
        node = simplex_fbm(level)
        assert node.simd_level == level
        assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), GOLDEN_2D)
        assert np.array_equal(node.gen_position_array_3d(X, Y, Z, SEED), GOLDEN_3D)
    # Levels the library is not compiled for, or the CPU lacks, would crash it.
    for level in ("SCALAR", "AVX", "nope"):
        with pytest.raises(ValueError, match="max_simd_level"):
            fn2.Node("Simplex", max_simd_level=level)


def test_default_level_is_the_best_of_the_cpu() -> None:
    if platform.machine() in ("arm64", "aarch64"):
        best = "AARCH64"
    elif sys.platform == "linux":
        flags = re.search(r"^flags\s*:(.*)$", Path("/proc/cpuinfo").read_text(), re.MULTILINE)
        assert flags is not None
        best = "SSE2"
        # The CPU features FastSIMD asks for, from level to level.
        for level, features in {
            "SSE4.1": {"pni", "ssse3", "sse4_1"},
            "AVX2": {"sse4_2", "popcnt", "xsave", "avx", "avx2"},
            "AVX512": {"avx512f", "avx512vl", "avx512dq", "avx512bw"},
        }.items():
            if not features <= set(flags[1].split()):
                break
            best = level
    else:
        pytest.skip("no list of the CPU features of this machine")
    assert fn2.simd_levels()[-1] == best
    assert fn2.Node("Simplex").simd_level == best


def test_levels_of_a_feature_set() -> None:
    # FastSIMD::FeatureSet values of an SSE2, an SSE4.2, an AVX2, an AVX-512 and an arm64 CPU.
    for feature_set, levels in {
        0x0000E: ("SSE2",),
        0x000FE: ("SSE2", "SSE4.1"),
        0x003FE: ("SSE2", "SSE4.1", "AVX2"),
        0x03FFE: ("SSE2", "SSE4.1", "AVX2", "AVX512"),
        0x1C000: ("AARCH64",),
    }.items():
        assert _binding._levels_within(feature_set) == levels


def lone_node(node_type: str, max_simd_level: str | None = None) -> fn2.Node:
    """A node of ``node_type`` that has its node members set and is not constant."""
    seed_offsets = itertools.count()

    def source() -> fn2.Node:
        simplex = fn2.Node("Simplex", max_simd_level=max_simd_level)
        simplex.set("Feature Scale", 1.0)
        # Each source of a node has its own values.
        simplex.set("Seed Offset", next(seed_offsets))
        return simplex

    node = fn2.Node(node_type, max_simd_level=max_simd_level)
    for name, member in fn2.members(node_type).items():
        if name == "Feature Scale":
            node.set(name, 1.0)
        elif name == "RHS":
            node.set(name, 0.37)
        # PowFloat is constant unless one of its two hybrid members is a node.
        elif member.kind == "node" or name == "LHS" or (node_type, name) == ("PowFloat", "Pow"):
            simplex = source()
            try:
                node.set(name, simplex)
            except TypeError:
                warp = fn2.Node("DomainWarpGradient", max_simd_level=max_simd_level)
                warp.set("Source", simplex)
                node.set(name, warp)
    for name, value in LONE_NODE_MEMBERS.get(node_type, {}).items():
        node.set(name, value)
    return node


def lattice_digest(node: fn2.Node) -> str:
    """Digest of the values of ``node`` on a 2D and a 3D lattice."""
    x, y, z = grid_positions((-3.7, 2.1, 0.4), (8, 8, 4), (0.37, 0.41, 0.53))
    values = [node.gen_position_array_2d(x, y, SEED).ravel()]
    # FastNoise2 1.1.1: AddDimension at 3D positions differs at AVX2 from the other levels.
    if node.node_type != "AddDimension":
        values.append(node.gen_position_array_3d(x, y, z, SEED).ravel())
    bits = np.concatenate(values).view(np.uint32)
    # Equal values: SSE2 returns 0.0 where the other levels return -0.0 for Terrace.
    bits[bits == NEGATIVE_ZERO] = 0
    return hashlib.sha256(bits.tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("node_type", fn2.node_types())
def test_node_type_at_every_simd_level(node_type: str) -> None:
    for level in fn2.simd_levels():
        assert lattice_digest(lone_node(node_type, level)) == GOLDEN_DIGESTS[node_type], level


def test_golden_digests_tell_the_node_types_apart() -> None:
    digests = dict(GOLDEN_DIGESTS)
    # A cache returns the values of its source.
    assert digests.pop("GeneratorCache") == digests["Simplex"]
    assert len(set(digests.values())) == len(digests)
    x, y, _ = grid_positions((-3.7, 2.1, 0.4), (8, 8, 4), (0.37, 0.41, 0.53))
    for node_type in set(digests) - {"Constant"}:
        values = lone_node(node_type).gen_position_array_2d(x, y, SEED)
        assert len(np.unique(values.view(np.uint32))) > 1, node_type


def test_generator_cache_returns_the_values_of_its_source() -> None:
    x, y, z = grid_positions((-3.7, 2.1, 0.4), (8, 8, 4), (0.37, 0.41, 0.53))
    for level in fn2.simd_levels():
        source = lone_node("Simplex", level)
        cache = fn2.Node("GeneratorCache", max_simd_level=level)
        cache.set("Source", source)
        # The second use of the cache at a position takes the values kept by the first.
        twice = fn2.Node("Add", max_simd_level=level)
        twice.set("LHS", cache)
        twice.set("RHS", cache)
        once = source.gen_position_array_2d(x, y, SEED)
        assert np.array_equal(twice.gen_position_array_2d(x, y, SEED), once + once)
        once = source.gen_position_array_3d(x, y, z, SEED)
        assert np.array_equal(twice.gen_position_array_3d(x, y, z, SEED), once + once)


def test_tree_of_mixed_simd_levels_is_rejected() -> None:
    levels = fn2.simd_levels()
    if levels[0] == levels[-1]:
        pytest.skip("this CPU has one SIMD level")
    source = fn2.Node("Simplex", max_simd_level=levels[0])
    with pytest.raises(ValueError, match="SIMD level"):
        fn2.Node("FractalFBm").set("Source", source)


def test_position_array_matches_single() -> None:
    node = simplex_fbm()
    values_2d = node.gen_position_array_2d(X, Y, SEED)
    values_3d = node.gen_position_array_3d(X, Y, Z, SEED)
    assert values_2d.dtype == values_3d.dtype == np.float32
    for index, (x, y, z) in enumerate(zip(X.tolist(), Y.tolist(), Z.tolist(), strict=True)):
        assert node.gen_single_2d(x, y, SEED) == values_2d[index]
        assert node.gen_single_3d(x, y, z, SEED) == values_3d[index]


def test_position_array_offsets() -> None:
    node = simplex_fbm()
    shifted = node.gen_position_array_2d(X, Y, SEED, x_offset=0.5, y_offset=-2.0)
    expected = node.gen_position_array_2d(X + np.float32(0.5), Y - np.float32(2.0), SEED)
    assert np.array_equal(shifted, expected)
    shifted = node.gen_position_array_3d(X, Y, Z, SEED, x_offset=0.5, y_offset=-2.0, z_offset=0.25)
    expected = node.gen_position_array_3d(
        X + np.float32(0.5), Y - np.float32(2.0), Z + np.float32(0.25), SEED
    )
    assert np.array_equal(shifted, expected)


def test_uniform_grid_matches_position_array() -> None:
    node = simplex_fbm()
    offsets, counts, steps = (0.3, -1.7, 4.2), (23, 11, 5), (0.0437, 0.0291, 0.113)

    grid = node.gen_uniform_grid_2d(
        x_offset=offsets[0],
        y_offset=offsets[1],
        x_count=counts[0],
        y_count=counts[1],
        x_step=steps[0],
        y_step=steps[1],
        seed=SEED,
    )
    assert grid.shape == (11, 23)
    x, y = grid_positions(offsets[:2], counts[:2], steps[:2])
    assert np.array_equal(grid, node.gen_position_array_2d(x, y, SEED))

    grid = node.gen_uniform_grid_3d(
        x_offset=offsets[0],
        y_offset=offsets[1],
        z_offset=offsets[2],
        x_count=counts[0],
        y_count=counts[1],
        z_count=counts[2],
        x_step=steps[0],
        y_step=steps[1],
        z_step=steps[2],
        seed=SEED,
    )
    assert grid.shape == (5, 11, 23)
    x, y, z = grid_positions(offsets, counts, steps)
    assert np.array_equal(grid, node.gen_position_array_3d(x, y, z, SEED))


def test_out() -> None:
    node = simplex_fbm()
    x, y, z = (np.ascontiguousarray(a.reshape(3, 4)) for a in (X, Y, Z))

    out = np.full((3, 4), np.nan, dtype=np.float32)
    assert node.gen_position_array_2d(x, y, SEED, out=out) is out
    assert np.array_equal(out.ravel(), GOLDEN_2D)
    assert node.gen_position_array_3d(x, y, z, SEED, out=out) is out
    assert np.array_equal(out.ravel(), GOLDEN_3D)

    grid = node.gen_uniform_grid_2d(**GRID_2D)
    assert node.gen_uniform_grid_2d(**GRID_2D, out=out) is out
    assert np.array_equal(out, grid)
    out = np.empty((2, 3, 4), dtype=np.float32)
    grid = node.gen_uniform_grid_3d(**GRID_3D)
    assert node.gen_uniform_grid_3d(**GRID_3D, out=out) is out
    assert np.array_equal(out, grid)


def test_no_values() -> None:
    node = simplex_fbm()
    empty = np.empty((0, 3), dtype=np.float32)
    assert node.gen_position_array_2d(empty, empty, SEED).shape == (0, 3)
    assert node.gen_position_array_3d(empty, empty, empty, SEED).shape == (0, 3)
    assert node.gen_uniform_grid_2d(**GRID_2D | {"x_count": 0}).shape == (3, 0)
    assert node.gen_uniform_grid_3d(**GRID_3D | {"y_count": 0}).shape == (2, 0, 4)


def read_only(array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    array = array.copy()
    array.flags.writeable = False
    return array


def float32(*shape: int, order: Literal["C", "F"] = "C") -> npt.NDArray[np.float32]:
    return np.empty(shape, dtype=np.float32, order=order)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda n: n.gen_position_array_2d(X.astype(np.float64), Y, SEED), TypeError),
        (lambda n: n.gen_position_array_2d(X, Y.astype(">f4"), SEED), TypeError),
        (lambda n: n.gen_position_array_2d(X.tolist(), Y, SEED), TypeError),
        (lambda n: n.gen_position_array_2d(X, Y[:-1], SEED), ValueError),
        (lambda n: n.gen_position_array_2d(X[::2], Y[::2], SEED), ValueError),
        (lambda n: n.gen_position_array_3d(X, Y, Z.reshape(3, 4), SEED), ValueError),
        (lambda n: n.gen_position_array_3d(X, Y, Z.astype(np.int32), SEED), TypeError),
        (lambda n: n.gen_position_array_2d(X, Y, SEED, out=float32(11)), ValueError),
        (lambda n: n.gen_position_array_2d(X, Y, SEED, out=np.empty(12)), TypeError),
        (lambda n: n.gen_position_array_2d(X, Y, SEED, out=read_only(X)), ValueError),
        # The library reads positions again after it has written values.
        (lambda n: n.gen_position_array_2d(X, Y, SEED, out=X), ValueError),
        (lambda n: n.gen_position_array_2d(Y, X, SEED, out=X), ValueError),
        (lambda n: n.gen_position_array_3d(X, Y, Z, SEED, out=Z), ValueError),
        (lambda n: n.gen_position_array_2d(X, Y, 2**31), ValueError),
        (lambda n: n.gen_position_array_2d(X, Y, -(2**31) - 1), ValueError),
        (lambda n: n.gen_position_array_2d(X, Y, 1.0), TypeError),
        (lambda n: n.gen_position_array_2d(X, Y, SEED, x_offset="0"), TypeError),
        (lambda n: n.gen_position_array_2d(X, Y, SEED, x_offset=10**400), ValueError),
        (lambda n: n.gen_single_2d(0.0, "0", SEED), TypeError),
        (lambda n: n.gen_single_3d(0.0, 0.0, 0.0, 2**31), ValueError),
        (lambda n: n.gen_uniform_grid_2d(**GRID_2D | {"x_count": -1}), ValueError),
        (lambda n: n.gen_uniform_grid_2d(**GRID_2D | {"x_count": 4.0}), TypeError),
        (lambda n: n.gen_uniform_grid_2d(**GRID_2D | {"y_step": "1"}), TypeError),
        (lambda n: n.gen_uniform_grid_3d(**GRID_3D | {"z_offset": None}), TypeError),
        # More values than a C int counts.
        (lambda n: n.gen_uniform_grid_2d(**GRID_2D | {"x_count": 2**31}), ValueError),
        (lambda n: n.gen_uniform_grid_3d(**GRID_3D | {"x_count": 2**30}), ValueError),
        # The first axis of the output is y.
        (lambda n: n.gen_uniform_grid_2d(**GRID_2D, out=float32(4, 3)), ValueError),
        (lambda n: n.gen_uniform_grid_2d(**GRID_2D, out=float32(3, 4, order="F")), ValueError),
        (lambda n: n.gen_uniform_grid_3d(**GRID_3D, out=float32(3, 4)), ValueError),
    ],
)  # fmt: skip
def test_generation_rejects(call: Callable[[fn2.Node], object], error: type[Exception]) -> None:
    with pytest.raises(error):
        call(simplex_fbm())


def test_unset_node_member() -> None:
    fbm = fn2.Node("FractalFBm")
    for generate in (
        lambda: fbm.gen_single_2d(0.0, 0.0, SEED),
        lambda: fbm.gen_single_3d(0.0, 0.0, 0.0, SEED),
        lambda: fbm.gen_position_array_2d(X, Y, SEED),
        lambda: fbm.gen_position_array_3d(X, Y, Z, SEED),
        lambda: fbm.gen_uniform_grid_2d(**GRID_2D),
        lambda: fbm.gen_uniform_grid_3d(**GRID_3D),
    ):
        with pytest.raises(RuntimeError, match="set 'Source'"):
            generate()
    # A node with an unset node member cannot become part of a tree either.
    with pytest.raises(ValueError, match="set 'Source'"):
        fn2.Node("DomainScale").set("Source", fbm)
    fbm.set("Source", fn2.Node("Simplex"))
    fbm.gen_single_2d(0.0, 0.0, SEED)


def test_cycle_is_rejected() -> None:
    inner = fn2.Node("DomainScale")
    inner.set("Source", fn2.Node("Simplex"))
    outer = fn2.Node("DomainScale")
    outer.set("Source", inner)
    with pytest.raises(ValueError, match="depends on this node"):
        inner.set("Source", outer)
    with pytest.raises(ValueError, match="depends on this node"):
        outer.set("Source", outer)
    fbm = simplex_fbm()
    with pytest.raises(ValueError, match="depends on this node"):
        fbm.set("Gain", fbm)


def chain(length: int, node_type: str = "Abs", member: str = "Source") -> list[fn2.Node]:
    """A chain of ``length`` nodes, root first: ``node_type`` nodes above a Simplex node."""
    nodes = [fn2.Node("Simplex")]
    for _ in range(length - 1):
        nodes.insert(0, fn2.Node(node_type))
        nodes[0].set(member, nodes[1])
    return nodes


def test_depth_limit() -> None:
    nodes = chain(MAX_DEPTH - 1)
    # The source of the lowest Abs node becomes a tree of two nodes.
    nodes[-2].set("Source", simplex_fbm())
    root = nodes[0]
    assert np.array_equal(root.gen_position_array_2d(X, Y, SEED), np.abs(GOLDEN_2D))
    with pytest.raises(ValueError, match="more than 128 nodes deep"):
        fn2.Node("Abs").set("Source", root)
    with pytest.raises(ValueError, match="more than 128 nodes deep"):
        fn2.Node("PowFloat").set("Pow", root)
    assert np.array_equal(root.gen_position_array_2d(X, Y, SEED), np.abs(GOLDEN_2D))


def test_depth_limit_counts_the_nodes_above() -> None:
    upper, lower = chain(3), chain(MAX_DEPTH - 1)
    # The two Abs nodes of upper above lower without its root.
    upper[1].set("Source", lower[1])
    with pytest.raises(ValueError, match="more than 128 nodes deep"):
        upper[1].set("Source", lower[0])
    with pytest.raises(ValueError, match="more than 128 nodes deep"):
        lower[-2].set("Source", chain(2)[0])
    # A tree that is too deep because it contains the node is refused for that.
    with pytest.raises(ValueError, match="depends on this node"):
        lower[-2].set("Source", upper[0])
    upper[0].gen_single_2d(0.0, 0.0, SEED)


@pytest.mark.parametrize("short_path_first", [True, False])
def test_cycle_through_a_node_reached_by_two_paths_is_rejected(short_path_first: bool) -> None:
    shared = fn2.Node("Abs")
    shared.set("Source", fn2.Node("Simplex"))
    above = fn2.Node("Abs")
    above.set("Source", shared)
    top = fn2.Node("Abs")
    top.set("Source", above)
    # Reaches `shared` as its own member and through `top` and `above`.
    both = fn2.Node("Add")
    both.set("LHS", shared if short_path_first else top)
    both.set("RHS", top if short_path_first else shared)
    # Below ten nodes, so that one set() makes all of these deeper, `shared` by two paths.
    nodes = chain(10)
    nodes[-2].set("Source", both)
    for upper in (above, top):
        with pytest.raises(ValueError, match="depends on this node"):
            shared.set("Source", upper)
    nodes[0].gen_single_2d(0.0, 0.0, SEED)


def test_refused_node_member_changes_nothing() -> None:
    warp = fn2.Node("DomainWarpGradient")
    warp.set("Source", fn2.Node("Simplex"))
    fractal = fn2.Node("DomainWarpFractalProgressive")
    fractal.set("Domain Warp Source", warp)
    # `fractal` is 100 nodes deep.
    upper = chain(100)
    upper[-2].set("Source", fractal)
    lower = chain(20)
    # Only a domain warp node fits here.
    with pytest.raises(TypeError, match="does not accept"):
        fractal.set("Domain Warp Source", lower[0])
    # `lower` got no deeper and is no member of `fractal`: all of `upper` fits below it.
    lower[-2].set("Source", upper[0])
    assert np.array_equal(
        lower[0].gen_position_array_2d(X, Y, SEED), upper[0].gen_position_array_2d(X, Y, SEED)
    )


def test_deepest_tree_fits_a_small_stack() -> None:
    # Cellular nodes with a fourth coordinate, from AddDimension, have the largest stack
    # frames. The stack is twice what the README names. The values are not looked at:
    # FastNoise2 1.1.1 returns arbitrary ones for AddDimension at 3D positions at AVX2.
    shapes: list[tuple[int, ...]] = []
    previous = threading.stack_size(2 * 1024 * 1024)
    try:
        for level in fn2.simd_levels():
            source = fn2.Node("CellularDistance", max_simd_level=level)
            for _ in range(MAX_DEPTH - 2):
                node = fn2.Node("CellularDistance", max_simd_level=level)
                node.set("Grid Jitter", source)
                source = node
            root = fn2.Node("AddDimension", max_simd_level=level)
            root.set("Source", source)
            thread = threading.Thread(
                target=lambda root=root: shapes.append(
                    root.gen_position_array_3d(X, Y, Z, SEED).shape
                )
            )
            thread.start()
            thread.join()
    finally:
        threading.stack_size(previous)
    assert shapes == [X.shape] * len(fn2.simd_levels())


def test_concurrent_set_cannot_build_a_cycle() -> None:
    def chain() -> fn2.Node:
        node = fn2.Node("Abs")
        node.set("Source", fn2.Node("Simplex"))
        return node

    def link(parent: fn2.Node, source: fn2.Node) -> None:
        barrier.wait()
        try:
            parent.set("Source", source)
        except ValueError:
            refused.append(parent)

    interval = sys.getswitchinterval()
    # Switch threads as often as possible, also between a cycle check and its set.
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(2_000):
            first, second = chain(), chain()
            barrier = threading.Barrier(2)
            refused: list[fn2.Node] = []
            threads = [
                threading.Thread(target=link, args=(first, second)),
                threading.Thread(target=link, args=(second, first)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert len(refused) == 1
    finally:
        sys.setswitchinterval(interval)


def test_concurrent_set_of_one_hybrid_member() -> None:
    def constant() -> fn2.Node:
        node = fn2.Node("Constant")
        node.set("Value", 0.25)
        return node

    def to_node() -> None:
        source = constant()
        barrier.wait()
        node.set("Gain", source)

    def to_float() -> None:
        barrier.wait()
        node.set("Gain", 0.5)

    node = simplex_fbm()
    node.set("Gain", constant())
    with_node = node.gen_position_array_2d(X, Y, SEED)
    assert not np.array_equal(with_node, GOLDEN_2D)
    interval = sys.getswitchinterval()
    # Switch threads as often as possible, also between the two halves of a set().
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(2_000):
            barrier = threading.Barrier(2)
            threads = [threading.Thread(target=to_node), threading.Thread(target=to_float)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            # The member is what the later set() gave it, in FastNoise2 as in the node.
            values = node.gen_position_array_2d(X, Y, SEED)
            assert np.array_equal(values, with_node if "Gain" in node._sources else GOLDEN_2D)
    finally:
        sys.setswitchinterval(interval)
    assert not _binding._dropped


def test_source_deleted_before_use() -> None:
    source = fn2.Node("Simplex")
    source.set("Feature Scale", 1.0)
    source.set("Output Min", -PEAK)
    source.set("Output Max", PEAK)
    node = fn2.Node("FractalFBm")
    node.set("Source", source)
    node.set("Octaves", 5)
    del source
    gc.collect()
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), GOLDEN_2D)


def test_collected_nodes_are_freed() -> None:
    # FastNoise2 gives the memory of a freed node to a later one, so the handles of
    # dropped trees repeat. Unfreed, the trees below would have 20,000 handles per level.
    for level in fn2.simd_levels():
        handles = set()
        for _ in range(10_000):
            node = simplex_fbm(level)
            handles |= {node._handle, node._sources["Source"]._handle}
        assert len(handles) < 5_000, level
    del node
    # Freed when dropped, not by the next call.
    assert not _binding._dropped


@pytest.mark.skipif(
    getattr(sys, "_is_gil_enabled", lambda: True)(), reason="the interpreter has a GIL"
)
def test_threads_drop_nodes_without_a_gil(monkeypatch: pytest.MonkeyPatch) -> None:
    unraisable: list[BaseException | None] = []
    # A finalizer, which frees its node, reports an error here.
    monkeypatch.setattr(sys, "unraisablehook", lambda error: unraisable.append(error.exc_value))
    deadline = time.monotonic() + 2.0

    def churn() -> None:
        while time.monotonic() < deadline:
            fn2.Node("Simplex")
            simplex_fbm()

    with ThreadPoolExecutor(max_workers=8) as pool:
        for thread in [pool.submit(churn) for _ in range(8)]:
            thread.result()
    assert not unraisable
    assert not _binding._dropped


def test_node_whose_creation_fails_late_is_freed(monkeypatch: pytest.MonkeyPatch) -> None:
    freed: list[int] = []
    drop = _binding._drop

    def record(handle: int) -> None:
        freed.append(handle)
        drop(handle)

    def fail() -> None:
        raise MemoryError

    with monkeypatch.context() as patch:
        patch.setattr(_binding, "_drop", record)
        # Runs when Node() lets go of the lock, after it has created the native node.
        patch.setattr(_binding, "_free_dropped", fail)
        with pytest.raises(MemoryError) as raised:
            fn2.Node("Simplex")
    # The traceback kept the node alive.
    del raised
    gc.collect()
    assert len(freed) == 1
    assert not _binding._dropped


def test_node_replaced_as_a_member_is_freed() -> None:
    node = fn2.Node("Abs")
    node.set("Source", simplex_fbm())
    # Nothing else refers to the first source, so its two nodes are dropped inside set(),
    # which holds the lock the frees need.
    node.set("Source", simplex_fbm())
    assert not _binding._dropped
    node = simplex_fbm()
    node.set("Gain", fn2.Node("Constant"))
    node.set("Gain", 0.5)
    assert not _binding._dropped


def test_callback_inside_set_cannot_create_or_change_nodes() -> None:
    node, other = fn2.Node("Abs"), fn2.Node("Simplex")
    refused: list[str] = []

    def use_nodes(_: object) -> None:
        for use in (
            lambda: fn2.Node("Simplex"),
            lambda: fn2.Node.from_encoded(FBM_ENCODED),
            lambda: node.set("Source", other),
            lambda: other.set("Feature Scale", 2.0),
        ):
            try:
                use()
            except RuntimeError as error:
                refused.append(str(error))

    source = fn2.Node("Simplex")
    node.set("Source", source)
    watch = weakref.ref(source, use_nodes)
    del source
    # Ends a test run in which the callback waits for the lock that its thread holds.
    signal.alarm(30)
    try:
        # The first source is dropped inside set(), which runs the callback there.
        node.set("Source", simplex_fbm())
    finally:
        signal.alarm(0)
    assert watch() is None
    assert len(refused) == 4
    assert all("already creating or changing" in message for message in refused)
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), np.abs(GOLDEN_2D))
    assert not _binding._dropped


@pytest.mark.skipif(sys.platform != "linux", reason="fork without exec is only safe on Linux")
def test_fork_in_a_callback_inside_set() -> None:
    pids: list[int] = []
    refused: list[str] = []

    def fork(_: object) -> None:
        pids.append(os.fork())
        if pids[-1] == 0:
            # Ends a child that waits for a lock.
            signal.alarm(10)
        # Parent and child are still inside set().
        try:
            fn2.Node("Simplex")
        except RuntimeError as error:
            refused.append(str(error))

    node, source = fn2.Node("Abs"), fn2.Node("Simplex")
    node.set("Source", source)
    watch = weakref.ref(source, fork)
    del source
    status = 1
    try:
        # Ends a test run in which the fork waits for the lock that its thread holds.
        signal.alarm(30)
        node.set("Source", simplex_fbm())
        # Parent and child have left set(): each can build a tree and use the one it has.
        if (
            watch() is None
            and len(refused) == 1
            and np.array_equal(simplex_fbm().gen_position_array_3d(X, Y, Z, SEED), GOLDEN_3D)
            and np.array_equal(node.gen_position_array_2d(X, Y, SEED), np.abs(GOLDEN_2D))
        ):
            status = 0
    finally:
        signal.alarm(0)
        if pids == [0]:
            os._exit(status)
    assert status == 0
    assert os.waitstatus_to_exitcode(os.waitpid(pids[0], 0)[1]) == 0


def test_node_outlives_the_trees_it_was_in() -> None:
    def churn() -> None:
        """Collect what was dropped and hand its memory to other nodes."""
        gc.collect()
        for _ in range(2_000):
            mixed_tree()

    node = simplex_fbm()
    gain = fn2.Node("Constant")
    gain.set("Value", 0.25)

    parent = fn2.Node("Abs")
    parent.set("Source", node)
    del parent
    churn()
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), GOLDEN_2D)

    parent = fn2.Node("Abs")
    parent.set("Source", node)
    parent.set("Source", fn2.Node("Simplex"))
    churn()
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), GOLDEN_2D)

    node.set("Gain", gain)
    node.set("Gain", 0.5)
    churn()
    assert gain.gen_single_2d(0.0, 0.0, SEED) == 0.25
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), GOLDEN_2D)


def cached_simplex(feature_scale: float, level: str | None = None) -> tuple[fn2.Node, fn2.Node]:
    """A GeneratorCache node and its Simplex source."""
    source = fn2.Node("Simplex", max_simd_level=level)
    source.set("Feature Scale", feature_scale)
    cache = fn2.Node("GeneratorCache", max_simd_level=level)
    cache.set("Source", source)
    return cache, source


def test_generator_cache_follows_its_source() -> None:
    # One position again and again: the first batch of a call then equals the last batch
    # of the call before, which is what the cache compares.
    x, y, z = (np.full(96, value, dtype=np.float32) for value in (0.25, -0.5, 0.75))

    def values(node: fn2.Node) -> list[Any]:
        return [
            node.gen_single_2d(0.25, -0.5, SEED),
            node.gen_single_3d(0.25, -0.5, 0.75, SEED),
            node.gen_position_array_2d(x, y, SEED).tolist(),
            node.gen_position_array_3d(x, y, z, SEED).tolist(),
        ]

    for level in fn2.simd_levels():
        cache, source = cached_simplex(1.0, level)
        expected = [values(cached_simplex(scale, level)[1]) for scale in (1.0, 3.0)]
        assert expected[0] != expected[1]
        assert values(cache) == expected[0]
        source.set("Feature Scale", 3.0)
        assert values(cache) == expected[1]

    # With a coordinate added above the cache. FastNoise2 1.1.1 returns arbitrary values
    # for AddDimension at 3D positions at AVX2, so that case runs at the lowest level.
    level = fn2.simd_levels()[0]
    cache, source = cached_simplex(1.0, level)
    wider = fn2.Node("AddDimension", max_simd_level=level)
    wider.set("Source", cache)
    before = values(wider)
    source.set("Feature Scale", 3.0)
    after = values(wider)
    assert all(old != new for old, new in zip(before, after, strict=True))


def test_generator_cache_does_not_return_values_of_a_freed_node() -> None:
    # A source mostly gets the address of the one freed just before it.
    scales = np.random.default_rng(0).uniform(0.5, 5.0, 3_000).tolist()
    for scale in scales:
        cache, source = cached_simplex(scale)
        assert cache.gen_single_2d(0.25, -0.5, SEED) == source.gen_single_2d(0.25, -0.5, SEED)
        del cache, source


def test_node_is_not_picklable_or_copyable() -> None:
    node = simplex_fbm()
    for duplicate in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError, match="cannot be pickled or copied"):
            duplicate(node)


def test_from_encoded() -> None:
    node = fn2.Node.from_encoded(FBM_ENCODED)
    assert node.node_type == "FractalFBm"
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), GOLDEN_2D)
    assert np.array_equal(node.gen_position_array_3d(X, Y, Z, SEED), GOLDEN_3D)

    node = fn2.Node.from_encoded(MIXED_ENCODED)
    assert node.node_type == "DomainOffset"
    expected = mixed_tree()
    assert np.array_equal(
        node.gen_position_array_2d(X, Y, SEED), expected.gen_position_array_2d(X, Y, SEED)
    )
    assert np.array_equal(
        node.gen_position_array_3d(X, Y, Z, SEED),
        expected.gen_position_array_3d(X, Y, Z, SEED),
    )

    lowest = fn2.simd_levels()[0]
    node = fn2.Node.from_encoded(FBM_ENCODED, max_simd_level=lowest)
    assert node.simd_level == lowest
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), GOLDEN_2D)


@pytest.mark.parametrize(
    "encoded",
    [
        "",
        "not base64!",
        "/w==",
        FBM_ENCODED[:-4],
        FBM_ENCODED + "AAAA",
        # FastNoise2's own decoder turns these into trees that crash when used.
        "DQAF@B6CQY@ACAxAIIQS/GAghBD8M",
        "DQAF@BC1Y@ACAP0AIIQS/GAghBD8M",
        "LsOu5azZCwqA3Obc4ipLnkNH",
        "HUzBD47vzdHu7qm0Ri2IJcqYzH5lenpj0HAB2EYdNK8R",
        pytest.param("DQAF@!", id="run of a character outside the alphabet"),
        pytest.param("/g==", id="unknown node type"),
        pytest.param("/wAA", id="reference to a node that is not there"),
        pytest.param("BvgAAIA/BA==", id="variable index beyond the node type"),
        pytest.param("KAAHAAAAEQYEBgQE", id="enum value beyond its options"),
        pytest.param("DREGBAQ=", id="more node members than the node type has"),
        pytest.param("EwkGBAQ=", id="node member of the wrong node type"),
        pytest.param("DQQ=", id="node member left unset"),
        pytest.param("DQkGBPsAAIA/BA==", id="hybrid index beyond the node type"),
        pytest.param("BgUE", id="unknown member kind"),
        pytest.param("KBEGDAYE", id="a node follows the end of its parent"),
        pytest.param("Bgw=", id="the root ends more nodes than there are"),
    ],
)
def test_from_encoded_rejects(encoded: str) -> None:
    with pytest.raises(ValueError, match="not a FastNoise2 node tree"):
        fn2.Node.from_encoded(encoded)


def test_from_encoded_rejects_other_arguments() -> None:
    with pytest.raises(TypeError, match="encoded must be a str"):
        fn2.Node.from_encoded(FBM_ENCODED.encode())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="^max_simd_level must be one of"):
        fn2.Node.from_encoded(FBM_ENCODED, max_simd_level="AVX")


def test_from_encoded_depth_limit() -> None:
    # Member bytes in the layout of FastNoise2.
    one_node_member, first_hybrid_node, node_end = 1 | 1 << 3, 2, 4
    abs_id, add_id, simplex_id = (
        fn2.node_types().index(name) for name in ("Abs", "Add", "Simplex")
    )

    def chain(depth: int) -> str:
        """A chain of Abs nodes above a Simplex node."""
        nodes = [abs_id, one_node_member] * (depth - 1) + [simplex_id] + [node_end] * depth
        return base64.b64encode(bytes(nodes)).decode()

    def sums(depth: int) -> list[int]:
        """Add nodes above ``2 ** (depth - 1)`` Simplex nodes."""
        if depth == 1:
            return [simplex_id, node_end]
        lower = sums(depth - 1)
        return [add_id, one_node_member, *lower, first_hybrid_node, *lower, node_end]

    simplex = fn2.Node("Simplex").gen_position_array_2d(X, Y, SEED)
    node = fn2.Node.from_encoded(chain(MAX_DEPTH))
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), np.abs(simplex))
    with pytest.raises(ValueError, match="more than 128 nodes deep"):
        fn2.Node.from_encoded(chain(MAX_DEPTH + 1))
    # Refused before the decoder has followed the string to its end.
    with pytest.raises(ValueError, match="more than 128 nodes deep"):
        fn2.Node.from_encoded(chain(100_000))
    # More nodes than the limit, but only eight deep.
    node = fn2.Node.from_encoded(base64.b64encode(bytes(sums(8))).decode())
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), np.float32(128.0) * simplex)


def test_generation_from_threads() -> None:
    node = simplex_fbm()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: node.gen_position_array_2d(X, Y, SEED), range(64)))
    assert all(np.array_equal(result, GOLDEN_2D) for result in results)


@pytest.mark.skipif(sys.platform != "linux", reason="fork without exec is only safe on Linux")
def test_node_survives_fork() -> None:
    node = simplex_fbm()
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            # Ends a child that waits for a lock.
            signal.alarm(10)
            os.close(read_end)
            values = node.gen_position_array_2d(X, Y, SEED)
            # The child owns a copy of the node: it can release it and build new ones.
            del node
            gc.collect()
            values_3d = simplex_fbm().gen_position_array_3d(X, Y, Z, SEED)
            os.write(write_end, values.tobytes() + values_3d.tobytes())
            status = 0
        finally:
            os._exit(status)
    os.close(write_end)
    with os.fdopen(read_end, "rb") as pipe:
        received = pipe.read()
    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0
    assert received == GOLDEN_2D.tobytes() + GOLDEN_3D.tobytes()
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), GOLDEN_2D)


@pytest.mark.skipif(sys.platform != "linux", reason="fork without exec is only safe on Linux")
@pytest.mark.filterwarnings("ignore:.*multi-threaded.*fork:DeprecationWarning")
def test_fork_while_a_thread_builds_nodes() -> None:
    stop = threading.Event()

    def build() -> None:
        while not stop.is_set():
            simplex_fbm()

    thread = threading.Thread(target=build)
    thread.start()
    try:
        for _ in range(50):
            pid = os.fork()
            if pid == 0:
                status = 1
                try:
                    # Ends a child that waits for a lock the other thread held when it forked.
                    signal.alarm(10)
                    if simplex_fbm().gen_single_2d(0.25, -0.5, SEED) == GOLDEN_2D[1]:
                        status = 0
                finally:
                    os._exit(status)
            assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0
    finally:
        stop.set()
        thread.join()


def test_distribution_ships_all_licenses() -> None:
    files = importlib.metadata.files("pyfastnoise2")
    assert files is not None
    licenses = {file.name: file.read_text() for file in files if "licenses" in file.parts}
    assert set(licenses) == {"LICENSE", "FastNoise2-LICENSE", "FastSIMD-LICENSE"}
    assert "Copyright (c) 2026 Zetta AI" in licenses["LICENSE"]
    assert "Copyright (c) 2020 Jordan Peck" in licenses["FastNoise2-LICENSE"]
    assert "Copyright (c) 2022 Jordan Peck" in licenses["FastSIMD-LICENSE"]
