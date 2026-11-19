"""Tests of the binding. They run against the installed distribution."""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import os
import pickle
import signal
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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

# lattice_digest(lone_node(node_type)) of every node type, the same on every platform and
# at every SIMD level.
GOLDEN_DIGESTS = {
    "Constant": "fef951e6c76ad6a0",
    "White": "32de67f7b31ca1b3",
    "Checkerboard": "4badd1e8307ad842",
    "SineWave": "ceb40b0a84ac4658",
    "Gradient": "e5a00aa9991ac8a5",
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
    "PowFloat": "c1ef1bf1d7bf8161",
    "PowInt": "60bb5e62f58b40d4",
    "DomainScale": "9ddce4109cea39b0",
    "DomainOffset": "9ddce4109cea39b0",
    "DomainRotate": "9ddce4109cea39b0",
    "DomainAxisScale": "9ddce4109cea39b0",
    "SeedOffset": "2aadd9127526a013",
    "ConvertRGBA8": "502f677b198142c7",
    "GeneratorCache": "9ddce4109cea39b0",
    "Fade": "9ddce4109cea39b0",
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


def test_hybrid_takes_node_or_float() -> None:
    expected = simplex_fbm().gen_position_array_2d(X, Y, SEED)
    gain = fn2.Node("Constant")
    gain.set("Value", 0.25)
    node = simplex_fbm()
    node.set("Gain", gain)
    assert not np.array_equal(node.gen_position_array_2d(X, Y, SEED), expected)
    node.set("Gain", 0.5)
    assert np.array_equal(node.gen_position_array_2d(X, Y, SEED), expected)


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


def lone_node(node_type: str, max_simd_level: str | None = None) -> fn2.Node:
    """A node of ``node_type`` that has its node members set and is not constant."""

    def source() -> fn2.Node:
        simplex = fn2.Node("Simplex", max_simd_level=max_simd_level)
        simplex.set("Feature Scale", 1.0)
        return simplex

    node = fn2.Node(node_type, max_simd_level=max_simd_level)
    for name, member in fn2.members(node_type).items():
        if name == "Feature Scale":
            node.set(name, 1.0)
        elif name == "RHS":
            node.set(name, 0.37)
        elif member.kind == "node" or name == "LHS":
            try:
                node.set(name, source())
            except TypeError:
                warp = fn2.Node("DomainWarpGradient", max_simd_level=max_simd_level)
                warp.set("Source", source())
                node.set(name, warp)
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
    shifted = node.gen_position_array_3d(X, Y, Z, SEED, z_offset=0.25)
    expected = node.gen_position_array_3d(X, Y, Z + np.float32(0.25), SEED)
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
    ],
)
def test_from_encoded_rejects(encoded: str) -> None:
    with pytest.raises(ValueError, match="not a FastNoise2 node tree"):
        fn2.Node.from_encoded(encoded)


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
