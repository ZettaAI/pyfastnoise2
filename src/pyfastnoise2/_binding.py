"""ctypes binding to the FastNoise2 C API (``FastNoise/FastNoise_C.h``)."""

from __future__ import annotations

import base64
import ctypes
import math
import numbers
import operator
import os
import re
import string
import struct
import sys
import threading
import weakref
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, NoReturn, Union

import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]
MemberValue = Union[float, str, "Node"]

_LIBRARY: Final = "libFastNoise.dylib" if sys.platform == "darwin" else "libFastNoise.so"
# CDLL releases the GIL for the duration of every call.
_lib: Final = ctypes.CDLL(str(Path(__file__).with_name(_LIBRARY)))

_INT: Final = ctypes.c_int
_UINT: Final = ctypes.c_uint
_FLOAT: Final = ctypes.c_float
_BOOL: Final = ctypes.c_bool
_NAME: Final = ctypes.c_char_p
# Node handles, and float arrays passed as the addresses of validated numpy buffers.
_POINTER: Final = ctypes.c_void_p

_SIGNATURES: Final[dict[str, tuple[Any, tuple[Any, ...]]]] = {
    "fnNewFromMetadata": (_POINTER, (_INT, _UINT)),
    "fnDeleteNodeRef": (None, (_POINTER,)),
    "fnGetActiveFeatureSet": (_UINT, (_POINTER,)),
    "fnGetMetadataCount": (_INT, ()),
    "fnGetMetadataName": (_NAME, (_INT,)),
    "fnGetMetadataVariableCount": (_INT, (_INT,)),
    "fnGetMetadataVariableName": (_NAME, (_INT, _INT)),
    "fnGetMetadataVariableDimensionIdx": (_INT, (_INT, _INT)),
    "fnGetMetadataVariableType": (_INT, (_INT, _INT)),
    "fnGetMetadataVariableDefaultFloat": (_FLOAT, (_INT, _INT)),
    "fnGetMetadataVariableDefaultIntEnum": (_INT, (_INT, _INT)),
    "fnGetMetadataEnumCount": (_INT, (_INT, _INT)),
    "fnGetMetadataEnumName": (_NAME, (_INT, _INT, _INT)),
    "fnGetMetadataNodeLookupCount": (_INT, (_INT,)),
    "fnGetMetadataNodeLookupName": (_NAME, (_INT, _INT)),
    "fnGetMetadataNodeLookupDimensionIdx": (_INT, (_INT, _INT)),
    "fnGetMetadataHybridCount": (_INT, (_INT,)),
    "fnGetMetadataHybridName": (_NAME, (_INT, _INT)),
    "fnGetMetadataHybridDimensionIdx": (_INT, (_INT, _INT)),
    "fnGetMetadataHybridDefault": (_FLOAT, (_INT, _INT)),
    "fnSetVariableFloat": (_BOOL, (_POINTER, _INT, _FLOAT)),
    "fnSetVariableIntEnum": (_BOOL, (_POINTER, _INT, _INT)),
    "fnSetNodeLookup": (_BOOL, (_POINTER, _INT, _POINTER)),
    "fnSetHybridNodeLookup": (_BOOL, (_POINTER, _INT, _POINTER)),
    "fnSetHybridFloat": (_BOOL, (_POINTER, _INT, _FLOAT)),
    # node, out, offsets, counts, steps, seed, min/max out
    "fnGenUniformGrid2D": (
        None,
        (_POINTER, _POINTER, *[_FLOAT] * 2, *[_INT] * 2, *[_FLOAT] * 2, _INT, _POINTER),
    ),
    "fnGenUniformGrid3D": (
        None,
        (_POINTER, _POINTER, *[_FLOAT] * 3, *[_INT] * 3, *[_FLOAT] * 3, _INT, _POINTER),
    ),
    # node, out, count, positions, offsets, seed, min/max out
    "fnGenPositionArray2D": (
        None,
        (_POINTER, _POINTER, _INT, *[_POINTER] * 2, *[_FLOAT] * 2, _INT, _POINTER),
    ),
    "fnGenPositionArray3D": (
        None,
        (_POINTER, _POINTER, _INT, *[_POINTER] * 3, *[_FLOAT] * 3, _INT, _POINTER),
    ),
    "fnGenSingle2D": (_FLOAT, (_POINTER, _FLOAT, _FLOAT, _INT)),
    "fnGenSingle3D": (_FLOAT, (_POINTER, _FLOAT, _FLOAT, _FLOAT, _INT)),
    "fnGenSingle4D": (_FLOAT, (_POINTER, _FLOAT, _FLOAT, _FLOAT, _FLOAT, _INT)),
}
for _name, (_restype, _argtypes) in _SIGNATURES.items():
    _function = getattr(_lib, _name)
    _function.restype = _restype
    _function.argtypes = list(_argtypes)

_INT_MAX: Final = 2**31 - 1
# Nodes in the longest chain of a tree, each a member of the one before. Generating
# recurses along a chain with up to 8 KiB of native stack per node, so a tree of this
# depth stays within 1 MiB.
_MAX_DEPTH: Final = 128
# FastSIMD::FeatureSet::Max: the best level the CPU supports.
_MAX_FEATURE_SET: Final = 0xFFFFFFFF
# FastSIMD::FeatureSet values of the levels FastNoise2 compiles, lowest first, under
# the names FastSIMD gives them. A level below the lowest one compiled, or one the CPU
# lacks, crashes the process, so no other value is ever passed to the library. The
# NEON level that arm64 builds also contain is left out: it divides by approximation,
# so it differs from AARCH64, the level every arm64 CPU runs.
_FEATURE_SETS: Final = {
    "SSE2": 0x0000E,
    "SSE4.1": 0x0007E,
    "AVX2": 0x003FE,
    "AVX512": 0x03FFE,
    "AARCH64": 0x1C000,
}
_FEATURE_SET_NAMES: Final = {value: name for name, value in _FEATURE_SETS.items()}

_DIMENSIONS: Final = "XYZW"
_VARIABLE_KINDS: Final = ("float", "int", "enum")
_BASE64: Final = string.ascii_uppercase + string.ascii_lowercase + string.digits + "+/"
# Kinds of the member bytes of an encoded node tree.
_VARIABLE, _NODE_LOOKUPS, _HYBRID_NODE, _HYBRID_FLOAT, _NODE_END = range(5)
# In place of a node type id: a reference to an earlier node of the tree.
_REFERENCE: Final = 0xFF


@dataclass(frozen=True)
class Member:
    """A settable member of a node type, as listed by :func:`members`.

    :param kind: ``"float"``, ``"int"`` or ``"enum"`` for a variable, ``"hybrid"`` for
        a member that takes a float or a node, ``"node"`` for one that must be a node.
    :param default: Initial value: the option name of an ``"enum"``, ``None`` for a
        ``"node"``.
    :param options: Option names of an ``"enum"``.
    """

    kind: Literal["float", "int", "enum", "hybrid", "node"]
    default: float | str | None
    options: tuple[str, ...] = ()


class _NodeType(NamedTuple):
    type_id: int
    name: str
    members: dict[str, Member]
    # Index of each member among the members of its C API category.
    indices: dict[str, int]

    def names(self, *kinds: str) -> list[str]:
        """Names of the members of ``kinds``, ordered by index."""
        return [name for name, member in self.members.items() if member.kind in kinds]


def _member_name(category: str, type_id: int, index: int) -> str:
    """A member's name as FastNoise2 displays it: per-dimension members share one name
    in the metadata and are told apart by a dimension prefix."""
    name: bytes = getattr(_lib, f"fnGetMetadata{category}Name")(type_id, index)
    dimension = getattr(_lib, f"fnGetMetadata{category}DimensionIdx")(type_id, index)
    return ("" if dimension < 0 else f"{_DIMENSIONS[dimension]} ") + name.decode()


def _variable(type_id: int, index: int) -> Member:
    kind = _VARIABLE_KINDS[_lib.fnGetMetadataVariableType(type_id, index)]
    if kind == "float":
        return Member("float", _lib.fnGetMetadataVariableDefaultFloat(type_id, index))
    default = _lib.fnGetMetadataVariableDefaultIntEnum(type_id, index)
    if kind == "int":
        return Member("int", default)
    options = tuple(
        _lib.fnGetMetadataEnumName(type_id, index, option).decode()
        for option in range(_lib.fnGetMetadataEnumCount(type_id, index))
    )
    return Member("enum", options[default], options)


def _read_node_type(type_id: int) -> _NodeType:
    name = _lib.fnGetMetadataName(type_id).decode()
    members_: dict[str, Member] = {}
    indices: dict[str, int] = {}
    for category in ("Variable", "NodeLookup", "Hybrid"):
        for index in range(getattr(_lib, f"fnGetMetadata{category}Count")(type_id)):
            member_name = _member_name(category, type_id, index)
            if member_name in members_:
                raise RuntimeError(f"FastNoise2 {name} has two members {member_name!r}")
            if category == "Variable":
                members_[member_name] = _variable(type_id, index)
            elif category == "NodeLookup":
                members_[member_name] = Member("node", None)
            else:
                default = _lib.fnGetMetadataHybridDefault(type_id, index)
                members_[member_name] = Member("hybrid", default)
            indices[member_name] = index
    return _NodeType(type_id, name, members_, indices)


# Metadata ids belong to the library build: node types are addressed by name.
_NODE_TYPES: Final = {
    node_type.name: node_type
    for node_type in map(_read_node_type, range(_lib.fnGetMetadataCount()))
}
_NODE_TYPES_BY_ID: Final = {node_type.type_id: node_type for node_type in _NODE_TYPES.values()}


def _node_type(name: str) -> _NodeType:
    node_type = _NODE_TYPES.get(name) if isinstance(name, str) else None
    if node_type is None:
        raise ValueError(f"unknown node type {name!r}; node_types() lists the names")
    return node_type


# Held for every native call that allocates or frees nodes, since the node pools of
# FastNoise2 do not survive a free next to another allocation or free, and for all of a
# set(), since two at once could each pass the cycle check of the other. The code that
# holds it is a section. The collector runs finalizers, __del__ methods and weakref
# callbacks anywhere, also inside a section, where a thread would wait forever for a
# plain lock that it holds itself. An RLock lets its holder through, at every instant
# from when it is taken, and _enter() then turns the thread away: no section is entered
# twice.
_lock: Final = threading.RLock()
# Whether the holder of the lock is inside a section. Only the holder reads or writes it.
_inside = False
# Handles of collected nodes that are still to be freed. A finalizer only queues its
# handle here: the collector runs it anywhere, also inside a section that holds the lock.
_dropped: Final[deque[int]] = deque()
# fnDeleteNodeRef gives a reference back without freeing the node, which FastNoise2 only
# does when another node lets go of it. This node, never generated from, takes each
# dropped node as a member for that.
_RELEASER_TYPE: Final = _NODE_TYPES["Add"]
_RELEASER: Final[int] = _lib.fnNewFromMetadata(_RELEASER_TYPE.type_id, _MAX_FEATURE_SET)
_RELEASER_MEMBER: Final = _RELEASER_TYPE.indices["RHS"]


def _enter(blocking: bool = True) -> bool:
    """Take the lock and enter the section it guards. ``False`` if this thread is inside
    the section already or, without ``blocking``, if another thread holds the lock."""
    global _inside
    if not _lock.acquire(blocking=blocking):
        return False
    if _inside:
        # The collector ran the caller inside a section of this thread.
        _lock.release()
        return False
    _inside = True
    return True


def _leave() -> None:
    global _inside
    _inside = False
    _lock.release()


def _free_dropped() -> None:
    """Free the nodes in ``_dropped``, unless the lock is held: its holder frees them when
    it lets go."""
    while _dropped and _enter(blocking=False):
        try:
            # Another thread may have emptied the queue since the check above. Only the
            # holder of the lock takes from it.
            if not _dropped:
                continue
            handle = _dropped.popleft()
            # The releaser takes its own reference first, so that the node is freed
            # exactly once, also by a FastNoise2 whose fnDeleteNodeRef frees a node
            # nothing refers to anymore.
            _lib.fnSetHybridNodeLookup(_RELEASER, _RELEASER_MEMBER, handle)
            _lib.fnDeleteNodeRef(handle)
            _lib.fnSetHybridFloat(_RELEASER, _RELEASER_MEMBER, 0.0)
        finally:
            _leave()


def _drop(handle: int) -> None:
    """Finalizer of a :class:`Node`."""
    _dropped.append(handle)
    _free_dropped()


class _Exclusive:
    """Context of a section that holds the lock."""

    def __enter__(self) -> None:
        if not _enter():
            raise RuntimeError(
                "this thread is already creating or changing a pyfastnoise2 node: a "
                "__del__ method or weakref callback, which the garbage collector may run "
                "inside that call, must not create or change nodes"
            )

    def __exit__(self, *error: object) -> None:
        _leave()
        # Those dropped inside the section.
        _free_dropped()


_exclusive: Final = _Exclusive()


def _after_fork() -> None:
    _lock.release()
    _free_dropped()


# A child forked while another thread is inside a section would inherit the lock as held,
# and the node pools in the middle of a call. The fork holds the lock without entering a
# section: it changes no node, and code that the collector runs inside a section may fork.
os.register_at_fork(before=_lock.acquire, after_in_parent=_after_fork, after_in_child=_after_fork)


# A GeneratorCache node keeps the values of its last batch per thread and returns them
# again for the same source address, seed and positions: also after the source has
# changed, or was freed and its address given to another node. Generating from one of
# these cache nodes, one per SIMD level with a source that no other node has, leaves a
# thread with a cached address that no other cache node matches.
_CACHE_TYPE: Final = _NODE_TYPES["GeneratorCache"]
_cache_resets: Final[dict[int, int]] = {}


def _new_cache_reset(feature_set: int) -> int:
    reset: int = _lib.fnNewFromMetadata(_CACHE_TYPE.type_id, feature_set)
    source = _lib.fnNewFromMetadata(_NODE_TYPES["Constant"].type_id, feature_set)
    _lib.fnSetNodeLookup(reset, _CACHE_TYPE.indices["Source"], source)
    return reset


def node_types() -> tuple[str, ...]:
    """Names of FastNoise2's node types, such as ``"Simplex"`` and ``"FractalFBm"``."""
    return tuple(_NODE_TYPES)


def members(node_type: str) -> dict[str, Member]:
    """Members of a node type by the names :meth:`Node.set` takes.

    The names are FastNoise2's own. A member that exists once per dimension carries the
    dimension as a prefix, as in ``"X Offset"``.
    """
    return dict(_node_type(node_type).members)


def _levels_within(feature_set: int) -> tuple[str, ...]:
    """Names of the levels that use no CPU feature beyond those of ``feature_set``."""
    return tuple(name for name, value in _FEATURE_SETS.items() if value & feature_set == value)


# The releaser runs at the level FastSIMD picks for this CPU.
_SIMD_LEVELS: Final = _levels_within(_lib.fnGetActiveFeatureSet(_RELEASER))


def simd_levels() -> tuple[str, ...]:
    """SIMD levels nodes can run at on this CPU, lowest first. The last is the default.

    FastNoise2 is built to return the same values at every level, so a lower level only
    costs speed.
    """
    return _SIMD_LEVELS


def _max_feature_set(max_simd_level: str | None) -> int:
    if max_simd_level is None:
        return _MAX_FEATURE_SET
    if max_simd_level not in _SIMD_LEVELS:
        raise ValueError(
            f"max_simd_level must be one of {_SIMD_LEVELS} on this CPU, not {max_simd_level!r}"
        )
    return _FEATURE_SETS[max_simd_level]


def _c_int(name: str, value: Any, lowest: int = -_INT_MAX - 1) -> int:
    """``value`` if a C ``int`` holds it; ctypes would silently truncate a larger one."""
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}") from None
    if not lowest <= number <= _INT_MAX:
        raise ValueError(f"{name} must be in [{lowest}, {_INT_MAX}], not {number}")
    return number


def _c_float(name: str, value: Any) -> float:
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    try:
        return float(value)
    except OverflowError:
        raise ValueError(f"{name} is beyond the range of a float") from None


def _member_number(what: str, member: Member, value: Any) -> float:
    """``value`` as the C number of a member that is not set to a node."""
    if member.kind == "node":
        raise TypeError(f"{what} must be a Node, not {type(value).__name__}")
    if member.kind in ("hybrid", "float"):
        return _c_float(what, value)
    if member.kind == "int":
        return _c_int(what, value)
    if isinstance(value, str):
        if value not in member.options:
            raise ValueError(f"{what} must be one of {member.options}, not {value!r}")
        return member.options.index(value)
    # The library does not check the range of an enum.
    number = _c_int(what, value, lowest=0)
    if number >= len(member.options):
        raise ValueError(f"{what} has {len(member.options)} options, not {number}")
    return number


def _float32_array(name: str, array: Any) -> Float32Array:
    """``array`` if the library may read it through a bare ``float*``."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, not {type(array).__name__}")
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32, not {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return array


def _output(out: Float32Array | None, shape: tuple[int, ...]) -> Float32Array:
    """``out``, or a new array, to hold one call's values in ``shape``."""
    if math.prod(shape) > _INT_MAX:
        raise ValueError(f"one call generates at most {_INT_MAX} values, not {shape}")
    if out is None:
        return np.empty(shape, dtype=np.float32)
    if _float32_array("out", out).shape != shape:
        raise ValueError(f"out must have shape {shape}, not {out.shape}")
    if not out.flags.writeable:
        raise ValueError("out must be writeable")
    return out


def _position_output(out: Float32Array | None, **coordinates: Float32Array) -> Float32Array:
    """``out``, or a new array, to hold the values at ``coordinates``."""
    shape = _float32_array("x", coordinates["x"]).shape
    for name, array in coordinates.items():
        if _float32_array(name, array).shape != shape:
            raise ValueError(f"{name} must have the shape of x, {shape}, not {array.shape}")
    out = _output(out, shape)
    for name, array in coordinates.items():
        # The library reads the last positions again after it has written other values.
        if np.may_share_memory(out, array):
            raise ValueError(f"out must not overlap {name}")
    return out


class _EncodedNodeTree:
    """Reader of an encoded node tree, in the layout of ``Metadata.cpp`` of FastNoise2.

    FastNoise2's own decoder accepts damaged strings and returns node trees that crash
    when used, so the tree is built here, node by node, through the checks of
    :meth:`Node.set`.
    """

    def __init__(self, encoded: str, max_simd_level: str | None) -> None:
        # Base64 in which "@" and one character stand for a run of "A".
        expanded = re.sub(
            "@(.)", lambda run: "A" * (_BASE64.index(run[1]) + 3), encoded, flags=re.DOTALL
        )
        self._data = base64.b64decode(expanded, validate=True)
        self._position = 0
        # Node ends still to report: one member byte ends several nested nodes.
        self._ends = 0
        self._max_simd_level = max_simd_level
        # Nodes that are being read: each is a member of the one before.
        self._open = 0
        # Finished nodes, which later ones refer to by position.
        self._nodes: list[Node] = []

    def root(self) -> Node:
        node = self._node()
        if self._ends or self._position != len(self._data):
            raise ValueError("data after the root node")
        return node

    def _read(self, layout: str) -> Any:
        size = struct.calcsize(layout)
        if self._ends or self._position + size > len(self._data):
            raise ValueError("the node tree is cut short")
        (value,) = struct.unpack_from(layout, self._data, self._position)
        self._position += size
        return value

    def _member(self) -> tuple[int, int]:
        """Kind and index of the next member byte."""
        if self._ends:
            self._ends -= 1
            return _NODE_END, 0
        byte = self._read("B")
        kind, index = byte & 7, byte >> 3
        if kind == _NODE_END:
            self._ends = index
        return kind, index

    def _node(self) -> Node:
        type_id = self._read("B")
        if type_id == _REFERENCE:
            position: int = self._read("<H")
            return self._nodes[position]
        metadata = _NODE_TYPES_BY_ID[type_id]
        self._open += 1
        if self._open > _MAX_DEPTH:
            raise ValueError(f"the tree is more than {_MAX_DEPTH} nodes deep")
        node = Node(metadata.name, max_simd_level=self._max_simd_level)
        kind, index = self._member()
        variables = metadata.names("float", "int", "enum")
        while kind == _VARIABLE:
            name = variables[index]
            is_float = metadata.members[name].kind == "float"
            node.set(name, self._read("<f" if is_float else "<i"))
            kind, index = self._member()
        if kind == _NODE_LOOKUPS:
            node_lookups = metadata.names("node")
            if index != len(node_lookups):
                raise ValueError(f"{metadata.name} has {len(node_lookups)} node members")
            for name in node_lookups:
                node.set(name, self._node())
            kind, index = self._member()
        hybrids = metadata.names("hybrid")
        while kind != _NODE_END:
            if kind == _HYBRID_NODE:
                node.set(hybrids[index], self._node())
            elif kind == _HYBRID_FLOAT:
                node.set(hybrids[index], self._read("<f"))
            else:
                raise ValueError(f"unexpected member kind {kind}")
            kind, index = self._member()
        node._require_members()
        self._open -= 1
        self._nodes.append(node)
        return node


class Node:
    """A FastNoise2 node. A node whose members are other nodes is the root of a node tree.

    Generating is thread-safe and runs without the GIL: threads may generate from one
    tree at the same time. Creating, changing and collecting nodes is thread-safe too,
    with one exception: do not call :meth:`set` on a node while another thread generates
    from a tree that contains it. FastNoise2 reads the members of a node without a lock,
    and a node that :meth:`set` replaces may be freed while that thread still uses it.
    A ``__del__`` method or weakref callback must not create or change nodes: it raises
    ``RuntimeError`` when the garbage collector runs it while its thread does the same.

    A tree is at most 128 nodes deep, since generating recurses through a chain of
    members on the native stack: :meth:`set` refuses a member that would make a longer
    chain. A node counts with the deepest place it has had in any tree.

    A node wraps a native handle, so it cannot be pickled or copied: build it in the
    process that uses it. The native node is freed when the node is collected and no
    other node has it as a member: a tree keeps the nodes it is built from. A node does
    survive ``os.fork()``. Parent and child then each own an independent copy, and
    FastNoise2 holds no threads or file descriptors.

    :param node_type: One of :func:`node_types`.
    :param max_simd_level: One of :func:`simd_levels`, to run this node below the best
        level of the CPU. All nodes of a tree must run at the same level.
    """

    __slots__ = (
        "__weakref__",
        "_depth",
        "_feature_set",
        "_finalizer",
        "_handle",
        "_sources",
        "_type",
        "_unset",
    )

    def __init__(self, node_type: str, *, max_simd_level: str | None = None) -> None:
        metadata = _node_type(node_type)
        feature_set = _max_feature_set(max_simd_level)
        with _exclusive:
            handle = _lib.fnNewFromMetadata(metadata.type_id, feature_set)
            if handle:
                # Before the lock is let go: no error on the way out of here may leave
                # the native node without the finalizer that frees it.
                self._finalizer = weakref.finalize(self, _drop, handle)
        if not handle:
            raise RuntimeError(f"FastNoise2 did not create a {node_type} node")
        self._handle: int = handle
        # Not at interpreter exit: a daemon thread may still be generating from the node,
        # and the memory goes back to the system with the process.
        self._finalizer.atexit = False
        self._type = metadata
        self._feature_set: int = _lib.fnGetActiveFeatureSet(handle)
        if metadata is _CACHE_TYPE:
            with _exclusive:
                if self._feature_set not in _cache_resets:
                    _cache_resets[self._feature_set] = _new_cache_reset(self._feature_set)
        # Nodes set as members; the library holds its own references to them.
        self._sources: dict[str, Node] = {}
        # Place in the longest chain from a root down to this node, over every tree the
        # node has been in: it only grows, so a member is always deeper than its node.
        self._depth = 1
        # "node" members not set yet: generating without them crashes.
        self._unset = metadata.names("node")

    @staticmethod
    def from_encoded(encoded: str, *, max_simd_level: str | None = None) -> Node:
        """The root of the node tree in ``encoded``, a string from the FastNoise2 Node
        Editor or from ``FastNoise::Metadata::SerialiseNodeData``.

        The string holds node type ids, which belong to a FastNoise2 version: decode
        strings of the version this package is built from.

        A damaged string is refused, but member values are taken as they are, since
        FastNoise2 does not limit them: a string from an untrusted source can describe a
        tree that takes practically forever to generate, with a billion octaves for one.
        """
        if not isinstance(encoded, str):
            raise TypeError(f"encoded must be a str, not {type(encoded).__name__}")
        _max_feature_set(max_simd_level)
        # Raises in a thread that is creating or changing a node already, which is not an
        # error of the string.
        with _exclusive:
            pass
        try:
            return _EncodedNodeTree(encoded, max_simd_level).root()
        except (ValueError, TypeError, LookupError, RuntimeError) as error:
            raise ValueError(f"encoded is not a FastNoise2 node tree: {error}") from error

    @property
    def node_type(self) -> str:
        """Name of the node type."""
        return self._type.name

    @property
    def simd_level(self) -> str:
        """SIMD level the node runs at, one of :func:`simd_levels`."""
        return _FEATURE_SET_NAMES[self._feature_set]

    def set(self, name: str, value: MemberValue) -> None:
        """Set the member ``name``, one of ``members(node.node_type)``.

        :param name: Name of the member.
        :param value: A number for a ``"float"`` or ``"int"`` member, an option name or
            index for an ``"enum"``, a number or a :class:`Node` for a ``"hybrid"``, a
            :class:`Node` for a ``"node"``. A node given here must have all of its own
            ``"node"`` members set.
        """
        member = self._type.members.get(name) if isinstance(name, str) else None
        if member is None:
            raise ValueError(
                f"{self.node_type} has no member {name!r}; its members are "
                f"{', '.join(map(repr, self._type.members))}"
            )
        index = self._type.indices[name]
        what = f"{self.node_type} member {name!r}"
        if isinstance(value, Node):
            with _exclusive:
                self._set_source(what, name, member, index, value)
            return
        # Before the lock is taken: __float__ and __index__ are code of the caller, which
        # may create or change nodes.
        number = _member_number(what, member, value)
        with _exclusive:
            if member.kind == "hybrid":
                accepted = _lib.fnSetHybridFloat(self._handle, index, number)
                if accepted:
                    # A constant replaces the node the library held for this member.
                    self._sources.pop(name, None)
            elif member.kind == "float":
                accepted = _lib.fnSetVariableFloat(self._handle, index, number)
            else:
                accepted = _lib.fnSetVariableIntEnum(self._handle, index, number)
        if not accepted:
            raise RuntimeError(f"FastNoise2 rejected {value!r} for {what}")

    def _set_source(self, what: str, name: str, member: Member, index: int, source: Node) -> None:
        if member.kind not in ("node", "hybrid"):
            raise TypeError(f"{what} must be {member.kind}, not a Node")
        if source._feature_set != self._feature_set:
            raise ValueError(
                f"{what}: the {source.node_type} node runs at SIMD level "
                f"{source.simd_level}, this node at {self.simd_level}"
            )
        if source._unset:
            raise ValueError(
                f"{what}: set {', '.join(map(repr, source._unset))} of the "
                f"{source.node_type} node first"
            )
        depths = self._depths_below(source, what)
        if member.kind == "node":
            accepted = _lib.fnSetNodeLookup(self._handle, index, source._handle)
        else:
            accepted = _lib.fnSetHybridNodeLookup(self._handle, index, source._handle)
        if not accepted:
            raise TypeError(f"{what} does not accept a {source.node_type} node")
        for node, depth in depths.items():
            node._depth = depth
        self._sources[name] = source
        if name in self._unset:
            self._unset.remove(name)

    def _depths_below(self, source: Node, what: str) -> dict[Node, int]:
        """The new depths of the nodes that get deeper when the tree of ``source`` becomes
        a member of this node. Refuses a tree that contains this node, which would recurse
        without end, and one that would reach deeper than ``_MAX_DEPTH``."""
        depths: dict[Node, int] = {}
        pending = [(source, self._depth + 1)]
        while pending:
            node, depth = pending.pop()
            # Nothing changes for a node that is this deep already, nor below it. It is
            # deeper than this node then, so this node is not below it either.
            if depth <= max(node._depth, depths.get(node, 0)):
                continue
            if node is self:
                raise ValueError(f"{what}: the {source.node_type} node depends on this node")
            depths[node] = depth
            pending.extend((member, depth + 1) for member in node._sources.values())
        if max(depths.values(), default=0) > _MAX_DEPTH:
            raise ValueError(f"{what}: the tree would be more than {_MAX_DEPTH} nodes deep")
        return depths

    def _require_members(self) -> None:
        if self._unset:
            raise RuntimeError(
                f"set {', '.join(map(repr, self._unset))} of the {self.node_type} node "
                "before generating"
            )

    def _ready(self) -> int:
        """The handle, once generating from it is safe."""
        self._require_members()
        if _cache_resets:
            reset = _cache_resets.get(self._feature_set)
            if reset is not None:
                # One cache per number of coordinates; AddDimension reaches the higher.
                _lib.fnGenSingle2D(reset, 0.0, 0.0, 0)
                _lib.fnGenSingle3D(reset, 0.0, 0.0, 0.0, 0)
                _lib.fnGenSingle4D(reset, 0.0, 0.0, 0.0, 0.0, 0)
        return self._handle

    def gen_uniform_grid_2d(
        self,
        *,
        x_count: int,
        y_count: int,
        x_step: float,
        y_step: float,
        seed: int,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        out: Float32Array | None = None,
    ) -> Float32Array:
        """Noise on a regular grid: ``result[j, i]`` is the value at
        ``(x_offset + i * x_step, y_offset + j * y_step)``, computed in float32.

        :param out: C-contiguous float32 array of shape ``(y_count, x_count)`` to fill.
        :return: ``out``, or a new array.
        """
        handle = self._ready()
        offsets = (_c_float("x_offset", x_offset), _c_float("y_offset", y_offset))
        counts = (_c_int("x_count", x_count, 0), _c_int("y_count", y_count, 0))
        steps = (_c_float("x_step", x_step), _c_float("y_step", y_step))
        seed = _c_int("seed", seed)
        out = _output(out, counts[::-1])
        if out.size:
            _lib.fnGenUniformGrid2D(handle, out.ctypes.data, *offsets, *counts, *steps, seed, None)
        return out

    def gen_uniform_grid_3d(
        self,
        *,
        x_count: int,
        y_count: int,
        z_count: int,
        x_step: float,
        y_step: float,
        z_step: float,
        seed: int,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        out: Float32Array | None = None,
    ) -> Float32Array:
        """Noise on a regular grid: ``result[k, j, i]`` is the value at
        ``(x_offset + i * x_step, y_offset + j * y_step, z_offset + k * z_step)``,
        computed in float32.

        :param out: C-contiguous float32 array of shape ``(z_count, y_count, x_count)``
            to fill.
        :return: ``out``, or a new array.
        """
        handle = self._ready()
        offsets = (
            _c_float("x_offset", x_offset),
            _c_float("y_offset", y_offset),
            _c_float("z_offset", z_offset),
        )
        counts = (
            _c_int("x_count", x_count, 0),
            _c_int("y_count", y_count, 0),
            _c_int("z_count", z_count, 0),
        )
        steps = (
            _c_float("x_step", x_step),
            _c_float("y_step", y_step),
            _c_float("z_step", z_step),
        )
        seed = _c_int("seed", seed)
        out = _output(out, counts[::-1])
        if out.size:
            _lib.fnGenUniformGrid3D(handle, out.ctypes.data, *offsets, *counts, *steps, seed, None)
        return out

    def gen_position_array_2d(
        self,
        x: Float32Array,
        y: Float32Array,
        seed: int,
        *,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        out: Float32Array | None = None,
    ) -> Float32Array:
        """Noise at the positions ``(x + x_offset, y + y_offset)``, added in float32.

        :param x: C-contiguous float32 array of any shape.
        :param y: C-contiguous float32 array of the shape of ``x``.
        :param out: C-contiguous float32 array of the shape of ``x`` to fill. It must
            not overlap ``x`` or ``y``.
        :return: ``out``, or a new array.
        """
        handle = self._ready()
        offsets = (_c_float("x_offset", x_offset), _c_float("y_offset", y_offset))
        seed = _c_int("seed", seed)
        out = _position_output(out, x=x, y=y)
        if out.size:
            _lib.fnGenPositionArray2D(
                handle,
                out.ctypes.data,
                out.size,
                x.ctypes.data,
                y.ctypes.data,
                *offsets,
                seed,
                None,
            )
        return out

    def gen_position_array_3d(
        self,
        x: Float32Array,
        y: Float32Array,
        z: Float32Array,
        seed: int,
        *,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        out: Float32Array | None = None,
    ) -> Float32Array:
        """Noise at the positions ``(x + x_offset, y + y_offset, z + z_offset)``, added
        in float32.

        :param x: C-contiguous float32 array of any shape.
        :param y: C-contiguous float32 array of the shape of ``x``.
        :param z: C-contiguous float32 array of the shape of ``x``.
        :param out: C-contiguous float32 array of the shape of ``x`` to fill. It must
            not overlap ``x``, ``y`` or ``z``.
        :return: ``out``, or a new array.
        """
        handle = self._ready()
        offsets = (
            _c_float("x_offset", x_offset),
            _c_float("y_offset", y_offset),
            _c_float("z_offset", z_offset),
        )
        seed = _c_int("seed", seed)
        out = _position_output(out, x=x, y=y, z=z)
        if out.size:
            _lib.fnGenPositionArray3D(
                handle,
                out.ctypes.data,
                out.size,
                x.ctypes.data,
                y.ctypes.data,
                z.ctypes.data,
                *offsets,
                seed,
                None,
            )
        return out

    def gen_single_2d(self, x: float, y: float, seed: int) -> float:
        """Noise at one position, a float32 value. Far slower per value than arrays."""
        handle = self._ready()
        position = (_c_float("x", x), _c_float("y", y))
        value: float = _lib.fnGenSingle2D(handle, *position, _c_int("seed", seed))
        return value

    def gen_single_3d(self, x: float, y: float, z: float, seed: int) -> float:
        """Noise at one position, a float32 value. Far slower per value than arrays."""
        handle = self._ready()
        position = (_c_float("x", x), _c_float("y", y), _c_float("z", z))
        value: float = _lib.fnGenSingle3D(handle, *position, _c_int("seed", seed))
        return value

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "a pyfastnoise2.Node wraps a native handle and cannot be pickled or copied; "
            "build the node in the process that uses it"
        )

    def __repr__(self) -> str:
        return f"<pyfastnoise2.Node {self.node_type} {self.simd_level}>"
