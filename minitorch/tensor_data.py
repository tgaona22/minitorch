from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """

    return np.sum(index * strides)


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # It should be made clearer here that 'ordinal' refers
    # to a position in the storage array.

    # example: arange(12) with shape (2, 3, 2), contiguous stride.
    # the indices for each ordinal are in order:
    # 000, 001, 010, 011, 020, 021,
    # 100, 101, 110, 111, 120, 121
    # so for instance, ordinal 8 -> 110
    # observe that
    # 8 = 0 * 1 + 1 * 2 + 1 * (2*3)
    # 9 = 1 * 1 + 1 * 2 + 1 * (2*3)
    # 11 -> 121 and 11 = 1 * 1 + 2 * 2 + 1 * (2*3)
    # from which the pattern becomes apparent.
    # For a shape (d1, d2, ..., dn), let x_i
    # be the product d_{i+1} * ... d_n.
    # so x_n = 1. Then an ordinal 0 <= x < size can be written
    # in the form x = sum_{i=1}^n a_i*x_i.
    # The digits a_i provide the index in the shape.
    # From this, an algorithm to convert an ordinal to the index is
    # similar to the division algorithm.
    # For example, 9 / 6 = 1 with remainder 3.
    # 3 / 2 = 1 with remainder 1. 1 / 1 = 1 with remainder 0.
    # Hence 9 -> 111. Likewise, 8 / 6 = 1 remainder 2,
    # 2 / 2 = 1 remainder 0, and 0 / 1 = 0.

    if False:
        x = np.zeros(len(shape))
        o = np.zeros(len(shape))
        o[0] = ordinal
        for i in range(len(x)):
            x[i] = np.prod(shape[i + 1 :])
            if i == 0:
                o[0] = ordinal
            else:
                o[i] = o[i - 1] % x[i]
        for i in range(len(out_index)):
            out_index[i] = o[i] // x[i]

    n = len(shape)
    x = np.empty(n, dtype=np.int64)

    # Precompute products of dimensions
    product = 1
    for i in range(n - 1, -1, -1):
        x[i] = product
        product *= shape[i]

    # Compute indices for each dimension in parallel
    # Note: Numba's parallel=True may not speed up this loop due to overhead and simplicity of operations
    for i in range(n):
        if i == 0:
            out_index[i] = ordinal // x[i] % shape[i]
        else:
            out_index[i] = (ordinal // x[i]) % shape[i]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    # warning, this was written by gpt4o.
    # but after thinking for a while it makes sense.
    # Initialize out_index with zeros
    for i in range(len(out_index)):
        out_index[i] = 0

    # Calculate the offset to align the shapes
    offset = len(big_shape) - len(shape)

    # Map the big_index to out_index following broadcasting rules
    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = 0
        else:
            out_index[i] = big_index[i + offset]


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    # right align the shapes
    # if shapes don't have same number of dims,
    # add dims of length 1 to the smaller shape.
    if len(shape1) < len(shape2):
        diff = len(shape2) - len(shape1)
        s1 = [1] * diff + list(shape1)
        s2 = list(shape2)
    elif len(shape1) > len(shape2):
        diff = len(shape1) - len(shape2)
        s1 = list(shape1)
        s2 = [1] * diff + list(shape2)
    else:
        s1 = shape1
        s2 = shape2

    assert len(s1) == len(s2)

    union = list(s1)
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            if s1[i] == 1:
                union[i] = s2[i]
            elif s2[i] == 1:
                union[i] = s1[i]
            else:
                raise IndexingError(
                    f"Cannot broadcast shapes {shape1} and {shape2}"
                )
    return tuple(union)


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(
                    f"Index {aindex} out of range {self.shape}."
                )
            if ind < 0:
                raise IndexingError(
                    f"Negative indexing for {aindex} not supported."
                )

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = [0] * len(self.shape)
        new_stride = [0] * len(self.shape)
        for i in range(len(new_shape)):
            new_shape[i] = self.shape[order[i]]
            new_stride[i] = self.strides[order[i]]
        return TensorData(self._storage, tuple(new_shape), tuple(new_stride))

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
