from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor: ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(
        fn: Callable[[float], float]
    ) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
            :class:`TensorData` : new tensor data
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # simple version
        if len(out_shape) == len(in_shape):
            assert np.all(out_shape == in_shape)
            # loop over the indices for the in tensor.
            in_idx = np.zeros(len(in_shape), dtype=np.int32)
            for i in range(len(in_storage)):
                to_index(i, in_shape, in_idx)
                in_pos = index_to_position(in_idx, in_strides)
                val = fn(in_storage[in_pos])
                # where you write val depends on the
                # stride of the output!
                out_pos = index_to_position(in_idx, out_strides)
                out[out_pos] = val
        else:
            # assuming that in_shape broadcasts to out_shape.
            # can loop over all indices in out_shape.
            # determine the corresponding index in in_shape.
            # and repeat as above.
            out_idx = np.zeros(len(out_shape), dtype=np.int32)
            in_idx = np.zeros(len(in_shape), dtype=np.int32)
            for o_i in range(len(out)):
                to_index(o_i, out_shape, out_idx)
                in_idx = broadcast_index(out_idx, out_shape, in_shape, in_idx)
                in_pos = index_to_position(in_idx, in_strides)
                in_val = in_storage[in_pos]
                out_val = fn(in_val)
                out_pos = index_to_position(out_idx, out_strides)
                out[out_pos] = out_val

    return _map


def tensor_zip(fn: Callable[[float, float], float]) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides],
    None,
]:
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if np.all(a_shape == b_shape):
            # simple, loop over indices in the a/b/out tensors
            idx = np.zeros(len(out_shape), dtype=np.int32)
            for i in range(len(out)):
                to_index(i, out_shape, idx)
                a_pos = index_to_position(idx, a_strides)
                b_pos = index_to_position(idx, b_strides)
                val = fn(a_storage[a_pos], b_storage[b_pos])
                out_pos = index_to_position(idx, out_strides)
                out[out_pos] = val
        else:
            # assume that a_shape and b_shape both
            # broadcast to out_shape.
            # similar idea, loop over indices to out,
            # convert to indices to a and b and repeat.
            out_idx = np.zeros(len(out_shape), dtype=np.int32)
            a_idx = np.zeros(len(a_shape), dtype=np.int32)
            b_idx = np.zeros(len(b_shape), dtype=np.int32)
            for i in range(len(out)):
                to_index(i, out_shape, out_idx)
                broadcast_index(out_idx, out_shape, a_shape, a_idx)
                broadcast_index(out_idx, out_shape, b_shape, b_idx)
                a_pos = index_to_position(a_idx, a_strides)
                b_pos = index_to_position(b_idx, b_strides)
                val = fn(a_storage[a_pos], b_storage[b_pos])
                out_pos = index_to_position(out_idx, out_strides)
                out[out_pos] = val
                # there is some code duplication here which can be fixed
                # later, for now we'll leave it until tests pass.
                # the main difference in the two cases is whether one
                # index suffices for all 3 tensors or whether separate
                # indices are needed and broadcasting is required.

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float

    Returns:
        Tensor reduce function.
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # loop over indices of out.
        out_idx = np.zeros(len(out_shape), dtype=np.int32)
        a_idx = np.zeros(len(a_shape), dtype=np.int32)
        for i in range(len(out)):
            to_index(i, out_shape, out_idx)
            # the starting index for the reduction
            # is the same as the out index.
            to_index(i, out_shape, a_idx)

            # get the first position in a.
            a_pos = index_to_position(a_idx, a_strides)
            acc = a_storage[a_pos]
            # reduce along the reduce_dim.
            for j in range(1, a_shape[reduce_dim]):
                a_idx[reduce_dim] += 1
                a_pos = index_to_position(a_idx, a_strides)
                acc = fn(acc, a_storage[a_pos])

            out_pos = index_to_position(out_idx, out_strides)
            out[out_pos] = acc

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
