import random
from typing import Callable, Dict, Iterable, List, Tuple
import numpy as np
import numba
import pytest
import hypothesis
from hypothesis import given, settings
from hypothesis.strategies import (
    DataObject,
    data,
    integers,
    lists,
    permutations,
)

import minitorch
from minitorch import MathTestVariable, Tensor, TensorBackend, grad_check

from tests.strategies import assert_close, small_floats
from tests.tensor_strategies import (
    assert_close_tensor,
    shaped_tensors,
    tensors,
)


from numba import njit
from minitorch.cuda_ops import CudaOps
from minitorch.testing import MathTest

one_arg, two_arg, red_arg = MathTestVariable._comp_testing()
# The tests in this file only run the main mathematical functions.
# The difference is that they run with different tensor ops backends.

SimpleBackend = minitorch.TensorBackend(minitorch.SimpleOps)
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
shared: Dict[str, TensorBackend] = {"fast": FastTensorBackend}

# ## Task 3.1
backend_tests = [pytest.param("fast", marks=pytest.mark.task3_1)]

# ## Task 3.2
matmul_tests = [pytest.param("fast", marks=pytest.mark.task3_2)]


if numba.cuda.is_available():
    # ## Task 3.3
    backend_tests.append(pytest.param("cuda", marks=pytest.mark.task3_3))

    # ## Task 3.4
    matmul_tests.append(pytest.param("cuda", marks=pytest.mark.task3_4))
    shared["cuda"] = minitorch.TensorBackend(minitorch.CudaOps)


# ## Task 3.1 and 3.3


@given(lists(small_floats, min_size=1))
@pytest.mark.parametrize("backend", backend_tests)
def test_create(backend: str, t1: List[float]) -> None:
    "Create different tensors."
    t2 = minitorch.tensor(t1, backend=shared[backend])
    for i in range(len(t1)):
        assert t1[i] == t2[i]


def test_parallel() -> None:

    t1 = minitorch.zeros((3,))
    g = MathTest.addConstant

    index = np.zeros((2,), dtype=np.int32)
    index[0] = 1
    index[1] = 1
    strides = np.zeros((2,), dtype=np.int32)
    strides[0] = 2
    strides[1] = 1
    shape = np.zeros((2,), dtype=np.int32)
    shape[0] = 3
    shape[1] = 2

    f1 = minitorch.fast_ops.index_to_position
    f2 = minitorch.fast_ops.to_index
    f3 = minitorch.fast_ops.broadcast_index

    print(f1(index, strides))

    out = np.zeros((2,), dtype=np.int32)
    f2(4, shape, out)
    print(out)

    big_index = np.zeros((3,), dtype=np.int32)
    big_shape = np.zeros((3,), dtype=np.int32)
    out = np.zeros((2,), dtype=np.int32)
    big_index[0] = 0
    big_index[1] = 1
    big_index[2] = 1
    big_shape[0] = 4
    big_shape[1] = 3
    big_shape[2] = 2
    f3(big_index, big_shape, shape, out)
    print(big_index)
    print(out)

    # the above code runs, confirming that the njit'ed
    # versions of the three indexing functions works.
    # so there is a problem with tensor_map.

    tmap = minitorch.fast_ops.tensor_map(njit()(g))
    out = minitorch.zeros((3, 3))
    tmap(*out.tuple(), *t1.tuple())
    print(out)
    # assert out[0] == 5.0

    a = minitorch.tensor([1, 2, 3])
    b = minitorch.tensor([[1, 2, 3], [2, 3, 4]])
    out = minitorch.zeros((2, 3))
    g = minitorch.testing.MathTest.add2
    tzip = minitorch.fast_ops.tensor_zip(njit()(g))
    tzip(*out.tuple(), *a.tuple(), *b.tuple())
    print(out)
    # assert out[0] == 2.0
    # assert out[1] == 4.0
    # assert out[2] == 6.0

    tr = minitorch.fast_ops.tensor_reduce(njit()(g))
    a = minitorch.tensor([[1, 2, 3], [4, 5, 6]])
    print(a)
    out = minitorch.zeros((2, 1))
    tr(*out.tuple(), *a.tuple(), 1)
    print(out)


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_one_args(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run forward for all one arg functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, base_fn, tensor_fn = fn
    t2 = tensor_fn(t1)
    for ind in t2._tensor.indices():
        assert_close(t2[ind], base_fn(t1[ind]))


def cuda_test() -> None:
    t1 = minitorch.zeros((3,))
    g = MathTest.addConstant
    tmap = CudaOps.map(g)
    # tmap = minitorch.cuda_ops.tensor_map(cuda.jit(device=True)(g))
    out = minitorch.zeros((3, 3))
    tmap(t1, out)
    print(out)
    # assert out[0] == 5.0

    a = minitorch.tensor([1, 2, 3])
    b = minitorch.tensor([[1, 2, 3], [2, 3, 4]])
    out = minitorch.zeros((2, 3))
    g = minitorch.testing.MathTest.add2
    tzip = CudaOps.zip(g)
    out = tzip(a, b)
    print(out)
    # assert out[0] == 2.0
    # assert out[1] == 4.0
    # assert out[2] == 6.0

    tr = minitorch.fast_ops.tensor_reduce(njit()(g))
    a = minitorch.tensor([[1, 2, 3], [4, 5, 6]])
    print(a)
    out = minitorch.zeros((2, 1))
    tr(*out.tuple(), *a.tuple(), 1)
    print(out)


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_two_args(
    fn: Tuple[
        str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]
    ],
    backend: str,
    data: DataObject,
) -> None:
    "Run forward for all two arg functions above."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, base_fn, tensor_fn = fn
    t3 = tensor_fn(t1, t2)
    for ind in t3._tensor.indices():
        assert_close(t3[ind], base_fn(t1[ind], t2[ind]))


@given(data())
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_one_derivative(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all one arg functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(data())
@settings(
    max_examples=10,
    suppress_health_check=[hypothesis.HealthCheck.data_too_large],
)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_two_grad(
    fn: Tuple[
        str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]
    ],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all two arg functions above."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1, t2)


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", red_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_reduce(
    fn: Tuple[
        str, Callable[[Iterable[float]], float], Callable[[Tensor], Tensor]
    ],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all reduce functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


if numba.cuda.is_available():

    @pytest.mark.task3_3
    def test_sum_practice() -> None:
        x = [random.random() for i in range(16)]
        b = minitorch.tensor(x)
        s = b.sum()[0]
        b2 = minitorch.tensor(x, backend=shared["cuda"])
        out = minitorch.sum_practice(b2)
        print(out._storage[0])
        assert_close(s, out._storage[0])

    @pytest.mark.task3_3
    def test_sum_practice2() -> None:
        x = [random.random() for i in range(64)]
        b = minitorch.tensor(x)
        s = b.sum()[0]
        b2 = minitorch.tensor(x, backend=shared["cuda"])
        out = minitorch.sum_practice(b2)
        assert_close(s, out._storage[0] + out._storage[1])

    @pytest.mark.task3_3
    def test_sum_practice3() -> None:
        x = [random.random() for i in range(48)]
        b = minitorch.tensor(x)
        s = b.sum()[0]
        b2 = minitorch.tensor(x, backend=shared["cuda"])
        out = minitorch.sum_practice(b2)
        # out = b2.sum()[0]
        assert_close(s, out._storage[0] + out._storage[1])
        # assert_close(s, out)

    @pytest.mark.task3_3
    def test_sum_practice4() -> None:
        x = [random.random() for i in range(32)]
        b = minitorch.tensor(x)
        s = b.sum()[0]
        b2 = minitorch.tensor(x, backend=shared["cuda"])
        out = b2.sum(0)
        assert_close(s, out[0])

    @pytest.mark.task3_3
    def test_sum_practice5() -> None:
        x = [random.random() for i in range(500)]
        b = minitorch.tensor(x)
        s = b.sum()[0]
        b2 = minitorch.tensor(x, backend=shared["cuda"])
        out = b2.sum(0)
        assert_close(s, out[0])

    @pytest.mark.task3_3
    def test_sum_practice_other_dims() -> None:
        x = [
            [[random.random() for i in range(32)] for j in range(16)]
            for k in range(8)
        ]
        b = minitorch.tensor(x)
        s = b.sum(1)
        b2 = minitorch.tensor(x, backend=shared["cuda"])
        out = b2.sum(1)
        for i in range(16):
            for k in range(8):
                assert_close(s[k, 0, i], out[k, 0, i])

    @pytest.mark.task3_4
    def test_mul_practice1() -> None:
        x1 = [[random.random() for i in range(2)] for j in range(2)]
        y1 = [[random.random() for i in range(2)] for j in range(2)]
        z = minitorch.tensor(x1, backend=shared["fast"]) @ minitorch.tensor(
            y1, backend=shared["fast"]
        )

        x = minitorch.tensor(x1, backend=shared["cuda"])
        y = minitorch.tensor(y1, backend=shared["cuda"])
        z2 = minitorch.mm_practice(x, y)
        for i in range(2):
            for j in range(2):
                assert_close(z[i, j], z2._storage[2 * i + j])

    @pytest.mark.task3_4
    def test_mul_practice2() -> None:
        x1 = [[random.random() for i in range(32)] for j in range(32)]
        y1 = [[random.random() for i in range(32)] for j in range(32)]
        z = minitorch.tensor(x1, backend=shared["fast"]) @ minitorch.tensor(
            y1, backend=shared["fast"]
        )

        x = minitorch.tensor(x1, backend=shared["cuda"])
        y = minitorch.tensor(y1, backend=shared["cuda"])
        z2 = minitorch.mm_practice(x, y)
        for i in range(32):
            for j in range(32):
                assert_close(z[i, j], z2._storage[32 * i + j])

    @pytest.mark.task3_4
    def test_mul_practice3() -> None:
        "Small real example"
        x1 = [[random.random() for i in range(2)] for j in range(2)]
        y1 = [[random.random() for i in range(2)] for j in range(2)]
        z = minitorch.tensor(x1, backend=shared["fast"]) @ minitorch.tensor(
            y1, backend=shared["fast"]
        )

        x = minitorch.tensor(x1, backend=shared["cuda"])
        y = minitorch.tensor(y1, backend=shared["cuda"])
        z2 = x @ y

        for i in range(2):
            for j in range(2):
                assert_close(z[i, j], z2[i, j])

    @pytest.mark.task3_4
    def test_mul_practice4() -> None:
        "Extend to require 2 blocks"
        size = 33
        x1 = [[random.random() for i in range(size)] for j in range(size)]
        y1 = [[random.random() for i in range(size)] for j in range(size)]
        z = minitorch.tensor(x1, backend=shared["fast"]) @ minitorch.tensor(
            y1, backend=shared["fast"]
        )

        x = minitorch.tensor(x1, backend=shared["cuda"])
        y = minitorch.tensor(y1, backend=shared["cuda"])
        z2 = x @ y

        for i in range(size):
            for j in range(size):
                assert_close(z[i, j], z2[i, j])

    @pytest.mark.task3_4
    def test_mul_practice5() -> None:
        "Extend to require a batch"
        size = 33
        x1 = [
            [[random.random() for i in range(size)] for j in range(size)]
            for _ in range(2)
        ]
        y1 = [
            [[random.random() for i in range(size)] for j in range(size)]
            for _ in range(2)
        ]
        z = minitorch.tensor(x1, backend=shared["fast"]) @ minitorch.tensor(
            y1, backend=shared["fast"]
        )

        x = minitorch.tensor(x1, backend=shared["cuda"])
        y = minitorch.tensor(y1, backend=shared["cuda"])
        z2 = x @ y

        for b in range(2):
            for i in range(size):
                for j in range(size):
                    assert_close(z[b, i, j], z2[b, i, j])

    @pytest.mark.task3_4
    def test_mul_practice6() -> None:
        "Extend to require a batch"
        size_a = 45
        size_b = 40
        size_in = 33
        x1 = [
            [[random.random() for i in range(size_in)] for j in range(size_a)]
            for _ in range(2)
        ]
        y1 = [
            [[random.random() for i in range(size_b)] for j in range(size_in)]
            for _ in range(2)
        ]
        z = minitorch.tensor(x1, backend=shared["fast"]) @ minitorch.tensor(
            y1, backend=shared["fast"]
        )

        x = minitorch.tensor(x1, backend=shared["cuda"])
        y = minitorch.tensor(y1, backend=shared["cuda"])
        z2 = x @ y

        for b in range(2):
            for i in range(size_a):
                for j in range(size_b):
                    print(i, j)
                    assert_close(z[b, i, j], z2[b, i, j])


@given(data())
@settings(
    max_examples=25,
    suppress_health_check=[hypothesis.HealthCheck.data_too_large],
)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_two_grad_broadcast(
    fn: Tuple[
        str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]
    ],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all two arg functions above with broadcast."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, base_fn, tensor_fn = fn

    grad_check(tensor_fn, t1, t2)

    # broadcast check
    grad_check(tensor_fn, t1.sum(0), t2)
    grad_check(tensor_fn, t1, t2.sum(0))


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("backend", backend_tests)
def test_permute(backend: str, data: DataObject) -> None:
    "Check permutations for all backends."
    t1 = data.draw(tensors(backend=shared[backend]))
    permutation = data.draw(permutations(range(len(t1.shape))))

    def permute(a: Tensor) -> Tensor:
        return a.permute(*permutation)

    minitorch.grad_check(permute, t1)


@pytest.mark.task3_2
def test_mm2() -> None:
    a = minitorch.rand((2, 3), backend=FastTensorBackend)
    b = minitorch.rand((3, 4), backend=FastTensorBackend)
    c = a @ b

    c2 = (a.view(2, 3, 1) * b.view(1, 3, 4)).sum(1).view(2, 4)

    for ind in c._tensor.indices():
        assert_close(c[ind], c2[ind])

    minitorch.grad_check(lambda a, b: a @ b, a, b)


# ## Task 3.2 and 3.4

# Matrix Multiplication


@given(data())
@pytest.mark.parametrize("backend", matmul_tests)
def test_bmm(backend: str, data: DataObject) -> None:
    small_ints = integers(min_value=2, max_value=4)
    A, B, C, D = (
        data.draw(small_ints),
        data.draw(small_ints),
        data.draw(small_ints),
        data.draw(small_ints),
    )
    a = data.draw(tensors(backend=shared[backend], shape=(D, A, B)))
    b = data.draw(tensors(backend=shared[backend], shape=(1, B, C)))

    c = a @ b
    c2 = (
        (a.contiguous().view(D, A, B, 1) * b.contiguous().view(1, 1, B, C))
        .sum(2)
        .view(D, A, C)
    )
    assert_close_tensor(c, c2)
