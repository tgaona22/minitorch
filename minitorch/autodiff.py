from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(
    f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6
) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # need to apply f to (x_0, \ldots, x_i + eps/2, \ldots x_n-1)
    # and to (x_0, \ldots, x_i - eps/2, \ldots x_n-1)
    right = [val for val in vals]
    left = [val for val in vals]
    right[arg] += epsilon / 2
    left[arg] -= epsilon / 2
    return (f(*right) - f(*left)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    marked = []
    ordered_list = []

    def _visit(var: Variable) -> None:
        if var.unique_id in marked or var.is_constant():
            return

        marked.append(var.unique_id)

        for parent in var.parents:
            _visit(parent)

        ordered_list.append(var)

    _visit(variable)

    return list(reversed(ordered_list))


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    ordered_list = topological_sort(variable)
    derivs = {ordered_list[0].unique_id: deriv}
    for var in ordered_list:
        if not var.is_leaf():
            assert derivs[var.unique_id] is not None
            back = var.chain_rule(derivs[var.unique_id])

            for var, deriv in back:
                if var.unique_id not in derivs:
                    derivs[var.unique_id] = 0
                derivs[var.unique_id] += deriv
        else:
            var.accumulate_derivative(derivs[var.unique_id])

    return


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
