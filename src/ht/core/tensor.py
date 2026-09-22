from __future__ import annotations

import torch
import numpy as np
from typing import Tuple
from ht.core.nograd import NoGrad
from ht.core.context import context


# TODO: implement overrides for division, subtraction, negation
# implement overrides for assignment operators (throw on requires_grad = true)

__all__ = ["Tensor", "zeros_like", "ones_like"]


def _mk_children(children: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
    return children if context.is_grad_enabled else ()


class Tensor:
    def __init__(
        self,
        data,
        requires_grad=False,
        _children=(),
        _op: str = "",
        _label: str = "",
    ):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None

        self._prev = _children
        self._op = _op
        self._label = _label
        self._grad_fn = None

    def __add__(self, o):
        other = o if isinstance(o, Tensor) else Tensor(o)

        sum_data = self.data + other.data

        out = Tensor(
            sum_data,
            _children=_mk_children((self, other)),
            _op="+",
            _label=f"({self._label}) + ({other._label})",
        )

        if self.requires_grad and context.is_grad_enabled:
            assert other.requires_grad == True

            def grad_fn():
                assert out.grad is not None
                assert isinstance(out.grad, Tensor)
                assert not out.grad.requires_grad

                self.grad = (
                    zeros_like(self) if self.grad is None else self.grad
                ) + out.grad

                other.grad = (
                    zeros_like(other) if other.grad is None else other.grad
                ) + out.grad

            out.requires_grad = True
            out._grad_fn = grad_fn

        return out

    # Component-wise multiplication (Hadamard Product)
    def __mul__(self, o):
        other = o if isinstance(o, Tensor) else Tensor(o)

        mul_data = np.multiply(self.data, other.data)

        out = Tensor(
            mul_data,
            _children=_mk_children((self, other)),
            _op="*",
            _label=f"({self._label}) * ({other._label})",
        )

        if self.requires_grad and context.is_grad_enabled:
            assert other.requires_grad == True

            def grad_fn():
                assert out.grad is not None
                assert isinstance(out.grad, Tensor)
                assert not out.grad.requires_grad

                self.grad = (
                    ones_like(self) if self.grad is None else self.grad
                ) * out.grad

                other.grad = (
                    ones_like(other) if other.grad is None else other.grad
                ) * out.grad

            out.requires_grad = True
            out._grad_fn = grad_fn

        return out

    def __matmul__(self, o):
        other = o if isinstance(o, Tensor) else Tensor(o)

        mm_data = np.matmul(self.data, other.data)

        out = Tensor(
            mm_data,
            _children=_mk_children((self, other)),
            _op="@",
            _label=f"({self._label}) @ ({other._label})",
        )

        if self.requires_grad and context.is_grad_enabled:
            assert other.requires_grad == True

            def grad_fn():
                assert out.grad is not None
                assert isinstance(out.grad, Tensor)
                assert not out.grad.requires_grad

                with NoGrad():
                    self.grad = out.grad @ other.T()
                    other.grad = self.T() @ out.grad

            out.requires_grad = True
            out._grad_fn = grad_fn

        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rmatmul__(self, o):
        other = o if isinstance(o, Tensor) else Tensor(o)

        return other @ self

    def T(self):
        out = Tensor(
            self.data.T,
            _children=_mk_children((self,)),
            _op="^T",
            _label=f"({self._label})^T",
        )

        if self.requires_grad and context.is_grad_enabled:

            def grad_fn():
                assert out.grad is not None
                assert out.requires_grad == False
                assert isinstance(out.grad, Tensor)

                self.grad = out.grad.T()

            out.requires_grad = True
            out._grad_fn = grad_fn

        return out

    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad} grad={self.grad})"


def zeros_like(t: Tensor):
    return Tensor(np.zeros_like(t.data))


def ones_like(t: Tensor):
    return Tensor(np.ones_like(t.data))
