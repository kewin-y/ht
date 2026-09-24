from __future__ import annotations

import torch
import numpy as np
from typing import Tuple, Callable
from ht.core.nograd import NoGrad
from ht.core.context import context


# TODO: implement overrides for division, subtraction, negation
# implement overrides for assignment operators (throw on requires_grad = true)

__all__ = ["Tensor", "zeros_like", "ones_like"]


def _mk_children(children: Tuple[Tensor, ...], requires_grad: bool) -> Tuple[Tensor, ...]:
    return children if context.is_grad_enabled and requires_grad else ()


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

        assert self.data.ndim <= 2

        self.requires_grad = requires_grad
        self.grad = None

        self._prev = _children
        self._op = _op
        self._label = _label
        self._grad_fn: Callable[[], None] | None = None

    def __add__(self, o):
        rg = self.requires_grad
        other = o if isinstance(o, Tensor) else Tensor(o, requires_grad=rg)
        assert o.requires_grad is rg

        sum_data = self.data + other.data

        out = Tensor(
            sum_data,
            _children=_mk_children((self, other), rg),
            _op="+",
            _label=f"({self._label}) + ({other._label})",
        )

        if rg and context.is_grad_enabled:
            assert other.requires_grad == True

            def grad_fn():
                assert out.grad is not None
                assert isinstance(out.grad, Tensor)
                assert not out.grad.requires_grad

                with NoGrad():
                    self.grad = (zeros_like(self) if self.grad is None else self.grad) + out.grad

                    other.grad = (zeros_like(other) if other.grad is None else other.grad) + out.grad

            out.requires_grad = True
            out._grad_fn = grad_fn

        return out

    # Component-wise multiplication (Hadamard Product)
    def __mul__(self, o):
        rg = self.requires_grad
        other = o if isinstance(o, Tensor) else Tensor(o, requires_grad=rg)
        assert o.requires_grad is rg

        mul_data = np.multiply(self.data, other.data)

        out = Tensor(
            mul_data,
            _children=_mk_children((self, other), self.requires_grad),
            _op="*",
            _label=f"({self._label}) * ({other._label})",
        )

        if rg and context.is_grad_enabled:
            assert other.requires_grad == True

            def grad_fn():
                assert out.grad is not None
                assert isinstance(out.grad, Tensor)
                assert not out.grad.requires_grad

                with NoGrad():
                    self.grad = (ones_like(self) if self.grad is None else self.grad) * out.grad

                    other.grad = (ones_like(other) if other.grad is None else other.grad) * out.grad

            out.requires_grad = True
            out._grad_fn = grad_fn

        return out

    def __matmul__(self, o):
        rg = self.requires_grad
        other = o if isinstance(o, Tensor) else Tensor(o, requires_grad=rg)
        assert rg is o.requires_grad

        mm_data = np.matmul(self.data, other.data)

        out = Tensor(
            mm_data,
            _children=_mk_children((self, other), self.requires_grad),
            _op="@",
            _label=f"({self._label}) @ ({other._label})",
        )

        if rg and context.is_grad_enabled:
            assert other.requires_grad == True

            def grad_fn():
                print("matmul grad is being called")
                assert out.grad is not None
                assert isinstance(out.grad, Tensor)
                assert not out.grad.requires_grad

                with NoGrad():
                    # numpy does not support @ if any of the operands has ndim of 0

                    if self.ndim() == 1 and other.ndim() == 1:
                        # out.ndim = 0
                        # vector dot product
                        self.grad = out.grad * other
                        other.grad = out.grad * self
                    elif self.ndim() == 1 and other.ndim() == 2:
                        # LHS acts as a row vector.
                        # e.g., (n,) @ (n, m) becomes (1, n) @ (n, m)
                        # result is a row vector shape (should be (1, m)), but has shape (m,)
                        # out.ndim = 1

                        # shape(other.T) = (m, n)
                        self.grad = out.grad @ other.T()

                        # we need (n, 1) @ (1, m)
                        other.grad = self.reshape((1, -1)).T() @ out.grad.reshape((1, -1))
                    elif self.ndim() == 2 and other.ndim() == 1:
                        # Matrix-vector product
                        # e.g., (n, m) @ (m,) becomes (n, m) @ (m, 1)
                        # result is a column vector (should be (n, 1)), but has shape (n,)
                        # out.ndim = 1

                        # we need (n, 1) @ (1, m)
                        self.grad = out.grad.reshape((-1, 1)) @ other.reshape((-1, 1)).T()

                        # shape(self.T) = (m, n)
                        other.grad = self.T() @ out.grad
                    else:
                        self.grad = out.grad @ other.T()
                        other.grad = self.T() @ out.grad

            out.requires_grad = True
            out._grad_fn = grad_fn

        return out

    def T(self):
        out = Tensor(
            self.data.T.copy(),
            _children=_mk_children((self,), self.requires_grad),
            _op="^T",
            _label=f"({self._label})^T",
        )

        if self.requires_grad and context.is_grad_enabled:

            def grad_fn():
                assert out.grad is not None
                assert out.requires_grad == False
                assert isinstance(out.grad, Tensor)

                with NoGrad():
                    self.grad = out.grad.T()

            out.requires_grad = True
            out._grad_fn = grad_fn

        return out

    def reshape(self, shape: Tuple[int, ...]):
        out = Tensor(
            self.data.reshape(shape, copy=True),
            _children=_mk_children((self,), self.requires_grad),
            _op=f"resshape({shape})",
            _label=f"(reshape({self._label}, {shape})",
        )

        if self.requires_grad and context.is_grad_enabled:
            return NotImplemented

        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rmatmul__(self, o):
        other = o if isinstance(o, Tensor) else Tensor(o)

        return other @ self

    def shape(self):
        return self.data.shape

    def ndim(self):
        return len(self.shape())

    def backward(self):
        assert self.requires_grad

        seen = set()
        nodes: list[Tensor] = []

        def search(n: Tensor):
            if n in seen:
                return

            seen.add(n)

            for node in n._prev:
                search(node)

            nodes.append(n)

        self.grad = ones_like(self)
        search(self)
        for i in range(len(nodes) - 1, 0, -1):
            fn = nodes[i]._grad_fn
            if fn is not None:
                fn()

        self.grad = None

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad} grad={self.grad})"


def zeros_like(t: Tensor, requires_grad=False):
    return Tensor(np.zeros_like(t.data), requires_grad=requires_grad)


def ones_like(t: Tensor, requires_grad=False):
    return Tensor(np.ones_like(t.data), requires_grad=requires_grad)
