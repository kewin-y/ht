from __future__ import annotations

import torch
import numpy as np
from typing import Tuple


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

    def __add__(self, o):
        other = o if isinstance(o, Tensor) else Tensor(o)

        sum_data = self.data + other.data
        out = Tensor(
            sum_data,
            _children=(self, other),
            _op="+",
            _label=f"({self._label}) + ({other._label})",
        )
        return out

    # component-wise multiplication
    def __mul__(self, o):
        other = o if isinstance(o, Tensor) else Tensor(o)

        mul_data = np.multiply(self.data, other.data)
        out = Tensor(
            mul_data,
            _children=(self, other),
            _op="*",
            _label=f"({self._label}) * ({other._label})",
        )

        return out

    def __matmul__(self, o):
        other = o if isinstance(o, Tensor) else Tensor(o)

        mm_data = np.matmul(self.data, other.data)
        out = Tensor(
            mm_data,
            _children=(self, other),
            _op="@",
            _label=f"({self._label}) @ ({other._label})",
        )

        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rmatmul__(self, o):
        other = o if isinstance(o, Tensor) else Tensor(o)

        return other @ self

    def __repr__(self):
        return f"Tensor(data={self.data}, poop={self.requires_grad} grad={self.grad})"
