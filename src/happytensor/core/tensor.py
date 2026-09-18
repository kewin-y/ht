from __future__ import annotations

import torch
import numpy as np
from typing import Tuple


class Tensor:
    def __init__(
        self,
        data: np.typing.ArrayLike,
        requires_grad: bool = false,
        _children: Tuple[Tensor, ...] = (),
        _op: str = "",
        _label: str = "",
    ):
        self.data = np.array(data)
        self.grad = None

        self._prev = _children
        self._op = _op
        self._label = _label

    def __add__(self, o: np.typing.ArrayLike | Tensor) -> Tensor:
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
    def __mul__(self, other: np.typing.ArrayLike | Tensor) -> Tensor:
        o = other if isinstance(other, Tensor) else Tensor(other)

        mul_data = np.multiply(self.data, o.data)
        out = Tensor(
            mul_data,
            _children=(self, o),
            _op="*",
            _label=f"({self._label}) * ({o._label})",
        )

        return out

    def __matmul__(self, o: np.typing.ArrayLike | Tensor) -> Tensor:
        other = o if isinstance(o, Tensor) else Tensor(o)
        
        mm_data = np.matmul(self.data, other.data)
        out = Tensor(
            mm_data,
            _children=(self, other),
            _op="@",
            _label=f"({self._label}) @ ({other._label})",
        )
        return out


    def __rmul__(self, other: np.typing.ArrayLike | Tensor) -> Tensor:
        print("Calling __rmul__")
        return self * other

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad} grad={self.grad})"

