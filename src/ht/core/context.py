__all__ = ["context"]

class _Context:
    def __init__(self):
        self.is_grad_enabled = True

context = _Context()

