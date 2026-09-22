from ht.core.context import context


__all__ = ["NoGrad"]


class NoGrad:
    def __enter__(self):
        context.is_grad_enabled = False

    def __exit__(self, _exc_type, _exc_value, _exc_traceback):
        context.is_grad_enabled = True
