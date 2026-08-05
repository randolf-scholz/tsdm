import logging
from collections.abc import Callable as Fn
from functools import wraps


def debug[**P, R](func: Fn[P, R], /) -> Fn[P, R]:  # +R
    r"""Print the function signature and return value."""
    logger = logging.getLogger(f"debug@{func.__name__}")

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        args_repr = [f"{type(a)}" for a in args]
        kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
        passed_args = ", ".join(args_repr + kwargs_repr)
        logger.info("Calling with arguments %s", passed_args)
        value = func(*args, **kwargs)
        logger.info("Return value %s", value)
        return value

    return __wrapper
