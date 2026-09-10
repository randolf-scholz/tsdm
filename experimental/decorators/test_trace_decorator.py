import logging
from collections.abc import Callable as Fn
from functools import wraps


def trace[**P, R](func: Fn[P, R], /) -> Fn[P, R]:  # +R
    r"""Log entering and exiting of function."""
    logger = logging.getLogger("trace")

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.info(
            "%s",
            "\n\t".join(
                (
                    f"{func.__qualname__}: ENTERING",
                    f"args={tuple(type(arg).__name__ for arg in args)}",
                    f"kwargs={ {k: type(v).__name__ for k, v in kwargs.items()}!s}",
                )
            ),
        )
        try:
            logger.info("%s: EXECUTING", func.__qualname__)
            result = func(*args, **kwargs)
        except Exception as exc:
            logger.exception("Execution of %s failed!", func.__qualname__)
            raise RuntimeError(
                f"Function execution failed with Exception {exc}"
            ) from exc
        logger.info(
            "%s: SUCCESS with result=%s", func.__qualname__, type(result).__name__
        )
        logger.info("%s", "\n\t".join((f"{func.__qualname__}: EXITING",)))
        return result

    return __wrapper
