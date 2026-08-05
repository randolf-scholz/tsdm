import gc
import logging
from collections.abc import Callable as Fn
from functools import wraps
from time import perf_counter_ns

from tsdm.decorator import decorator


@decorator
def timefun[**P, R](  # +R
    func: Fn[P, R], /, *, loglevel: int = logging.WARNING
) -> Fn[P, tuple[R, float]]:
    r"""Log the execution time of the function. Use as decorator.

    By default, appends the execution time (in seconds) to the function call.

    `outputs, time_elapse = timefun(f, append=True)(inputs)`

    If the function call failed, `outputs=None` and `time_elapsed=float('nan')` are returned.

    If `append=True`, then the decorated function will return a tuple of the form `(func(x), time_elapsed)`.
    """
    timefun_logger = logging.getLogger("timefun")

    @wraps(func)
    def __wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
        gc.collect()
        gc.disable()
        try:
            start_time = perf_counter_ns()
            result = func(*args, **kwargs)
            end_time = perf_counter_ns()
            elapsed = (end_time - start_time) / 10**9
            timefun_logger.log(
                loglevel, "%s executed in %.4f s", func.__qualname__, elapsed
            )
        except Exception as exc:
            timefun_logger.exception("Execution of %s failed!", func.__qualname__)
            raise RuntimeError("Function execution failed") from exc
        finally:
            gc.enable()

        return result, elapsed

    return __wrapper
