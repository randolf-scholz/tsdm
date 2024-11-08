r"""Show which torch builtins can be scripted directly."""

import torch
from torch.jit.frontend import NotSupportedError, UnsupportedNodeError


def show_incompatible_builtins() -> None:
    for name in torch.__all__:
        obj = getattr(torch, name)
        if name.startswith("_") or isinstance(obj, type) or not callable(obj):
            continue
        try:
            torch.jit.script(obj)
        except RuntimeError as exc:
            msg = str(exc)
            if "Python builtin" in msg:
                continue
            if "attribute lookup" in msg:
                continue
            if "Unknown type constructor" in msg:
                continue
            if "builtin cannot be used as a value" in msg:
                continue
            if "previously had type" in msg:
                continue
            if "Arguments for call are not valid." in msg:
                continue
            if "Expression of type dots cannot be used in a type expression" in msg:
                continue
            if "object has no attribute or method" in msg:
                continue
            if "hasattr's second argument must be a string literal" in msg:
                continue
            if "cannot be directly compiled because it is overloaded" in msg:
                continue
        except TypeError as exc:
            msg = str(exc)
            if "is a built-in class" in msg:
                continue
            print(f"TypeError torch.{name}")
        except AssertionError as exc:
            if " Unsupported annotation" in str(exc):
                continue
        except (NotSupportedError, UnsupportedNodeError):
            continue


if __name__ == "__main__":
    print(f"torch=={torch.__version__}")
    show_incompatible_builtins()
