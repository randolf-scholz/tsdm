r"""Show which torch builtins can be scripted directly."""

import torch


def show_incompatible_builtins() -> None:
    for name in dir(torch):
        obj = getattr(torch, name)
        if name.startswith("_") or isinstance(obj, type) or not callable(obj):
            continue
        try:
            torch.jit.script(obj)
        except RuntimeError:
            print(f"torch.{name}")
        except Exception:
            pass


if __name__ == "__main__":
    print(f"torch=={torch.__version__}")
    show_incompatible_builtins()
