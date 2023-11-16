"""Clean namespace of module (unused)."""

# import logging
# from datetime import datetime
# c_now=datetime.now()
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] :: %(message)s",
#     handlers=[
#         logging.StreamHandler(),
#         # logging.FileHandler("../logs/log_file_{}-{}-{}-{}.log"
#         # .format(c_now.year,c_now.month,c_now.day,c_now.hour))
#     ]
# )
# DEBUG_LEVEL_NUM = 99
# logging.addLevelName(DEBUG_LEVEL_NUM, "CUSTOM")
# def custom_level(message, *args, **kws):
#     logging.Logger._log(logging.root,DEBUG_LEVEL_NUM, message, args, **kws)
# logging.custom_level = custom_level
#
# logging.custom_level("demo")

# #!/usr/bin/env python
# # encoding: utf-8
# import logging
# # now we patch Python code to add color support to logging.StreamHandler
# def add_coloring_to_emit_windows(fn):
#         # add methods we need to the class
#     def _out_handle(self):
#         import ctypes
#         return ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)
#     out_handle = property(_out_handle)
#
#     def _set_color(self, code):
#         import ctypes
#         # Constants from the Windows API
#         self.STD_OUTPUT_HANDLE = -11
#         hdl = ctypes.windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)
#         ctypes.windll.kernel32.SetConsoleTextAttribute(hdl, code)
#
#     setattr(logging.StreamHandler, '_set_color', _set_color)
#
#     def new(*args):
#         FOREGROUND_BLUE      = 0x0001 # text color contains blue.
#         FOREGROUND_GREEN     = 0x0002 # text color contains green.
#         FOREGROUND_RED       = 0x0004 # text color contains red.
#         FOREGROUND_INTENSITY = 0x0008 # text color is intensified.
#         FOREGROUND_WHITE     = FOREGROUND_BLUE|FOREGROUND_GREEN |FOREGROUND_RED
#         STD_INPUT_HANDLE = -10
#         STD_OUTPUT_HANDLE = -11
#         STD_ERROR_HANDLE = -12
#
#         FOREGROUND_BLACK     = 0x0000
#         FOREGROUND_BLUE      = 0x0001
#         FOREGROUND_GREEN     = 0x0002
#         FOREGROUND_CYAN      = 0x0003
#         FOREGROUND_RED       = 0x0004
#         FOREGROUND_MAGENTA   = 0x0005
#         FOREGROUND_YELLOW    = 0x0006
#         FOREGROUND_GREY      = 0x0007
#         FOREGROUND_INTENSITY = 0x0008 # foreground color is intensified.
#
#         BACKGROUND_BLACK     = 0x0000
#         BACKGROUND_BLUE      = 0x0010
#         BACKGROUND_GREEN     = 0x0020
#         BACKGROUND_CYAN      = 0x0030
#         BACKGROUND_RED       = 0x0040
#         BACKGROUND_MAGENTA   = 0x0050
#         BACKGROUND_YELLOW    = 0x0060
#         BACKGROUND_GREY      = 0x0070
#         BACKGROUND_INTENSITY = 0x0080 # background color is intensified.
#
#         levelno = args[1].levelno
#         if(levelno>=50):
#             color = BACKGROUND_YELLOW | FOREGROUND_RED | FOREGROUND_INTENSITY
#             | BACKGROUND_INTENSITY
#         elif(levelno>=40):
#             color = FOREGROUND_RED | FOREGROUND_INTENSITY
#         elif(levelno>=30):
#             color = FOREGROUND_YELLOW | FOREGROUND_INTENSITY
#         elif(levelno>=20):
#             color = FOREGROUND_GREEN
#         elif(levelno>=10):
#             color = FOREGROUND_MAGENTA
#         else:
#             color =  FOREGROUND_WHITE
#         args[0]._set_color(color)
#
#         ret = fn(*args)
#         args[0]._set_color( FOREGROUND_WHITE )
#         #print "after"
#         return ret
#     return new
#
# def add_coloring_to_emit_ansi(fn):
#     # add methods we need to the class
#     def new(*args):
#         levelno = args[1].levelno
#         if(levelno>=50):
#             color = '\x1b[31m' # red
#         elif(levelno>=40):
#             color = '\x1b[31m' # red
#         elif(levelno>=30):
#             color = '\x1b[33m' # yellow
#         elif(levelno>=20):
#             color = '\x1b[32m' # green
#         elif(levelno>=10):
#             color = '\x1b[35m' # pink
#         else:
#             color = '\x1b[0m' # normal
#         args[1].msg = color + args[1].msg +  '\x1b[0m'  # normal
#         #print "after"
#         return fn(*args)
#     return new
#
# import platform
# if platform.system()=='Windows':
#     # Windows does not support ANSI escapes and we are using API calls to set the console color
#     logging.StreamHandler.emit = add_coloring_to_emit_windows(logging.StreamHandler.emit)
# else:
#     # all non-Windows platforms are supporting ANSI escapes so we use them
#     logging.StreamHandler.emit = add_coloring_to_emit_ansi(logging.StreamHandler.emit)
#     #log = logging.getLogger()
#     #log.addFilter(log_filter())
#     #//handler = logging.StreamHandler()
#     #//handler.setFormatter(formatter())

#
# GREY = "\x1b[38;21m"
# YELLOW = "\x1b[33;21m"
# RED = "\x1b[31;21m"
# BOLD_RED = "\x1b[31;1m"
# RESET = "\x1b[0m"
# GREEN = "\x1b[1;32m"
#
#
# class CustomFormatter(logging.Formatter):
#
#     format = (
#         "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
#     )
#
#     FORMATS = {
#         logging.DEBUG: GREY + format + RESET,
#         logging.INFO: GREY + format + RESET,
#         logging.WARNING: YELLOW + format + RESET,
#         logging.ERROR: RED + format + RESET,
#         logging.CRITICAL: BOLD_RED + format + RESET,
#     }
#
#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno)
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)
#
#
# class MyLogger(logging.Logger):
#     #
#     # def __init__(self, *args, **kwargs):
#     #     super().__init__(*args, **kwargs)
#
#     def message(self, msg, *args, **kwargs):
#         if self.isEnabledFor(MESSAGE):
#             self._log(MESSAGE, msg, args, **kwargs)
#
#     def passed(self, message, *args, **kws):
#         # Yes, logger takes its '*args' as 'args'.
#         message = GREEN + "✔ " + message + " PASSED ✔ " + RESET
#         self._log(logging.INFO, message, args, **kws)
#
#     def failed(self, message, *args, **kws):
#         # Yes, logger takes its '*args' as 'args'.
#         message = BOLD_RED + "✘ " + message + " FAILED ✘ " + RESET
#         self._log(logging.ERROR, message, args, **kws)
#
# logging.setLoggerClass(MyLogger)
#
# from typing_extensions import cast
#
# LOGGER: MyLogger = cast(MyLogger, logging.getLogger(__name__))
#
# from typing_extensions import TypeVar, Generic
# T = TypeVar("T", bound="Foo")
#
# class Foo():
#
#     def make_demo(self: T) -> T:
#         cls = type(self)
#         return cls()
#
#
# class Bar(Foo):
#     ...
#
# m = Bar()
#
# x: Bar = m.make_demo()
# y = m.make_demo()

# create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
#
# ch.setFormatter(CustomFormatter())
# __logger__.addHandler(ch)

# logging.Logger.passed = passed
# logging.Logger.failed = failed

# setattr(logging.getLoggerClass(), "passed", passed)
# setattr(logging.getLoggerClass(), "failed", failed)
#
# __logger__ = logging.getLogger(__name__)
# setattr(logging.Logger, "passed", passed)
# setattr(logging.Logger, "failed", failed)
# setattr(__logger__, "passed", passed)
# setattr(__logger__, "failed", failed)
#
#
# __logger__.debug("debug message")
# __logger__.info("info message")
# __logger__.warning("warning message")
# __logger__.error("error message")
# __logger__.critical("critical message")
# __logger__.log(logging.INFO, "success message")
# __logger__.log(logging.ERROR, "failed message")
#
#
# class demo:
#
#     def __getattr__(self, item) -> object:
#         return ...
#
#
# x = demo()
# print(x.omega)


# region cleanup -----------------------------------------------------------------------
# region cleanup -----------------------------------------------------------------------


# import logging
# from types import ModuleType
#
# __logger__: logging.Logger = logging.getLogger(__name__)
#
#
# def _clean_namespace(module: ModuleType) -> None:
#     r"""Recursively cleans up the namespace.
#
#     Sets `obj.__module__` equal to `obj.__package__` for all objects listed in
#     `package.__all__` that are originating from private submodules (`package/_module.py`).
#     """
#     __logger__.info("Cleaning module=%s", module)
#     variables = vars(module)
#
#     def is_private(s: str) -> bool:
#         return s.startswith("_") and not s.startswith("__")
#
#     def get_module(obj_ref: object) -> str:
#         return obj_ref.__module__.rsplit(".", maxsplit=1)[-1]
#
#     assert hasattr(module, "__name__"), f"{module=} has no __name__ ?!?!"
#     assert hasattr(module, "__package__"), f"{module=} has no __package__ ?!?!"
#     assert hasattr(module, "__all__"), f"{module=} has no __all__!"
#     assert module.__name__ == module.__package__, f"{module=} is not a package!"
#
#     max_length = max((len(key) for key in variables))
#
#     def _format(key: str) -> str:
#         return key.ljust(max_length)
#
#     for key in list(variables):
#         key_repr = _format(key)
#         obj = variables[key]
#         # ignore _clean_namespace and ModuleType
#         if key in ("ModuleType", "_clean_namespace"):
#             __logger__.debug("key=%s  skipped! - protected object!", key_repr)
#             continue
#         # ignore dunder keys
#         if key.startswith("__") and key.endswith("__"):
#             __logger__.debug("key=%s  skipped! - dunder object!", key_repr)
#             continue
#         # special treatment for ModuleTypes
#         if isinstance(obj, ModuleType):
#             if obj.__package__ is None:
#                 __logger__.debug(
#                     "key=%s  skipped! Module with no __package__!", key_repr
#                 )
#                 continue
#             # subpackage!
#             if obj.__package__.rsplit(".", maxsplit=1)[0] == module.__name__:
#                 __logger__.debug("key=%s  recursion!", key_repr)
#                 _clean_namespace(obj)
#             # submodule!
#             elif obj.__package__ == module.__name__:
#                 __logger__.debug("key=%s  skipped! Sub-Module!", key_repr)
#                 continue
#             # 3rd party!
#             else:
#                 __logger__.debug("key=%s  skipped! 3rd party Module!", key_repr)
#                 continue
#         # key is found:
#         if key in module.__all__:
#             # set __module__ attribute to __package__ for functions/classes
#             # originating from private modules.
#             if isinstance(obj, type) or callable(obj):
#                 mod = get_module(obj)
#                 if is_private(mod):
#                     __logger__.debug(
#                         "key=%s  changed {obj.__module__=} to {module.__package__}!",
#                         key_repr,
#                     )
#                     obj.__module__ = str(module.__package__)
#         else:
#             # kill the object
#             delattr(module, key)
#             __logger__.debug("key=%s  killed!", key_repr)
#     # Post Loop - clean up the rest
#     for key in ("ModuleType", "_clean_namespace"):
#         if key in variables:
#             key_repr = _format(key)
#             delattr(module, key)
#             __logger__.debug("key=%s  killed!", key_repr)
#
#
# # recursively clean namespace from self.
# _clean_namespace(__import__(__name__))

# endregion cleanup --------------------------------------------------------------------
