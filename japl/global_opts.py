import os
import importlib.util



JAPL_HOME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

__PLOTTING_BACKEND = "pyqtgraph"


def get_plotlib() -> str:
    global __PLOTTING_BACKEND
    return __PLOTTING_BACKEND


def set_plotlib(plotlib: str) -> None:
    global __PLOTTING_BACKEND
    if plotlib in ["matplotlib", "mpl"]:
        __PLOTTING_BACKEND = "matplotlib"
    elif plotlib in ["pyqtgraph", "qt"]:
        __PLOTTING_BACKEND = "pyqtgraph"
    else:
        raise Exception(f"{plotlib} not an available plotting backend. use \"matplotlib\" or \"pyqtgraph\"")


def get_root_dir():
    # Find the module spec for the japl package
    spec = importlib.util.find_spec("japl")
    if spec is None or spec.origin is None:
        raise RuntimeError("japl package is not installed or cannot be found.")
    # Get the root directory of the japl package
    root_dir = os.path.dirname(spec.origin)
    return os.path.dirname(root_dir)
