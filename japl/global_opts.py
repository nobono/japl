import os



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
