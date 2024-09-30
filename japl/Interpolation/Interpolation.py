import numpy as np
from typing import Any
from scipy.interpolate import RegularGridInterpolator
import linterp  # type:ignore



class LinearInterp:

    """This class provides interpolation for multidimensional
    data. This is primarily used in DataTable class."""

    def __init__(self, axes: tuple[np.ndarray, ...], table: np.ndarray) -> None:
        """
        Args:
        -------------------------------------------------------------------------
            - axes: tuple of axis dimension arrays for example:
                if the data to interpolate has axes of alpha, mach, altitutude,
                then the axes arguments is a tuple of arrays: (alpha, mach, altitude).
            - table: the data table to be interpolated.
        -------------------------------------------------------------------------
        """
        try:
            self._interp_obj = self.interp = self.create_linterp(axes, table)
        except Exception as e:
            print("linterp extension module not found. using RegularGridInterpolator", e)
            self._interp_obj = self.interp = RegularGridInterpolator(axes, table)


    def __call__(self, args: tuple[np.ndarray, ...]) -> np.ndarray:
        return self._interp_obj(args)


    @staticmethod
    def create_linterp(axes: tuple[np.ndarray, ...], table: np.ndarray) -> Any:
        """This method creates a linear interpolation object from the \"linterp\"
        c++ library. It is a drop-in replacement fro RegularGridInterpolator and
        provides a 2x performance speed up.
        """
        naxes = len(axes)
        match naxes:
            case 1:
                return linterp.Interp1d(axes, table)
            case 2:
                return linterp.Interp2d(axes, table)
            case 3:
                return linterp.Interp3d(axes, table)
            case 4:
                return linterp.Interp4d(axes, table)
            case 5:
                return linterp.Interp5d(axes, table)
            case _:
                raise Exception("unhanded case.")
