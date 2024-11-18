from japl.Interpolation.Interpolation import LinearInterp
from typing import Optional, Union
import numpy as np

ArgType = Union[float, list, np.ndarray]



class DataTable(np.ndarray):

    """This class inherits from numpy.ndarray and is also a wrapper
    for LinearInterp but includes additional checks for __call__()
    arguments."""

    def __new__(cls, input_array, axes: dict):
        input_array = cls.check_input_data(input_array)
        data_table = np.asarray(input_array).view(cls)
        obj = data_table
        # allow None initialization for invalid DataTable
        # invalid DataTable will always return zero.
        if input_array is None:
            obj.axes = {}
            obj.interp = None
        else:
            obj.axes = axes.copy()
            _axes = obj._get_table_args(**axes)
            obj.interp = LinearInterp(_axes, data_table)
        return obj


    def __array_finalize__(self, obj):
        # Called when the object is created, and when a view or slice is created
        if obj is None:
            return
        self.axes = getattr(obj, "axes", {})
        self.interp: Optional[LinearInterp] = None


    def __repr__(self) -> str:
        ret = super().__repr__()
        ret += f"\nAxis Info: {str(self.axes)}"
        return ret


    def __call__(self, **kwargs) -> float|np.ndarray:
        """Checks if kwargs matches table axes then calls LinearInterp.
        Arguments
        ----------
        kwargs:
            keyword args which should match table.axes dict

        Returns
        -------
        float | numpy.ndarray
        """
        args = self._get_table_args(**kwargs)
        if len(args) != len(self.axes):
            raise Exception(f"missing DataTable arguments for: {list(self.axes.keys())[len(args):]}")
        if self.interp is None:
            # default return value: 0
            return 0.0
        else:
            ret = self.interp(args)
        if len(ret.shape) < 1:
            return ret.item()  # type:ignore
        else:
            return ret


    def swap_to_label_order(self, labels : list[str]|tuple[str]) -> None:
        # NOTE: This isnt used
        DEFAULT_LABEL_ORDER = ["alpha", "phi", "mach", "alt", "iota"]
        id_swap_order = []
        for label in DEFAULT_LABEL_ORDER:
            if label in labels:
                id_swap_order += [labels.index(label)]
        self.transpose(id_swap_order)


    @staticmethod
    def check_input_data(input_array):
        """This method handles some specifics of varying
        input data formats."""
        if input_array is None:
            return input_array
        if hasattr(input_array, "table"):
            input_array = input_array.table
        # check for any nan values
        if np.isnan(input_array).any():
            raise Exception("NaN value found in input_array initializing DataTable")
        return input_array


    @staticmethod
    def pad_with(vector, pad_width, iaxis, kwargs):
        flip_axis = kwargs["flip_axis"]
        if flip_axis == iaxis:
            if pad_width[0] > 0:
                vec_to_reflect = vector[pad_width[0]:]
                vector[:] = np.concatenate([-vec_to_reflect[::-1][:-1], vec_to_reflect])
            elif pad_width[1] > 0:
                vec_to_reflect = vector[pad_width[0]:]
                vector[:] = np.concatenate([vec_to_reflect, -vec_to_reflect[::-1][:-1]])


    def mirror_axis(self, axis_name: str) -> "DataTable":
        """This method mirrors a table axis and appends it to the top
        of the table. This is mainly to deal with increment arrays not reflected
        accross zero point."""
        if axis_name not in self.axes:
            raise Exception(f"cannot find axis name {axis_name} in table axes.")

        flip_axis = list(self.axes.keys()).index(axis_name)
        axis = self.axes[axis_name]

        # padding information
        # see numpy.pad docs
        pad_axes = ()
        naxes = len(self.shape)
        for iaxis in range(naxes):
            if iaxis == flip_axis:
                npadding = len(self.axes[axis_name]) - 1
                if axis[0] == 0:
                    pad_axes += ((npadding, 0),)
                elif axis[-1] == 0:
                    pad_axes += ((0, npadding),)
            else:
                pad_axes += ((0, 0),)

        mirrored_table = np.pad(self, pad_axes, self.pad_with, flip_axis=flip_axis)
        axis_array = self.axes[axis_name]
        mirrored_axis = np.concatenate([-axis_array[::-1][:-1], axis_array])

        axes = self.axes.copy()
        axes[axis_name] = mirrored_axis
        return DataTable(mirrored_table, axes=axes)


    def isnone(self) -> bool:
        """This method return True if the DataTable is not valid
        "None". This method is neccessary because a DataTable initalized
        as None is a single-valued numpy array containing None which does
        not work with the "is" / "is not" operator."""
        if self.shape == ():
            if self.item(0) is None:
                return True
            else:
                return False
        else:
            return False


    def _get_table_args(self, **kwargs) -> tuple:
        """This method handles arguments passed to DataTables dynamically
        according to the arguments passed and the axes of the table
        being accessed."""
        args = ()
        for label in self.axes:
            arg_val = kwargs.get(label, None)
            if arg_val is not None:
                args += (arg_val,)
        return args
