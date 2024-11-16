from japl.Interpolation.Interpolation import LinearInterp
from typing import Optional, Union
import numpy as np

ArgType = Union[float, list, np.ndarray]



class DataTable(np.ndarray):

    DEBUG = True  # debug flag allows extra checks for input args

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
            _axes = obj._get_table_args(table=data_table, **axes)
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


    # def __call__(self,
    #              alpha: Optional[ArgType] = None,
    #              phi: Optional[ArgType] = None,
    #              mach: Optional[ArgType] = None,
    #              alt: Optional[ArgType] = None,
    #              iota: Optional[ArgType] = None) -> float|np.ndarray:
        # TODO do this better
        # lower boundary on altitude
        # if (alt is not None) and ("alt" in self.axes):
        #     alt = np.clip(alt, self.axes["alt"].min(), self.axes["alt"].max())

        # TODO do this better
        # protection / boundary for mach
        # if (mach is not None) and ("mach" in self.axes):
        #     mach = np.clip(mach, self.axes["mach"].min(), self.axes["mach"].max())


    def __call__(self, *args, **kwargs) -> float|np.ndarray:
        # args = DataTable._get_table_args(table=self, alpha=alpha, phi=phi, mach=mach, alt=alt, iota=iota)
        args = DataTable._get_table_args(table=self, *args, **kwargs)
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


    def mirror_axis(self, axis_name: str) -> "DataTable":
        """This method mirrors a table axis and appends it to the top
        of the table. This is mainly to deal with increment arrays not reflected
        accross zero point."""
        if axis_name not in self.axes:
            raise Exception(f"cannot find axis name {axis_name} in table axes.")
        axis_id = list(self.axes.keys()).index(axis_name)
        # reflect table axis across its zero-axis
        slice_tuple = np.array([slice(None) for _ in range(len(self.shape))], dtype=object)
        slice_tuple.put(axis_id, slice(None, -1))  # type:ignore
        # reflect the axes info
        axis_name = list(self.axes.keys())[axis_id]  # type:ignore
        axis_array = self.axes[axis_name]
        mirrored_array = np.concatenate([-axis_array[::-1][:-1], axis_array])
        self.axes[axis_name] = mirrored_array
        return DataTable(np.concatenate([-self[::-1][*slice_tuple], self]), axes=self.axes)


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


    @staticmethod
    def _get_table_args(table: "DataTable", **kwargs) -> tuple:
        """This method handles arguments passed to DataTables dynamically
        according to the arguments passed and the axes of the table
        being accessed."""
        args = ()
        for label in table.axes:
            arg_val = kwargs.get(label, None)
            if arg_val is not None:
                args += (arg_val,)
        return args
