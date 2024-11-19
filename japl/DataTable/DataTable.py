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
        # self.interp: Optional[LinearInterp] = None
        self.interp: Optional[LinearInterp] = getattr(obj, "interp", None)


    def __repr__(self) -> str:
        ret = super().__repr__()
        axis_info = [k + ': ' + str(v.shape) for k, v in self.axes.items()]
        ret += f"\nShape: {self.shape}"
        ret += f"\nAxis Info: {str(axis_info)}"
        return ret


    def __call__(self, *args, **kwargs) -> float|np.ndarray:
        """Checks if kwargs matches table axes then calls LinearInterp.
        Arguments
        ----------
        kwargs:
            keyword args which should match table.axes dict

        Returns
        -------
        float | numpy.ndarray
        """
        args = self._get_table_args(*args, **kwargs)
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


    def swap_to_label_order(self, set_labels : list[str]|tuple[str]):
        # NOTE: This isnt used
        # DEFAULT_LABEL_ORDER = ["alpha", "phi", "mach", "alt", "iota"]
        current_labels = list(self.axes.keys())
        id_swap_order = []
        for label in set_labels:
            if label in current_labels:
                id_swap_order += [current_labels.index(label)]
        self.axes = {key: self.axes[key] for key in set_labels if key in self.axes}
        return self.transpose(id_swap_order)


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


    def _get_table_args(self, *args, **kwargs) -> tuple:
        """This method handles arguments passed to DataTables dynamically
        according to the arguments passed and the axes of the table
        being accessed."""
        # args = ()
        for label in self.axes:
            arg_val = kwargs.get(label, None)
            if arg_val is not None:
                args += (arg_val,)
        return args[:len(self.axes)]


    @staticmethod
    def _op_align_axes(table1: "DataTable",
                       table2: "DataTable") -> tuple[np.ndarray, np.ndarray, dict]:
        """operations between tables will be applied across dimensions of the same axis
        definition.

            a table with dimensions: ['alpha', 'mach'] + ['alpha', 'mach', 'alt']
        will return: ['alpha', 'mach', 'alt'].
        """
        s1 = table1.shape
        s2 = table2.shape
        mapping = {}
        used_axes_s2 = set()
        unmatched_s1 = []

        s1_sizes = list(s1)
        s2_sizes = list(s2)

        s1_labels = list(table1.axes.keys())
        s2_labels = list(table2.axes.keys())

        # map matching sizes between tables
        for idx_s1, label_s1 in enumerate(s1_labels):
            found = False
            for idx_s2, label_s2 in enumerate(s2_labels):
                if label_s1 == label_s2 and idx_s2 not in used_axes_s2:
                    mapping[label_s1] = label_s2
                    used_axes_s2.add(idx_s2)
                    found = True
                    break
            if not found:
                unmatched_s1.append((idx_s1, label_s1))

        # create new shape for table1
        new_shape1 = []
        new_shape2 = list(s2)

        new_s1_labels = s1_labels.copy()
        new_s2_labels = s2_labels.copy()
        for idx, label in unmatched_s1:
            new_s2_labels.insert(idx, label)

        # for each axis in table2, place size from table1 if mapped, else 1
        for label_s2 in s2_labels:
            # check if any axis in table1 maps to idx_s2
            mapped_s1_axes = [label_s1 for label_s1, label_s2_mapped in mapping.items() if label_s2_mapped == label_s2]
            if mapped_s1_axes:
                label_s1 = mapped_s1_axes[0]
                idx_s1 = s1_labels.index(label_s1)
                size_s1 = s1_sizes[idx_s1]
                new_shape1.append(size_s1)
            else:
                new_shape1.append(1)
                new_s1_labels.append(label_s2)

        for idx, label in unmatched_s1:
            if label in table1.axes:
                size = table1.axes[label].size
            else:
                size = table2.axes[label].size
            new_shape1.insert(idx, size)

        # reshape table1 to new shape
        new_table1 = table1.reshape(new_shape1)

        # build new shape for table2 by inserting size-1 dimensions for unmatched table1 dimensions
        for idx, label in unmatched_s1:
            if label in table1.axes:
                size = table1.axes[label].size
            else:
                size = table2.axes[label].size
            new_shape2.insert(idx, 1)

        new_table2 = table2.reshape(new_shape2)

        # some checks
        # if new_s1_labels != new_s2_labels:
        #     # try to rearange table1
        #     _combined_axes = {**table1.axes, **table2.axes}
        #     # sort dict according to new labels1
        #     _axes = {k: _combined_axes[k] for k in new_s1_labels}
        #     table1 = DataTable(new_table1, axes=_axes)
        #     raise Exception(f"Error when aligning axes {new_s1_labels} != {new_s2_labels}")

        combined_axes = {}
        for label in new_s2_labels:
            if label in table1.axes:
                combined_axes[label] = table1.axes[label]
            else:
                combined_axes[label] = table2.axes[label]

        return (np.asarray(new_table1), np.asarray(new_table2), combined_axes)


    def __add__(self, other) -> "DataTable":
        """addition operations between tables will be applied across dimensions of the same axis
        definition.
            a table with dimensions: ['alpha', 'mach'] + ['alpha', 'mach', 'alt']
        will return: ['alpha', 'mach', 'alt'].
        """
        if isinstance(other, DataTable):
            table1, table2, new_axes = self._op_align_axes(self, other)
            return DataTable(table1 + table2, axes=new_axes)
        else:
            return DataTable(np.asarray(self) + other, axes=self.axes)


    def __mul__(self, other) -> "DataTable":
        if isinstance(other, DataTable):
            table1, table2, new_axes = self._op_align_axes(self, other)
            return DataTable(table1 * table2, axes=new_axes)
        else:
            return DataTable(np.asarray(self) * other, axes=self.axes)


    def __matmul__(self, other) -> "DataTable":
        if isinstance(other, DataTable):
            table1, table2, new_axes = self._op_align_axes(self, other)
            return DataTable(table1 @ table2, axes=new_axes)
        else:
            return DataTable(np.asarray(self) @ other, axes=self.axes)
