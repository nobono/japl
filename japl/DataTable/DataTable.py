import numpy as np



class DataTable(np.ndarray):
    def __new__(cls, input_array, axis_labels: list[str]|tuple):
        input_array = cls.check_input_data(input_array)
        obj = np.asarray(input_array).view(cls)
        obj.axis_labels = axis_labels
        return obj


    def __array_finalize__(self, obj):
        # Called when the object is created, and when a view or slice is created
        if obj is None:
            return
        self.axis_labels = getattr(obj, 'axis_labels', [])


    def __repr__(self) -> str:
        ret = super().__repr__()
        ret += f"\nAxis Labels: {str(self.axis_labels)}"
        return ret


    def swap_to_label_order(self, labels : list[str]|tuple[str]) -> None:
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


    def mirror_axis(self, axis: int) -> "DataTable":
        """This method mirrors a table axis and appends it to the top
        of the table. This is mainly to deal with increment arrays not reflected
        accross zero point."""
        slice_tuple = np.array([slice(None) for _ in range(len(self.shape))], dtype=object)
        slice_tuple.put(axis, slice(None, -1))  # type:ignore
        return DataTable(np.concatenate([-self[::-1][*slice_tuple], self]), axis_labels=self.axis_labels)


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
