import re
from pathlib import Path
from scipy.io import loadmat
import numpy as np
from typing import Any, Callable



class MatStruct:

    """This class will recreate the structure of a matlab struct object from the
    provided object output from scipy.io.loadmat().
    """

    __attribute_ignores = ("find", "findall", "is_struct", "safe_unpack",
                           "get_attributes")

    def __init__(self, data: np.ndarray) -> None:
        # checks
        names = data.dtype.names
        if data.shape == (1, 1):
            vals = data.item()
            assert len(names) == len(vals)
        elif data.shape[0] > 1 or data.shape[1] > 1:
            # Extract each field as a float array and concatenate them into a single matrix
            vals = np.row_stack([np.array([float(x[0]) for x in data[name]]) for name in names])
            assert len(names) == vals.shape[0]
        else:
            raise Exception("unhandled case.")

        for name, val in zip(names, vals):
            if self.is_struct(val):
                val = MatStruct(val)     # recursively process struct within this struct
            else:
                val = self.safe_unpack(val)
            self.__setattr__(name, val)


    def get_attributes(self) -> list[str]:
        """returns a string of the attributes parsed from the file."""
        file_attrs = [i for i in dir(self) if "__" not in i]
        return [i for i in file_attrs if i not in self.__attribute_ignores]


    @staticmethod
    def safe_unpack(data: np.ndarray) -> np.ndarray:
        """This method will attempt to dispense with unnecessary array dimensions while
        unpacking matlab file data."""
        try:
            data = data.squeeze()
        except Exception as e:
            Warning(e)
        try:
            data = data.item()
        except Exception as e:
            Warning(e)
        return data


    @staticmethod
    def is_struct(data) -> bool:
        """This method detects whether an object returned by scipy.io.loadmat
        is a matlab struct object. Matlab struct objects are loaded as np.ndarrays
        with multiple named dtypes."""
        if not isinstance(data, np.ndarray):
            return False
        return len(data.dtype) > 0


    def __repr__(self) -> str:
        attrs = [i for i in dir(self) if "__" not in i and not isinstance(getattr(self, i), Callable)]
        return "\n".join(attrs)


    def find(self, keys: str|list[str], case_sensitive: bool = False, default: Any = None,
             expected_types: list[type]|tuple[type, ...] = []):
        """Searches MatFile for possible keys. This is useful
        when different files have slightly different namespaces
        for contained items.

        From the list of provided keys, the first found instance is
        returned.
        -------------------------------------------------------------------

        Parameters:
            keys: list of keys to find

            case_sensitive: require key case case sensitivity. if False,
                            matfile will be searched for both .lower() and
                            .upper() case of keys.

            default: return if no matching attributes can be found in the
                     MatFile

            expected_types: (optional) specific data types to find in
                            addition to the provided search key / pattern.
                            This is necessary when the file structure may
                            return an item of the correct key; but is,
                            itself, another struct. when providing expected
                            types, this method will attempt to further
                            unpack any arbitrary file structure to return
                            what you are looking for.

                            *If the file structure to extract contains
                            multiple values of the same type, the first
                            found instance will be returned.*

        -------------------------------------------------------------------
        """
        if isinstance(keys, str):
            keys = [keys]

        file_attrs = self.get_attributes()
        for attr_key in file_attrs:
            attr = getattr(self, attr_key)
            if isinstance(attr, MatStruct):  # recursive search
                ret = attr.find(keys, case_sensitive=case_sensitive, default=None)
                if ret is not None:
                    return ret

            for key in keys:
                if case_sensitive:
                    if key in file_attrs:
                        ret = getattr(self, key)

                        # if here, the correct key was found but
                        # must check if it is of the expected type
                        if expected_types and isinstance(ret, MatStruct):
                            struct_attrs = ret.get_attributes()
                            for s_attr_key in struct_attrs:
                                s_attr = getattr(ret, s_attr_key)
                                if isinstance(s_attr, tuple(expected_types)):
                                    return s_attr
                        else:
                            return ret

                else:
                    file_attrs_lower = [i.lower() for i in self.get_attributes()]
                    if key.lower() in file_attrs_lower:
                        idx = file_attrs_lower.index(key.lower())
                        ret = getattr(self, file_attrs[idx])

                        # if here, the correct key was found but
                        # must check if it is of the expected type
                        if expected_types and isinstance(ret, MatStruct):
                            struct_attrs = ret.get_attributes()
                            for s_attr_key in struct_attrs:
                                s_attr = getattr(ret, s_attr_key)
                                if isinstance(s_attr, tuple(expected_types)):
                                    return s_attr
                        else:
                            return ret

        return default


class MatFile:

    """This class loads a matlab \".mat\" file given a user-defined path
    and unpacks the output of scipy.io.loadmat() into a user-friendly data structure.
    """

    __attribute_ignores = ("find", "findall", "is_struct", "safe_unpack",
                           "get_attributes")

    def __init__(self, path: str|Path) -> None:
        self._raw_data = loadmat(path)
        if isinstance(self._raw_data, dict):
            for k, v in self._raw_data.items():
                if "__" not in k:
                    if self.is_struct(v):
                        self.__setattr__(k, MatStruct(v))  # MatStruct is recursive
                    else:
                        self.__setattr__(k, self.safe_unpack(v))
        elif isinstance(self._raw_data, np.ndarray):
            self.__setattr__("data", MatStruct(self._raw_data))


    def __contains__(self, attr_name: str) -> bool:
        return hasattr(self, attr_name)


    def get(self, key: str, default=None) -> Any:
        if hasattr(self, key):
            return self.__getattribute__(key)
        else:
            return default


    def get_attributes(self) -> list[str]:
        """returns a string of the attributes parsed from the file."""
        file_attrs = [i for i in dir(self) if "__" not in i]
        return [i for i in file_attrs if i not in self.__attribute_ignores]


    @staticmethod
    def safe_unpack(data: np.ndarray) -> np.ndarray:
        """This method will attempt to dispense with unnecessary array dimensions while
        unpacking matlab file data."""
        return MatStruct.safe_unpack(data)


    @staticmethod
    def is_struct(data: np.ndarray) -> bool:
        """This method detects whether an object returned by scipy.io.loadmat
        is a matlab struct object. Matlab struct objects are loaded as np.ndarrays
        with multiple named dtypes."""
        return MatStruct.is_struct(data)


    def __repr__(self) -> str:
        attrs = [i for i in dir(self) if "__" not in i and not isinstance(getattr(self, i), Callable)]
        return "\n".join(attrs)


    def findall(self, pattern: str, case_sensitive: bool = False) -> dict:
        """Searches MatFile for possible keys. This is useful
        when different files have slightly different namespaces
        for contained items.

        From the list of provided keys, the first found instance is
        returned.
        -------------------------------------------------------------------

        Parameters:
            pattern: regular expression (regex) pattern to filter for

            case_sensitive: require key case case sensitivity

        Returns:
            list of found attributes in MatFile

        -------------------------------------------------------------------
        """
        found_attrs = {}
        file_attrs = [i for i in dir(self) if "__" not in i]
        for attr in file_attrs:
            if case_sensitive:
                _pattern = pattern.replace("*", ".*") + r'\b'  # apply word boundaries
                if (match := re.match(_pattern, attr)) is not None:
                    found_attrs[match.string] = getattr(self, match.string)
            else:
                _pattern = pattern.replace("*", ".*") + r'\b'  # apply word boundaries
                if (match := re.match(_pattern, attr, re.IGNORECASE)) is not None:
                    found_attrs[match.string] = getattr(self, match.string)
        return found_attrs


    def find(self, keys: str|list[str], case_sensitive: bool = False, default: Any = None,
             expected_types: list[type]|tuple[type, ...] = []):
        """Searches MatFile for possible keys. This is useful
        when different files have slightly different namespaces
        for contained items.

        From the list of provided keys, the first found instance is
        returned.
        -------------------------------------------------------------------

        Parameters:
            keys: list of keys to find

            case_sensitive: require key case case sensitivity. if False,
                            matfile will be searched for both .lower() and
                            .upper() case of keys.

            default: return if no matching attributes can be found in the
                     MatFile

            expected_types: (optional) specific data types to find in
                            addition to the provided search key / pattern.
                            This is necessary when the file structure may
                            return an item of the correct key; but is,
                            itself, another struct. when providing expected
                            types, this method will attempt to further
                            unpack any arbitrary file structure to return
                            what you are looking for.

                            *If the file structure to extract contains
                            multiple values of the same type, the first
                            found instance will be returned.*

        -------------------------------------------------------------------
        """
        if isinstance(keys, str):
            keys = [keys]

        file_attrs = self.get_attributes()
        for attr_key in file_attrs:
            attr = getattr(self, attr_key)
            if isinstance(attr, MatStruct):  # recursive search
                ret = attr.find(keys, case_sensitive=case_sensitive, default=None)
                if ret is not None:
                    return ret

            for key in keys:
                if case_sensitive:
                    if key in file_attrs:
                        ret = getattr(self, key)

                        # if here, the correct key was found but
                        # must check if it is of the expected type
                        if expected_types and isinstance(ret, MatStruct):
                            struct_attrs = ret.get_attributes()
                            for s_attr_key in struct_attrs:
                                s_attr = getattr(ret, s_attr_key)
                                if isinstance(s_attr, tuple(expected_types)):
                                    return s_attr
                        else:
                            return ret

                else:
                    file_attrs_lower = [i.lower() for i in self.get_attributes()]
                    if key.lower() in file_attrs_lower:
                        idx = file_attrs_lower.index(key.lower())
                        ret = getattr(self, file_attrs[idx])

                        # if here, the correct key was found but
                        # must check if it is of the expected type
                        if expected_types and isinstance(ret, MatStruct):
                            struct_attrs = ret.get_attributes()
                            for s_attr_key in struct_attrs:
                                s_attr = getattr(ret, s_attr_key)
                                if isinstance(s_attr, tuple(expected_types)):
                                    return s_attr
                        else:
                            return ret

        return default
