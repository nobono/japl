# import astropy.units as u
from astropy.units.quantity import Quantity



def assert_physical_type(obj: Quantity, physical_type: str):
    assert isinstance(obj, Quantity)
    if physical_type not in obj.unit.physical_type:  # type:ignore
        raise ValueError(f"obj required to be of type {physical_type}")
