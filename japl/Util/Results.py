import os
import dill
from japl import SimObject
import numpy as np



class Results:
    def __init__(self, time, simobj: SimObject) -> None:
        self.set('t', time)
        for name in simobj.model.state_register.keys():
            id = simobj.model.get_state_id(name)
            self.set(name, simobj.Y[:, id])
        for name in simobj.model.input_register.keys():
            id = simobj.model.get_input_id(name)
            self.set(name, simobj.U[:, id])

    def set(self, key: str, val):
        setattr(self, key, val)


    def save(self, path: str):
        print("saving results to path:", path)
        # remove existing file if exists
        if os.path.isfile(path):
            os.remove(path)
        with open(path, 'ab') as f:
            dill.dump(self, f)


    @classmethod
    def load(cls, path: str) -> "Results":
        with open(path, 'rb') as f:
            obj = dill.load(f)
        return obj


    def __repr__(self) -> str:
        members = []
        for i in dir(self):
            attr = getattr(self, i)
            if isinstance(attr, np.ndarray):
                try:
                    members += [str(i)]
                except Exception as e:  # noqa
                    members += [str(i)]
        return "\n".join(members)
