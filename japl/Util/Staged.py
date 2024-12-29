class Staged:

    """Base class for Staged subclasses. inheriting
    from this class makes the child class a container for
    other child-classes of the same type.

    For example, `Staged` is used as a parent class for
    AeroTable. subsequent AeroTables can be added to the top
    container class as stages.

    [Parent] AeroTable
        - [Child] AeroTable stage 1
        - [Child] AeroTable stage 2
        - ...etc

    the current child stage can be accessed through the
    parent container. The child stage can be set with
    set_stage(), get_stage().

    NOTE:
        for a Staged subclass, in order to correctly
        access the current stage, methods must call
        self.get_stage():

        (example: `self.get_stage().my_object`)
    """

    stages: list
    stage_id: int
    is_stage: bool

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.stages = []
        obj.stage_id = 0
        obj.is_stage = True
        return obj


    def add_stage(self, child) -> None:
        """Adds a child AeroTable object as an ordered child
        of this object."""
        self.is_stage = False
        child.is_stage = True
        self.stages += [child]

    def set_stage(self, stage: int) -> None:
        """Set the current stage index for the aerotable. This is
        so that `get_stage()` will return the corresponding aerotable."""
        if int(stage) >= len(self.stages):
            raise Exception(f"cannot access stage {int(stage)} "
                            f"for container of size {len(self.stages)}")
        self.stage_id = int(stage)

    def get_stage(self):
        """Returns the current aerotable corresponding to the stage_id."""
        if self.is_stage:
            return self
        else:
            return self.stages[self.stage_id]
