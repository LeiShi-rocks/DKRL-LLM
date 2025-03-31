# Minimal TestUnitWrapper that only provides get_treatment_ids()
class TestUnitWrapper:
    def __init__(self, unit):
        # unit is a dictionary representing one test unit.
        self.unit = unit
    def get_treatment_ids(self):
        return [self.unit["treatment_id"]]
    # Optionally, you can add __getitem__ if you need to index into the unit:
    def __getitem__(self, key):
        return self.unit[key]
    # Do not override __getattr__ to avoid recursion issues.