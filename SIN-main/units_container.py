# units_container.py

class UnitsContainer:
    def __init__(self, data):
        """
        data: a dictionary of arrays with keys like 'features', 'treatments', 
              'outcomes', 'edges', and 'edge_types'
        """
        self.data = data

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    def __len__(self):
        return len(next(iter(self.data.values())))

    def get_covariates(self):
        # Return the 'features' array if present; otherwise, try 'covariates'
        if "features" in self.data:
            return self.data["features"]
        elif "covariates" in self.data:
            return self.data["covariates"]
        else:
            raise ValueError("UnitsContainer does not contain 'features' or 'covariates'.")
