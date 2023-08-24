import pathlib
import numpy as np
import substratools as tools
import pandas as pd


class Opener(tools.Opener):
    def fake_data(self, n_samples=None):
        pass

    def get_data(self, folders):
        # get npy files
        p = pathlib.Path(folders[0])
        df = pd.read_csv(next(p.iterdir()))

        df = df.dropna(subset=["PM25", "age", "sex", "cbmi", "blood_pre"])

        df["sex"] = df["sex"].replace("female", 0)
        df["sex"] = df["sex"].replace("male", 1)

        # load data
        data = {"data": df[["PM25", "age", "sex", "cbmi"]].to_numpy(), "targets": df["blood_pre"].to_numpy()}

        return data
