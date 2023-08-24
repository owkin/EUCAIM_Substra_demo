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
        df = pd.read_csv(p / (p.name + ".csv"))

        # load data
        data = {"data": df[["PM25", "age", "sex", "cbmi"]], "targets": df["blood_pre"]}

        return data
