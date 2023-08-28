import pathlib
import numpy as np
import substratools as tools
from PIL import Image


class Opener(tools.Opener):
    def fake_data(self, n_samples=None):
        pass

    def get_data(self, folders):
        # get npy files
        p = pathlib.Path(folders[0])

        image_db = []
        label_db = []

        for image_path in p.iterdir():
            if image_path.name.startswith("NORMAL"):
                label_db.append(0)
            elif image_path.name.startswith("PNEUMONIA"):
                label_db.append(1)
            else:
                raise Exception(
                    f"Illegal image name found: {image_path.name}. The image name must start with NORMAL or PNEUMONIA."
                )

            image = Image.open(image_path)

            image_db.append(np.asarray(image))
            breakpoint()

        # load data
        data = {"data": image_db, "targets": label_db}

        return data
