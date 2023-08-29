from pathlib import Path
import pandas as pd
import shutil
from . import logger


def extract_data(data_ids: Path, image_path: Path, destination_path: Path, n_org: int):
    try:
        destination_path.mkdir(exist_ok=False)
    except FileExistsError:
        logger.info(f"{destination_path.name} folder already exists in the given path. Skipping data extraction.")
        return

    for i in range(n_org):
        (destination_path / f"org_{i+1}").mkdir(exist_ok=True)

        # Train
        train_path = destination_path / f"org_{i+1}" / "train"
        train_path.mkdir(exist_ok=True)

        train_normal_file = f"train.nrm.{n_org}_{i+1}.csv"
        train_pneumonia_file = f"train.pnm.{n_org}_{i+1}.csv"

        train_normal_df = pd.read_csv(data_ids / train_normal_file)
        copy_image_from_df(df=train_normal_df, dest_folder=train_path, image_path=image_path / "train", label="NORMAL")

        train_pneumonia_df = pd.read_csv(data_ids / train_pneumonia_file)
        copy_image_from_df(
            df=train_pneumonia_df, dest_folder=train_path, image_path=image_path / "train", label="PNEUMONIA"
        )

        # Test
        test_path = destination_path / f"org_{i+1}" / "test"
        test_path.mkdir(exist_ok=True)

        test_normal_file = f"test.nrm.{n_org}_{i+1}.csv"
        test_pneumonia_file = f"test.pnm.{n_org}_{i+1}.csv"

        test_normal_df = pd.read_csv(data_ids / test_normal_file)
        copy_image_from_df(df=test_normal_df, dest_folder=test_path, image_path=image_path / "test", label="NORMAL")

        test_pneumonia_df = pd.read_csv(data_ids / test_pneumonia_file)
        copy_image_from_df(
            df=test_pneumonia_df, dest_folder=test_path, image_path=image_path / "test", label="PNEUMONIA"
        )


def copy_image_from_df(df: pd.DataFrame, dest_folder: Path, image_path: Path, label: str):
    for _, row in df.iterrows():
        image_name = row["image_name"]
        shutil.copyfile(image_path / label / image_name, dest_folder / (label + "-" + image_name))
