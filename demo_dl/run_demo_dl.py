NUM_DATA_PROVIDER = 3

####################
# Data preparation #
####################

from pathlib import Path
from substra_assets.dataset import setup_dataset

_data_ids = Path.cwd() / "demo_dl_data" / "data_ids"

IMAGE_PATH = Path.cwd() / "demo_dl_data" / "chest_xray"
SUBSTRA_DATA_PATH = Path.cwd() / f"substra_data_{NUM_DATA_PROVIDER}_split"

if NUM_DATA_PROVIDER == 2:
    DATA_IDS = _data_ids / "two_datasites_scenario"
elif NUM_DATA_PROVIDER == 3:
    DATA_IDS = _data_ids / "three_dataseties_scenario"
else:
    raise Exception("Illegal number of data provider. Must be 2 or 3.")

setup_dataset.extract_data(
    data_ids=DATA_IDS, image_path=IMAGE_PATH, destination_path=SUBSTRA_DATA_PATH, n_org=NUM_DATA_PROVIDER
)
