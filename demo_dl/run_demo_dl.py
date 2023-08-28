NUM_DATA_PROVIDER = 3

####################
# Data preparation #
####################

from pathlib import Path
from substra_dl_assets.dataset import setup_dataset

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

##########################
# Create substra clients #
##########################

from substra import Client

# We create 1 more client than the number of data provider
clients_list = [Client(client_name=f"org-{i+1}", backend_type="subprocess") for i in range(NUM_DATA_PROVIDER + 1)]
clients = {client.organization_info().organization_id: client for client in clients_list}
# Store organization IDs
ORGS_ID = list(clients)
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]  # Data provider orgs are the last two organizations.

#####################
# Data registration #
#####################

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

assets_directory = Path.cwd() / "substra_dl_assets"
dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    client = clients[org_id]

    permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

    # DatasetSpec is the specification of a dataset. It makes sure every field
    # is well-defined, and that our dataset is ready to be registered.
    # The real dataset object is created in the add_dataset method.

    dataset = DatasetSpec(
        name="Chest X-ray",
        type="jpeg",
        data_opener=assets_directory / "dataset" / "opener.py",
        description=assets_directory / "dataset" / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Dataset key"

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=SUBSTRA_DATA_PATH / f"org_{i+1}" / "train",
    )
    train_datasample_keys[org_id] = client.add_data_sample(data_sample)

    # Add the testing data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=SUBSTRA_DATA_PATH / f"org_{i+1}" / "test",
    )
    test_datasample_keys[org_id] = client.add_data_sample(data_sample)
