from pathlib import Path
from substra_assets.dataset import setup_dataset

####################
# Data preparation #
####################

NUM_DATA_PROVIDER = 2
_data_path = Path.cwd() / "demo_ml_data" / "data"

if NUM_DATA_PROVIDER == 2:
    DATA_PATH = _data_path / "two_datasites_scenario"
elif NUM_DATA_PROVIDER == 3:
    DATA_PATH = _data_path / "three_dataseties_scenario"
else:
    raise Exception("Illegal number of data provider. Must be 2 or 3.")

setup_dataset.extract_data(data_path=DATA_PATH)

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

assets_directory = Path.cwd() / "substra_assets"

permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

dataset = DatasetSpec(
    name="Iris",
    type="npy",
    data_opener=assets_directory / "dataset" / "opener.py",
    description=assets_directory / "dataset" / "description.md",
    permissions=permissions_dataset,
    logs_permission=permissions_dataset,
)

dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    client = clients[org_id]

    # Add the dataset to the client to provide access to the opener in each organization.
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Missing data manager key"

    client = clients[org_id]

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=DATA_PATH / f"z{NUM_DATA_PROVIDER}_{i+1}",
    )
    train_datasample_keys[org_id] = client.add_data_sample(
        data_sample,
        local=True,
    )

####################
# Model definition #
####################

from sklearn import linear_model

SEED = 42

cls = linear_model.LogisticRegression(random_state=SEED, warm_start=True, max_iter=3)

###################################
# SubstraFL FL objects definition #
###################################

from substra_assets.sklearn_algo import SklearnLogisticRegression
from substrafl.strategies import FedAvg

strategy = FedAvg(algo=SklearnLogisticRegression(model=cls, seed=SEED))


from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode

#####################################
# Where to train where to aggregate #
#####################################

aggregation_node = AggregationNode(ALGO_ORG_ID)

# Create the Train Data Nodes (or training tasks) and save them in a list
train_data_nodes = [
    TrainDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    for org_id in DATA_PROVIDER_ORGS_ID
]
