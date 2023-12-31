import argparse

parser = argparse.ArgumentParser(prog="Demo DL EUCAIM")
parser.add_argument(
    "-r",
    "--remote",
    action="store_true",
    help="Client in remote mode.",
)

parser.add_argument(
    "--n-split",
    type=int,
    default=3,
    help="Number of data provider to work with for the demo.",
)

args = parser.parse_args()

NUM_DATA_PROVIDER = args.n_split

####################
# Data preparation #
####################

from pathlib import Path
from substra_ml_assets.dataset import setup_dataset

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

if args.remote:
    config_file = Path.cwd() / "config_files" / f"config-{NUM_DATA_PROVIDER}-client.yaml"
else:
    config_file = None

clients_list = [Client(client_name=f"org-{i+1}", configuration_file=config_file) for i in range(NUM_DATA_PROVIDER)]
clients = {client.organization_info().organization_id: client for client in clients_list}

# Store organization IDs
ORGS_ID = list(clients)
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID  # All organization provides data in this demo orgs are the last two organizations.


# Store organization IDs
ORGS_ID = list(clients)
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID  # Data provider orgs are the last two organizations.


#####################
# Data registration #
#####################

from substra.sdk.schemas import Permissions

from substra_ml_assets.dataset.register_dataset import register_dataset

permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

dataset_keys, train_datasample_keys = register_dataset(
    permissions=permissions_dataset,
    clients=clients,
    orgs_id=DATA_PROVIDER_ORGS_ID,
    data_path=DATA_PATH,
    asset_path=Path.cwd() / "substra_ml_assets" / "dataset",
    num_data_provider=NUM_DATA_PROVIDER,
    use_cache=args.remote,
)

####################
# Model definition #
####################

from sklearn import linear_model

SEED = 42

cls = linear_model.LinearRegression()

###################################
# SubstraFL FL objects definition #
###################################

from substra_ml_assets.sklearn_algo import SklearnLogisticRegression
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

##########################
# Running the experiment #
##########################

from substrafl.experiment import execute_experiment
from substrafl.dependency import Dependency

# Number of times to apply the compute plan.
NUM_ROUNDS = 1

dependencies = Dependency(
    pypi_dependencies=["pandas==2.0.3", "scikit-learn==1.3.0"], local_code=[Path.cwd() / "substra_ml_assets"]
)

compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=None,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=dependencies,
    name=f"EUCAIM demo ML - {NUM_DATA_PROVIDER} split",
    clean_models=False,
)


####################
# Download results #
####################

from substrafl.model_loading import download_algo_state

# The results will be available once the compute plan is completed
client_to_download_from = clients[DATA_PROVIDER_ORGS_ID[0]]
client_to_download_from.wait_compute_plan(compute_plan.key)

algo = download_algo_state(
    client=client_to_download_from,
    compute_plan_key=compute_plan.key,
)

cls = algo.model

print("Coefs: ", cls.coef_)
print("Intercepts: ", cls.intercept_)
