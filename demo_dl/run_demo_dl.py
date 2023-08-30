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

parser.add_argument(
    "--n-round",
    type=int,
    default=5,
    help="Number of round of the compute plan.",
)

args = parser.parse_args()

NUM_DATA_PROVIDER = args.n_split

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

#####################
# Data registration #
#####################

from substra_dl_assets.dataset.register_dataset import register_dataset

from substra.sdk.schemas import Permissions

permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

dataset_keys, train_datasample_keys, test_datasample_keys = register_dataset(
    permissions=permissions_dataset,
    clients=clients,
    orgs_id=DATA_PROVIDER_ORGS_ID,
    data_path=SUBSTRA_DATA_PATH,
    asset_path=Path.cwd() / "substra_dl_assets" / "dataset",
    num_data_provider=NUM_DATA_PROVIDER,
    use_cache=args.remote,
)

######################
# Metrics definition #
######################

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import numpy as np


def accuracy(datasamples, predictions_path):
    y_true = datasamples["targets"]
    y_pred = np.load(predictions_path)

    return accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))


def binary_cross_entropy(datasamples, predictions_path):
    y_true = datasamples["targets"]
    y_pred = np.load(predictions_path)

    return log_loss(y_true, y_pred)


####################
# Model definition #
####################

import torch
from torch import nn

seed = 42
torch.manual_seed(seed)


class CnnModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CnnModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=32 * 112 * 112, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = output.view(-1, 32 * 112 * 112)
        output = self.fc(output)
        return output


model = CnnModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

########################################
# Specifying on how much data to train #
########################################

from substrafl.index_generator import NpIndexGenerator

# Number of model updates between each FL strategy aggregation.
NUM_UPDATES = 20

# Number of samples per update.
BATCH_SIZE = 32

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)

############################
# Torch Dataset definition #
############################


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, datasamples, is_inference: bool):
        self.x = datasamples["data"]
        self.y = datasamples["targets"]
        self.is_inference = is_inference

    def __getitem__(self, idx):
        if self.is_inference:
            x = torch.FloatTensor(self.x[idx]) / 255
            x = x.permute(2, 0, 1)
            return x

        else:
            x = torch.FloatTensor(self.x[idx]) / 255
            x = x.permute(2, 0, 1)

            y = torch.tensor(self.y[idx]).type(torch.float32)

            return x, y

    def __len__(self):
        return len(self.x)


#############################
# SubstraFL algo definition #
#############################

from substrafl.algorithms.pytorch import TorchFedAvgAlgo


class TorchCNN(TorchFedAvgAlgo):
    def __init__(self):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TorchDataset,
            seed=seed,
        )


#################################
# Federated Learning strategies #
#################################

from substrafl.strategies import FedAvg

strategy = FedAvg(algo=TorchCNN())

#####################################
# Where to train where to aggregate #
#####################################

from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode


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
# Where and when to test #
##########################

from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy

# Create the Test Data Nodes (or testing tasks) and save them in a list
test_data_nodes = [
    TestDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        test_data_sample_keys=[test_datasample_keys[org_id]],
        metric_functions={"Accuracy": accuracy, "Binary Cross Entropy": binary_cross_entropy},
    )
    for org_id in DATA_PROVIDER_ORGS_ID
]


# Test at the end of every round
my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, eval_frequency=1)

##########################
# Where and when to test #
##########################

from substrafl.experiment import execute_experiment
from substrafl.dependency import Dependency

dependencies = Dependency(pypi_dependencies=["torch==2.0.1", "scikit-learn==1.3.0", "Pillow==10.0.0"])

# A round is defined by a local training step followed by an aggregation operation
NUM_ROUNDS = args.n_round

compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=dependencies,
    clean_models=False,
    name=f"EUCAIM demo DL - {NUM_DATA_PROVIDER} split",
)


#######################
# Explore the results #
#######################

import pandas as pd
import matplotlib.pyplot as plt

# The results will be available once the compute plan is completed
client_to_inspect = clients[ALGO_ORG_ID]
client_to_inspect.wait_compute_plan(compute_plan.key)

performances_df = pd.DataFrame(client_to_inspect.get_performances(compute_plan.key).dict())
print("\nPerformance Table: \n")
print(performances_df[["worker", "round_idx", "identifier", "performance"]])


fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Test dataset results")

axs[0].set_title("Accuracy")
axs[1].set_title("Binary Cross Entropy")

for ax in axs.flat:
    ax.set(xlabel="Rounds", ylabel="Score")


for org_id in DATA_PROVIDER_ORGS_ID:
    org_df = performances_df[performances_df["worker"] == org_id]
    acc_df = org_df[org_df["identifier"] == "Accuracy"]
    axs[0].plot(acc_df["round_idx"], acc_df["performance"], label=org_id)

    auc_df = org_df[org_df["identifier"] == "Binary Cross Entropy"]
    axs[1].plot(auc_df["round_idx"], auc_df["performance"], label=org_id)

plt.legend(loc="lower right")
plt.show()
