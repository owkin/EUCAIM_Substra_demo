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
