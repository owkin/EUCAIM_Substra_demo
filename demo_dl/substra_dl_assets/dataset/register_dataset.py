from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.exceptions import NotFound
from substra import Client

from pathlib import Path
from typing import List
import json

from . import logger


def register_dataset(
    *,
    permissions: Permissions,
    clients: List[Client],
    orgs_id: List[str],
    data_path: Path,
    asset_path: Path,
    num_data_provider: int,
    use_cache: bool = False,
):
    json_cache = asset_path / "cache_keys" / f"cache_keys_{num_data_provider}_split.json"

    if use_cache:
        if not json_cache.exists():
            logger.info("No cache found for dataset and datasample keys. The registration will proceed without cache.")
        else:
            with json_cache.open(mode="r") as f:
                dict_keys = json.load(f)

            dataset_keys = dict_keys["dataset_keys"]
            train_datasample_keys = dict_keys["train_datasample_keys"]
            test_datasample_keys = dict_keys["test_datasample_keys"]

            try:
                for org_id in orgs_id:
                    clients[org_id].get_dataset(dataset_keys[org_id])

                return dataset_keys, train_datasample_keys, test_datasample_keys

            except NotFound:
                logger.info("Keys in cache were not found. The registration will proceed without cache.")

    dataset_keys = {}
    train_datasample_keys = {}
    test_datasample_keys = {}

    for i, org_id in enumerate(orgs_id):
        client = clients[org_id]

        # DatasetSpec is the specification of a dataset. It makes sure every field
        # is well-defined, and that our dataset is ready to be registered.
        # The real dataset object is created in the add_dataset method.

        dataset = DatasetSpec(
            name="Chest X-ray",
            type="jpeg",
            data_opener=asset_path / "opener.py",
            description=asset_path / "description.md",
            permissions=permissions,
            logs_permission=permissions,
        )
        dataset_keys[org_id] = client.add_dataset(dataset)
        assert dataset_keys[org_id], "Dataset key"

        # Add the training data on each organization.
        data_sample = DataSampleSpec(
            data_manager_keys=[dataset_keys[org_id]],
            path=data_path / f"org_{i+1}" / "train",
        )
        train_datasample_keys[org_id] = client.add_data_sample(data_sample)

        # Add the testing data on each organization.
        data_sample = DataSampleSpec(
            data_manager_keys=[dataset_keys[org_id]],
            path=data_path / f"org_{i+1}" / "test",
        )
        test_datasample_keys[org_id] = client.add_data_sample(data_sample)

    if use_cache:
        dict_keys = {
            "dataset_keys": dataset_keys,
            "train_datasample_keys": train_datasample_keys,
            "test_datasample_keys": test_datasample_keys,
        }
        with json_cache.open(mode="w", encoding="UTF-8") as f:
            json.dump(dict_keys, f)

    return dataset_keys, train_datasample_keys, test_datasample_keys