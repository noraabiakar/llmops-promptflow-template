"""
This module executes experiment jobs/bulk-runs using standard flows.

Args:
--subscription_id: The Azure subscription ID.
This argument is required for identifying the Azure subscription.
--data_purpose: The data identified by its purpose.
This argument is required to specify the purpose of the data.
--flow_to_execute: The name of the flow use case.
This argument is required to specify the name of the flow for execution.
--env_name: The environment name for execution and deployment.
This argument is required to specify the environment (dev, test, prod)
for execution or deployment.
"""

import argparse
import hashlib
import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from typing import Optional

from llmops.common.experiment_cloud_config import ExperimentCloudConfig
from llmops.common.experiment import load_experiment
from llmops.common.logger import llmops_logger


logger = llmops_logger("register_data_asset")


def generate_file_hash(file_path):
    """
    Generate hash of a file.

    Returns:
        hash as string
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        file_content = file.read()
        sha256.update(file_content)

    return sha256.hexdigest()


def register_data_asset(
    base_path: str,
    exp_filename: Optional[str],
    subscription_id: Optional[str],
    env_name: Optional[str],
):
    environment_name = env_name
    config = ExperimentCloudConfig(subscription_id=subscription_id, env_name=env_name)
    experiment = load_experiment(
        filename=exp_filename, base_path=base_path, env=config.environment_name
    )
    ml_client = MLClient(
        DefaultAzureCredential(),
        config.subscription_id,
        config.resource_group_name,
        config.workspace_name,
    )

    # TODO, check source of dataset when creating Dataset?

    data_config = json.load(config_file)

    for elem in data_config["datasets"]:
        if "DATA_PURPOSE" in elem and "ENV_NAME" in elem:
            if (
                data_purpose == elem["DATA_PURPOSE"]
                and environment_name == elem["ENV_NAME"]
            ):
                data_path = f"{args.flow_to_execute}/{elem['DATA_PATH']}"
                dataset_desc = elem["DATASET_DESC"]
                dataset_name = elem["DATASET_NAME"]

                data_hash = generate_file_hash(data_path)
                print("Hash of the folder:", data_hash)

                aml_dataset = Data(
                    path=data_path,
                    type=AssetTypes.URI_FILE,
                    description=dataset_desc,
                    name=dataset_name,
                    tags={"data_hash": data_hash},
                )

                try:
                    data_info = ml_client.data.get(name=dataset_name, label="latest")

                    m_hash = dict(data_info.tags).get("data_hash")
                    if m_hash is not None:
                        if m_hash != data_hash:
                            ml_client.data.create_or_update(aml_dataset)
                    else:
                        ml_client.data.create_or_update(aml_dataset)
                except Exception:
                    ml_client.data.create_or_update(aml_dataset)

                aml_dataset_unlabeled = ml_client.data.get(
                    name=dataset_name, label="latest"
                )

                logger.info(aml_dataset_unlabeled.version)
                logger.info(aml_dataset_unlabeled.id)


def main():
    parser = argparse.ArgumentParser("register data assets")
    parser.add_argument(
        "--file",
        type=str,
        help="The experiment file. Default is 'experiment.yaml'",
        required=False,
        default="experiment.yaml",
    )
    parser.add_argument(
        "--subscription_id",
        type=str,
        help="Subscription ID, overrides the SUBSCRIPTION_ID environment variable",
        default=None,
    )
    parser.add_argument(
        "--base_path",
        type=str,
        help="Base path of the use case",
        required=True,
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help="environment name(dev, test, prod) for execution and deployment, overrides the ENV_NAME environment variable",
        default=None,
    )

    args = parser.parse_args()

    register_data_asset(args.file, args.base_path, args.subscription_id, args.env_name)


if __name__ == "__main__":
    # Load variables from .env file into the environment
    load_dotenv(override=True)

    main()
