import os
from typing import Any, List, Optional, Tuple

import yaml

_FLOW_DAG_FILENAME = "flow.dag.yaml"
_DEFAULT_FLOWS_DIR = "flows"
_DEFAULT_DATA_DIR = "data"


class Dataset:
    """
    Defines an Azure ML Dataset or a local dataset.

    :param name: Unique name given to the dataset.
    :type name: str
    :param source: Azure ML name and version (example: azureml:<dataset_name_in_azureml>:<version>) or path to file.
    :type source: str
    :param description: Description of the Azure ML dataset (only required if source is a path to file).
    :type source: str
    """

    def __init__(
        self,
        name: str,
        source: str,
        description: Optional[str],
        reference: Optional[str],
    ):
        self.name = name
        self.source = source
        self.description = description
        self.reference = reference
        self.remote_source = self.source.startswith("azureml:")

    def with_mappings(self, mappings: dict[str, str]) -> "MappedDataset":
        return MappedDataset(mappings, self)

    def is_eval(self):
        return self.reference is not None

    def get_remote_name(self):
        if self.remote_source:
            return self.source
        return f"azureml:{self.name}:latest"

    # def _resolve_flow_dir(base_path: Optional[str], flow: str) -> str:
    #     # Check if the flow path can be deducted from base path
    #
    #     if os.path.isfile(os.path.join(safe_base_path, flow, _FLOW_DAG_FILENAME)):
    #         return os.path.abspath(os.path.join(safe_base_path, flow))

    #     return os.path.join(safe_base_path, _DEFAULT_FLOWS_DIR, flow)

    def get_local_source(self, base_path: Optional[str] = None):
        if self.remote_source:
            return None
        safe_base_path = base_path or ""
        data_path = os.path.join(safe_base_path, self.source)
        if os.path.exists(data_path):
            return os.path.abspath(data_path)
        return os.path.join(safe_base_path, _DEFAULT_DATA_DIR, self.source)

    # Define equality operation
    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return NotImplemented

        return (
            self.name == other.name
            and self.source == other.source
            and self.description == other.description
            and self.reference == other.reference
        )


class MappedDataset:
    """
    Defines an Azure ML Mapped dataset referencing a Dataset object and a dictionary mapping prompt flow inputs to
    dataset columns or previous run outputs.

    :param dataset: Dataset to be mapped.
    :type dataset: Dataset
    :param mappings: Dictionary mapping prompt flow inputs to dataset columns or previous run outputs
    (example: {"col_0": "${data:col_0}", "col_1": "${run.outputs.col_1}"}).
    :type mappings: dict[str, str]
    """

    def __init__(self, mappings: dict[str, str], dataset: Dataset):
        self.dataset = dataset
        self.mappings = mappings

    # Define equality operation
    def __eq__(self, other):
        if not isinstance(other, MappedDataset):
            return NotImplemented

        return self.dataset == other.dataset and self.mappings == other.mappings


class Evaluator:
    """
    Defines a prompt flow evaluator flow.

    :param name: Name of the evaluation flow.
    :type name: str
    :param path: Path to the evaluation flow. Default value is "flows".
    :type path: str
    :param datasets: List of mapped datasets used for the evaluator flow.
    :type datasets: List[MappedDataset]
    """

    def __init__(
        self, name: str, datasets: list[MappedDataset], path: Optional[str] = None
    ):
        self.name = name
        self.path = path or os.path.join("flows", name)
        self.datasets = datasets

    def find_dataset_with_reference(self, dataset_name: str) -> Optional[MappedDataset]:
        for dataset in self.datasets:
            if dataset.dataset.reference == dataset_name:
                return dataset
        return None

    # Define equality operation
    def __eq__(self, other):
        if not isinstance(other, Evaluator):
            return NotImplemented

        return (
            self.name == other.name
            and self.datasets == other.datasets
            and self.path == other.path
        )


class FlowDetail:
    """
    Contains the details of a flow (location, nodes, variants)

    :param flow_path: Path to a prompt flow flow.
    :type flow_path: str
    :param all_variants: List of dictionaries, one per llm node, from llm node variant name to llm node name.
    :type all_variants: list[dict[str, Any]]
    :param all_llm_nodes: Set of llm node names.
    :type all_llm_nodes: set
    :param default_variants: Dictionary from llm node name to default node variant.
    :type default_variants: dict[str, str]
    """

    def __init__(
        self,
        flow_path: str,
        all_variants: list[dict[str, Any]],
        all_llm_nodes: set,
        default_variants: dict,
    ):
        self.flow_path = flow_path
        self.all_variants = all_variants
        self.all_llm_nodes = all_llm_nodes
        self.default_variants = default_variants


class Experiment:
    """
    Contains the details of an experiment (name, flow, path, datasets, evaluators)

    :param base_path: Path to the directory containing the yaml file defining the experiment. Default value is current
    working directory.
    :type base_path: Optional[str]
    :param name: Name of the experiment.
    :type name: str
    :param flow: Name of the flow. Default value is :param name:.
    :type flow: Optional[str]
    :param datasets: List of datasets.
    :type datasets: List[Dataset]
    :param evaluators: List of evaluators.
    :type evaluators: List[Evaluator]
    :param runtime: Name of the Prompt flow runtime to use in Azure ML. If not provided, use automatic runtime.
    :type runtime: str
    """

    def __init__(
        self,
        base_path: Optional[str],
        name: str,
        flow: Optional[str],
        datasets: list[MappedDataset],
        evaluators: list[Evaluator],
        runtime: Optional[str],
    ):
        self.base_path = base_path
        self.name = name
        self.flow = flow or name
        self.datasets = datasets
        self.evaluators = evaluators
        self.runtime = runtime
        self._flow_detail: Optional[FlowDetail] = None

    def get_flow_detail(self) -> FlowDetail:
        self._flow_detail = self._flow_detail or self._load_flow_detail()
        return self._flow_detail

    def _load_flow_detail(self) -> FlowDetail:
        """
        Load flow details from the yaml files describing the experiment and the flow.
        """
        # Load flow data
        flow_path = _resolve_flow_dir(self.base_path, self.flow)
        flow_file_path = os.path.join(flow_path, _FLOW_DAG_FILENAME)
        if not os.path.exists(flow_file_path):
            raise ValueError(
                f"Could not open prompt flow file in path {flow_file_path}"
            )

        yaml_data: dict
        with open(flow_file_path, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        # Find prompt variants and nodes
        all_variants: list[dict[str, Any]] = []
        all_llm_nodes = set()
        default_variants = {}
        for node_name, node_data in yaml_data.get("node_variants", {}).items():
            node_variant_mapping = {}
            variants = node_data.get("variants", {})
            default_variant: str = node_data["default_variant_id"]
            default_variants[node_name] = default_variant
            for variant_name, _ in variants.items():
                node_variant_mapping[variant_name] = node_name
                all_llm_nodes.add(node_name)
            all_variants.append(node_variant_mapping)

        for nodes in yaml_data["nodes"]:
            node_variant_mapping = {}
            if nodes.get("type", {}) == "llm":
                all_llm_nodes.add(nodes["name"])

        return FlowDetail(flow_path, all_variants, all_llm_nodes, default_variants)


# Helper function for raising errors
def _raise_error_if_missing_keys(keys: List[str], config: dict, message: str):
    for key in keys:
        if key not in config:
            raise ValueError(f"{message}: {key}")


def _raise_error_if_existing_keys(keys: List[str], config: dict, message: str):
    for key in keys:
        if key in config:
            raise ValueError(f"{message}: {key}")


def _create_datasets_and_default_mappings(
    raw: list[dict],
) -> Tuple[dict[str, Dataset], list[MappedDataset]]:
    """
    Create datasets and mapped datasets from list of dictionaries.

    :param raw: List of dictionaries containing the description of the experiment datasets.
    :type raw: list[dict]
    :return: Tuple of dictionary from dataset name to dataset and list of mapped datasets
    :rtype: Tuple[dict[str, Dataset], list[MappedDataset]]
    """
    datasets: dict[str, Dataset] = {}
    mappings: list[MappedDataset] = []
    for ds in raw:
        # Raise error if expected dataset configuration missing
        _raise_error_if_missing_keys(
            ["name", "source", "mappings"],
            ds,
            message=f"Dataset '{ds.get('name')}' config missing parameter",
        )
        # Raise error if unexpected dataset configuration found
        _raise_error_if_existing_keys(
            ["reference"],
            ds,
            message=f"Unexpected parameter found in dataset '{ds.get('name')}' description",
        )
        dataset = Dataset(
            ds["name"], ds["source"], ds.get("description"), ds.get("reference")
        )
        datasets[dataset.name] = dataset
        mappings.append(dataset.with_mappings(ds["mappings"] or {}))

    return datasets, mappings


def _create_eval_datasets_and_default_mappings(
    raw: list[dict], existing_datasets: dict[str, Dataset]
) -> list[MappedDataset]:
    """
    Create mapped datasets from list of evaluation datasets.

    :param raw: List of dictionaries containing the description of the evaluation datasets.
    :type raw: list[dict]
    :param datasets: Dictionary from dataset name to Dataset object.
    :type datasets: dict[str, Dataset]
    :return: List of mapped datasets
    :rtype: list[MappedDataset]
    """
    mappings: list[MappedDataset] = []

    # The datasets here are "evaluation" datasets, used to run the evaluation flows.
    # The datasets are either a direct reference to an existing dataset (in practice this means
    # that the the evaluation flow and standard flow are running on the same dataset);
    # or they are new - in which case the new dataset they must still "reference" an existing
    # dataset (in practice  this means that the evaluation flow and standard flow are running on
    # different but related datasets.)

    for ds in raw:
        # Check that the common keys are available
        _raise_error_if_missing_keys(
            ["name", "mappings"],
            ds,
            message=f"Dataset '{ds.get('name')}' config missing parameter",
        )
        ds_name = ds["name"]

        # Create or get dataset
        dataset: Dataset = None
        if ds_name in existing_datasets:
            _raise_error_if_existing_keys(
                ["source", "reference"],
                ds,
                message=f"Dataset '{ds_name}' config is referencing an existing dataset so it doesn't support parameter",
            )
            dataset = existing_datasets.get(ds_name)
        else:
            _raise_error_if_missing_keys(
                ["source", "reference"],
                ds,
                message=f"Dataset '{ds_name}' config missing parameter",
            )
            dataset = Dataset(
                ds_name, ds.get("source"), ds.get("description"), ds.get("reference")
            )

            # Validate that the reference dataset exists
            if dataset.reference not in existing_datasets:
                raise ValueError(
                    f"Referenced dataset '{dataset.reference}' not defined"
                )

        # Collect mappings
        mappings.append(dataset.with_mappings(ds["mappings"] or {}))

    return mappings


def _create_evaluators(
    raw_evaluators: list[dict], datasets: dict[str, Dataset], base_path: Optional[str]
) -> list[Evaluator]:
    """
    Create evaluators from a list of dictionaries describing the raw evaluators, a dictionary of existing datasets,
    and the path to the evaluator flow.

    :param raw_evaluators: List of dictionaries containing the description of the experiment evaluators.
    :type raw_evaluators: list[dict]
    :param datasets: Dictionary from dataset name to Dataset object.
    :type datasets: dict[str, Dataset]
    :param base_path: Path to the evaluator flow directory containing the yaml description.
    Default value is current working directory.
    :type base_path: Optional[str]
    :return: List of evaluators.
    :rtype: list[Evaluator]
    """
    evaluators: list[Evaluator] = []
    for raw_evaluator in raw_evaluators:
        # Raise error if expected evaluator configuration missing
        _raise_error_if_missing_keys(
            ["name", "datasets"],
            raw_evaluator,
            message=f"Evaluator '{raw_evaluator.get('name')}' config missing parameter",
        )
        evaluator_datasets = _create_eval_datasets_and_default_mappings(
            raw_evaluator["datasets"], datasets
        )
        eval_name = raw_evaluator["name"]
        flow = raw_evaluator.get("flow") or eval_name
        flow_path = _resolve_flow_dir(base_path, flow)
        evaluator = Evaluator(
            name=eval_name, datasets=evaluator_datasets, path=flow_path
        )
        evaluators.append(evaluator)
    return evaluators


def _resolve_flow_dir(base_path: Optional[str], flow: str) -> str:
    """
    Resolve path to the yaml file describing the flow. Tries multiple resolution methods.
    - If base_path not provided, uses the current working directory
    - Looks for "flow.dag.yaml" in the base_path/flow, returns that path if found; otherwise returns
      base_path/flows/flow.

    :param base_path: Path to the flow directory. Default value is current working directory.
    :type base_path: Optional[str]
    :param flow: Name of the flow.
    :type flow: str
    :return: Path to flow.
    :rtype: str
    """
    # Check if the flow path can be deducted from base path
    safe_base_path = base_path or ""
    if os.path.isfile(os.path.join(safe_base_path, flow, _FLOW_DAG_FILENAME)):
        return os.path.abspath(os.path.join(safe_base_path, flow))

    return os.path.join(safe_base_path, _DEFAULT_FLOWS_DIR, flow)


def _load_base_experiment(exp_file_path: str, base_path: Optional[str]) -> Experiment:
    exp_config: dict
    with open(exp_file_path, "r") as yaml_file:
        exp_config = yaml.safe_load(yaml_file)

    # Read base raw datasets and create base datasets and mappings
    raw_datasets: list[dict] = exp_config.get("datasets")
    if not raw_datasets:
        raise ValueError("No datasets configured for experiment")
    datasets, mappings = _create_datasets_and_default_mappings(raw_datasets)

    # Read base raw evaluators and create base evaluators
    raw_evaluators: list[dict] = exp_config.get("evaluators")
    evaluators: list[Evaluator] = []
    if raw_evaluators is not None and len(raw_evaluators) > 0:
        evaluators = _create_evaluators(raw_evaluators, datasets, base_path)

    runtime = exp_config.get("runtime")

    # Create experiment
    return Experiment(
        base_path=base_path,
        name=exp_config["name"],
        flow=exp_config.get("flow"),
        datasets=mappings,
        evaluators=evaluators,
        runtime=runtime,
    )


def _apply_overlay(
    experiment: Experiment, overlay_file_path: str, base_path: Optional[str]
):
    overlay_config: dict
    with open(overlay_file_path, "r") as yaml_file:
        overlay_config = yaml.safe_load(yaml_file)

    experiment_dataset_map: dict[str, Dataset] = {
        ds.dataset.name: ds.dataset for ds in experiment.datasets
    }
    # Read env raw datasets and create env datasets and mappings
    if "datasets" in overlay_config:
        overlay_raw_datasets: list[dict] = overlay_config["datasets"]
        if overlay_raw_datasets:
            overlay_datasets, overlay_mappings = _create_datasets_and_default_mappings(
                overlay_raw_datasets
            )
            # Override experiment datasets
            experiment.datasets = overlay_mappings
            experiment_dataset_map = overlay_datasets
        else:
            experiment.datasets = []

    # Read env raw evaluators and create env evaluators

    if "evaluators" in overlay_config:
        overlay_raw_evaluators: list[dict] = overlay_config["evaluators"]
        if overlay_raw_evaluators:
            experiment.evaluators = _create_evaluators(
                overlay_raw_evaluators, experiment_dataset_map, base_path
            )
        else:
            experiment.evaluators = []

    if "runtime" in overlay_config:
        experiment.runtime = overlay_config["runtime"]


def load_experiment(
    filename: Optional[str] = None,
    base_path: Optional[str] = None,
    env: Optional[str] = None,
) -> Experiment:
    """
    Load an experiment from a YAML file.

    :param filename: The experiment filename. Default is "experiment.yaml"
    :type filename: Optional[str]
    :param base_path: The base path of the experiment. If not specified, the current working directory is used.
    :type base_path: Optional[str]
    :param env: The environment to use. If specified, look for an overlay experiment file named
    <experiment_name>.<env>.yaml and use it to override experiment dataset sources.
    :type env: Optional[str]
    """

    safe_base_path = base_path or ""
    experiment_file_name = filename or "experiment.yaml"

    # Validate the experiment file name
    file_parts = os.path.splitext(experiment_file_name)
    if len(file_parts) != 2:  # noqa: PLR2004
        raise ValueError(f"Invalid experiment filename '{experiment_file_name}'")
    env_experiment_file_name = f"{file_parts[0]}.{env}{file_parts[1]}"

    # Create base experiment
    exp_file_path = os.path.join(safe_base_path, experiment_file_name)
    if not os.path.exists(exp_file_path):
        raise ValueError(f"Could not open experiment file in path {exp_file_path}")
    experiment = _load_base_experiment(exp_file_path, safe_base_path)

    # Apply environment overlay
    env_exp_file_path = os.path.join(safe_base_path, env_experiment_file_name)
    if os.path.exists(env_exp_file_path):
        _apply_overlay(experiment, env_exp_file_path, base_path)

    return experiment
