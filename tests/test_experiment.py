import os
from pathlib import Path
from typing import Any, List

import pytest
from llmops.common.experiment import (
    Dataset,
    Evaluator,
    Experiment,
    MappedDataset,
    _create_datasets_and_default_mappings,
    _create_eval_datasets_and_default_mappings,
    _create_evaluators,
    _load_base_experiment,
    _apply_overlay,
    load_experiment,
)

THIS_PATH = Path(__file__).parent
RESOURCE_PATH = THIS_PATH / "resources"


def check_lists_equal(actual: List[Any], expected: List[Any]):
    assert len(actual) == len(expected)
    assert all(any(a == e for a in actual) for e in expected)
    assert all(any(a == e for e in expected) for a in actual)


def test_create_datasets_and_default_mappings():
    # Prepare inputs
    g_name = "groundedness"
    g_source = "azureml:groundedness:1"
    g_mappings = {"claim": "claim_mapping"}
    g_dataset = Dataset(g_name, g_source, None, None)

    r_name = "recall"
    r_source = "recall_source"
    r_description = "recall_description"
    r_mappings = {"input": "input_mapping", "gt": "gt_mapping"}
    r_dataset = Dataset(r_name, r_source, r_description, None)

    raw_datasets = [
        {"name": g_name, "source": g_source, "mappings": g_mappings},
        {
            "name": r_name,
            "source": r_source,
            "description": r_description,
            "mappings": r_mappings,
        },
    ]

    # Prepare expected outputs
    expected_datasets = {g_name: g_dataset, r_name: r_dataset}
    expected_mapped_datasets = [
        MappedDataset(g_mappings, g_dataset),
        MappedDataset(r_mappings, r_dataset),
    ]

    # Check outputs
    [datasets, mapped_datasets] = _create_datasets_and_default_mappings(raw_datasets)
    assert datasets == expected_datasets
    check_lists_equal(mapped_datasets, expected_mapped_datasets)

    assert datasets[g_name].is_remote()
    assert not datasets[g_name].is_eval()
    assert not datasets[r_name].is_remote()
    assert not datasets[r_name].is_eval()


@pytest.mark.parametrize(
    ("raw_datasets", "error"),
    [
        ([{}], "Dataset 'None' config missing parameter: name"),
        (
            [
                {
                    "name": "groundedness",
                }
            ],
            "Dataset 'groundedness' config missing parameter: source",
        ),
        (
            [
                {
                    "name": "groundedness",
                    "source": "groundedness_source",
                }
            ],
            "Dataset 'groundedness' config missing parameter: mappings",
        ),
        (
            [
                {
                    "name": "groundedness",
                    "source": "groundedness_source",
                    "mappings": [],
                    "reference": "recall",
                }
            ],
            "Unexpected parameter found in dataset 'groundedness' description: reference",
        ),
    ],
)
def test_create_datasets_and_default_mappings_missing_parameters(
    raw_datasets: List[dict], error: str
):
    # Check that datasets with missing parameters raise an exception
    with pytest.raises(ValueError, match=error):
        _create_datasets_and_default_mappings(raw_datasets)


def test_create_eval_datasets_and_default_mappings():
    # Prepare inputs

    # Evaluation datasets
    g_name = "groundedness"
    g_source = "azureml:groundedness:1"
    g_reference = "groundedness_ref"
    g_mappings = {"claim": "claim_mapping"}
    g_dataset = Dataset(g_name, g_source, None, g_reference)

    r_name = "recall"
    r_source = "recall_source"
    r_description = "recall_description"
    r_reference = "recall_ref"
    r_mappings = {"input": "input_mapping", "gt": "gt_mapping"}
    r_dataset = Dataset(r_name, r_source, r_description, r_reference)

    a_name = "accuracy"
    a_source = "accuracy_source"
    a_mappings = {"text": "text_mapping"}
    a_dataset = Dataset(a_name, a_source, None, None)

    # Reference datasets
    g_ref_source = "groundedness_ref_source"
    g_ref_dataset = Dataset(g_reference, g_ref_source, None, None)

    r_ref_source = "recall_ref_source"
    r_ref_dataset = Dataset(r_reference, r_ref_source, None, None)

    existing_datasets = {
        g_reference: g_ref_dataset,
        r_reference: r_ref_dataset,
        a_name: a_dataset,
    }

    raw_eval_datasets = [
        {
            "name": g_name,
            "source": g_source,
            "reference": g_reference,
            "mappings": g_mappings,
        },
        {
            "name": r_name,
            "source": r_source,
            "description": r_description,
            "reference": r_reference,
            "mappings": r_mappings,
        },
        {
            "name": a_name,
            "mappings": a_mappings,
        },
    ]

    # Prepare expected outputs
    expected_mapped_datasets = [
        MappedDataset(g_mappings, g_dataset),
        MappedDataset(r_mappings, r_dataset),
        MappedDataset(a_mappings, a_dataset),
    ]

    # Check outputs
    mapped_datasets = _create_eval_datasets_and_default_mappings(
        raw_eval_datasets, existing_datasets
    )
    check_lists_equal(mapped_datasets, expected_mapped_datasets)

    assert mapped_datasets[0].dataset.is_remote()
    assert mapped_datasets[0].dataset.is_eval()

    assert not mapped_datasets[1].dataset.is_remote()
    assert mapped_datasets[1].dataset.is_eval()

    assert not mapped_datasets[2].dataset.is_remote()
    assert not mapped_datasets[2].dataset.is_eval()


@pytest.mark.parametrize(
    ("raw_datasets", "datasets", "error"),
    [
        ([{}], {}, "Dataset 'None' config missing parameter: name"),
        (
            [{"name": "groundedness"}],
            {},
            "Dataset 'groundedness' config missing parameter: mappings",
        ),
        (
            [
                {
                    "name": "groundedness",
                    "mappings": {},
                }
            ],
            {},
            "Dataset 'groundedness' config missing parameter: source",
        ),
        (
            [
                {
                    "name": "groundedness",
                    "source": "groundedness_source",
                    "mappings": {},
                }
            ],
            {},
            "Dataset 'groundedness' config missing parameter: reference",
        ),
        (
            [
                {
                    "name": "groundedness",
                    "source": "groundedness_source_2",
                    "mappings": {},
                }
            ],
            {
                "groundedness": Dataset(
                    "groundedness", "groundedness_source", None, None
                )
            },
            "Dataset 'groundedness' config is referencing an existing dataset so it doesn't support parameter: source",
        ),
        (
            [
                {
                    "name": "groundedness",
                    "reference": "groundedness_ref",
                    "mappings": {},
                }
            ],
            {
                "groundedness": Dataset(
                    "groundedness", "groundedness_source", None, None
                )
            },
            "Dataset 'groundedness' config is referencing an existing dataset so it doesn't support parameter: reference",
        ),
        (
            [
                {
                    "name": "groundedness",
                    "source": "groundedness_source",
                    "reference": "groundedness_ref",
                    "mappings": {},
                }
            ],
            {},
            "Referenced dataset 'groundedness_ref' not defined",
        ),
    ],
)
def test_create_eval_datasets_and_default_mappings_missing_parameters(
    raw_datasets: List[dict], datasets: dict[str:Dataset], error: str
):
    # Check that datasets with missing parameters raise an exception
    with pytest.raises(ValueError, match=error):
        _create_eval_datasets_and_default_mappings(raw_datasets, datasets)


def test_create_evaluators():
    # Prepare inputs
    g_name = "groundedness"
    g_flow = "groundedness_eval"
    g_source = "groundedness_source"
    g_mappings = {"claim": "claim_mapping"}
    g_dataset = Dataset(g_name, g_source, None, None)

    r_name = "recall"
    r_source = "recall_source"
    r_mappings = {"input": "input_mapping", "gt": "gt_mapping"}
    r_dataset = Dataset(r_name, r_source, None, None)

    available_datasets = {g_name: g_dataset, r_name: r_dataset}

    raw_evaluators = [
        {
            "name": g_name,
            "flow": g_flow,
            "datasets": [{"name": g_dataset.name, "mappings": g_mappings}],
        },
        {
            "name": r_name,
            "datasets": [{"name": r_dataset.name, "mappings": r_mappings}],
        },
    ]

    # Test with base_path
    base_path = "/path/to/flow/"

    # Prepare expected outputs
    expected_evaluators = [
        Evaluator(
            g_name,
            [MappedDataset(g_mappings, g_dataset)],
            os.path.join(base_path, "flows", g_flow),
        ),
        Evaluator(
            r_name,
            [MappedDataset(r_mappings, r_dataset)],
            os.path.join(base_path, "flows", r_name),
        ),
    ]

    # Check outputs
    evaluators = _create_evaluators(raw_evaluators, available_datasets, base_path)
    assert evaluators == expected_evaluators
    assert evaluators[0].find_dataset_with_reference(g_name) is None

    # Test without base_path
    base_path = None

    # Prepare expected outputs
    expected_evaluators = [
        Evaluator(
            g_name,
            [MappedDataset(g_mappings, g_dataset)],
            os.path.join("flows", g_flow),
        ),
        Evaluator(
            r_name,
            [MappedDataset(r_mappings, r_dataset)],
            os.path.join("flows", r_name),
        ),
    ]

    # Check outputs
    evaluators = _create_evaluators(raw_evaluators, available_datasets, base_path)
    assert evaluators == expected_evaluators


@pytest.mark.parametrize(
    "raw_evaluators",
    [
        [{}],
        [
            {
                "name": "groundedness",
            }
        ],
        [
            {
                "name": "groundedness",
                "datasets": [{"name": "groundedness"}],
            }
        ],
    ],
)
def test_create_evaluators_missing_parameters(raw_evaluators: List[dict]):
    available_datasets = {
        "groundedness": Dataset("groundedness", "groundedness_source", None, None),
    }
    # Check that evaluators with missing parameters raise an exception
    with pytest.raises(ValueError, match=".*missing parameter"):
        _create_evaluators(raw_evaluators, available_datasets, None)


def test_create_evaluators_invalid_dataset():
    # Prepare inputs
    eval_name = "groundedness"
    dataset_name = "groundedness"

    raw_evaluators = [
        {
            "name": eval_name,
            "flow": "groundedness_eval",
            "datasets": [
                {"name": dataset_name, "mappings": {"claim": "claim_mapping"}}
            ],
        }
    ]

    available_datasets = {}

    # Check that evaluators with invalid datasets (datasets not in the available_datasets dict) raise an exception
    with pytest.raises(
        ValueError, match=f"Dataset '{dataset_name}' config missing parameter: source"
    ):
        _create_evaluators(raw_evaluators, available_datasets, None)


def test_experiment_creation():
    # Prepare inputs
    base_path = str(RESOURCE_PATH)
    name = "exp_name"
    flow = "exp_flow"

    # Prepare expected outputs
    expected_flow_variants = [
        {"var_0": "node_var_0", "var_1": "node_var_0"},
        {"var_3": "node_var_1", "var_4": "node_var_1"},
    ]
    expected_flow_default_variants = {"node_var_0": "var_0", "node_var_1": "var_3"}
    expected_flow_llm_nodes = {
        "node_var_0",
        "node_var_1",
    }

    # Check outputs
    experiment = Experiment(base_path, name, flow, [], [])
    flow_detail = experiment.get_flow_detail()

    assert flow_detail.flow_path == os.path.join(base_path, "flows", flow)
    assert flow_detail.all_variants == expected_flow_variants
    assert flow_detail.default_variants == expected_flow_default_variants
    assert flow_detail.all_llm_nodes == expected_flow_llm_nodes


# def test_load_experiment():
#     # Prepare inputs
#     base_path = str(RESOURCE_PATH)

#     # Prepare expected outputs
#     expected_name = "exp"
#     expected_flow = "exp_flow"

#     expected_dataset_names = ["ds1", "ds2"]
#     expected_dataset_sources = ["ds1_source", "ds2_source"]
#     expected_dataset_mappings = [
#         {"ds1_input": "ds1_mapping"},
#         {"ds2_input": "ds2_mapping"},
#     ]
#     expected_datasets = [
#         Dataset(expected_dataset_names[0], expected_dataset_sources[0], None, None),
#         Dataset(expected_dataset_names[1], expected_dataset_sources[1], None, None),
#     ]
#     expected_mapped_datasets = [
#         MappedDataset(expected_dataset_mappings[0], expected_datasets[0]),
#         MappedDataset(expected_dataset_mappings[1], expected_datasets[1]),
#     ]

#     expected_evaluator_dataset_mappings = [
#         {"ds1_input": "ds1_mapping", "ds1_extra": "ds1_extra_mapping"},
#         {"ds2_extra": "ds2_extra_mapping"},
#         {"ds2_input": "ds2_diff_mapping"},
#     ]
#     expected_evaluator_mapped_datasets = [
#         [
#             MappedDataset(expected_evaluator_dataset_mappings[0], expected_datasets[0]),
#             MappedDataset(expected_evaluator_dataset_mappings[1], expected_datasets[1]),
#         ],
#         [MappedDataset(expected_evaluator_dataset_mappings[2], expected_datasets[1])],
#     ]
#     expected_evaluators = [
#         Evaluator(
#             "eval1",
#             expected_evaluator_mapped_datasets[0],
#             os.path.join(base_path, "flows", "eval1"),
#         ),
#         Evaluator(
#             "eval2",
#             expected_evaluator_mapped_datasets[1],
#             os.path.join(base_path, "flows", "eval2"),
#         ),
#     ]

#     # Test with no environment overrides
#     # Check outputs
#     experiment = load_experiment(base_path=base_path)
#     assert experiment.base_path == base_path
#     assert experiment.name == expected_name
#     assert experiment.flow == expected_flow
#     assert experiment.datasets == expected_mapped_datasets
#     assert experiment.evaluators == expected_evaluators

#     # Test with environment overrides
#     # Modify expected outputs
#     expected_overridden_dataset_source = "overridden_ds1_source"
#     expected_overridden_dataset = Dataset(
#         expected_dataset_names[0], expected_overridden_dataset_source, None, None
#     )
#     expected_overridden_mapped_datasets = [
#         MappedDataset(expected_dataset_mappings[0], expected_overridden_dataset),
#         MappedDataset(expected_dataset_mappings[1], expected_datasets[1]),
#     ]
#     expected_overridden_evaluator_mapped_datasets = [
#         MappedDataset(
#             expected_evaluator_dataset_mappings[0], expected_overridden_dataset
#         ),
#         MappedDataset(expected_evaluator_dataset_mappings[1], expected_datasets[1]),
#     ]
#     expected_overridden_evaluators = [
#         Evaluator(
#             "eval1",
#             expected_overridden_evaluator_mapped_datasets,
#             os.path.join(base_path, "flows", "eval1"),
#         ),
#         Evaluator(
#             "eval2",
#             expected_evaluator_mapped_datasets[1],
#             os.path.join(base_path, "flows", "eval2"),
#         ),
#     ]

#     # Check outputs
#     experiment = load_experiment(base_path=base_path, env="dev")
#     assert experiment.base_path == base_path
#     assert experiment.name == expected_name
#     assert experiment.flow == expected_flow
#     assert experiment.datasets == expected_overridden_mapped_datasets
#     assert experiment.evaluators == expected_overridden_evaluators
