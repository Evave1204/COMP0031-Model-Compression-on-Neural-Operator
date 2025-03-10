from neuralop.data.datasets.darcy import load_darcy_flow_small
from neuralop.models.codano import CODANO
from neuralop.models.gino import GINO
from neuralop.models.deeponet import DeepONet
from neuralop.models.fno import FNO
from compression.utils import compare_models
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.layer_pruning.layer_pruning import GlobalLayerPruning
import sys
import os
import torch
import torch.nn as nn
import yaml
import copy
import wandb

# --- Helper to download artifacts from WandB ---


def download_artifact(artifact_path, artifact_type="model"):
    run = wandb.init(project="ucl-neural-operator", job_type="download", reinit=True)
    artifact = run.use_artifact(artifact_path, type=artifact_type)
    local_path = artifact.download()
    run.finish()
    return local_path




# --- Adjust sys.path so that the repository root is found ---
repo_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# --- FNO configuration loader ---


def load_fno_config(config_path="config/darcy_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Assume the YAML file contains keys: n_modes, in_channels, out_channels, hidden_channels.
    return (
        config.get("n_modes", (16, 16)),
        config.get("in_channels", 1),
        config.get("out_channels", 1),
        config.get("hidden_channels", 32)
    )


# --- Load the Darcy dataset ---
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=16,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[16, 16],
    encode_input=False,
    encode_output=False,
)

# --- Updated load_and_prune_model function using compare_models ---


def load_and_prune_model(ModelClass, weight_path, test_loaders, data_processor, device, prune_ratio=0.2, technique="layer"):
    """
    Load a model from weight_path, create a deep copy for pruning, apply the selected pruning technique,
    and then evaluate and compare using compare_models.
    """
    # Instantiate the model with required parameters.
    if ModelClass.__name__ == "FNO":
        n_modes, in_channels, out_channels, hidden_channels = load_fno_config()
        base_model = ModelClass(n_modes, in_channels,
                                out_channels, hidden_channels)
    elif ModelClass.__name__ == "DeepONet":
        # Use appropriate default parameters; adjust these as needed.
        base_model = ModelClass(train_resolution=16, in_channels=1, out_channels=1,
                                hidden_channels=32, branch_layers=3, trunk_layers=3)
    elif ModelClass.__name__ == "GINO":
        # Adjust as needed.
        base_model = ModelClass(in_channels=1, out_channels=1)
    elif ModelClass.__name__ == "CODANO":
        base_model = ModelClass(in_channels=1, output_variable_codimension=1,
                                hidden_variable_codimension=2, lifting_channels=4,
                                # Example defaults.
                                branch_layers=3, trunk_layers=3)
    else:
        base_model = ModelClass()

    # Load the model weights onto the base model.
    base_model.load_state_dict(torch.load(weight_path, map_location=device))
    base_model = base_model.to(device)
    base_model.eval()

    # Create a deep copy for pruning.
    pruned_model = copy.deepcopy(base_model)

    # Apply the selected pruning technique.
    if technique == "layer":
        pruner = GlobalLayerPruning(pruned_model)
        pruner.layer_prune(prune_ratio=prune_ratio)
    elif technique == "magnitude":
        pruner = GlobalMagnitudePruning(pruned_model, prune_ratio=prune_ratio)
        pruner.prune()  # Adjust this call if necessary.
    else:
        raise ValueError(
            "Unknown compression technique: choose 'layer' or 'magnitude'")

    # Evaluate using the real evaluation routine.
    results = compare_models(
        model1=base_model,
        model2=pruned_model,
        test_loaders=test_loaders,
        data_processor=data_processor,
        device=device,
        verbose=False
    )

    return base_model, pruned_model, results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    technique = "layer"  # Choose either "layer" or "magnitude"

    # --- Section 1: Evaluate all 4 models ---
    # Download artifacts using their IDs.
    fno_weight_path = download_artifact(
        "model-fno-darcy-16-resolution-2025-03-04-18-48:v0")
    deeponet_weight_path = download_artifact(
        "model-deeponet-darcy-128-resolution-2025-03-04-18-53:v0")
    gino_weight_path = download_artifact(
        "model-gino-carcfd-32-resolution-2025-03-04-20-01:v0")
    # For Codano, if an artifact is available, use download_artifact; otherwise, use a local placeholder.
    # Update this when the artifact is available.
    codano_weight_path = "path/to/codano_weights.pth"

    models_info = {
        "FNO": {"class": FNO, "weight": fno_weight_path},
        "DeepONet": {"class": DeepONet, "weight": deeponet_weight_path},
        "GINO": {"class": GINO, "weight": gino_weight_path},
        "Codano": {"class": CODANO, "weight": codano_weight_path},
    }

    results_section1 = {}
    print("=== Section 1: Evaluating all 4 models using {} pruning ===".format(technique))
    for model_name, info in models_info.items():
        print(f"\nProcessing {model_name}...")
        model_class = info["class"]
        weight_path = info["weight"]

        try:
            base_model, pruned_model, results = load_and_prune_model(
                model_class, weight_path, test_loaders, data_processor, device, prune_ratio=0.2, technique=technique
            )
            results_section1[model_name] = results
            print(f"{model_name} evaluation results: {results}")
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    # --- Section 2: Compare 'Our' vs 'Their' weights for FNO and Codano ---
    # Download or set paths for our vs. their weights.
    # Replace with artifact for "our" run if different.
    our_fno_weight = download_artifact(
        "model-fno-darcy-16-resolution-2025-03-04-18-48:v0")
    # Replace with artifact for "their" run.
    their_fno_weight = download_artifact(
        "model-fno-darcy-16-resolution-2025-03-04-18-48:v0")
    # For Codano, update these paths when the artifacts become available.
    our_codano_weight = "path/to/our_codano_weights.pth"
    their_codano_weight = "path/to/their_codano_weights.pth"

    models_info_section2 = {
        "FNO": {
            "our": our_fno_weight,
            "theirs": their_fno_weight
        },
        "Codano": {
            "our": our_codano_weight,
            "theirs": their_codano_weight
        }
    }

    results_section2 = {}
    print("\n=== Section 2: Comparing 'Our' vs 'Their' weights for FNO and Codano using {} pruning ===".format(technique))
    for model_name, weight_paths in models_info_section2.items():
        results_section2[model_name] = {}
        ModelClass = FNO if model_name == "FNO" else CODANO

        for variant, weight_path in weight_paths.items():
            print(f"\nProcessing {model_name} ({variant})...")
            try:
                base_model, pruned_model, results = load_and_prune_model(
                    ModelClass, weight_path, test_loaders, data_processor, device, prune_ratio=0.2, technique=technique
                )
                results_section2[model_name][variant] = results
                print(f"{model_name} ({variant}) evaluation results: {results}")
            except Exception as e:
                print(f"Error processing {model_name} ({variant}): {e}")

    print("\nSection 1 Results:")
    print(results_section1)
    print("\nSection 2 Results:")
    print(results_section2)


if __name__ == "__main__":
    main()
