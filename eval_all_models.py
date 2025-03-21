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

# Import safe globals for torch.load
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarfloat import ScalarFloat

# --- Adjust sys.path so that the repository root is found ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# --- FNO configuration loader ---
def load_fno_config(config_path="config/darcy_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Expecting keys: n_modes, in_channels, out_channels, hidden_channels.
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
    test_resolutions=[128],
    n_tests=[100],
    test_batch_sizes=[16],
    encode_input=False,
    encode_output=False,
)

def load_and_prune_model(ModelClass, weight_path, test_loaders, data_processor, device, prune_ratio=0.2, technique="layer"):
    # Instantiate the model with training parameters.
    if ModelClass.__name__ == "FNO":
        # FNO-specific parameters (from your code)
        n_modes = (32, 32)
        in_channels = 1
        out_channels = 1
        hidden_channels = 64
        n_layers = 5
        base_model = ModelClass(
            n_modes, in_channels, out_channels, hidden_channels,
            n_layers=n_layers, skip="linear", norm="group_norm",
            implementation="factorized", projection_channel_ratio=2,
            separable=False, dropout=0.0, rank=1.0
        )
    elif ModelClass.__name__ == "DeepONet":
        # DeepONet-specific parameters (from your code)
        train_resolution = 128
        in_channels = 1
        out_channels = 1
        hidden_channels = 64
        branch_layers = [256, 256, 256, 256, 128]
        trunk_layers = [256, 256, 256, 256, 128]
        base_model = ModelClass(train_resolution, in_channels, out_channels, hidden_channels, branch_layers, trunk_layers)
    else:
        base_model = ModelClass()

    # Load the model weights safely.
    with torch.serialization.safe_globals([ScalarInt, ScalarFloat]):
        state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    base_model.load_state_dict(state_dict, strict=False)
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
        pruner.prune()
    else:
        raise ValueError("Unknown compression technique: choose 'layer' or 'magnitude'")

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

# --- Updated load_and_prune_model functionfor GINO AND CODANO ---
    # Instantiate the model with training parameters.
    # elif ModelClass.__name__ == "GINO": # ENABLE WHEN NEEDED
    #     # Use parameters from gino_carcfd_config.yaml (adjust as necessary)
    #     n_modes = (16, 16, 16)
    #     in_channels = 1
    #     out_channels = 1
    #     hidden_channels = 64
    #     base_model = ModelClass(n_modes, in_channels, out_channels, hidden_channels)
    # elif ModelClass.__name__ == "CODANO": # ENABLE WHEN NEEDED
    #     # Use parameters from darcy_config_codano.yaml
    #     in_channels = 1
    #     output_variable_codimension = 1
    #     hidden_variable_codimension = 2
    #     lifting_channels = 4
    #     base_model = ModelClass(
    #         in_channels=in_channels,
    #         output_variable_codimension=output_variable_codimension,
    #         hidden_variable_codimension=hidden_variable_codimension,
    #         lifting_channels=lifting_channels
    #     )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    technique = "layer"  # Choose "layer" or "magnitude"

    # --- Section 1: Eval all models ---
    fno_weight_path = "models/model-fno-darcy-16-resolution-2025-03-04-18-48.pt"
    deeponet_weight_path = "models/model-deeponet-darcy-128-resolution-2025-03-04-18-53.pt"
    # gino_weight_path = "models/model-gino-carcfd-32-resolution-2025-03-04-20-01.pt"
    # codano_weight_path = "models/model-codano-darcy-16-resolution-2025-02-11-21-13.pt"
    
    models_info = {
        "FNO": {"class": FNO, "weight": fno_weight_path},
        "DeepONet": {"class": DeepONet, "weight": deeponet_weight_path},
        # "GINO": {"class": GINO, "weight": gino_weight_path},
        # "Codano": {"class": CODANO, "weight": codano_weight_path},
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
    # For demonstration, if separate files are not available, use the same file.
    our_fno_weight = fno_weight_path  # Replace with separate path if available.
    their_fno_weight = fno_weight_path  # Replace with separate path if available.
    # our_codano_weight = codano_weight_path  # Replace with separate path if available.
    # their_codano_weight = codano_weight_path  # Replace with separate path if available.

    # models_info_section2 = {
    #     "FNO": {
    #         "our": our_fno_weight,
    #         "theirs": their_fno_weight
    #     },
    #     # "Codano": {
    #         # "our": our_codano_weight,
    #         # "theirs": their_codano_weight
    #     # }
    # }

    # results_section2 = {}
    # print("\n=== Section 2: Comparing 'Our' vs 'Their' weights for FNO and Codano using {} pruning ===".format(technique))
    # for model_name, weight_paths in models_info_section2.items():
    #     results_section2[model_name] = {}
    #     ModelClass = FNO if model_name == "FNO" else CODANO

    #     for variant, weight_path in weight_paths.items():
    #         print(f"\nProcessing {model_name} ({variant})...")
    #         try:
    #             base_model, pruned_model, results = load_and_prune_model(
    #                 ModelClass, weight_path, test_loaders, data_processor, device, prune_ratio=0.2, technique=technique
    #             )
    #             results_section2[model_name][variant] = results
    #             print(f"{model_name} ({variant}) evaluation results: {results}")
    #         except Exception as e:
    #             print(f"Error processing {model_name} ({variant}): {e}")

    print("\nSection 1 Results:")
    print(results_section1)
    # print("\nSection 2 Results:")
    # print(results_section2)

if __name__ == "__main__":
    main()
