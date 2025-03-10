import torch
import yaml
from neuralop.models.deeponet import DeepONet

def load_deeponet_config(config_path="config/deeponet_darcy_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    deeponet_config = config.get("deeponet", {})
    train_resolution = deeponet_config.get("train_resolution", 128)
    in_channels = deeponet_config.get("in_channels", 1)
    out_channels = deeponet_config.get("out_channels", 1)
    hidden_channels = deeponet_config.get("hidden_channels", 64)
    branch_layers = deeponet_config.get("branch_layers", [256,256,256,256,128])
    trunk_layers = deeponet_config.get("trunk_layers", [256,256,256,256,128])
    return train_resolution, in_channels, out_channels, hidden_channels, branch_layers, trunk_layers

def main():
    train_resolution, in_channels, out_channels, hidden_channels, branch_layers, trunk_layers = load_deeponet_config()
    model = DeepONet(train_resolution, in_channels, out_channels, hidden_channels, branch_layers, trunk_layers)
    
    weight_path = "models/model-deeponet-darcy-128-resolution-2025-03-04-18-53.pt"
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print("DeepONet loaded successfully!")
    
    # Create a dummy input (adjust dimensions according to your model's forward method)
    dummy_input = torch.randn(1, in_channels, train_resolution)  # example: 1 x 1 x 128
    output = model(dummy_input)
    print("Output:", output)

if __name__ == "__main__":
    main()
