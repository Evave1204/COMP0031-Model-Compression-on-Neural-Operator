import torch
import yaml
from neuralop.models.codano import CODANO

def load_codano_config(config_path="config/darcy_config_codano.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    codano_config = config.get("codano", {})
    in_channels = codano_config.get("data_channels", 1)
    output_variable_codimension = codano_config.get("output_variable_codimension", 1)
    hidden_variable_codimension = codano_config.get("hidden_variable_codimension", 2)
    lifting_channels = codano_config.get("lifting_channels", 4)
    return in_channels, output_variable_codimension, hidden_variable_codimension, lifting_channels

def main():
    in_channels, output_variable_codimension, hidden_variable_codimension, lifting_channels = load_codano_config()
    # Instantiate CODANO with the configuration parameters.
    model = CODANO(
        in_channels=in_channels,
        output_variable_codimension=output_variable_codimension,
        hidden_variable_codimension=hidden_variable_codimension,
        lifting_channels=lifting_channels
    )
    weight_path = "models/model-codano-darcy-16-resolution-2025-02-11-21-13.pt"
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print("CODANO loaded successfully!")
    
    # According to the codano config, train_resolution is 16.
    dummy_input = torch.randn(1, in_channels, 16, 16)
    output = model(dummy_input)
    print("Output:", output)

if __name__ == "__main__":
    main()
