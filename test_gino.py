import torch
import yaml
from neuralop.models.gino import GINO
from ruamel.yaml.scalarfloat import ScalarFloat

def load_gino_config(config_path="config/gino_carcfd_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    cfd_config = config.get("cfd", {})
    gino_config = cfd_config.get("gino", {})
    n_modes = tuple(gino_config.get("fno_n_modes", [16, 16, 16]))
    in_channels = 1
    out_channels = gino_config.get("out_channels", 1)
    hidden_channels = gino_config.get("fno_hidden_channels", 64)
    resolution_scaling_factor = gino_config.get("fno_resolution_scaling_factor", 1)
    return n_modes, in_channels, out_channels, hidden_channels, resolution_scaling_factor

def main():
    n_modes, in_channels, out_channels, hidden_channels, resolution_scaling_factor = load_gino_config()
    model = GINO(n_modes, in_channels, out_channels, hidden_channels,
                 resolution_scaling_factor=resolution_scaling_factor)
    weight_path = "models/model-gino-carcfd-32-resolution-2025-03-04-20-01.pt"
    with torch.serialization.safe_globals([ScalarFloat]):
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.eval()
    print("GINO loaded successfully!")
    dummy_input = torch.randn(1, in_channels, 32, 32)
    output = model(dummy_input)
    if isinstance(output, (list, tuple)):
        print("Output shapes:", [o.shape for o in output])
    else:
        print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
