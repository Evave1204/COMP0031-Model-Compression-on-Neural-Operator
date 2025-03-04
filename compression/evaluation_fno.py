from neuralop.models import FNO
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.base import CompressedModel
from neuralop.data.datasets import load_darcy_flow_small
from compression.utils import evaluate_model, compare_models


fno_model = FNO(
    in_channels=1,
    out_channels=1,
    n_modes=(16, 16),
    hidden_channels=32,
    projection_channel_ratio=2,
    n_layers=4,
    skip="linear",
    norm="group_norm",
    implementation="factorized",
    separable=False,
    factorization=None,
    rank=1.0,
    domain_padding=None,
    stabilizer=None,
    dropout=0.0)

# fno_model = FNO(
#     in_channels=1,
#     out_channels=1,
#     n_modes=(32, 32),
#     hidden_channels=64,
#     projection_channel_ratio=2,
#     n_layers=5,
#     skip="linear",
#     norm="group_norm",
#     implementation="factorized",
#     separable=False,
#     factorization=None,
#     rank=1.0,
#     domain_padding=None,
#     stabilizer=None,
#     dropout=0.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fno_model.load_state_dict(torch.load("models/model-fno-darcy-16-resolution-2025-02-05-19-55.pt", weights_only=False))
fno_model.eval()
fno_model = fno_model.to(device)

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=16,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[16, 16],
    encode_input=False, 
    encode_output=False,
)
# train_loader, test_loaders, data_processor = load_darcy_flow_small(
#     n_train=100,
#     batch_size=16,
#     test_resolutions=[128],
#     n_tests=[10000],
#     test_batch_sizes=[1],
#     encode_input=False, 
#     encode_output=False,
# )

pruned_model = CompressedModel(
    model=fno_model,
    compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.5),
    create_replica=True
)
pruned_model = pruned_model.to(device)

lowrank_model = CompressedModel(
    model=fno_model,
    compression_technique=lambda model: SVDLowRank(model, rank_ratio=0.5,
                                                 min_rank=4, max_rank=128),
    create_replica=True
)
lowrank_model = lowrank_model.to(device)

dynamic_quant_model = CompressedModel(
    model=fno_model,
    compression_technique=lambda model: DynamicQuantization(model),
    create_replica=True
)
dynamic_quant_model = dynamic_quant_model.to(device)


# Start Compression ..

print("\n"*2)
print("Pruning.....")
compare_models(
    model1=fno_model,
    model2=pruned_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)

print("\n"*2)
print("Low Ranking.....")
compare_models(
    model1=fno_model,
    model2=lowrank_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device,
    track_performance = True
)

print("\n"*2)
print("Dynamic Quantization.....")
compare_models(
    model1=fno_model,               # The original model (it will be moved to CPU in evaluate_model)
    model2=dynamic_quant_model,     # The dynamically quantized model
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)
