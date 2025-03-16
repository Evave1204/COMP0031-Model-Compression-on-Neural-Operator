from neuralop.models.fno import FNO
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.base import CompressedModel
from neuralop.data.datasets import load_darcy_flow_small

from compression.utils.evaluation_util import evaluate_model, compare_models
from compression.utils.fno_util import optional_fno

fno_model, train_loader, test_loaders, data_processor = optional_fno(resolution="high")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fno_model = fno_model.to(device)


# Initialize models 
# pruned_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.5),
#     create_replica=True
# )
# pruned_model = pruned_model.to(device)

lowrank_model = CompressedModel(
    model=fno_model,
    compression_technique=lambda model: SVDLowRank(model, 
                                                   rank_ratio=0.8, # option = [0.2, 0.4, 0.6, 0.8]
                                                   min_rank=16,
                                                   max_rank=256, # option = [8, 16, 32, 64, 128, 256]
                                                   is_compress_conv1d=False,
                                                   is_compress_FC=False,
                                                   is_compress_spectral=True),
    create_replica=True
)
lowrank_model = lowrank_model.to(device)

# dynamic_quant_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: SVDLowRank(model, 
#                                                    rank_ratio=0.8, # option = [0.2, 0.4, 0.6, 0.8]
#                                                    min_rank=16,
#                                                    max_rank=256, # option = [8, 16, 32, 64, 128, 256]
#                                                    is_compress_conv1d=False,
#                                                    is_compress_FC=False,
#                                                    is_comrpess_spectral=True),
#     create_replica=True
# )
# lowrank_model = lowrank_model.to(device)

dynamic_quant_model = CompressedModel(
    model=fno_model,
    compression_technique=lambda model: DynamicQuantization(model),
    create_replica=True
)
dynamic_quant_model = dynamic_quant_model.to(device)


# Start Compression ..

# print("\n"*2)
# print("Pruning.....")
# compare_models(
#     model1=fno_model,
#     model2=pruned_model,
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device
# )

# print("\n"*2)
# print("Low Ranking.....")
# compare_models(
#     model1=fno_model,
#     model2=lowrank_model,
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device,
#     track_performance = True
# )


print("\n"*2)
print("Dynamic Quantization.....")
compare_models(
    model1=fno_model,               # The original model (it will be moved to CPU in evaluate_model)
    model2=dynamic_quant_model,     # The dynamically quantized model
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)


