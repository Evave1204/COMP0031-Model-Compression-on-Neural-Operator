from neuralop.models.fno import FNO
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.UniformQuant.uniform_quant import UniformQuantisation
from compression.base import CompressedModel
from neuralop.data.datasets import load_darcy_flow_small

from compression.utils.evaluation_util import evaluate_model, compare_models
from compression.utils.fno_util import optional_fno

fno_model, train_loader, test_loaders, data_processor = optional_fno(resolution="low")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fno_model = fno_model.to(device)


train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=100,
    batch_size=16,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[16, 16],
    encode_input=False, 
    encode_output=False,
)


# pruned_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.05),
#     create_replica=True
# )
# pruned_model = pruned_model.to(device)

# lowrank_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: SVDLowRank(model, rank_ratio=0.7, 
#                                                    min_rank=8, max_rank=16),
#     create_replica=True
# )
# lowrank_model = lowrank_model.to(device)

# lowrank_compare = compare_models(
#     model1=fno_model,
#     model2=lowrank_model,
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device,
#     track_performance=True
# )

stquantised_model = CompressedModel(
    model=fno_model,
    compression_technique=lambda model: UniformQuantisation(model, 
                                                            num_bits=16,
                                                            num_calibration_runs=32),
    create_replica=True
)
print(fno_model)
print("quuantised module",stquantised_model)
stquantised_model = stquantised_model.to(device)

stquantised_compare = compare_models(
    model1=fno_model,
    model2=stquantised_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device,
    #track_performance=True
)
print(stquantised_compare)

# dyquantised_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: DynamicQuantization(model),
#     create_replica=True
# )
# print(fno_model)
# print("quuantised module",dyquantised_model)
# dyquantised_model = dyquantised_model.to(device)

# dyquantised_compare = compare_models(
#     model1=fno_model,
#     model2=dyquantised_model,
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device,
#     # track_performance=True
# )