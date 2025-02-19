from neuralop.models import DeepONet
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.base import CompressedModel
from neuralop.data.datasets import load_darcy_flow_small
from compression.utils import evaluate_model, compare_models

deeponet_model = DeepONet(
    in_channels=1,
    out_channels=1, 
    hidden_channels=64,
    branch_layers=[64, 64, 64, 64, 64],
    trunk_layers=[64, 64, 64, 64, 64],
    positional_embedding='grid',
    non_linearity='gelu',
    norm='instance_norm',
    dropout=0.1
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deeponet_model.load_state_dict(torch.load("model-deeponet-darcy-128-resolution-2025-02-19-19-15.pt", weights_only=False))
deeponet_model.eval()
deeponet_model = deeponet_model.to(device)

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=16,
    test_resolutions=[16, 32, 128],
    n_tests=[100, 100, 100],
    test_batch_sizes=[16, 16],
    encode_input=False, 
    encode_output=False,
)

pruned_model = CompressedModel(
    model=deeponet_model,
    compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.05),
    create_replica=True
)
pruned_model = pruned_model.to(device)

lowrank_model = CompressedModel(
    model=deeponet_model,
    compression_technique=lambda model: SVDLowRank(model, rank_ratio=0.7,
                                                 min_rank=8, max_rank=16),
    create_replica=True
)
lowrank_model = lowrank_model.to(device)

compare_models(
    model1=deeponet_model,
    model2=pruned_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)

compare_models(
    model1=deeponet_model,
    model2=lowrank_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)