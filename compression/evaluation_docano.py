from neuralop.models import CODANO
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.base import CompressedModel
from neuralop.data.datasets import load_darcy_flow_small
from compression.utils import evaluate_model, compare_models
from neuralop.data.transforms.codano_processor import CODANODataProcessor

fno_model = CODANO(
    in_channels=1,
    output_variable_codimension=1,

    hidden_variable_codimension=2,
    lifting_channels=4,

    use_positional_encoding=False,
    positional_encoding_dim=1,
    positional_encoding_modes=[8, 8],

    use_horizontal_skip_connection=True,
    horizontal_skips_map={3: 1, 4: 0},

    n_layers=5,
    n_heads=[2, 2, 2, 2, 2],
    n_modes=[[8, 8], [8, 8], [8, 8], [8, 8], [8, 8]],
    attention_scaling_factors=[0.5, 0.5, 0.5, 0.5, 0.5],
    per_layer_scaling_factors=[[1, 1], [0.5, 0.5], [1, 1], [2, 2], [1, 1]],

    static_channel_dim=0,
    variable_ids=["a1"],
    enable_cls_token=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fno_model.load_model(torch.load("model-codano-darcy-16-resolution-2025-02-11-21-13.pt", weights_only=False))
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
test_loader_16 = test_loaders[16]
test_loader_32 = test_loaders[32]

# When creating data processor, use the imported class
data_processor = CODANODataProcessor(
    in_normalizer=data_processor.in_normalizer,
    out_normalizer=data_processor.out_normalizer
)

pruned_model = CompressedModel(
    model=fno_model,
    compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.05),
    create_replica=True
)
pruned_model = pruned_model.to(device)

compare_models(
    model1=fno_model,
    model2=pruned_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)