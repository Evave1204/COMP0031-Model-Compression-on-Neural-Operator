'''
from original but only evaluation...
'''
from neuralop.models.codano_gino import CodANO
import argparse
import torch
import random
import numpy as np
from neuralop.models.get_models import *
from neuralop.layers.variable_encoding import *
from neuralop.data.datasets.ns_dataset import *
from neuralop.data_utils.data_utils import MaskerNonuniformMesh, get_meshes
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.base import CompressedModel
from compression.utils import evaluate_model, compare_models, CodanoYParams
from neuralop.data.transforms.codano_processor import CODANODataProcessor
from compression.utils import missing_variable_testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="?", default="FSI", type=str)
    parser.add_argument("--config", nargs="?", default="codano_gno_NS_ES", type=str)
    parsed_args = parser.parse_args()

    if parsed_args.exp == "FSI":
        config_file = './config/ssl_ns_elastic.yaml'
    elif parsed_args.exp == "RB":
        config_file = './config/RB_config.yaml'
    else:
        raise ValueError("Unknown experiment type")

    config = parsed_args.config
    print("Loading config", config)
    params = CodanoYParams(config_file, config, print_params=False)
    # params.pretrain_ssl = True
    stage = StageEnum.RECONSTRUCTIVE
    torch.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)

    # nettype -> transformer, grid_type -> non_uniform
    # => ssl_model
    # evaluation only
    encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(params)

    # set the variable encoder
    variable_encoder = None
    token_expander = None
    if params.use_variable_encoding:
        variable_encoder = get_variable_encoder(params)
        token_expander = TokenExpansion(
            sum([params.equation_dict[i] for i in params.equation_dict.keys()]),
            params.n_encoding_channels,
            params.n_static_channels,
            params.grid_type == 'uniform'
        )

    model = SSLWrapper(params, encoder, decoder, contrastive, predictor, stage=stage)

    if params.grid_type != 'uniform':
        mesh = get_mesh(params)
        input_mesh = torch.from_numpy(mesh).float().cuda()
        model.set_initial_mesh(input_mesh)

    # load the weights
    model.load_state_dict(torch.load("models/codano_whole_model_weights.pt"), strict=False)
    model = model.cuda()
    model.eval()
    if variable_encoder is not None:
        variable_encoder = variable_encoder.cuda()
    if token_expander is not None:
        token_expander = token_expander.cuda()

    test_path = "neuralop/data/datasets/data/foundation_codano_test.pkl"
    test_dataset, test_dataloader, metadata = load_test_set(test_path)

    print("Evaluating model on test dataset...")
    missing_variable_testing(
        model,
        test_dataloader,
        augmenter=None,
        stage=stage,
        params=params,
        variable_encoder=variable_encoder,
        token_expander=token_expander,
        initial_mesh=input_mesh
    )


'''
ours
'''
# from neuralop.models.codano_gino import CodANO
# import argparse
# import torch
# import random
# import numpy as np
# from neuralop.models.get_models import *
# from neuralop.layers.variable_encoding import *
# from neuralop.data.datasets.ns_dataset import *
# from neuralop.data_utils.data_utils import MaskerNonuniformMesh, get_meshes
# from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
# from compression.LowRank.SVD_LowRank import SVDLowRank
# from compression.quantization.dynamic_quantization import DynamicQuantization
# from compression.base import CompressedModel
# from neuralop.data.datasets import load_darcy_flow_small
# from compression.utils import evaluate_model, compare_models, CodanoYParams
# from neuralop.data.transforms.codano_processor import CODANODataProcessor
# # parse config
# parser = argparse.ArgumentParser()
# parser.add_argument("--exp", nargs="?", default="FSI", type=str)
# parser.add_argument("--config", nargs="?", default="codano_gno_NS_ES", type=str)
# parser.add_argument("--ntrain", nargs="?", default=None, type=int)
# parser.add_argument("--epochs", nargs="?", default=None, type=int)
# parser.add_argument("--random_seed", nargs="?", default=42, type=int)
# parser.add_argument("--scheduler_step", nargs="?", default=None, type=int)
# parser.add_argument("--batch_size", nargs="?", default=None, type=int)
# parsed_args = parser.parse_args()

# config_file = "config/ssl_ns_elastic.yaml"
# config = parsed_args.config
# print("Loading config", config)
# params = CodanoYParams(config_file, config, print_params=False)

# # set random seeds
# if parsed_args.random_seed is not None:
#     params.random_seed = parsed_args.random_seed
#     print("Overriding random seed to", params.random_seed)
# torch.manual_seed(params.random_seed)
# random.seed(params.random_seed)
# np.random.seed(params.random_seed)

# params.config = config

# if params.pretrain_ssl:
#     stage = StageEnum.RECONSTRUCTIVE
# else:
#     stage = StageEnum.PREDICTIVE
# variable_encoder = get_variable_encoder(params)
# token_expander = TokenExpansion(sum([params.equation_dict[i] for i in params.equation_dict.keys(
#             )]), params.n_encoding_channels, params.n_static_channels, params.grid_type == 'uniform')

# # prepare model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(params)
# codano_model = SSLWrapper(params, encoder, decoder, contrastive, predictor,stage=stage)
# mesh = get_mesh(params)
# input_mesh = torch.from_numpy(mesh).type(torch.float).cuda()
# codano_evaluation_params = {"variable_encoder":variable_encoder, 
#                             "token_expander":token_expander, 
#                             "input_mesh":input_mesh,
#                             "params":params,
#                             "stage":stage}
# codano_model.set_initial_mesh(input_mesh)
# codano_model = codano_model.to(device)
# # load and assign weights
# weights = torch.load("models/codano_whole_model_weights.pt", map_location='cpu', weights_only=False)
# codano_model.load_state_dict(weights)
# codano_model.eval()

# print("model loaded success")

# test_path = "neuralop/data/datasets/data/foundation_codano_test.pkl"
# test_dataset, test_loader, metadata = load_test_set(test_path)
# print("test dataset loaded")

# lowrank_model = CompressedModel(
#     model=codano_model,
#     compression_technique=lambda model: SVDLowRank(model, 
#                                                    rank_ratio=0.8, # option = [0.2, 0.4, 0.6, 0.8]
#                                                    min_rank=16,
#                                                    max_rank=256, # option = [8, 16, 32, 64, 128, 256]
#                                                    is_full_rank=True,
#                                                    is_compress_conv1d=False,
#                                                    is_compress_FC=False,
#                                                    is_comrpess_spectral=True),
#     create_replica=True
# )

# lowrank_model = lowrank_model.to(device)

# print("\n"*2)
# print("Low Ranking.....")
# compare_models(
#     model1=codano_model,
#     model2=lowrank_model,
#     test_loaders=test_loader,
#     data_processor=None,
#     device=device,
#     track_performance = True,
#     evaluation_params = codano_evaluation_params
# )