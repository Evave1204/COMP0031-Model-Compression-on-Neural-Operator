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
from compression.utils.evaluation_util import evaluate_model, compare_models
from neuralop.data.transforms.codano_processor import CODANODataProcessor
from compression.utils.codano_util import missing_variable_testing, CodanoYParams

# we should use scatter to accerlaerate
import warnings
warnings.filterwarnings("ignore", message="use_scatter is True but torch_scatter is not properly built")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="?", default="FSI", type=str)
    parser.add_argument("--config", nargs="?", default="codano_gno_NS_ES", type=str)
    args = parser.parse_args()
    config_file = './config/ssl_ns_elastic.yaml'
    print("Loading config", args.config)
    params = CodanoYParams(config_file, args.config, print_params=False)
    torch.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    params.config = args.config
    #stage = StageEnum.PREDICTIVE
    stage = StageEnum.RECONSTRUCTIVE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    variable_encoder = None
    token_expander = None

    encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(params)
    if params.use_variable_encoding:
        variable_encoder = get_variable_encoder(params)
        token_expander = TokenExpansion(
            sum([params.equation_dict[i] for i in params.equation_dict.keys()]),
            params.n_encoding_channels, params.n_static_channels,
            params.grid_type == 'uniform'
        )

    codano_model = SSLWrapper(params, encoder, decoder, contrastive, predictor, stage=stage)

    print("Setting the Grid")
    mesh = get_mesh(params)
    input_mesh = torch.from_numpy(mesh).float().cuda()
    codano_model.set_initial_mesh(input_mesh)

    codano_model.load_state_dict(torch.load("models/codano_whole_model_weights.pt"), strict=False)
    codano_model = codano_model.cuda().eval()
    #print(codano_model)



    if variable_encoder is not None:
        variable_encoder = variable_encoder.cuda().eval()
    if token_expander is not None:
        token_expander = token_expander.cuda().eval()

    dataset = NsElasticDataset(
        params.data_location,
        equation=list(params.equation_dict.keys()),
        mesh_location=params.input_mesh_location,
        params=params
    )
    _, test_dataloader = dataset.get_dataloader(
        params.mu_list, params.dt,
        ntrain=params.get('ntrain'),
        ntest=params.get('ntest'),
        batch_size = 1,
        train_test_split = 0.0001, # just for test
        sample_per_inlet=params.sample_per_inlet
    )

    grid_non, grid_uni = get_meshes(params, params.grid_size)
    test_augmenter = None

    print("Evaluating model on test dataset...")
    # missing_variable_testing(
    #     model,
    #     test_dataloader,
    #     augmenter=test_augmenter,
    #     stage=stage,
    #     params=params,
    #     variable_encoder=variable_encoder,
    #     token_expander=token_expander,
    #     initial_mesh=input_mesh
    # )

    # codano_evaluation_params = {"variable_encoder": variable_encoder,
    #                             "token_expander": token_expander,
    #                             "params": params,
    #                             "stage": stage,
    #                             "input_mesh":input_mesh}

    # lowrank_model = CompressedModel(
    #     model=codano_model,
    #     compression_technique=lambda model: SVDLowRank(model, 
    #                                                 rank_ratio=0.8, # option = [0.2, 0.4, 0.6, 0.8]
    #                                                 min_rank=16,
    #                                                 max_rank=256, # option = [8, 16, 32, 64, 128, 256]
    #                                                 is_full_rank=False,
    #                                                 is_compress_conv1d=True,
    #                                                 is_compress_FC=True,
    #                                                 is_comrpess_spectral=True),
    #     create_replica=True
    # )

    # lowrank_model = lowrank_model.to(device)

    # print("\n"*2)
    # print("Low Ranking.....")
    # compare_models(
    #     model1=codano_model,
    #     model2=lowrank_model,
    #     test_loaders=test_dataloader,
    #     data_processor=None,
    #     device=device,
    #     track_performance = True,
    #     evaluation_params = codano_evaluation_params
    # )

dynamic_quant_model = CompressedModel(
    model=codano_model,
    compression_technique=lambda model: DynamicQuantization(model),
    create_replica=True
)
dynamic_quant_model = dynamic_quant_model.to(device)

print("\n"*2)
print("Dynamic Quantization.....")
compare_models(
    model1=codano_model,               # The original model (it will be moved to CPU in evaluate_model)
    model2=dynamic_quant_model,     # The dynamically quantized model
    test_loaders=test_dataloader,
    data_processor=None,
    device=device
)