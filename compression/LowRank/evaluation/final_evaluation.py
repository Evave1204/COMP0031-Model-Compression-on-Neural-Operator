from neuralop.models import FNO
import torch
from neuralop.data.datasets.darcy import load_darcy_flow_small_validation_test
import matplotlib.pyplot as plt
import random
import numpy as np
from neuralop.models.get_models import *
from neuralop.layers.variable_encoding import *
from neuralop.data.datasets.ns_dataset import *
from neuralop.data_utils.data_utils import MaskerNonuniformMesh, get_meshes
from compression.utils.codano_util import missing_variable_testing, CodanoYParams
from neuralop.models.foundation_fno import fno
from compression.LowRank.SVD_LowRank import SVDLowRank

from compression.base import CompressedModel
from compression.utils.fno_util import FNOYParams
from neuralop.data.datasets.mixed import get_data_val_test_loader
import argparse
import os
import logging
import torch.distributed as dist
from neuralop.models import CODANO
from neuralop.data.transforms.codano_processor import CODANODataProcessor
from neuralop.models import DeepONet

from compression.utils.evaluation_util import evaluate_model, compare_models
from compression.utils.fno_util import optional_fno
from compression.utils.graph_util import generate_graph

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------- INIT FFNO MODEL 
# ffnoparser = argparse.ArgumentParser()
# ffnoparser.add_argument("--yaml_config", default='config/mixed_config.yaml', type=str)
# ffnoparser.add_argument("--config", default='fno-foundational', type=str)
# ffnoparser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
# ffnoparser.add_argument("--run_num", default='0', type=str, help='sub run config')
# ffnoparser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
# ffnoargs = ffnoparser.parse_args()
# ffnoparams = FNOYParams(os.path.abspath(ffnoargs.yaml_config), ffnoargs.config, print_params=False)
# if dist.is_initialized():
#     dist.barrier()
# logging.info('DONE')
# ffnoparams['global_batch_size'] = params.batch_size
# ffnoparams['local_batch_size'] = int(ffnoparams.batch_size)

# ffnoparams['global_valid_batch_size'] = ffnoparams.valid_batch_size
# ffnoparams['local_valid_batch_size'] = int(ffnoparams.valid_batch_size)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cpu_device = torch.device('cpu')


# ffno_model = fno(ffnoparams).to(device)

# checkpoints = torch.load("models/ckpt_best.tar", map_location='cpu', weights_only=False)  
# ffno_model.load_state_dict(checkpoints['model_state'])
# ffno_model.eval()

# ffnovalidation_dataloaders, ffnotest_loaders, ffnodata_processor = get_data_val_test_loader(ffnoparams,
#                                                                 ffnoparams.test_path, 
#                                                                 dist.is_initialized(), 
#                                                                 train=False, 
#                                                                 pack=ffnoparams.pack_data)

# ------ INIT FCODANO MODEL
# fcodanoparser = argparse.ArgumentParser()
# fcodanoparser.add_argument("--exp", nargs="?", default="FSI", type=str)
# fcodanoparser.add_argument("--config", nargs="?", default="codano_gno_NS_ES", type=str)
# fcodanoargs = fcodanoparser.parse_args()
# fcodanoconfig_file = './config/ssl_ns_elastic.yaml'
# print("Loading config", fcodanoargs.config)
# fcodanoparams = CodanoYParams(fcodanoconfig_file, fcodanoargs.config, print_params=True)
# torch.manual_seed(fcodanoparams.random_seed)
# random.seed(fcodanoparams.random_seed)
# np.random.seed(fcodanoparams.random_seed)
# fcodanoparams.config = fcodanoargs.config
# stage = StageEnum.RECONSTRUCTIVE
# #stage = StageEnum.RECONSTRUCTIVE
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fcodanovariable_encoder = None
# fcodanotoken_expander = None

# fcodanoencoder, fcodanodecoder, fcodanocontrastive, fcodanopredictor = get_ssl_models_codano_gino(fcodanoparams)
# if fcodanoparams.use_variable_encoding:
#     fcodanovariable_encoder = get_variable_encoder(fcodanoparams)
#     fcodanotoken_expander = TokenExpansion(
#         sum([fcodanoparams.equation_dict[i] for i in fcodanoparams.equation_dict.keys()]),
#         fcodanoparams.n_encoding_channels, fcodanoparams.n_static_channels,
#         fcodanoparams.grid_type == 'uniform'
#     )

# fcodano_model = SSLWrapper(fcodanoparams, fcodanoencoder, fcodanodecoder, fcodanocontrastive, fcodanopredictor, stage=stage)

# print("Setting the Grid")
# fcodanomesh = get_mesh(fcodanoparams)
# fcodanoinput_mesh = torch.from_numpy(fcodanomesh).type(torch.float).cuda()
# fcodano_model.set_initial_mesh(fcodanoinput_mesh)

# stage = StageEnum.PREDICTIVE
# fcodano_model.load_state_dict(torch.load(fcodanoparams.model_path), strict=False)
# fcodano_model = fcodano_model.cuda().eval()
# print(fcodano_model)

# fcodanovariable_encoder.load_encoder(
#                 "NS", fcodanoparams.NS_variable_encoder_path)

# fcodanovariable_encoder.load_encoder(
#             "ES", fcodanoparams.ES_variable_encoder_path)

# if vfcodanoariable_encoder is not None:
#     fcodanovariable_encoder = fcodanovariable_encoder.cuda().eval()
# if fcodanotoken_expander is not None:
#     fcodanotoken_expander = fcodanotoken_expander.cuda().eval()

# fcodanodataset = NsElasticDataset(
#     fcodanoparams.data_location,
#     equation=list(fcodanoparams.equation_dict.keys()),
#     mesh_location=fcodanoparams.input_mesh_location,
#     params=fcodanoparams
# )

# fcodanovalidation_dataloader, fcodanotest_dataloader = dataset.get_validation_test_dataloader(
#     fcodanoparams.mu_list, fcodanoparams.dt,
#     ntrain=fcodanoparams.get('ntrain'),
#     ntest=40,
#     batch_size = 1,
#     train_test_split = 0.1, # just for test
#     sample_per_inlet=fcodanoparams.sample_per_inlet
# )

# fcodanogrid_non, fcodanogrid_uni = get_meshes(params, params.grid_size)
# fcodanotest_augmenter = None

# codano_evaluation_params = {"variable_encoder": fcodanovariable_encoder,
#                             "token_expander": fcodanotoken_expander,
#                             "params": fcodanoparams,
#                             "stage": stage,
#                             "input_mesh":fcodanoinput_mesh}

# ------ INIT CODANO MODEL
codano_model = CODANO(
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
cpu_device = torch.device('cpu')
codano_model.load_model(torch.load("models/model-codano-darcy-16-resolution-2025-02-11-21-13.pt", weights_only=False))
codano_model.eval()
codano_model = codano_model.to(device)

codanovalidation_loaders, codanotest_loaders, codanodata_processor = load_darcy_flow_small_validation_test(
    n_train=10,
    batch_size=16,
    test_resolutions=[16],
    n_tests=[10, 5],
    test_batch_sizes=[16, 16],
    encode_input=False, 
    encode_output=False,
)

codanotest_loader_16 = codanotest_loaders[16]

# When creating data processor, use the imported class
data_processor = CODANODataProcessor(
    in_normalizer=codanodata_processor.in_normalizer,
    out_normalizer=codanodata_processor.out_normalizer
)

# ------ INIT DEEPO MODEL
deeponet_model = DeepONet(
    train_resolution=128,
    in_channels=1,
    out_channels=1, 
    hidden_channels=64,
    branch_layers=[256, 256, 256, 256, 128],
    trunk_layers=[256, 256, 256, 256, 128],
    positional_embedding='grid',
    non_linearity='gelu',
    norm='instance_norm',
    dropout=0.1
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deeponet_model.load_state_dict(torch.load("models/model-deeponet-darcy-128-resolution-2025-02-19-22-23.pt", weights_only=False))
deeponet_model.eval()
deeponet_model = deeponet_model.to(device)

validation_loaders_deeponet, test_loaders_deeponet, data_processor_deeponet = load_darcy_flow_small_validation_test(
    n_train=10,
    batch_size=4,
    test_resolutions=[128],
    n_tests=[10],
    test_batch_sizes=[4, 4],
    encode_input=False, 
    encode_output=False,
)

# ------ INIT FNO 16x16 MODEL
fno_model_16, validation_loaders_fno16, test_loaders_fno16, data_processor_fno16 = optional_fno(resolution="low")
fno_model_16 = fno_model_16.to(device)

# ------ INIT FNO 32x32 MODEL
fno_model_32, validation_loaders_fno32, test_loaders_fno32, data_processor_fno32 = optional_fno(resolution="medium")
fno_model_16 = fno_model_16.to(device)

# ------ INIT FNO 128x128 MODEL
fno_model_128, validation_loaders_fno128, test_loaders_fno128, data_processor_fno128 = optional_fno(resolution="high")
fno_model_128 = fno_model_128.to(device)



# ------ INIT INFO
hyperparameters = {
    "FNO 16x16": [0.5, 0.55, 0.6, 0.65, 0.7],
    "FNO 32x32": [0.5, 0.55, 0.6, 0.65, 0.7],
    "DeepONet": [0.8, 0.85, 0.9, 0.95, 0.99]
}
number_of_hyperparameters = 5
results_by_model = {'FNO 16x16': {}, 'FNO 32x32': {}, 'FNO 128x128': {}, 'Codano': {},
                    'DeepONet': {}, 'Foundation FNO': {}, 'Foundation Codano': {}}

# ------ RUN COMPARISON
for i in range(number_of_hyperparameters):

    # ffnolowrank_model = CompressedModel(
    #     model=ffno_model,
    #     compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=ratio),
    #     create_replica=True
    # )

    # ffnolowrank_model = ffnolowrank_model.to(device)

    # ffnocompare = compare_models(
    #     model1=ffno_model,
    #     model2=ffnolowrank_model,
    #     test_loaders=test_loaders,
    #     data_processor=data_processor,
    #     device=device,
    #     track_performance = True
    # )

    # fcodanolowrank_model = CompressedModel(
    #     model=fcodano_model,
    #     compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=ratio),
    #     create_replica=True
    # )
    # fcodanolowrank_model = fcodanolowrank_model.to(device)

    # fcodano_compare = compare_models(
    #     model1=fcodano_model,
    #     model2=fcodanolowrank_model,
    #     test_loaders=fcodanotest_dataloader,
    #     data_processor=None,
    #     device=device,
    #     track_performance = True,
    #     evaluation_params = codano_evaluation_params
    # )
    # results_by_model["fcodano"][ratio] = fcodano_compare

    # codanolowrank_model = CompressedModel(
    #     model=fno_model,
    #     compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=ratio),
    #     create_replica=True
    # )
    # codanolowrank_model = codanolowrank_model.to(device)

    # codano_compare = compare_models(
    #     model1=codano_model,
    #     model2=codanolowrank_model,
    #     test_loaders=codanotest_loaders,
    #     data_processor=codanodata_processor,
    #     device=device,
    #     track_performance=True
    # )
    # results_by_model["codano"][ratio] = codano_compare

# ------------------------------------- DeepONet ---------------------------------------
    ratio = hyperparameters["DeepONet"][i]
    deepolowrank_model = CompressedModel(
        model=deeponet_model,
        compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio),
        create_replica=True
    )
    deepolowrank_model = deepolowrank_model.to(device)

    deepo_compare = compare_models(
        model1=deeponet_model,
        model2=deepolowrank_model,
        test_loaders=validation_loaders_deeponet,
        data_processor=data_processor_deeponet,
        device=device,
        track_performance = True
    )
    results_by_model["DeepONet"][ratio] = deepo_compare

# ------------------------------------- FNO 16 ---------------------------------------
    ratio = hyperparameters["FNO 16x16"][i]
    fnolowrank_model_16 = CompressedModel(
        model=fno_model_16,
        compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio),
        create_replica=True
    )
    fnolowrank_model_16 = fnolowrank_model_16.to(device)

    fnocompare_16 = compare_models(
        model1=fno_model_16,
        model2=fnolowrank_model_16,
        test_loaders=validation_loaders_fno16,
        data_processor=data_processor_fno16,
        device=device,
        track_performance=True
    )
    results_by_model["FNO 16x16"][ratio] = fnocompare_16

# ------------------------------------- FNO 32 ---------------------------------------
    ratio = hyperparameters["FNO 32x32"][i]
    fnolowrank_model_32 = CompressedModel(
        model=fno_model_32,
        compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio),
        create_replica=True
    )
    fnolowrank_model_32 = fnolowrank_model_32.to(device)

    fnocompare_32 = compare_models(
        model1=fno_model_32,
        model2=fnolowrank_model_32,
        test_loaders=validation_loaders_fno32,
        data_processor=data_processor_fno32,
        device=device,
        track_performance=True
    )
    results_by_model["FNO 32x32"][ratio] = fnocompare_32

# ------------------------------------- Final Evaluation ---------------------------------------
generate_graph(results_by_model, hyperparameters, "SVD_low_rank", "Rank Ratio", "%", savefile="compression/LowRank/results/all_lowrank_performance.png")