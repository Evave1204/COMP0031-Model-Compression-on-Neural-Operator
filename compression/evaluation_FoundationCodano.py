from neuralop.models.codano_gino import CodANO
import argparse
import torch
import random
import numpy as np
from neuralop.models.get_models import *
from neuralop.layers.variable_encoding import *
from neuralop.data_utils.data_utils import MaskerNonuniformMesh, get_meshes
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.base import CompressedModel
from neuralop.data.datasets import load_darcy_flow_small
from compression.utils import evaluate_model, compare_models, CodanoYParams
from neuralop.data.transforms.codano_processor import CODANODataProcessor
# parse config
parser = argparse.ArgumentParser()
parser.add_argument("--exp", nargs="?", default="FSI", type=str)
parser.add_argument("--config", nargs="?", default="codano_gno_NS_ES", type=str)
parser.add_argument("--ntrain", nargs="?", default=None, type=int)
parser.add_argument("--epochs", nargs="?", default=None, type=int)
parser.add_argument("--random_seed", nargs="?", default=42, type=int)
parser.add_argument("--scheduler_step", nargs="?", default=None, type=int)
parser.add_argument("--batch_size", nargs="?", default=None, type=int)
parsed_args = parser.parse_args()

config_file = "config/ssl_ns_elastic.yaml"
config = parsed_args.config
print("Loading config", config)
params = CodanoYParams(config_file, config, print_params=True)

# set random seeds
if parsed_args.random_seed is not None:
    params.random_seed = parsed_args.random_seed
    print("Overriding random seed to", params.random_seed)
torch.manual_seed(params.random_seed)
random.seed(params.random_seed)
np.random.seed(params.random_seed)

params.config = config

if params.pretrain_ssl:
    stage = StageEnum.RECONSTRUCTIVE
else:
    stage = StageEnum.PREDICTIVE

if params.nettype == 'transformer':
    if params.grid_type == 'uniform':
        encoder, decoder, contrastive, predictor = get_ssl_models_codano(params)
    else:
        encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(params)

    model = SSLWrapper(params, encoder, decoder, contrastive, predictor)

elif params.nettype in ['simple', 'gnn', 'deeponet', 'vit', 'unet', 'fno']:
    model = get_baseline_model(params)

print("model loaded success") 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cpu_device = torch.device('cpu')
# fno_model.load_model(torch.load("models/codano_whole_model_weights.pt", weights_only=False))
# fno_model.eval()
# fno_model = fno_model.to(device)
# print(fno_model)