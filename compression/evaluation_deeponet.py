import sys
sys.path.append("C:\\Users\\ahmed\\UCL-Ibsa-Ahmed\\Year_3\\COMP0030-31\\COMP0031-Model-Compression-on-Neural-Operator")
from neuralop.models import DeepONet
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.UniformQuant.uniform_quant import UniformQuantisation
from compression.base import CompressedModel
from neuralop.data.datasets import load_darcy_flow_small
from compression.utils import evaluate_model, compare_models
import matplotlib.pyplot as plt

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

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=16,
    test_resolutions=[128],
    n_tests=[100],
    test_batch_sizes=[16, 16],
    encode_input=False, 
    encode_output=False,
)

'''pruned_model = CompressedModel(
    model=deeponet_model,
    compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.5),
    create_replica=True
)
pruned_model = pruned_model.to(device)

lowrank_model = CompressedModel(
    model=deeponet_model,
    compression_technique=lambda model: SVDLowRank(model, rank_ratio=0.5,
                                                 min_rank=4, max_rank=128),
    create_replica=True
)
lowrank_model = lowrank_model.to(device)'''

quantised_model = CompressedModel(
    model=deeponet_model,
    compression_technique=lambda model: UniformQuantisation(model, num_bits=8),
    create_replica=True
)

'''print("Pruning.....")
compare_models(
    model1=deeponet_model,
    model2=pruned_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)

print("Low Ranking.....")
lowrank_compare = compare_models(
    model1=deeponet_model,
    model2=lowrank_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)'''

print("Quantising.....")
quantised_compare = compare_models(
    model1=deeponet_model,
    model2=quantised_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)

print(quantised_compare)

def plot_accuracy(*args):
    models = ['Original', 'Quantised']
    accuracies = [1-args[0]['128_base']['l2_loss']]
    for each in args:
        accuracies.append(1-each['128_compressed']['l2_loss'])

    plt.figure(figsize=(10, 6))
    #plt.bar(models, accuracies, color=['blue', 'orange'])
    print(models, accuracies)
    plt.plot(models, accuracies, color='blue')
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies Comparison')
    plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
    plt.show()

# Call the plot_accuracy function with the comparison results
plot_accuracy(quantised_compare)