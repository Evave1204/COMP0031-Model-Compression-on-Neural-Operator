import torch
import torch.nn as nn

class GlobalLayerPruning:
    def __init__(self, model):
        """
        Initialize with the model you wish to prune.
        """
        self.model = model

    def layer_prune(self, prune_ratio=0.2):
        """
        Prune entire layers based on an importance score.
        
        The importance score is computed as the average absolute value of the layer's weights.
        Layers with the lowest scores (i.e. least important) are replaced with nn.Identity().
        
        Parameters:
            prune_ratio (float): Fraction of candidate layers to prune (e.g., 0.2 to prune 20%).
        
        Returns:
            List of names of layers that were pruned.
        """
        # Collect candidate layers (example: nn.Linear and nn.Conv2d)
        candidate_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                candidate_layers.append((name, module))
                
        if not candidate_layers:
            print("No candidate layers for pruning found.")
            return []

        # Compute importance scores for each candidate layer
        scores = []
        for name, module in candidate_layers:
            if hasattr(module, 'weight') and module.weight is not None:
                score = torch.mean(torch.abs(module.weight)).item()
                scores.append((name, module, score))
            else:
                # If there is no weight, assign an infinite score to skip pruning it.
                scores.append((name, module, float('inf')))

        # Sort layers by importance score (least important first)
        scores.sort(key=lambda x: x[2])
        num_to_prune = int(len(scores) * prune_ratio)
        pruned_layers = scores[:num_to_prune]
        pruned_names = [entry[0] for entry in pruned_layers]

        # Replace selected layers with an identity mapping
        for name, module, score in pruned_layers:
            parts = name.split('.')
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            last_part = parts[-1]
            setattr(parent, last_part, nn.Identity())

        print(f"Pruned layers: {pruned_names}")
        return pruned_names

# Demonstration if run as a standalone script.
if __name__ == "__main__":
    # Define a simple model for demonstration purposes.
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x, conv_input):
            x = self.fc1(x)
            conv_out = self.conv1(conv_input)
            x = self.fc2(x)
            return x, conv_out

    # Instantiate and display the model before pruning.
    model = SimpleModel()
    print("Model before pruning:")
    print(model)

    # Create a pruner and prune 50% of candidate layers.
    pruner = GlobalLayerPruning(model)
    pruner.layer_prune(prune_ratio=0.5)

    print("\nModel after pruning:")
    print(model)
