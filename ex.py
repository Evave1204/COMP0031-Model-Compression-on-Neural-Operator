import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def generate_graph(results: dict, hyperparameters: dict, model_name: str, measureable: str, unit: str, savefile: str = None):
    metrics = ["l2_loss_increase", "model_size_reduction", 
               "run_time_speed_up", "peak_memory_reduction", "flops_reduction"]  
    fig = plt.figure(figsize=(15, 10)) 
    fig.suptitle(f'Comparison of {model_name.replace("_"," ").title()} Compression Performance', fontsize=16)

    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])  

    markers = ['o', 's', 'D', '^', 'v', 'p', '*']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f1c40f', '#ff5733', '#1abc9c'] 

    plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (1,2)]  
    axes = []  

    global_handles = [] 
    global_labels = []  

    for i, metric in enumerate(metrics):
        row, col = plot_positions[i]  
        ax = fig.add_subplot(gs[row, col]) 
        axes.append(ax)

        for j, model in enumerate(results.keys()):
            if model not in hyperparameters:
                continue  
            
            x_values = hyperparameters[model] 
            y_values = [
                results[model].get(ratio, {}).get("Comparison", {}).get(metric, None) 
                for ratio in x_values
            ]

            x_filtered = [x for x, y in zip(x_values, y_values) if y is not None]
            y_filtered = [y for y in y_values if y is not None]

            if not x_filtered or not y_filtered:
                print(f"Warning: No data found for {model} at {metric}")
                continue  

            line, = ax.plot(np.array(x_filtered) * 100, y_filtered,
                            f'--{markers[j]}', label=f'{model}',
                            color=colors[j % len(colors)], alpha=0.7)

            if i == 0 and line not in global_handles:
                global_handles.append(line)
                global_labels.append(model)

        ax.set_xlabel(f'{model_name.replace("_"," ").title()} {measureable} ({unit})')
        ax.set_ylabel(f"{metric.replace('_', ' ').title()} Ratio (%)")
        ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
        ax.grid(True, linestyle='--', alpha=0.7)

    if global_handles:
        fig.legend(handles=global_handles, labels=global_labels, 
                   loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=len(global_labels),
                   fontsize=12, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    # Determine the file path and create the directory if needed
    if savefile:
        save_path = savefile
    else:
        save_path = f'compression/utils/all_{model_name}_performance.png'
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Simulated hyperparameters (pruning ratios as decimals)
hyperparams_FNO = {"FNO": [0.1, 0.2, 0.3, 0.4, 0.5]}
hyperparams_DeepONet = {"DeepONet": [0.1, 0.2, 0.3, 0.4, 0.5]}

# Simulated results for FNO (values in percentages)
results_FNO = {"FNO": {
    0.1: {"Comparison": {
        "l2_loss_increase": 2,
        "model_size_reduction": 10,
        "run_time_speed_up": 5,
        "peak_memory_reduction": 8,
        "flops_reduction": 15,
    }},
    0.2: {"Comparison": {
        "l2_loss_increase": 4,
        "model_size_reduction": 20,
        "run_time_speed_up": 10,
        "peak_memory_reduction": 16,
        "flops_reduction": 30,
    }},
    0.3: {"Comparison": {
        "l2_loss_increase": 6,
        "model_size_reduction": 30,
        "run_time_speed_up": 15,
        "peak_memory_reduction": 24,
        "flops_reduction": 45,
    }},
    0.4: {"Comparison": {
        "l2_loss_increase": 8,
        "model_size_reduction": 40,
        "run_time_speed_up": 20,
        "peak_memory_reduction": 36,
        "flops_reduction": 60,
    }},
    0.5: {"Comparison": {
        "l2_loss_increase": 10,
        "model_size_reduction": 50,
        "run_time_speed_up": 25,
        "peak_memory_reduction": 50,
        "flops_reduction": 75,
    }},
}}

# Simulated results for DeepONet (values in percentages)
results_DeepONet = {"DeepONet": {
    0.1: {"Comparison": {
        "l2_loss_increase": 1,
        "model_size_reduction": 10,
        "run_time_speed_up": 3,
        "peak_memory_reduction": 10,
        "flops_reduction": 20,
    }},
    0.2: {"Comparison": {
        "l2_loss_increase": 2,
        "model_size_reduction": 20,
        "run_time_speed_up": 6,
        "peak_memory_reduction": 20,
        "flops_reduction": 40,
    }},
    0.3: {"Comparison": {
        "l2_loss_increase": 3,
        "model_size_reduction": 30,
        "run_time_speed_up": 9,
        "peak_memory_reduction": 30,
        "flops_reduction": 60,
    }},
    0.4: {"Comparison": {
        "l2_loss_increase": 4,
        "model_size_reduction": 40,
        "run_time_speed_up": 12,
        "peak_memory_reduction": 40,
        "flops_reduction": 80,
    }},
    0.5: {"Comparison": {
        "l2_loss_increase": 5,
        "model_size_reduction": 50,
        "run_time_speed_up": 15,
        "peak_memory_reduction": 50,
        "flops_reduction": 100,
    }},
}}

# Generate the graph for FNO
generate_graph(results_FNO, hyperparams_FNO, model_name="FNO", measureable="Pruning Ratio", unit="%")

# Generate the graph for DeepONet
generate_graph(results_DeepONet, hyperparams_DeepONet, model_name="DeepONet", measureable="Pruning Ratio", unit="%")
