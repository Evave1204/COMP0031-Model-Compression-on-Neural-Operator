import matplotlib.pyplot as plt
import numpy as np

def clean_compare(data):
    reformatted_data = {}
    for key, value in data.items():
        resolution, model_type = key.split('_')
        if resolution not in reformatted_data:
            reformatted_data[resolution] = {}
        if model_type not in reformatted_data[resolution]:
            reformatted_data[resolution][model_type] = {}
        reformatted_data[resolution][model_type] = value
    return reformatted_data

def plot_compare(data, compression_stats):
    resolutions = sorted(data.keys())
    metrics = list(next(iter(next(iter(data.values())).values())).keys())

    num_metrics = len(metrics)
    num_cols = 2 
    num_rows = (num_metrics + num_cols - 1) // num_cols  
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten() 
    
    for i, metric in enumerate(metrics):
        base_values = [data[res]['base'][metric] for res in resolutions]
        compressed_values = [data[res]['compressed'][metric] for res in resolutions]
        
        ax = axes[i]
        ax.plot(resolutions, base_values, label='Base Model', color='blue', marker='o')
        ax.plot(resolutions, compressed_values, label='Compressed Model', color='orange', marker='o')
        
        ax.set_xlabel('Resolution')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()