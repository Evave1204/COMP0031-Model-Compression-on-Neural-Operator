import matplotlib.pyplot as plt
import numpy as np

def generate_graph(results : dict, x_points : list, model_name : str, measureable : str, unit : str, savefile : str = None):
    metrics = list(next(iter(next(iter(next(iter(results.values())).values())).values())).keys())
    temp_results = results.copy()
    for model in results.keys():
        if results[model] == {}:
            temp_results.pop(model)
    results = temp_results
    num_metrics = len(metrics)
    num_cols = 2
    num_rows = (num_metrics + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    fig.suptitle(f'Comparison of {model_name.replace("_"," ").title()} Compression Performance', fontsize=16)
    axes = axes.flatten()

    markers = ['o', 's', 'D', '^', 'v']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f1c40f']

    for i, metric in enumerate(metrics):
        for j, model in enumerate(results.keys()):
            compressed_values = [results[model][x_point][key][metric] 
                        for x_point in x_points 
                        for key in results[model][x_point].keys() 
                        if "compressed" in key]

            ax = axes[i]
            ax.plot(compressed_values, np.array(x_points) * 100,
                    f'--{markers[j]}', label=f'{model} {metric}', 
                    color=colors[j], alpha=0.7)
        
        ax.set_ylabel(f'{model_name.replace("_"," ").title()} {measureable} ({unit})')
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust the spacing between subplots
    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'compression/utils/all_{model_name}_performance.png', dpi=300, bbox_inches='tight')
    plt.show()