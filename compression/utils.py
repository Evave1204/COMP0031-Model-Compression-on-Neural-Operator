import torch
from neuralop.losses import LpLoss, H1Loss

def evaluate_model(model, dataloader, data_processor, device='cuda'):
    """
    Evaluates model performance in a way that exactly mirrors the Trainer's behavior.
    
    This implementation uses loss functions set to `reduction='mean'` to match the 
    trainer's default behavior. The loss is averaged over both the batch and spatial 
    dimensions.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    dataloader : torch.utils.data.DataLoader
        The DataLoader for evaluation.
    data_processor : Module or None
        Optional data processor for any preprocessing/postprocessing.
    device : str
        The device on which to run the evaluation (default 'cuda').
        
    Returns
    -------
    dict
        Dictionary containing the averaged 'l2_loss' and 'h1_loss'.
    """
    model.eval()
    total_l2_loss = 0.0
    total_h1_loss = 0.0

    model = model.to(device)
    if data_processor is not None:
        data_processor = data_processor.to(device)
        data_processor.eval()

    l2_loss = LpLoss(d=2, p=2, reduction='mean')
    h1_loss = H1Loss(d=2, reduction='mean')

    with torch.no_grad():
        for batch in dataloader:
            if data_processor is not None:
                processed_data = data_processor.preprocess(batch)
            else:
                processed_data = {k: v.to(device)
                                  for k, v in batch.items() if torch.is_tensor(v)}

            out = model(**processed_data)

            if data_processor is not None:
                out, processed_data = data_processor.postprocess(out, processed_data)

            total_l2_loss += l2_loss(out, processed_data['y']).item()
            total_h1_loss += h1_loss(out, processed_data['y']).item()

    avg_l2_loss = total_l2_loss / len(dataloader)
    avg_h1_loss = total_h1_loss / len(dataloader)

    return {'l2_loss': avg_l2_loss, 'h1_loss': avg_h1_loss}


def compare_models(model1, model2, test_loaders, data_processor, device, 
                  model1_name="Original Model", model2_name="Compressed Model",
                  verbose=True):
    """Compare performance between two models across different resolutions.
    
    Args:
        model1: First model to evaluate (e.g., original model)
        model2: Second model to evaluate (e.g., compressed model)
        test_loaders: Dict of test loaders for different resolutions
        data_processor: Data processor for the dataset
        device: Device to run evaluation on
        model1_name: Name for the first model (default: "Original Model")
        model2_name: Name for the second model (default: "Compressed Model")
        verbose: Whether to print detailed results (default: True)
    """
    results = {}
    
    if verbose:
        print("\n" + "="*50)
        print(f"{model1_name.upper()} EVALUATION")
        print("="*50)
    
    for resolution, loader in test_loaders.items():
        if verbose:
            print(f"\nResults on {resolution}x{resolution} resolution")
            print("-"*30)
        results[f"{resolution}_base"] = evaluate_model(model1, loader, data_processor, device)
        if verbose:
            print(f"L2 Loss: {results[f'{resolution}_base']['l2_loss']:.6f}")
            print(f"H1 Loss: {results[f'{resolution}_base']['h1_loss']:.6f}")
    
    if verbose:
        print("\n" + "="*50)
        print(f"{model2_name.upper()} EVALUATION")
        print("="*50)
    
    if hasattr(model2, 'get_compression_stats') and verbose:
        stats = model2.get_compression_stats()
        print(f"\nModel sparsity: {stats['sparsity']:.2%}")
        # # for dynamic quantization method, we need compare model size
        # print(f"Original model size: {stats['original_size'] / (1024*1024):.2f} MB")
        # print(f"Quantized model size: {stats['quantized_size'] / (1024*1024):.2f} MB")
        # print(f"Compression ratio: {stats['compression_ratio']:.2%}")
    
    for resolution, loader in test_loaders.items():
        if verbose:
            print(f"\nResults on {resolution}x{resolution} resolution")
            print("-"*30)
        results[f"{resolution}_compressed"] = evaluate_model(model2, loader, data_processor, device)
        if verbose:
            print(f"L2 Loss: {results[f'{resolution}_compressed']['l2_loss']:.6f}")
            print(f"H1 Loss: {results[f'{resolution}_compressed']['h1_loss']:.6f}")
    
    if verbose:
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        print("\nRelative increase in error (compressed vs original):")
        print("-"*50)
    
        for resolution in test_loaders.keys():
            base_results = results[f"{resolution}_base"]
            comp_results = results[f"{resolution}_compressed"]
            print(f"{resolution}x{resolution} - L2: {(comp_results['l2_loss']/base_results['l2_loss'] - 1)*100:.2f}%")
            print(f"{resolution}x{resolution} - H1: {(comp_results['h1_loss']/base_results['h1_loss'] - 1)*100:.2f}%")
    
    return results