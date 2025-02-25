import torch
import torch.nn as nn
from typing import Dict, Optional
from compression.base import CompressionTechnique

class StaticQuantization(CompressionTechnique):
    """
    Post-training static quantization technique.
    Inserts observers, calibrates activation ranges on a small dataset,
    and converts the model to int8 for efficient inference on CPU.
    """
    def __init__(
        self, 
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        data_processor: Optional[nn.Module] = None,
        num_calibration_batches: int = 8,
        backend: str = 'qnnpack'
    ):
        self.model = model
        self.calibration_loader = calibration_loader
        self.data_processor = data_processor
        self.num_calibration_batches = num_calibration_batches
        self.backend = backend

    def compress(self) -> None:
        # 1) Move model to CPU & set eval
        self.model.cpu()
        self.model.eval()

        # 2) Assign a default QConfig for the chosen backend
        torch.backends.quantized.engine = self.backend
        self.model.qconfig = torch.quantization.get_default_qconfig(self.backend)

        # 3) Prepare the model
        torch.quantization.prepare(self.model, inplace=True)

        # 4) Calibration
        with torch.no_grad():
            if self.data_processor is not None:
                self.data_processor.cpu()
                self.data_processor.eval()

            for batch_idx, batch_data in enumerate(self.calibration_loader):
                if batch_idx >= self.num_calibration_batches:
                    break

                # First move raw batch to CPU
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor):
                        batch_data[k] = v.cpu()

                # Preprocess
                if self.data_processor is not None:
                    processed_data = self.data_processor.preprocess(batch_data)
                else:
                    processed_data = batch_data

                # --- NEW STEP: Move the result back to CPU again ---
                for k, v in processed_data.items():
                    if isinstance(v, torch.Tensor):
                        processed_data[k] = v.cpu()

                # Now the model forward pass uses CPU inputs
                _ = self.model(**processed_data)


        # 5) Convert to int8
        torch.quantization.convert(self.model, inplace=True)

    def get_compression_stats(self) -> Dict[str, float]:
        # Include a sparsity key so compare_models doesn't crash
        return {
            "sparsity": 0.0,
            # add anything else you want
        }
