import numpy as np
import torch
from lib.swinir.models.network_swinir import SwinIR


class Upsampler:
    def __init__(self, weights: str, device: torch.device) -> None:
        model = SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="1conv",
        ).to(device)
        pretrained_model = torch.load(weights)
        model.load_state_dict(
            pretrained_model["params_ema"]
            if "params_ema" in pretrained_model.keys()
            else pretrained_model,
            strict=True,
        )
        self.model = model
        self.device = device

    def upsample(self, low_quality_img: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.tensor(low_quality_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        y = self.model(x)
        return y.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
