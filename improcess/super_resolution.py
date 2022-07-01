import numpy as np
import torch
from lib.swinir.models.network_swinir import SwinIR


class Upsampler:
    def __init__(self, weights: str, sr_scale: int = 4) -> None:
        self.model = SwinIR(
            upscale=sr_scale,
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
        )
        pretrained_model = torch.load(weights)
        self.model.load_state_dict(
            pretrained_model["params_ema"]
            if "params_ema" in pretrained_model.keys()
            else pretrained_model,
            strict=True,
        )

    def upsample(self, low_quality_img: np.ndarray) -> np.ndarray:
        # print(low_quality_img.shape)
        # exit(1)
        img_tensor: torch.Tensor = self.model(torch.tensor(low_quality_img).permute(2, 0, 1).unsqueeze(0))
        return img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
