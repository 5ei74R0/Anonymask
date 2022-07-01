from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from lib.mae.models_mae import MaskedAutoencoderViT


class Inpainter():
    def __init__(self, checkpoints_path: str, device: torch.device = torch.device("cuda:0")):
        model = MaskedAutoencoder(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                                  decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                  mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        checkpoint = torch.load(checkpoints_path, map_location='cpu')
        model.load_state_dict(checkpoint["model"], strict=True)
        self.model = model.to(device)

    def __call__(self, x: np.ndarray, mask_pos: np.ndarray) -> np.ndarray:
        """ Execute inpainting
        Args:
            x (np.ndarray): a target image
            mask_pos (np.ndarray): positions of bboxes (size: [2,2])

        Returns:
            np.ndarray: inpainted image
        """
        # prepare input
        self.model.eval()
        self.model.set_mask_position(mask_pos)
        x = torch.Tensor(x).cuda()
        x = x.permute(2, 0, 1).unsqueeze(0)

        # inference
        _, y, mask = self.model(x)
        y = self.model.unpatchify(y)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.model.patch_size**2 * 3)
        mask = self.model.unpatchify(mask).detach()
        z = x * (1 - mask) + y * mask
        img = z.detach().cpu().numpy()[0, :, :, :].transpose(1, 2, 0)

        return img


class MaskedAutoencoder(MaskedAutoencoderViT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            decoder_embed_dim,
            decoder_depth,
            decoder_num_heads,
            mlp_ratio,
            norm_layer,
            norm_pix_loss)

        self.patch_size = patch_size
        self.img_size = img_size

    def set_mask_position(self, pos: np.ndarray):
        self.pos = pos

    def make_noise(self, shape: Tuple[int, int, int], device: torch.device):
        N, L, _ = shape
        p = self.patch_size
        x1, y1 = self.pos[0, :]
        x2, y2 = self.pos[1, :]
        H, W = self.img_size // p, self.img_size // p

        noise = torch.zeros((N, L))
        ids = {k: set() for k in ["mask", "nonmask"]}
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                n, m = x // p, y // p
                z = W * n + m
                noise[0, z] = 1
                ids["mask"].add(z)

        ids["nonmask"] = set(range(W * H)) - ids["mask"]
        return noise.to(device), ids

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        N, L, D = x.shape
        assert N == 1, "Only support batch size 1"

        # noise in [0, 1]: 1 is masked
        noise, ids = self.make_noise(x.shape, device=x.device)
        len_keep = L - int(noise.sum())
        ids_shuffle = list(ids["nonmask"]) + sorted(list(ids["mask"]))
        ids_shuffle = torch.Tensor(ids_shuffle).to(torch.int64).unsqueeze(0).to(x.device)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
