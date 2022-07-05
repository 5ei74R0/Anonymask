

import cv2 as cv
import numpy as np
import torch
from cv2 import resize
from torchinfo import summary

from detection import Detector
from improcess.inpaint import Inpainter
from improcess.super_resolution import Upsampler
from improcess.utils import gaussian_filter, correct_img
from util.data import Checkpoints
from util.argparse import get_argparser, get_checkpoints
from test import test_integrate, test_mae, test_swinir, test_yolo


class Anonymask:
    def __init__(self, checkpoints: Checkpoints, device: str):
        print("Prepare models...")
        device = torch.device(device)
        self.detector = Detector(checkpoints.yolo, device=device)
        self.inpainter = Inpainter(checkpoints.mae, device=device)
        self.upsampler = Upsampler(checkpoints.swinir, sr_scale=4)
        self.device = device
        self.summary()

    def __call__(self, img: np.ndarray) -> np.ndarray:  # img should be RGB
        # init
        SEED = 42
        torch.manual_seed(SEED)
        torch.random.manual_seed(SEED)
        np.random.seed(SEED)
        H, W = img.shape[:2]

        # yolo detector
        print("Detecting...")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x = cv.resize(img, (640, 640))
        input_imgs: torch.Tensor = (
            torch.tensor(x).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        )

        # scale bboxes
        print("Process bounding boxes...")
        scale = 224. / 640.
        bboxes = self.detector.get_bboxes(input_imgs)
        if bboxes.sum() == 0:
            print("No bboxes")
            return img

        bboxes = bboxes[0, :, :]
        count = bboxes.shape[0]
        print(f"> found {count} bboxes !")
        for i in range(count):
            print(f"=========== {i+1} / {count} ===========")
            bbox = bboxes[i, :]
            bbox_224 = bbox.float() * scale
            bbox_224 = bbox_224.view(2, 2).detach().cpu().numpy()
            bbox_224 = bbox_224.astype(np.int32)[:, ::-1]

            # execute inpainter
            print("Execute inpainter...")
            x = resize(img, (224, 224))
            x = x.astype(np.float32) / 255.
            y = self.inpainter(x, bbox_224)

            # apply Gaussian filter
            print("Apply Gaussian filter...")
            y = gaussian_filter(y)
            y = correct_img(y)

            # apply super resolution
            print("Apply super resolution...")
            z = self.upsampler.upsample(y)  # upsample
            z = cv.resize(z, (W, H))  # downsample
            z = correct_img(z)

            # reconstruct image
            print("Reconstruct image...")
            bbox = bbox.flatten().detach().cpu().numpy()
            bbox[::2] = (bbox[::2] * W / 640.).astype(np.int32)
            bbox[1::2] = (bbox[1::2] * H / 640.).astype(np.int32)
            y1, x1, y2, x2 = list(bbox)
            img[x1:x2, y1:y2, :] = z[x1:x2, y1:y2, :] * 255

        # output
        print("Complete!")
        return img

    def summary(self):
        print("Summary of models:")
        models = {"detector": self.detector.model, "inpainter": self.inpainter.model, "upsampler": self.upsampler.model}
        for name, model in models.items():
            print(f"> {name}")
            summary(model)


def main():
    parser = get_argparser()
    args = parser.parse_args()
    checkpoints = get_checkpoints(args)
    anonymask = Anonymask(checkpoints, device=args.device)

    if args.mode == "train":
        pass
    elif args.mode == "prod":
        img = cv.imread(args.img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = anonymask(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imshow("img", img)
        cv.waitKey(10 * 1000)
    elif args.mode == "test_mae":
        test_mae(args)
    elif args.mode == "test_swinir":
        test_swinir(args)
    elif args.mode == "test_yolo":
        test_yolo(args)
    elif args.mode == "test_integrate":
        test_integrate(args)
    else:
        assert False, "Unknown mode"


if __name__ == "__main__":
    main()
