import argparse
from test import test_integrate, test_mae, test_swinir, test_yolo

import cv2 as cv
import numpy as np
import torch
from cv2 import resize

from detection import Detector
from improcess.inpaint import Inpainter
from improcess.super_resolution import Upsampler
from improcess.utils import gaussian_filter


def anonymask(args: argparse.Namespace):
    # init
    SEED = 42
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    img = cv.imread(args.img_path)
    H, W = img.shape[:2]

    # yolo detector
    print("Detecting...")
    det = Detector(args.yolo_checkpoint, device=device)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    x = cv.resize(img, (640, 640))
    input_imgs: torch.Tensor = (
        torch.tensor(x).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    )

    # scale bboxes
    print("Process bounding boxes...")
    scale = 224. / 640.
    bboxes = det.get_bboxes(input_imgs)
    if bboxes.sum() == 0:
        print("No bboxes")
        return
    bboxes = bboxes[0, :, :]
    bboxes_224 = bboxes.float() * scale
    bboxes_224 = bboxes_224.view(2, 2).detach().cpu().numpy()
    bboxes_224 = bboxes_224.astype(np.int32)[:, ::-1]

    # execute inpainter
    print("Execute inpainter...")
    inpainter = Inpainter(args.mae_checkpoint, device=device)
    x = resize(img, (224, 224))
    x = x.astype(np.float32) / 255.
    y = inpainter(x, bboxes_224)

    # apply Gaussian filter
    print("Apply Gaussian filter...")
    y = gaussian_filter(y)

    # apply super resolution
    print("Apply super resolution...")
    upsampler = Upsampler(args.swinir_checkpoint, sr_scale=4)
    z = upsampler.upsample(y)  # upsample
    z = cv.resize(z, (W, H))  # downsample

    # reconstruct image
    print("Reconstruct image...")
    bboxes = bboxes.flatten().detach().cpu().numpy()
    bboxes[::2] = (bboxes[::2] * W / 640.).astype(np.int32)
    bboxes[1::2] = (bboxes[1::2] * H / 640.).astype(np.int32)
    y1, x1, y2, x2 = list(bboxes)
    img[x1:x2, y1:y2, :] = z[x1:x2, y1:y2, :] * 255

    # output
    print("Complete!")
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imshow("img", img)
    cv.imwrite("../prod.png", img)
    cv.waitKey(10 * 1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="prod")
    parser.add_argument("--yolo_checkpoint", default="checkpoints/yolo_checkpoint.pt")
    parser.add_argument('--mae_checkpoint', default="checkpoints/mae_checkpoint.pt")
    parser.add_argument('--swinir_checkpoint', default="checkpoints/swinir_checkpoint.pt")
    parser.add_argument("--img_path", default="data/openlogo/test/images/logos32plus_000626.jpg")
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    if args.mode == "train":
        pass
    elif args.mode == "test_mae":
        test_mae(args)
    elif args.mode == "test_swinir":
        test_swinir(args)
    elif args.mode == "test_yolo":
        test_yolo(args)
    elif args.mode == "test_integrate":
        test_integrate(args)
    elif args.mode == "prod":
        anonymask(args)
    else:
        assert False, "Unknown mode"


if __name__ == "__main__":
    main()
