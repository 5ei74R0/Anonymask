import argparse
from test import test_integrate, test_mae, test_swinir, test_yolo

import cv2 as cv
import numpy as np
import torch
from cv2 import resize

from detection import Detector
from mask.inpaint import Inpainter
from mask.super_resolution import Upsampler


def anonymask(args: argparse.Namespace):
    # init
    SEED = 42
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)

    # yolo detector
    print("Detecting...")
    det = Detector(args.yolo_checkpoint, device=device)
    img = cv.imread(args.test_img_path)
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
    bboxes = bboxes[0, :, :].float() * scale
    bboxes = bboxes.view(2, 2).detach().cpu().numpy()
    bboxes = bboxes.astype(np.int32)[:, ::-1]

    # execute inpainter
    print("Execute inpainter...")
    inpainter = Inpainter(args.mae_checkpoint, device=device)
    x = resize(img, (224, 224))
    x = x.astype(np.float32) / 255.
    y = inpainter(x, bboxes)

    # apply super resolution
    print("Apply super resolution...")
    upsampler = Upsampler(sr_scale=4)
    y = upsampler.upsample(y)
    y = cv.resize(y, (640, 640))

    # output
    print("complete!")
    x = cv.cvtColor(x, cv.COLOR_RGB2BGR)
    y = cv.cvtColor(y, cv.COLOR_RGB2BGR)
    cv.imshow("img", y)
    cv.waitKey(10 * 1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--yolo_checkpoint", default="checkpoints/yolo_checkpoint.pt")
    parser.add_argument('--mae_checkpoint', default="checkpoints/mae_checkpoint.pt")
    parser.add_argument("--test_img_path", default="data/openlogo/test/images/logos32plus_000626.jpg")
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
