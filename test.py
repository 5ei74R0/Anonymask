import argparse

import cv2 as cv
import numpy as np
import torch
from cv2 import resize

from detection import Detector
from mask.inpaint import Inpainter
from mask.super_resolution import Upsampler


def test_yolo(args: argparse.Namespace):
    DEVICE = torch.device("cuda:0")
    det: Detector = Detector(args.yolo_checkpoint, device=DEVICE)
    img = cv.imread(args.test_img_path)
    img = cv.resize(img, (640, 640))
    input_imgs: torch.Tensor = (
        torch.tensor(img).permute(2, 0, 1).unsqueeze(0).expand(3, -1, -1, -1).to(DEVICE)
        / 255.0
    )
    print(input_imgs.shape)
    print(det.get_bboxes(input_imgs))


def test_swinir(args: argparse.Namespace):
    DEVICE = torch.device("cuda:0")
    upsampler = Upsampler(sr_scale=4)
    img = cv.imread(args.test_img_path)
    input_imgs: torch.Tensor = (
        torch.tensor(img).permute(2, 0, 1).unsqueeze(0).expand(3, -1, -1, -1).to(DEVICE)
        / 255.0
    )
    upsampler.upsample(img)


def test_mae(args: argparse.Namespace):
    inpainter = Inpainter(args.mae_checkpoint)
    x = cv.imread(args.test_img_path)
    x = resize(x, (224, 224))
    x = x.astype(np.float32) / 255.
    y = inpainter(cv.cvtColor(x, cv.COLOR_BGR2RGB), np.array([[0, 0], [40, 40]]))
    y = cv.cvtColor(y, cv.COLOR_RGB2BGR)
    cv.imshow("img", np.hstack([x, y]))
    cv.waitKey(0)


def test_integrate(args: argparse.Namespace, save=True):
    SEED = 42
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    DEVICE = torch.device("cuda:0")
    det: Detector = Detector(args.yolo_checkpoint, device=DEVICE)
    img = cv.imread(args.test_img_path)
    x = cv.resize(img, (640, 640))
    input_imgs: torch.Tensor = (
        torch.tensor(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        / 255.0
    )

    scale = 224. / 640.
    bboxes = det.get_bboxes(input_imgs)
    if bboxes.sum() == 0:
        print("No bboxes")
        return
    bboxes = bboxes[0, :, :].float() * scale
    bboxes = bboxes.view(2, 2).detach().cpu().numpy()
    bboxes = bboxes.astype(np.int32)[:, ::-1]

    print(bboxes)
    inpainter = Inpainter(args.mae_checkpoint)
    x = resize(img, (224, 224))
    x = x.astype(np.float32) / 255.
    y = inpainter(cv.cvtColor(x, cv.COLOR_BGR2RGB), bboxes)
    y = cv.cvtColor(y, cv.COLOR_RGB2BGR)
    cv.imshow("img", np.hstack([x, y]))
    if save:
        cv.imwrite("../output.jpg", (np.hstack([x, y]) * 255).astype(np.int32))
    cv.waitKey(0)
