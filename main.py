import torch
import argparse
import cv2 as cv
import numpy as np
from cv2 import resize

from detection import Detector
from mask.inpaint import Inpainter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--yolo_checkpoint", default="checkpoints/yolo_checkpoint.pth")
    parser.add_argument('--mae_checkpoint', default="checkpoints/mae_checkpoint.pth")
    parser.add_argument("--test_img_path", default="data/openlogo/test/images/zara2.jpg")

    args = parser.parse_args()

    if args.mode == "train":
        pass
    elif args.mode == "test_mae":
        test_mae(args)
    elif args.mode == "test_yolo":
        test_yolo(args)
    elif args.mode == "test_integrate":
        test_integrate(args)
    elif args.mode == "prod":
        pass
    else:
        assert False, "Unknown mode"


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


def test_mae(args: argparse.Namespace):
    inpainter = Inpainter(args.mae_checkpoint)
    x = cv.imread(args.test_img_path)
    x = resize(x, (224, 224))
    x = x.astype(np.float32) / 255.
    y = inpainter(cv.cvtColor(x, cv.COLOR_BGR2RGB), np.array([[0, 0], [40, 40]]))
    y = cv.cvtColor(y, cv.COLOR_RGB2BGR)
    cv.imshow("img", np.hstack([x, y]))
    cv.waitKey(0)


def test_integrate(args: argparse.Namespace):
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
    # cv.imwrite("../output.jpg", (np.hstack([x, y]) * 255).astype(np.int32))
    cv.waitKey(0)


if __name__ == "__main__":
    main()
