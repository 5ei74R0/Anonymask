import torch
from lib.yolo.models.common import DetectMultiBackend as Yolov5Model
from lib.yolo.utils.general import non_max_suppression


class Detector:
    def __init__(self, weights: str = "yolov5s.pt", device=None):
        self.model = Yolov5Model(
            weights, device=device if device is not None else torch.device("cpu")
        )

    def get_bboxes(
        self, img: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45
    ) -> torch.Tensor:
        """B x C x H x W -> B x #bboxes x 4"""
        self.model.eval()
        preds = self.model(img)
        preds = non_max_suppression(preds, conf_thres, iou_thres)
        for i, pred in enumerate(preds):
            preds[i] = pred[..., :4].long()
        return torch.stack(preds)
