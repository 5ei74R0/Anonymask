"""
Callback server
"""
import io
import uvicorn
import cv2 as cv
import numpy as np

from fastapi import FastAPI, File, Form, UploadFile
from starlette.responses import StreamingResponse
from typing import List
from torchvision import transforms
from PIL import Image


class CallbackServer:
    @staticmethod
    def get_tensor_image(img_buff):
        transform = transforms.Compose([transforms.ToTensor()])
        img = Image.frombytes(mode="RGB", size=(256, 256), data=img_buff)
        img = transform(img)
        img = img[0, :, :].unsqueeze(0)
        return img.unsqueeze(0)

    @staticmethod
    def start(callback):
        """
        Function of start http server
        """
        fapi = FastAPI()

        @fapi.post("/anonymask")
        def execute_oneshot(
            image: UploadFile = File(description="target image"),
        ):
            img = np.asarray(Image.open(io.BytesIO(image.file.read())))
            img = callback(img)
            _, im_png = cv.imencode(".png", cv.cvtColor(img, cv.COLOR_RGB2BGR))
            return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

        host_name = "127.0.0.1"
        port_num = 8080
        uvicorn.run(fapi, host=host_name, port=port_num)
