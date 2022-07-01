## Dataset preparation
Place [`openlogo`](https://hangsu0730.github.io/qmul-openlogo/) into `data`
```sh
mkdir ../data/openlogo/labels
```
```sh
./xmlAnnotations2txtlabels.sh ../data/openlogo/Annotations ../data/openlogo/labels
```
```sh
./divide.sh ../data/openlogo
```

## Train yolov5
Use training script provided by yolov5
```sh
python train.py --img 640 --batch 40 --epochs 15 --data ../../data/openlogo.yaml --weights ../../models/yolov5s4logo.pt
```
