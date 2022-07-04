# Anonymask

Banish any LOGO from your sight 👀

## Demo

Coming soon...

## Getting Started

1. Clone this repository and run `cd Anonymask; git submodule update --init --recursive`.
2. `pip install -r requirements.txt`
3. [preprocess](#preprocess)
4. `sh run.sh`


### Preprocess

1. Download finetuned Yolov5, Masked Auto Encoder, SwinIR models from [[Google Drive](https://drive.google.com/drive/folders/1m-jA_p3aCg7G3MxJ9QH6R9cpS9mVRs9j?usp=sharing)], and put them into `./checkpoints/`.
2. Place [openlogo](https://hangsu0730.github.io/qmul-openlogo/) dir into `./data/`
3. `cd preprocess; mkdir ../data/openlogo/labels`
4. `./xmlAnnotations2txtlabels.sh ../data/openlogo/Annotations ../data/openlogo/labels`
5. `./divide.sh ../data/openlogo`
6. Let's finetune !
    - `cd ../lib/yolo`
    - `python train.py --img 640 --batch 40 --epochs 15 --data ../../data/openlogo.yaml --weights ../../models/yolov5s4logo.pt`

## Details
