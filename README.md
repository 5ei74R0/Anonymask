# Anonymask

Banish any LOGO from your sight 👀

## Demo

![Screenshot from 2022-07-04 22-54-57](https://user-images.githubusercontent.com/51681991/177169271-b0ac2c5b-3ea5-482d-a027-018e817ccf2d.png)


## Getting Started

1. Clone this repository and run `cd Anonymask; git submodule update --init --recursive`.
2. `pip install -r requirements.txt`
3. [preprocess](#preprocess)
4. `sh run.sh`


### Preprocess

1. Download finetuned Yolov5, Masked Auto Encoder, SwinIR models from [[release](https://github.com/5ei74R0/Anonymask/releases/tag/add-models)], and put them into `./checkpoints/`.
2. Place [openlogo](https://hangsu0730.github.io/qmul-openlogo/) dir into `./data/`
3. `cd preprocess; mkdir ../data/openlogo/labels`
4. `./xmlAnnotations2txtlabels.sh ../data/openlogo/Annotations ../data/openlogo/labels`
5. `./divide.sh ../data/openlogo`
6. Let's finetune !
    - `cd ../lib/yolo`
    - `python train.py --img 640 --batch 40 --epochs 15 --data ../../data/openlogo.yaml --weights ../../models/yolov5s4logo.pt`

## Details
