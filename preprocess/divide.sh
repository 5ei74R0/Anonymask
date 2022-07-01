openlogo=$1

mkdir -p $openlogo/train/images
mkdir -p $openlogo/train/labels
mkdir -p $openlogo/test/images
mkdir -p $openlogo/test/labels

while read r; do
    cp $openlogo/labels/$r.txt $openlogo/train/labels
    cp $openlogo/JPEGImages/$r.jpg $openlogo/train/images
done < $openlogo/ImageSets/Main/train_test/train_all.txt

while read r; do
    mv $openlogo/labels/$r.txt $openlogo/test/labels
    mv $openlogo/JPEGImages/$r.jpg $openlogo/test/images
done < $openlogo/ImageSets/Main/train_test/test_all.txt
