# EAST: An Efficient and Accurate Scene Text Detector

**Refernce Git** : https://github.com/argman/EAST  
**Paper** : https://arxiv.org/abs/1704.03155v2   
**Dataset ICDAR** : [2019](http://rrc.cvc.uab.es/?ch=13)   
- Use Google Drive Link: https://drive.google.com/drive/folders/1ShItNWXyiY1tFDM5W02bceHuJjyeeJl2 
and download the files

## Text Localization:
- All images are provided as JPEG or PNG files and the text files are UTF-8 files with CR/LF new line endings.
- The ground truth is given as separate text files (one per image) where each line specifies the coordinates of one 
word's bounding box and its transcription in a comma separated format 
- [2019](http://rrc.cvc.uab.es/?ch=13&com=tasks)

img_1.txt <-> img_01.txt

```sh
x1_1, y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1, transcript_1

x1_2,y1_2,x2_2,y2_2,x3_2,y3_2,x4_2,y4_2, transcript_2

x1_3,y1_3,x2_3,y2_3,x3_3,y3_3,x4_3,y4_3, transcript_3
```

## Data Setup

```sh
#Mannual way
mkdir /opt/data/icdar/2019/
tree .
.
├── 0325updated.task1train(626p)-20190701T072844Z-001.zip # unzip this to train foldr
├── task1&2_test(361p)-20190701T072809Z-001.zip           # unzip this to test folder
└── text.task1&2-test（361p)-20190701T072850Z-001.zip     # unzip this to test folder

cd /opt/data/icdar/2019
mkdir train
mkdir test
mkdir val

# 2019 train files
unzip 0325updated.task1train\(626p\)-20190701T072844Z-001.zip -d train/
cd train
mv 0325updated.task1train\(626p\)/* .
rm -rf 0325updated.task1train\(626p\)/
rm *\).txt
rm *\).jpg
mv `ls  | head -200` ../val/
cd ..
unzip task1\&2_test\(361p\)-20190701T072809Z-001.zip -d test
unzip text.task1\&2-test（361p\)-20190701T072850Z-001.zip -d test
cd ..
ls

#OR Through CLI

mkdir -p /opt/data/icdar/
mkdir -p 2019
cd 2019
mkdir -p train
mkdir -p test
mkdir -p val
cd ..
drive clone https://drive.google.com/drive/folders/1ShItNWXyiY1tFDM5W02bceHuJjyeeJl2
cd SROIE2019/
cp 0325updated.task1train\(626p\)/* ../2019/train/
cp task1\&2_test\(361p\)/* ../2019/test/
cp text.task1\&2-test（361p\)/* ../2019/test/
cd ../2019/train/
mv `ls  | head -200` ../2019/val/
cd ..
```

## Configuration

Check this [file](../../config/east_config.gin)!

##Compile lanms

```sh
cd lanms
make
```
If you get compilation issues with lanms, follow this [link](https://github.com/argman/EAST/issues/156#issuecomment-404166990)! to resolve it.

## Running

```sh

python vitaflow/bin/run_experiments.py \
	--mode=train \
	--config_file=vitaflow/config/east_config.gin 

#serving
export MODEL_NAME=EAST
export MODEL_PATH=/opt/tmp/icdar/east/EASTModel/exported/

tensorflow_model_server   \
--port=8500   \
--rest_api_port=8501   \
--model_name="$MODEL_NAME" \
--model_base_path="$MODEL_PATH"

python grpc_predict.py \
  --image /opt/tmp/test/img_967.jpg \
  --output_dir /opt/tmp/icdar/ \
  --model EAST  \
  --host "localhost" \
  --signature_name serving_default


python grpc_predict.py \
  --images_dir /opt/tmp/test/ \
  --output_dir /opt/tmp/icdar/ \
  --model EAST  \
  --host "localhost" \
  --signature_name serving_default 
```

### PS

- As compared to original EAST repo, we have used Tensorflow high level APIs tf.data and tf.Estimators
- This comes in handy when we move to big dataset or if we wanted to experiment with different models/data
- TF Estimator also takes care of exporting the model for serving! [Reference](https://medium.com/@yuu.ishikawa/serving-pre-modeled-and-custom-tensorflow-estimator-with-tensorflow-serving-12833b4be421)

### Similar Models / Gits:
- https://github.com/Michael-Xiu/ICDAR-SROIE
- https://github.com/xieyufei1993/FOTS
- https://github.com/hwalsuklee/awesome-deep-text-detection-recognition
- https://github.com/tangzhenyu/Scene-Text-Understanding

### References
- https://berlinbuzzwords.de/18/session/scalable-ocr-pipelines-using-python-tensorflow-and-tesseract
- http://machinelearninguru.com/deep_learning/data_preparation/tfrecord/tfrecord.html
- https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
- https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
- https://databricks.com/blog/2018/07/10/how-to-use-mlflow-tensorflow-and-keras-with-pycharm.html
- https://stackoverflow.com/questions/51455863/whats-the-difference-between-a-tensorflow-keras-model-and-estimator