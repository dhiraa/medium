'''
python pb_predict\
 --images_dir="/home/gaurishk/Desktop/preprocess/test"\
 --output_dir="/home/gaurishk/Desktop/preprocess/"\
 --model="/home/gaurishk/Projects/vitaFlow/vitaflow/annotate_server/static/data/east_models/east_airflow_demo/EASTModel/exported/1558013588"
'''

import argparse
import os

try:
    from vitaflow.models.image.east.grpc_predict import read_image, get_text_segmentation_pb
    from vitaflow.datasets.image.icdar.icdar_data import get_images
except:
    from vitaflow.models.image.east.grpc_predict import read_image, get_text_segmentation_pb
    from vitaflow.datasets.image.icdar.icdar_data import get_images


from tensorflow.contrib import predictor


def run(input_dir, output_dir, model_dir):
    images_dir = input_dir
    images = get_images(images_dir)
    predict_fn = predictor.from_saved_model(model_dir)
    for image_file_path in images:
        im, img_resized, ratio_h, ratio_w = read_image(image_file_path)
        result = predict_fn({'images': img_resized})
        get_text_segmentation_pb(img_mat=im,
                                 result=result,
                                 output_dir=output_dir,
                                 file_name=os.path.basename(image_file_path),
                                 ratio_h=ratio_h,
                                 ratio_w=ratio_w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', help='input images', type=str)
    parser.add_argument('--output_dir', help='out dir for images', type=str)
    parser.add_argument('--model', help='model exported path', type=str)

    args = parser.parse_args()
    run(input_dir=args.images_dir, output_dir=args.output_dir, model_dir=args.model)
