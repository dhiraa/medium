import argparse
import os
import time

import cv2
import numpy as np
from grpc.beta import implementations
try:
    from vitaflow.datasets.image.icdar.icdar_data import get_images
    from vitaflow.models.image.east.prediction import resize_image, sort_poly, detect
except:
    from vitaflow.datasets.image.icdar.icdar_data import get_images
    from vitaflow.models.image.east.prediction import resize_image, sort_poly, detect

from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


# https://sthalles.github.io/serving_tensorflow_models/
# https://medium.com/@yuu.ishikawa/serving-pre-modeled-and-custom-tensorflow-estimator-with-tensorflow-serving-12833b4be421

def read_image(path):
    im = cv2.imread(path)[:, :, ::-1]
    start_time = time.time()
    im_resized, (ratio_h, ratio_w) = resize_image(im)
    im_resized = np.expand_dims(im_resized, axis=0).astype(np.float32)
    return im, im_resized, ratio_h, ratio_w


def get_predictions(host, port, model_name, signature_name, input_placeholder_name, mat):
    channel = implementations.insecure_channel(host, port)._channel
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    request.inputs[input_placeholder_name].CopyFrom(make_tensor_proto(mat))
    result = stub.Predict(request)
    return result


# TODO GT this function and the one below are kind of similar but differ in the way
# they get the data .... refactor this
def get_text_segmentation(img_mat, ratio_h, ratio_w, result, output_dir, file_name):
    f_score = result.outputs["f_score"].float_val
    dims = result.outputs["f_score"].tensor_shape.dim
    shape = tuple(d.size for d in dims)
    score = np.asarray(f_score)
    score.resize(shape)

    f_geometry = result.outputs["f_geometry"].float_val
    dims = result.outputs["f_geometry"].tensor_shape.dim
    shape = tuple(d.size for d in dims)
    geometry = np.asarray(f_geometry)
    geometry.resize(shape)

    boxes, timer = detect(score_map=score, geo_map=geometry, timer=None)

    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    # save to file
    if boxes is not None:
        res_file = os.path.join(output_dir,
                                '{}.txt'.format(
                                    file_name.split('.')[0]))

        with open(res_file, 'w') as f:
            for box in boxes:
                # to avoid submitting errors
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2,
                                                                    0], box[2, 1], box[3, 0], box[3, 1],
                ))
                cv2.polylines(img_mat[:, :, ::-1], [box.astype(np.int32).reshape(
                    (-1, 1, 2))], True, color=(255, 255, 0), thickness=1)

    img_path = os.path.join(output_dir, file_name)
    # img_mat = np.squeeze(img_mat, axis=0)
    cv2.imwrite(img_path, img_mat[:, :, ::-1])


def get_text_segmentation_pb(img_mat, ratio_h, ratio_w, result, output_dir, file_name):
    f_score = result["f_score"]
    score = np.copy(f_score)
    f_geometry = result["f_geometry"]
    geometry = np.copy(f_geometry)
    boxes, timer = detect(score_map=score, geo_map=geometry, timer=None)

    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    # save to file
    if boxes is not None:
        res_file = os.path.join(output_dir,
                                '{}.txt'.format(
                                    file_name.split('.')[0]))

        with open(res_file, 'w') as f:
            for box in boxes:
                # to avoid submitting errors
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2,
                                                                    0], box[2, 1], box[3, 0], box[3, 1],
                ))
                cv2.polylines(img_mat[:, :, ::-1], [box.astype(np.int32).reshape(
                    (-1, 1, 2))], True, color=(255, 255, 0), thickness=1)

    img_path = os.path.join(output_dir, file_name)
    # img_mat = np.squeeze(img_mat, axis=0)
    cv2.imwrite(img_path, img_mat[:, :, ::-1])


def run(host, port, image, model, signature_name, output_dir):
    im, img_resized, ratio_h, ratio_w = read_image(image)
    result = get_predictions(host=host,
                             port=port,
                             model_name=model,
                             signature_name=signature_name,
                             input_placeholder_name="images",
                             mat=img_resized)
    get_text_segmentation(img_mat=im,
                          result=result,
                          output_dir=output_dir,
                          file_name=os.path.basename(image),
                          ratio_h=ratio_h,
                          ratio_w=ratio_w)


def run_on_images(host, port, images_dir, model, signature_name, output_dir):
    images = get_images(images_dir)

    for image_file_path in images:
        im, img_resized, ratio_h, ratio_w = read_image(image_file_path)
        result = get_predictions(host=host,
                                 port=port,
                                 model_name=model,
                                 signature_name=signature_name,
                                 input_placeholder_name="images",
                                 mat=img_resized)
        get_text_segmentation(img_mat=im,
                              result=result,
                              output_dir=output_dir,
                              file_name=os.path.basename(image_file_path),
                              ratio_h=ratio_h,
                              ratio_w=ratio_w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--host', help='Tensorflow server host name', default='localhost', type=str)
    parser.add_argument(
        '--port', help='Tensorflow server port number', default=8500, type=int)
    parser.add_argument('--image', help='input image', type=str)
    parser.add_argument('--images_dir', help='input images', type=str)
    parser.add_argument('--output_dir', help='out dir for images', type=str)
    parser.add_argument('--model', help='model name', type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model',
                        default='serving_default', type=str)

    args = parser.parse_args()
    if args.images_dir:
        run_on_images(args.host,
                      args.port,
                      args.images_dir,
                      args.model,
                      args.signature_name,
                      output_dir=args.output_dir)
    else:
        run(args.host, args.port, args.image, args.model, args.signature_name, output_dir=args.output_dir)
