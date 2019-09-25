import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from PIL import Image
import numpy as np
import os
import json
import time
import torch

from maskrcnn_benchmark.config import cfg
from .predictor import COCODemo


class MyDemo(COCODemo):

    center_score_weight = [
        0.0000, 0.0769, 0.1479, 0.2134, 0.2739, 0.3297, 0.3812, 0.4288, 0.4727, 0.5132,
        0.5507, 0.5852, 0.6171, 0.6465, 0.6737, 0.6988, 0.7220, 0.7433, 0.7631, 0.7813,
        0.7981, 0.8136, 0.8280, 0.8412, 0.8534, 0.8647, 0.8751, 0.8847, 0.8935, 0.9017,
        0.9093, 0.9163, 0.9227, 0.9286, 0.9341, 0.9392, 0.9439, 0.9482, 0.9522, 0.9558,
        0.9592, 0.9624, 0.9653, 0.9679, 0.9704, 0.9727, 0.9748, 0.9767, 0.9785, 0.9802,
        0.9817, 0.9831, 0.9844, 0.9856, 0.9867, 0.9877, 0.9887, 0.9895, 0.9903, 0.9911,
        0.9918, 0.9924, 0.9930, 0.9935, 0.9940, 0.9945, 0.9949, 0.9953, 0.9957, 0.9960,
        0.9963, 0.9966, 0.9968, 0.9971, 0.9973, 0.9975, 0.9977, 0.9979, 0.9981, 0.9982,
        0.9983, 0.9985, 0.9986, 0.9987, 0.9988, 0.9989, 0.9990, 0.9991, 0.9991, 0.9992,
        0.9993, 0.9993, 0.9994, 0.9994, 0.9995, 0.9995, 0.9995, 0.9996, 0.9996, 0.9996,
        0.9997
    ]
    area_score_weight = [
        0.0000, 0.1479, 0.2739, 0.3812, 0.4727, 0.5507, 0.6171, 0.6737, 0.7220, 0.7631,
        0.7981, 0.8280, 0.8534, 0.8751, 0.8935, 0.9093, 0.9227, 0.9341, 0.9439, 0.9522,
        0.9592, 0.9653, 0.9704, 0.9748, 0.9785, 0.9817, 0.9844, 0.9867, 0.9887, 0.9903,
        0.9918, 0.9930, 0.9940, 0.9949, 0.9957, 0.9963, 0.9968, 0.9973, 0.9977, 0.9981,
        0.9983, 0.9986, 0.9988, 0.9990, 0.9991, 0.9993, 0.9994, 0.9995, 0.9995, 0.9996,
        0.9997, 0.9997, 0.9998, 0.9998, 0.9998, 0.9998, 0.9999, 0.9999, 0.9999, 0.9999,
        0.9999, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000]

    def __init__(self,
                 path="model/final_model.pth",
                 config_file="configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml",
                 cuda=True,
                 threshold=0.7
                 ):
        # this makes our figures bigger
        pylab.rcParams['figure.figsize'] = 20, 12

        # update the config options with the config file
        cfg.merge_from_file(config_file)
        # manual override some options
        if cuda:
            cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        else:
            cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
        cfg.MODEL.WEIGHT = path
        super().__init__(cfg, min_image_size=400, confidence_threshold=threshold)

        self._init_weights()

    def _init_weights(self, k=-8, b=2):
        if k != -8 or b != 2:
            # todo: 计算新的权重
            pass

    def _new_scores(self, width, height, predictions, k=-8, b=2):
        # 计算每个bbox的中心点
        area = width * height
        minx, miny, maxx, maxy = predictions._split_into_xyxy()
        # predictions.convert("xywh")
        box_loc = []
        for i in range(len(predictions)):
            box_loc.append([minx[i][0], miny[i][0], maxx[i][0], maxy[i][0]])
        box_center = []
        for box in box_loc:
            box_center.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        # 根据中心点计算权重
        center_weight = []
        for point in box_center:
            # 计算相对位置
            xr = int(min(point[0] / width, 1 - point[0] / width) * 100)
            yr = int(min(point[1] / height, 1 - point[1] / height) * 100)
            # 计算位置权重
            center_weight.append(min(self.center_score_weight[xr], self.center_score_weight[yr]))
        # 计算每个bbox的大小
        box_area = []
        for box in box_loc:
            box_area.append(abs(box[2] - box[0] * box[3] - box[1]))
        # 根据大小计算权重
        area_weight = []
        for box in box_area:
            # 计算相对大小
            area_weight.append(self.area_score_weight[int(box / area * 100)])
        # 计算新评分
        scores = predictions.get_field("scores")
        new_scores = [score * center_weight[i] * area_weight[i] for i, score in enumerate(scores)]
        return torch.Tensor(new_scores)

    def _sort_predictions_by_salient(self, width, height, predictions):
        scores = self._new_scores(width, height, predictions)
        keep = []
        # keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        if len(keep) != 0:
            return predictions[keep]
        # scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx][[0]] if len(predictions) != 0 else predictions

    def _MO_cumputer(self, prediction, true_ground, mode='xyxy'):
        if len(prediction) == 0:
            return 0

        minx, miny, maxx, maxy = prediction._split_into_xyxy()
        predict = [minx[0], miny[0], maxx[0], maxy[0]]
        overlap = [max(predict[0], true_ground[0]), max(predict[1], true_ground[1]),
                   min(predict[2], true_ground[2]), min(predict[3], true_ground[3])]
        overlap_area = (overlap[2] - overlap[0] + 1) * (overlap[3] - overlap[1] + 1)
        if overlap_area < 0:
            overlap_area = 0
        covered_area = (predict[2] - predict[0] + 1) * (predict[3] - predict[1] + 1) + \
                       (true_ground[2] - true_ground[0] + 1) * (true_ground[3] - true_ground[1] + 1) - overlap_area

        return overlap_area / covered_area

    def predict(self, image, mask=True):
        pil_image = image.convert("RGB")
        array_image = np.array(pil_image)[:, :, [2, 1, 0]]

        predictions = self.compute_prediction(array_image)
        top_predictions = self._sort_predictions_by_salient(image.width, image.height, predictions)

        result = array_image.copy()
        result = self.overlay_boxes(result, top_predictions)
        if mask:
            result = self.overlay_mask(result, top_predictions)
        return result

    def test(self, annotation_file, root):
        MO_list = [i / 100 for i in range(50, 100, 5)]
        with open(annotation_file) as f:
            annotation = json.load(f)
            positive = [0 for _ in range(len(MO_list))]
            test_info = {'images': []}
            start_time = time.time()
            for i, image in enumerate(annotation['images'][:100]):
                image_info = {}

                img = Image.open(root + image['file_name'])
                image_info['file_name'] = root + image['file_name']
                pil_image = img.convert("RGB")
                array_image = np.array(pil_image)[:, :, [2, 1, 0]]
                true_ground = annotation['annotations'][i]['bbox']
                true_ground = [true_ground[0], true_ground[1], true_ground[0] + true_ground[2] - 1, true_ground[1] + true_ground[3] - 1]

                predictions = self.compute_prediction(array_image)
                top_prediction = self._select_top_prediction(img.width, img.height, predictions)
                overlap = self._MO_cumputer(top_prediction, true_ground)
                for j, MO in enumerate(MO_list):
                    positive[j] += 1 if overlap >= MO else 0
                image_info['overlap'] = float(overlap)
                test_info['images'].append(image_info)
                print(i)
            end_time = time.time()
            test_info['time'] = end_time - start_time

            test_info['AP'] = {}
            mean_AP = 0
            for i, MO in enumerate(MO_list):
                test_info['AP'][MO] = positive[i] / len(test_info['images'])
                mean_AP += test_info['AP'][MO]
            test_info['AP']['mean'] = mean_AP / len(MO_list)
            print(test_info['AP'])

            with open('output/test.json', 'w') as output:
                json.dump(test_info, output, indent=4, separators=(",", ": "))


if __name__ == "__main__":

    model = MyDemo()

    img_dirs = ("datasets/coco/test2014/",)
    try:
        os.makedirs("output/test2014_org/")
    except:
        pass

    for img_dir in img_dirs:
        for img_name in os.listdir(img_dir):
            if img_name[-4:] != ".jpg":
                continue

            image = Image.open(img_dir + img_name)
            # forward predict
            prediction = model.predict(image)

            # vis
            plt.imshow(prediction[:, :, ::-1])
            plt.axis('off')
            plt.savefig("output/test2014_org/%s" % img_name)

            print(img_name)

