import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from PIL import Image
import numpy as np
import os
import torch

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


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

    def __init__(self, config):
        super().__init__(config, min_image_size=400, confidence_threshold=0.7)
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

    def _select_top_predictions(self, width, height, predictions):
        # scores = self._new_scores(width, height, predictions)
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def predict(self, image, mask=True):
        pil_image = image.convert("RGB")
        array_image = np.array(pil_image)[:, :, [2, 1, 0]]

        predictions = self.compute_prediction(array_image)
        top_predictions = self._select_top_predictions(image.width, image.height, predictions)

        result = array_image.copy()
        result = model.overlay_boxes(result, top_predictions)
        if mask:
            result = model.overlay_mask(result, top_predictions)
        return result


if __name__ == "__main__":
    # this makes our figures bigger
    pylab.rcParams['figure.figsize'] = 20, 12

    config_file = "configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    model = MyDemo(cfg)

    img_dirs = ("datasets/coco/test2014/",)
    try:
        os.makedirs("output/test2014_org/")
    except:
        pass

    for img_dir in img_dirs:
        for img_name in os.listdir(img_dir)[:50]:
            if img_name[-4:] != ".jpg":
                continue

            image = Image.open(img_dir + img_name)
            image = np.array(image)[:, :, [2, 1, 0]]
            # forward predict
            prediction = model.run_on_opencv_image(image)

            # vis
            plt.imshow(prediction[:, :, ::-1])
            plt.axis('off')
            plt.savefig("output/test2014_org/%s" % img_name)

            print(img_name)

