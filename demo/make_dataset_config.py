# encode = utf-8

import json
import os
import numpy as np
from PIL import Image


class PictureDatasetConfig(object):

    info = dict()
    licences = list()
    images = list()
    annotations = list()
    categories = {
        "supercategory": "NULL",
        "id": 1,
        "name": "NULL",
    }

    def __init__(self):
        pass

    def _print_mask(self, mask, points):
        for y in range(len(mask)):
            for x in range(len(mask[0])):
                if [x, y] in points:
                    print("*", end="")
                else:
                    print("_", end="")
            print()
        print()

    def _clockwise(self, bias):
        if bias == [0, -1]:
            return [1, -1]
        elif bias == [1, -1]:
            return [1, 0]
        elif bias == [1, 0]:
            return [1, 1]
        elif bias == [1, 1]:
            return [0, 1]
        elif bias == [0, 1]:
            return [-1, 1]
        elif bias == [-1, 1]:
            return [-1, 0]
        elif bias == [-1, 0]:
            return [-1, -1]
        else:
            return [0, -1]

    def _reverse(self, bias):
        return [-bias[0], -bias[1]]

    def _next_points(self, mask, points, point, bias):
        new_bias = self._reverse(bias)
        cur = [point[0] + new_bias[0], point[1] + new_bias[1]]
        pre = []
        for _ in range(7):
            new_bias = self._clockwise(new_bias)
            pre = cur
            cur = [point[0] + new_bias[0], point[1] + new_bias[1]]
            cur_p = mask[cur[1]][cur[0]]
            pre_p = mask[pre[1]][pre[0]]
            if cur_p == 255 and pre_p == 0 and cur not in points:
                return cur, new_bias
        return point, self._reverse(bias)

    def _shrink(self, mask_array):
        mask = [list(line) for line in mask_array]
        # 在四周补0
        for row in range(len(mask)):
            mask[row].insert(0, 0)
            mask[row].append(0)
        mask.insert(0, [0 for _ in range(len(mask[0]))])
        mask.append([0 for _ in range(len(mask[0]))])

        new_mask = []
        # 3*3池化,step=2，多于3点为255则取255
        for y in range(1, len(mask) - 2, 2):
            new_line = []
            for x in range(1, len(mask[0]) - 2, 2):
                a = mask[y][x]
                point = int(mask[y - 1][x - 1]) + int(mask[y][x - 1]) + int(mask[y + 1][x - 1]) + \
                        int(mask[y - 1][x]) + int(mask[y][x]) + int(mask[y + 1][x]) + \
                        int(mask[y - 1][x + 1]) + int(mask[y][x + 1]) + int(mask[y + 1][x + 1])
                point = 255 if point >= 3 * 255 else 0
                new_line.append(point)
            new_mask.append(new_line)

        # 在四周补0
        for row in range(len(new_mask)):
            new_mask[row].insert(0, 0)
            new_mask[row].append(0)
        new_mask.insert(0, [0 for _ in range(len(new_mask[0]))])
        new_mask.append([0 for _ in range(len(new_mask[0]))])
        return new_mask

    def _skip_points(self, points):
        new_points = [[points[i][0] * 2 - 1, points[i][1] * 2 - 1] for i in range(0, len(points), 5)]
        i = 1
        while i < len(new_points):
            if (new_points[i][0] - new_points[i - 1][0]) ** 2 + (new_points[i][1] - new_points[i - 1][1]) ** 2 <= 100:
                del new_points[i]
                i -= 1
            i += 1
        i = 1
        while i < len(new_points) - 1:
            if abs((new_points[i][0] - new_points[i - 1][0]) * (new_points[i + 1][1] - new_points[i][1]) - (new_points[i + 1][0] - new_points[i][0]) * (new_points[i][1] - new_points[i - 1][1])) <= 0.2:
                del new_points[i]
            i += 1
        return new_points

    def _segmentation_points(self, mask_array):
        mask = self._shrink(mask_array)
        start_point = None
        current_point = None
        next_point = None
        points = []
        next_bias = [-1, 0]
        pre_bias = [0, 0]
        # 找到第一个点
        for row, line in enumerate(mask):
            for col, pixle in enumerate(line):
                if pixle == 255:
                    start_point = [col, row]
                    break
            if start_point:
                break
        # 找到所有点
        current_point = list(start_point)
        while True:
            # print(current_point, pre_bias)
            next_point, next_bias = self._next_points(mask, points, current_point, pre_bias)
            if next_point in points:
                break
            points.append(next_point)
            current_point = next_point
            pre_bias = next_bias

        points = self._skip_points(points)
        if len(points) > 17:
            return points
        else:
            # self._print_mask(mask_array, points)
            print(len(points))
            return None

    def _get_segmentation(self, mask_name):
        mask_image = Image.open(mask_name)
        mask_array = np.array(mask_image)
        segmentation = []
        points = self._segmentation_points(mask_array)
        if points is None:
            print("%s" % mask_name)
            # a = input()
            return None

        for point in points:
            segmentation.append(point[0])
            segmentation.append(point[1])
        return segmentation


    def _get_bbox(self, mask_name):

        mask_image = Image.open(mask_name)
        mask_array = np.array(mask_image)
        bbox = [mask_image.height, mask_image.width, 0, 0]

        for row, line in enumerate(mask_array):
            for col, pixle in enumerate(line):
                if pixle == 255:
                    if col < bbox[0]:
                        bbox[0] = col
                    if row < bbox[1]:
                        bbox[1] = row
                    if col > bbox[2]:
                        bbox[2] = col
                    if row > bbox[3]:
                        bbox[3] = row

        height = bbox[2] - bbox[0] + 1
        width = bbox[3] = bbox[1] + 1
        bbox[2] = height
        bbox[3] = width

        return bbox

    def append_image(self, file_name, mask_name):

        image = Image.open(file_name)

        image_info = {
            "file_name": file_name,
            "height": image.height,
            "width": image.width,
            "id": len(self.images),
        }
        annotation = {
            "segmentation": self._get_segmentation(mask_name),
            "area": image_info["height"] * image_info["width"],
            "image_id": len(self.images),
            "bbox": self._get_bbox(mask_name),
            "category_id": 1,
            "id": len(self.annotations),
        }
        if annotation["segmentation"] == None:
            return

        self.images.append(image_info)
        self.annotations.append(annotation)

    def to_json(self, file_name):
        config = {
            "info": {},
            "licenses": [],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

        json_file = json.dumps(config, indent=4, separators=(",", ": "))
        with open(file_name, 'w') as f:
            f.write(json_file)


if __name__ == "__main__":
    config = PictureDatasetConfig()

    # file_name = "../datasets/coco/train2014/3_115_115151.jpg"
    # mask_name = "../datasets/coco/mask/train2014/3_115_115151.png"
    # config.append_image(file_name, mask_name)


    picture_root = "../datasets/coco/train2014/"
    mask_root = "../datasets/coco/mask/train2014/"

    for file_name in os.listdir(picture_root):
        if file_name[-4:] == ".jpg":
            # print(file_name)
            config.append_image(picture_root + file_name, mask_root + file_name.replace(".jpg", ".png"))
    config.to_json("../datasets/coco/annotations/instance_train2014.json")

    picture_root = "../datasets/coco/val2014/"
    mask_root = "../datasets/coco/mask/val2014/"
    config = PictureDatasetConfig()

    for file_name in os.listdir(picture_root):
        if file_name[-4:] == ".jpg":
            config.append_image(picture_root + file_name, mask_root + file_name.replace(".jpg", ".png"))
            # print(file_name)
    config.to_json("../datasets/coco/annotations/instance_val2014_1.json")

    picture_root = "../datasets/coco/test2014/"
    mask_root = "../datasets/coco/mask/test2014/"
    config = PictureDatasetConfig()

    for file_name in os.listdir(picture_root):
        if file_name[-4:] == ".jpg":
            config.append_image(picture_root + file_name, mask_root + file_name.replace(".jpg", ".png"))
            # print(file_name)
    config.to_json("../datasets/coco/annotations/instance_test2014_1.json")
