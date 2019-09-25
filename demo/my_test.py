import time

from demo.my_demo import MyDemo

if __name__ == '__main__':
    # model = MyDemo(path='output/model_0010000.pth')
    model = MyDemo(path='output/model_0010000.pkl')
    model.test('datasets/coco/annotations/instances_test2014.json', 'datasets/coco/test2014/')