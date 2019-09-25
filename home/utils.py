# encode=utf-8

from django.conf import settings
from django.http import HttpResponseRedirect
from django.core.files.uploadedfile import InMemoryUploadedFile

from io import BytesIO
from PIL import Image

from demo.my_demo import MyDemo

my_model = MyDemo()

def flush_2_home():
    """
    清理用户信息并重定向到首页
    :return:
    """
    response = HttpResponseRedirect('/')
    response.delete_cookie('username')
    response.delete_cookie('password')
    return response


def predict(image):
    item = Image.open(image)
    # forward predict
    prediction = my_model.predict(item)
    prediction = Image.fromarray(prediction[:, :, ::-1])
    # 转化格式
    prediction_io = BytesIO()
    prediction.save(prediction_io, format='JPEG')
    # 再转化为InMemoryUploadedFile数据
    prediction_file = InMemoryUploadedFile(
        file=prediction_io,
        field_name=None,
        name=image.name,
        content_type=image.content_type,
        size=image.size,
        charset=None
    )
    return prediction_file
