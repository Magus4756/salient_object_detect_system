# encode=utf-8

from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.contrib.auth.hashers import (
    make_password, check_password,
)
from django.core.exceptions import ObjectDoesNotExist

from datetime import datetime
import time

from home.models import User, History

from .utils import (
    flush_2_home, predict,
)


def home(request):
    """
    主页。若已登录则显示上传页面，否则跳转到登录页面
    :param request:
    :return:
    """
    user = User.check_user(request)
    if user == 'KeyError' or user == 'PasswordNotMatch':
        return HttpResponseRedirect('/login/')
    if request.method == 'GET':
        return render(request, 'home.html', {'user': user, })
    else:
        try:
            images = request.FILES.getlist('img')
        except AttributeError:
            return render(request, 'home.html', {'user': user, })
        begin_time = time.time()
        results = []
        for img in images:
            start_time = time.time()
            image_ = History(
                user=user,
                image=img,
                name=img.name,
                upload_time=datetime.now(),
                prediction=predict(img),
                spend_time=time.time() - start_time,
            )
            image_.save()
            results.append(image_)
        end_time = time.time()
        return render(request, 'home.html', {'predictions': results, 'spend_time': '%.2fs' % (end_time - begin_time)})


def history(request):
    """
    用户个人界面，展示并管理各种记录
    :param request:
    :return:
    """
    user = User.check_user(request)
    if user == 'KeyError' or user == 'PasswordNotMatch':
        return flush_2_home()

    # 查询所需信息
    images = History.objects.filter(user=user, is_deleted=False).order_by("-upload_time")  # 从数据库中取出所有的图片路径
    return render(request, 'history.html', {'user': user, 'images': images})


def delete_all(request):
    """
    浅删除所有历史记录
    :param request:
    :return:
    """
    user = User.check_user(request)
    if user == 'KeyError' or user == 'PasswordNotMatch':
        return flush_2_home()

    histories = History.objects.filter(user=user)
    for item in histories:
        item.is_deleted = True
        item.deleted_time = datetime.now()
        item.save()
    return HttpResponseRedirect('/history/')


def delete_one(request):
    """
    浅删除一项历史记录
    :param request:
    :return:
    """
    user = User.check_user(request)
    if user == 'KeyError' or user == 'PasswordNotMatch':
        return flush_2_home()

    item_id = request.GET['id']
    item = History.objects.filter(user=user, id=item_id)
    for i in item:
        i.is_deleted = True
        i.deleted_time = datetime.now()
        i.save()
    return HttpResponseRedirect('/history/')


def login(request):
    """
    用户登录
    :param request:
    :return:
    """
    if request.method == 'GET':
        return render(request, 'login.html')

    elif request.method == 'POST':
        # 获取用户名和密码
        name = request.POST['username']
        password = request.POST['password']
        try:
            user = User.objects.get(name=name)
        except ObjectDoesNotExist:  # 用户名不存在
            return render(request, 'login.html', {'state': 'name_wrong'})

        if check_password(password, user.password) is False:  # 密码不匹配
            return render(request, 'login.html', {'state': 'password_wrong'})

        response = HttpResponseRedirect('/')
        response.set_cookie('username', user.name)
        response.set_cookie('password', user.password)
        return response

    else:
        return flush_2_home()


def logout(request):
    # 删除 Cookie 中的用户名和密码
    return flush_2_home()


def register(request):
    """
    注册用户
    :param request:
    :return:
    """
    if request.method == 'GET':
        return render(request, 'register.html', {'state': 0})

    elif request.method == 'POST':
        # 取得用户名和密码
        try:
            username = request.POST['username']
            password = request.POST['password']
        except KeyError:  # 错误处理
            return render(request, 'register.html', {'state': 1})

        # 查找重名用户
        user = User.objects.filter(name=username)
        if user:
            return render(request, 'register.html', {'state': 2})

        hashed = make_password(password)  # 密码以加密形式保存
        user = User(name=username, password=hashed)
        user.save()

        # 将用户信息存入cookie
        response = HttpResponseRedirect('/')
        response.set_cookie('username', username)
        response.set_cookie('password', hashed)
        return response

    return HttpResponseRedirect('/')
