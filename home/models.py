from django.db import models
from django.core.exceptions import ObjectDoesNotExist


class User(models.Model):
    name = models.CharField(max_length=20, primary_key=True)
    password = models.CharField(max_length=78)

    @staticmethod
    def check_user(request):
        """
        Check the user's identity
        if no this user return 'KeyError'
        if the password does not match return 'PasswordNotMatch'
        :param request:
        :return: User
        """
        # Get user's information
        try:
            name = request.COOKIES['username']
        except KeyError:
            return 'KeyError'

        try:
            password = request.COOKIES['password']
        except KeyError:
            return 'KeyError'

        # Match
        try:
            user = User.objects.get(name=name)
        except ObjectDoesNotExist:
            return 'UserNotExist'

        if password != user.password:  # 密码错误
            return 'PasswordNotMatch'

        return user


class History(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    image = models.ImageField(upload_to='images/')
    prediction = models.ImageField(upload_to='predictions/', null=True)
    upload_time = models.DateTimeField()
    spend_time = models.FloatField()
    # 最近一批提交的图像，需要在result页面展示
    recent = models.BooleanField(default=True)
    # 浅删除标记
    is_deleted = models.BooleanField(default=False)
    deleted_time = models.DateTimeField(null=True)
