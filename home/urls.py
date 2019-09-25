# coding=utf-8

from django.conf.urls import url

from .views import (
    home, history, delete_all, delete_one, login, logout, register
)

urlpatterns = [
    url(r'^$', home),
    url(r'^history/$', history),
    url(r'^history/del_all/$', delete_all),
    url(r'^history/del', delete_one),
    url(r'^login/$', login),
    url(r'^logout/$', logout),
    url(r'^register/$', register),
]
