# Generated by Django 2.2 on 2019-04-28 09:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0010_auto_20190427_1145'),
    ]

    operations = [
        migrations.AddField(
            model_name='history',
            name='spend_time',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
    ]
