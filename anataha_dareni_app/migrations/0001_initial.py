# Generated by Django 3.0 on 2021-08-21 06:05

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ModelFile',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('proba', models.FloatField(default=0.0, verbose_name='信頼度（確率）')),
            ],
        ),
    ]
