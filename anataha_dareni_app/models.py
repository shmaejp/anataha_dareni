from django.db import models

class ModelFile(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.ImageField(upload_to = 'documents/')
    label = models.IntegerField('推論結果（ラベル）', blank=True, null=True)
    proba = models.FloatField('信頼度（確率）', default=0.0)

    # def __str__(self):
    #     return self.title