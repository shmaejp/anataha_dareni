from django.db import models

class ModelFile(models.Model):
    famous_person_options = (
        (0, '阿部寛'),
        (1, 'イチロー'),
        (2, '木村拓哉'),
        (3, '堺正章'),
        (4, '田中圭'),
        (5, '中居正広'),
        (6, 'ダルビッシュ有'),
        (7, '福山雅治'),
        (8, '三浦翔平'),
        (9, '横浜流星'),
        (10, '新垣結衣'),
        (11, '綾瀬はるか'),
        (12, '有村架純'),
        (13, '石原さとみ'),
        (14, '北川景子'),
        (15, '戸田恵梨香'),
        (16, '二階堂ふみ'),
        (17, '橋本環奈'),
        (18, '浜辺美波'),
        (19, '広瀬すず'),
        (20, '該当者なし')
    )

    id = models.AutoField(primary_key=True)
    image = models.ImageField(upload_to = 'documents/', blank=True, null=True)
    label = models.IntegerField('推論結果（ラベル）', choices=famous_person_options, default=20)
    proba = models.FloatField('信頼度（確率）', default=0.0)

    # def __str__(self):
    #     return self.title