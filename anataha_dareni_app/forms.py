from django import forms
# from django.db.models.base import Model
from django import forms
from .models import ModelFile
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

class ImageForm(forms.ModelForm):
    class Meta:
        model = ModelFile
        fields = ('image',)
        exclude = ('id',)

# ログインフォーム
class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'
            field.widget.attrs['placeholder'] = field.label

class SignUpForm(UserCreationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #htmlの表示を変更可能にします
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'
            field.widget.attrs['placeholder'] = field.label

    class Meta:
        model = User
        fields = ('username', 'password1', 'password2', )