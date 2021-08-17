from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from .forms import ImageForm, LoginForm, SignUpForm
from .models import ModelFile
from model.anataha_dareni import Net
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# @login_required
def image_upload(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            image_name = request.FILES['image']
            image_url = 'media/documents/{}'.format(image_name)

            # 推論処理は別関数(inference)として定義
            y, y_proba = inference(image_url)
            # print('y:', y)
            # print('y_proba:', y_proba)

            # 推論処理後に、推論結果（ラベル）と信頼度（確率）を DB に格納する
            model = ModelFile.objects.order_by('id').reverse()[0]   # ModelFileの切り出し
            model.label = y[0]    # 推論結果（ラベル）
            model.proba = y_proba[0]    # 信頼度（確率）
            # print('y[0]:', y[0])
            # print('y_proba[0]:', y_proba[0])
            model.save() # データをDBに保存

            # 推論結果をHTMLに渡す
            return render(request, 'send_image_app/classify.html', {'y':y[0], 'y_proba':y_proba[0], 'image_url':image_url})
    else:
        form = ImageForm()
        return render(request, 'send_image_app/index.html', {'form':form})

def inference(image_url):
    # ネットワークの準備
    net = Net().cpu().eval()

    # 重みの読み込み
    net.load_state_dict(torch.load('model/send_image.pt', map_location=torch.device('cpu')))

    image = Image.open(image_url)
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.transpose(image, (2, 0, 1))
    data = torch.from_numpy(image)

    # 推論の実行
    # print(data)
    x = data
    # print(type(x))
    y = net(data.float().unsqueeze(0))
    y = F.softmax(y)
    
    y_proba, y = y.topk(1, dim=1)
    
    # 予測ラベル
    y = y.view(-1).to('cpu').detach().numpy()
    
    # 信頼度
    y_proba = y_proba.view(-1).to('cpu').detach().numpy()
    y_proba = np.round(y_proba * 100, 1)
    # print('y:', y)
    # print('y_proba:', y_proba)
    return y, y_proba

class Login(LoginView):
    from_class = LoginForm
    template_name = 'send_image_app/login.html'

# ログアウトページ
class Logout(LogoutView):
    template_name = 'send_image_app/base.html'

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            #フォームから'username'を読み取る
            username = form.cleaned_data.get('username')
            #フォームから'password1'を読み取る
            password = form.cleaned_data.get('password1')
            # 読み取った情報をログインに使用する情報として new_user に格納
            new_user = authenticate(username=username, password=password)
            if new_user is not None:
                # new_user の情報からログイン処理を行う
                login(request, new_user)
                # ログイン後のリダイレクト処理
                return redirect('imageupload')
        else:
            print('form is not valid.')
    # POST で送信がなかった場合の処理
    else:
        form = SignUpForm()
        return render(request, 'send_image_app/signup.html', {'form': form})