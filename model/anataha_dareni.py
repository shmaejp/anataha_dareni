import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.models import resnet18
from torchvision.models import densenet169

# class TrainNet(pl.LightningModule):

#     # @pl.data_loader
#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True)

#     def training_step(self, batch, batch_nb):
#         x, t = batch
#         x = x.float()
#         y = self.forward(x)
#         loss = self.lossfun(y, t)
#         results = {'loss' : loss}
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         self.log('train_acc', accuracy(y.softmax(dim=-1), t), on_step=True, on_epoch=True, prog_bar=True)
#         return results

# class ValidationNet(pl.LightningModule):

#     # @pl.data_loader
#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(train, self.batch_size)

#     def validation_step(self, batch, batch_nb):
#         x, t = batch
#         x = x.float()
#         y = self.forward(x)
#         loss = self.lossfun(y, t)
#         y_label = torch.argmax(y, dim=1)
#         acc = torch.sum(t == y_label) * 1.0 / len(t)
#         results = {'val_loss': loss, 'val_acc': acc}
#         self.log('val_loss', loss, on_step=False, on_epoch=True)
#         self.log('val_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
#         return results

#     def validation_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
#         results =  {'val_loss': avg_loss, 'val_acc': avg_acc}
#         return results

class Net():

    def __init__(self, hidden_size=50, output_size=50, batch_size=10, lr=0.1):
        super().__init__()
        self.lr = lr
        # self.feature = resnet18(pretrained=True)  # ResNet-18 を特徴抽出機として使用
        self.feature = densenet169(pretrained=True)  # DenseNet-169 を特徴抽出機として使用
        self.fc = nn.Linear(1000, output_size)
        self.batch_size = batch_size

    def lossfun(self, y, t):
        return F.cross_entropy(y, t)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer