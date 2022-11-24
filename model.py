import torch
torch.manual_seed(7)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torch.nn as nn


class SvhnModel(nn.Module):
    def __init__(self):
        super(SvhnModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),stride=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.f1 = nn.Linear(32 * 3 * 7, 11)
        self.f2 = nn.Linear(32 * 3 * 7, 11)
        self.f3 = nn.Linear(32 * 3 * 7, 11)
        self.f4 = nn.Linear(32 * 3 * 7, 11)
        self.f5 = nn.Linear(32 * 3 * 7, 11)
        self.f6 = nn.Linear(32 * 3 * 7, 11)

    def forward(self,img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0],-1)
        c1 = self.f1(feat)
        c2 = self.f2(feat)
        c3 = self.f3(feat)
        c4 = self.f4(feat)
        c5 = self.f5(feat)
        c6 = self.f6(feat)
        return c1,c2,c3,c4,c5,c6

model = SvhnModel()

