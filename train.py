import torch
import torch.nn as nn
from model import *
from dataloader import *

criterion = nn.CrossEntropyLoss()
learn_rate = 0.005
optimizer = torch.optim.Adam(model.parameters(),learn_rate)

loss_polt , c1_polt= [],[]
epochs = 10
for epoch in range(epochs):
    for data in train_loader:
        c1, c2, c3, c4, c5, c6 = model(data[0])
        loss = criterion(c1,data[1][:,0])+\
                criterion(c2,data[1][:,1])+\
                criterion(c3,data[1][:,2])+\
                criterion(c4,data[1][:,3])+\
                criterion(c5,data[1][:,5])+\
                criterion(c6,data[1][:,5])
        loss /=6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_polt.append(loss.item())
            # 看一下第一个数字的精度
        c1_polt.append((c1.argmax(1)==data[1][:,0]).sum().item()*1.0/c1.shape[0])

    print('epoch',epoch)


    def validate(val_loader, model, criterion):
        # 切换模型为预测模型
        model.eval()
        val_loss = []

        # 不记录模型梯度信息
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                c0, c1, c2, c3, c4, c5 = model(data[0])
                loss = criterion(c0, data[1][:, 0]) + \
                       criterion(c1, data[1][:, 1]) + \
                       criterion(c2, data[1][:, 2]) + \
                       criterion(c3, data[1][:, 3]) + \
                       criterion(c4, data[1][:, 4]) + \
                       criterion(c5, data[1][:, 5])
                loss /= 6
                val_loss.append(loss.item())
        return np.mean(val_loss)

