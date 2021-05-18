
# 이미지 추가후 확인
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device)
from matplotlib.pyplot import imshow

#%matplotlib inline

trans = transforms.Compose([
    transforms.Resize((64, 128))
])#사이즈 변경    여기
train_data = torchvision.datasets.ImageFolder(root='custom_data/origin_data', transform=trans) #파일주소 바꿔야함
# 저장해두고 다음번에 또쓰기 위해
'''
for num, value in enumerate(train_data):
    data, label = value
    print(num, data, label)

    if (label == 0):
        data.save('custom_data/train_data/gray/%d_%d.jpeg' % (num, label))
    else:
        data.save('custom_data/train_data/red/%d_%d.jpeg' % (num, label))
'''

data_loader = DataLoader(dataset = train_data, batch_size = 8, shuffle = True, num_workers=2)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16 * 13 * 29, 120),
            nn.ReLU(),
            nn.Linear(120, 2)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.layer3(out)
        return out

net = CNN().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.00005)
loss_func = nn.CrossEntropyLoss().to(device)

total_batch = len(data_loader)

epochs = 7
for epoch in range(epochs):
    avg_cost = 0.0
    for num, data in enumerate(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = net(imgs)
        loss = loss_func(out, labels)
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('[Epoch:{}] cost = {}'.format(epoch + 1, avg_cost))
print('Learning Finished!')

