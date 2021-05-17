import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init
import torchvision

import visdom
vis = visdom.Visdom()#해보고 안되면 서버 키기

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

vis.text("Hello, world!",env="main")# env 한번에 모든창을 끌때 사용
"""
a=torch.randn(3,200,200)
vis.image(a)
vis.images(torch.Tensor(3,3,28,28)) 여러개 이미지 띄우기
"""

print(device)
learning_rate = 0.001
training_epochs = 15
batch_size = 100
mnist_train = dsets.MNIST(root='F:/MNIST- 미해결/m_data',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='F:/MNIST- 미해결/m_data',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data = mnist_train.__getitem__(0)
vis.images(data[0], env="main")


data_loader = torch.utils.data.DataLoader(dataset = mnist_train,
                                          batch_size = 32,
                                          shuffle = False)
for num, value in enumerate(data_loader):
    value = value[0]
    print(value.shape)
    vis.images(value)
    break#여러개 불러오기

#그래프 그리기
Y_data = torch.randn(5)# x가 없으면 0~1로 표현된다
plt = vis.line (Y=Y_data)
X_data = torch.Tensor([1, 2, 3, 4, 5])
plt = vis.line(Y=Y_data, X=X_data)

#라인을 update하는법
Y_append = torch.randn(1)
X_append = torch.Tensor([6])
vis.line(Y=Y_append, X=X_append, win=plt, update='append')
#단일창에 여러 line 그리기
num = torch.Tensor(list(range(0,10)))
num = num.view(-1,1)
num = torch.cat((num,num),dim=1)

plt = vis.line(Y=torch.randn(10,2), X = num)

#line info 정보 넣기
plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', showlegend=True))

plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', legend = ['1번'],showlegend=True))
plt = vis.line(Y=torch.randn(10,2), X = num, opts=dict(title='Test', legend=['1번','2번'],showlegend=True))

#make function for update line
'''
def loss_tracker(loss_plot, loss_value, num):
    #num, loss_value, are Tensor
    vis.line(X=num,
             Y=loss_value,
             win = loss_plot,
             update='append'
             )
    plt = vis.line(Y=torch.Tensor(1).zero_())

    for i in range(500):
        loss = torch.randn(1) + i
        loss_tracker(plt, loss, torch.Tensor([i]))
'''
