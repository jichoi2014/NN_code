#multivariate Linear Regression
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
x_train = torch.FloatTensor([[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# Hypothesis y=Wx+b
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)

W = torch.zeros((3, 1), requires_grad=True)#초기화를 0으로 해주고 학습하고 싶다는 뜻/ (3, 1) 3으로들어가서 1로나오는거 표현
b = torch.zeros(1, requires_grad=True)#초기화를 0으로 해주고 학습하고 싶다는 뜻


#model = MultivariateLinearRegressionModel()
#cost = torch.mean((hypothesis-y_train)**2)#cost값 계산 ->mse와 같다
optimizer = torch.optim.SGD([W, b], lr=1e-5)# optimizer 설정 // 경사 하강법 같은거 gd

nb_epoch = 20

for epoch in range(1, nb_epoch+1):

    hypothesis = x_train.matmul(W)+b
    cost = torch.mean((hypothesis - y_train) ** 2)#함수 만들기
    optimizer.zero_grad()
    # gradient를 0으로 초기화 -> 그래야 이전에 영향을 안받음
    cost.backward()# cost값을 계산해서 gradient 계산
    optimizer.step()#gradient descent ->  학습 기울기에 lr을 곱해주면서 움직이는것
    print('Epoch {:4d}/{} hypothesis: {} Cost: {: .6f}'.format(epoch, nb_epoch, hypothesis.squeeze().detach(),
                                                               cost.item()))
print(W)
print(b)