#공부한 시간과 점수의 연관관계 -> 입력다수 출력은 적음

import torch
import numpy as np
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# Hypothesis y=Wx+b

W = torch.zeros(1, requires_grad=True)#초기화를 0으로 해주고 학습하고 싶다는 뜻
b = torch.zeros(1, requires_grad=True)#초기화를 0으로 해주고 학습하고 싶다는 뜻
#hypothesis = x_train*W+b# Hypothesis y=Wx+b

#cost = torch.mean((hypothesis-y_train)**2)#cost값 계산 ->mse와 같다
optimizer = torch.optim.SGD([W,b], lr=0.001)# optimizer 설정 // 경사 하강법 같은거 gd

nb_epoch = 100000

for epoch in range(1, nb_epoch+1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2) # 함수 만들기
    optimizer.zero_grad()# gradient를 0으로 초기화 -> 그래야 이전에 영향을 안받음
    cost.backward()# cost값을 계산해서 gradient 계산
    optimizer.step()#gradient descent ->  학습 기울기에 lr을 곱해주면서 움직이는것
print(W)
print(b)