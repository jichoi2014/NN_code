#backpropagation
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(0)
if device == 'cuda':
    torch.cuda.manual_seed_all(0)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)


linear1 = torch.nn.Linear(2, 2, bias=True)
linear2 = torch.nn.Linear(2, 1, bias=True)
'''
#nn layers
w1 = torch.Tensor(2, 2).to(device)
b1 - torch.Tensor(2).to(device)
w2 - torch.Tensor(2, 1).to(device)
b1 - torch.Tensor(2).to(device)
'''


sigmoid = torch.nn.Sigmoid()
'''
def sigmiod(x):
return 1.0/(1.0+torch.exp(-x))

def sigmoid_prime(x):
return sigmoid(x)*(1-sigmoid(x))
'''
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)

'''
#model
l1 = torch.matmul(X, w1), b1)
a1 = sigmoid(l1)
 l2 = torch.matmul(a1, w2), b2)
 Y_pred = sigmoid(l2)

'''

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)  # modified learning rate from 0.1 to 1

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    # cost/loss function
    cost = criterion(hypothesis, Y)#BCE
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, cost.item())

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('\nHypothesis: ', hypothesis.detach().cpu().numpy(), '\nCorrect: ', predicted.detach().cpu().numpy(), '\nAccuracy: ', accuracy.item())
