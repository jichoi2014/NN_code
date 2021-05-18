import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
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
# loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size, shuffle=True, drop_last=True)


# omdel 만들기
class CNN(nn.Module):

    def __init__(self):  # 초기화
        super(CNN, self).__init__()  # 필수 빼먹으면 학습이안된다.
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.fc1 = nn.Linear(4*4*128, 625, bias=True)
        self.layer4 = torch.nn.Sequential(self.fc1, torch.nn.ReLU())
        self.fc2 = nn.Linear(625, 10, bias=True)

        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out


model = CNN().to(device)


criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
total_batch = len(data_loader)  # rlfdl dkfdkqhrl
for epoch in range(training_epochs):
    avg_cost = 0
    for X, Y in data_loader:  # X는 이미지 Y는 label
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()  # 안하면 학습이 안된다.
        hypotesis = model(X)
        cost = criterion(hypotesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
print('Learning Finished!')

with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

