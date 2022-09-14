import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torchvision import datasets, transforms
import time
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output_layer = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        output = self.output_layer(x)
        return output

EPOCH = 2
BATCH_SIZE = 100
LR = 0.001

train_data = datasets.MNIST(root='./data',  train=True,  transform=transforms.ToTensor(), download=True)

# DataLoader
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

test_data = datasets.MNIST(root='./data',train=False)

cnn = CNN()
cnn.cuda()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

loss_func = nn.CrossEntropyLoss()

# 为了节约时间，只使用测试集前3000个数据
test_x = Variable(torch.unsqueeze(test_data.data, dim=1),volatile=True).type(torch.FloatTensor)[:3000] / 255 

test_y = test_data.targets[:3000]

# 将测试数据移到GPU上
test_x = test_x.cuda()
test_y = test_y.cuda()

start = time.time()

# 训练神经网络
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = cnn(test_x)
            predict_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = (predict_y == test_y).sum().item() / test_y.size(0)
            print('Epoch', epoch, '|', 'Step', step, '|', 'Loss', loss.data.item(), '|', 'Test Accuracy', accuracy)

end = time.time()

print('Time cost:', end - start, 's')

# 预测
test_output = cnn(test_x[:10])
predict_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()
real_y = test_y[:10].cpu().numpy()
print("预测：",predict_y)
print("实际：",real_y)
