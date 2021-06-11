import os
import tqdm
import numpy as np

import torch
import torchvision
from torchvision import transforms

from nets.nn import resnet50
from utils.loss import Loss
from utils.dataset import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root = '../Dataset/'
learning_rate = 0.001
num_epochs = 20
batch_size = 32
seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

net = resnet50()
print(net)
resnet = torchvision.models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()

net_dict = net.state_dict()
for k in new_state_dict.keys():
    if k in net_dict.keys() and not k.startswith('fc'):
        net_dict[k] = new_state_dict[k]
net.load_state_dict(net_dict)

print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

criterion = Loss(7, 2, 5, 0.5)

net = net.to(device)

if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)

net.train()

# different learning rate
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr': learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr': learning_rate}]

optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

with open('../Dataset/train.txt') as f:
    train_names = f.readlines()
train_dataset = Dataset(root, train_names, train=True, transform=[transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=os.cpu_count())

with open('../Dataset/test.txt') as f:
    test_names = f.readlines()
test_dataset = Dataset(root, test_names, train=False, transform=[transforms.ToTensor()])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False,
                                          num_workers=os.cpu_count())

print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
print(f'BATCH SIZE: {batch_size}')

num_iter = 0
best_test_loss = np.inf

for epoch in range(num_epochs):
    net.train()
    # if epoch == 1:
    #     learning_rate = 0.0005
    # if epoch == 2:
    #     learning_rate = 0.00075
    # if epoch == 3:
    #     learning_rate = 0.001
    if epoch == 30:
        learning_rate = 0.0001
    if epoch == 40:
        learning_rate = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    # training
    total_loss = 0.
    print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
    progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, target) in progress_bar:
        images = images.to(device)
        target = target.to(device)

        pred = net(images)
        loss = criterion(pred, target)
        total_loss += loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, num_epochs), total_loss / (i + 1), mem)
        progress_bar.set_description(s)

    # validation
    validation_loss = 0.0
    net.eval()
    progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
    for i, (images, target) in progress_bar:
        images = images.to(device)
        target = target.to(device)

        prediction = net(images)
        loss = criterion(prediction, target)
        validation_loss += loss.data
    validation_loss /= len(test_loader)

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss

        save = {'state_dict': net.state_dict()}
        torch.save(save, 'best.pth')

    save = {'state_dict': net.state_dict()}
    torch.save(save, 'yolo.pth')