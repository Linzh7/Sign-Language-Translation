import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import signdataset
import classifier
import labelmap
import torch.optim as optim

CLASS_NUM = len(labelmap.label_map)
N_EPOCHS = 10
DROP_OUT_RATE = 0.25

train_dataloader = DataLoader(signdataset.SignDataset('./train.csv'),
                              batch_size=1,
                              shuffle=True)
test_dataloader = DataLoader(signdataset.SignDataset('./test.csv'),
                             batch_size=1,
                             shuffle=True)
# examples = enumerate(test_dataloader)
# batch_index, (example_data, example_targets) = next(examples)

net = classifier.Classifier(CLASS_NUM, DROP_OUT_RATE)
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.5)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_dataloader.dataset) for i in range(N_EPOCHS + 1)]


def train(epoch):
    net.train()
    for batch_index, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_index % 500 == 0:
            print('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_dataloader.dataset),
                100. * batch_index / len(train_dataloader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_index * 64) +
                                 ((epoch - 1) * len(train_dataloader.dataset)))
            torch.save(net.state_dict(), './models/model.pth')
            torch.save(optimizer.state_dict(), './models/optimizer.pth')


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)
    print('[Test] Avg loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))


if __name__ == '__main__':
    test()
    for epoch in range(1, N_EPOCHS + 1):
        train(epoch)
        test()