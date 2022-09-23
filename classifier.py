import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, class_num, drop_out_rate=0.25):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(drop_out_rate)
        self.fc1 = nn.Linear(126, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, class_num)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)