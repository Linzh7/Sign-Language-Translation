from numpy import genfromtxt


class SignDataset():
    def __init__(self, file_path):
        self.data = genfromtxt(file_path, delimiter=',')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :-1], self.data[index, -1]
