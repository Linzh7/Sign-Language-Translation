import os
import LinzhUtil
from sklearn.model_selection import train_test_split

path = './dataset/'

file_list = LinzhUtil.getFileList('./dataset/')

with open('./train.csv', 'w') as train_file, open('./test.csv',
                                                  'w') as test_file:
    for file in file_list:
        if file.endswith('.csv'):
            with open(os.path.join(path, file), 'r') as read_file:
                records = read_file.readlines()
                records[-1] += '\n'
                train, test = train_test_split(records, test_size=0.25)
                train_file.writelines(train)
                test_file.writelines(test)
print('Done.')