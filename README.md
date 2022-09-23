# Sign-Language-Translation

This project is aimed to recognize and classify sign language, then translate it into spoken language.

## Stucture of project

```
.
├── LinzhUtil.py - my tools collection
├── README.md - the file you are reading
├── classifier.py - class of the model
├── dataset - there are datasets with lable name as file name
│   ├── bar.csv - named [label].csv
│   └── foo.csv
├── datasetmaker.py - use this to create datasets
├── datasetprocess.py - use this to combine datasets and split them in to train.csv and test.csv
├── handsdetect.py - class of keypoint detection
├── labelmap.py - add the sign you want to classify and their index here
├── main.py - you should run this to do the translation
├── models - storage the models we trained
│   ├── model.pth
│   └── optimizer.pth
├── requirements.txt - use for install libraries
├── signdataset.py - make a dataloader class for pytorch
├── test.csv - dataset for test
├── train.csv - dataset for train
├── train.py - train the model
└── visualdetect.py - to see how keypoint detection works
```

## How to get start

### Requirements

Make sure that u install all libraries.

We also provide a `requirements.txt` for pip. Use the following command to install.

``pip install -r requirements.txt``

NB: If you are using Macbook with Apple silicon, you can install `mediapipe-silicon` instead of `mediapipe`, but they said that mediapipe will support Apple silicon soon.

### Make dataset

1. run `datasetmaker.py`, such as `python datasetmaker.py` or `python3 datasetmaker.py`.
2. after see the window of camera cap, you can press **space** to capture this sign as a sample for dataset.
3. repeat 2, until you think they are enough.
4. press **esc** to exit.

### Process the dataset

1. run `datasetprocess.py`.
2. wait, until it exit.

### Train your model

1. run `train.py`.
2. wait, until it exit.

### Use model to predict

*working on it :)*