# Project 3: Artifical Neural Networks
Sophia Strano, Luke Foley, Ilana Whittaker

## Written Report

This project dives into building and training Deep Neural Networks (DNNs) for image classification: data preprocessing, model architecture and hyperparameter selection.
We are using a multi-layered convolutional neural network (CNN).When determining our  hyperparameters, the number of epochs is restricted to 30, as our model’s testing accuracy plateaus and then decreases after this number.
The hidden size is 300, as a higher hidden size continuously increases our testing accuracy, however this number greatly increases the time cost of running the algorithm the higher it is set at
Our scale factor is set at 100 because numbers higher or lower than this value decrease our testing accuracy.
The optimizer that we are using is Adam because it achieves a high initial accuracy, despite its performance gap between other optimizers.

