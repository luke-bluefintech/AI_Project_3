# Project 3: Machine Learning & Artifical Neural Networks
Sophia Strano, Luke Foley, Ilana Whittaker

## Written Report

### Model & Training Procedure
This project dives into building and training Deep Neural Networks (DNNs) for image classification: data preprocessing, model architecture and hyperparameter selection. We are using a multi-layered convolutional neural network (CNN). When determining our  hyperparameters, we restrict the number of epochs to 30, as our model’s testing accuracy plateaus and then decreases after this number. The hidden size is 300, as a higher hidden size continuously increases our testing accuracy, however this number greatly increases the time cost of running the algorithm the higher it is set at
Our scale factor is set at 100 because numbers higher or lower than this value decrease our testing accuracy.
The optimizer that we are using is Adam because it achieves a high initial accuracy, despite its performance gap between other optimizers.

In our CNN, we are using 40 filters to find patterns in our images, as a relatively high filter value increases our training accuracy. We experimented with various Our filter size is 3, because size these images are only 2x2, a size larger than this is too much to analyze to find a pattern, and a size smaller does not have enough information.Our pooling size is 5, which is relatively large, and is done because this helps us to find averages of values along long stretches of the image. Our model also has three layers, tanh, tanh, and relu activation functions respectively,with a high number of analyzed values to achieve the best performance. Smaller and larger networks tended to be less accurate for this data set, and analyzing small proportions of the data set would not affect the testing accuracy by much.

### Model Performance & Confusion Matrix

The aspects of our network that impact the testing accuracy the most are number of epochs, network architecture, and pooling size value. In our best performing model, we obtained accuracies ranging between 31%-63%, with our best accuracy being 63.48%. The output of our best performing model is located in the Model folder.

In our best performing model, our precision and accuracy were

Precision:  [0.67630701, 0.78548559, 0.50965961, 0.43686354, 0.52463504, 0.59572072,
 0.61793215, 0.78580815, 0.76869392, 0.70694319]
 
Recall:  [0.608, 0.736, 0.554, 0.429, 0.575, 0.529, 0.765, 0.598, 0.771, 0.784]

Here is a confusion matrix for the results of our best performing model:

![BestGraph](https://user-images.githubusercontent.com/64103447/195634316-eff6334d-7de5-4b64-9898-ac6eaa1dcd67.png)
![image](https://user-images.githubusercontent.com/64103447/195641687-a7fc1647-4333-4005-a458-4cf1a1546ad9.png)

To design our best performing model, we used the following activation functions in our training procedure:

```
model.add(Flatten())
model.add(Dense((args.hidden_size)*7/8, activation="relu"))  # first layer
model.add(Dense((args.hidden_size)*5/6, activation='relu'))  # second layer
model.add(Dense((args.hidden_size)*1/5, activation='selu'))  # third layer
```

### Training performance plot
The following plot represents our model's training accuracy and validation accuracy with respect to the number of training epochs (x axis) and accuracy (y axis).

 ![desmos-graph](https://user-images.githubusercontent.com/64103447/195652860-021b324f-a8e7-4d97-8c2b-2bed9264e743.png)



### Misclassified Visualizations


Here are three of the images our best performing model misclassified: 
The written report includes a visualization of three of the images that were misclassified by your best-performing model.


