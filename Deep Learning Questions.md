# Deep Learning Questions #

## Questions ##
* [Q1: What are autoencoders? Explain the different layers of autoencoders and mention three practical usages of them?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Deep%20Learning%20Questions.md#:~:text=Questions%20%26%20Answers-,Q1%3A%20What%20are%20autoencoders%3F%20Explain%20the%20different%20layers%20of%20autoencoders%20and%20mention%20three%20practical%20usages%20of%20them%3F,-Answer%3A)
* [Q2: What is an activation function and discuss the use of an activation function? Explain three different types of activation functions?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Deep%20Learning%20Questions.md#:~:text=version%20of%20PCA-,Q2%3A%20What%20is%20an%20activation%20function%20and%20discuss%20the%20use%20of%20an%20activation%20function%3F%20Explain%20three%20different%20types%20of%20activation%20functions%3F,-Answer%3A)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Questions & Answers ##

### Q1: What are autoencoders? Explain the different layers of autoencoders and mention three practical usages of them? ###

Answer:

Autoencoders are one of the deep learning types used for unsupervised learning. There are key layers of autoencoders, which are the input layer, encoder, bottleneck hidden layer, decoder, and output.

The three layers of the autoencoder are:-
1) Encoder - Compresses the input data to an encoded representation which is typically much smaller than the input data.
2) Latent Space Representation/ Bottleneck/ Code - Compact summary of the input containing the most important features
3) Decoder - Decompresses the knowledge representation and reconstructs the data back from its encoded form.
Then a loss function is used at the top to compare the input and output images.
NOTE- It's a requirement that the dimensionality of the input and output be the same. Everything in the middle can be played with.

Autoencoders have a wide variety of usage in the real world. The following are some of the popular ones:

1. Transformers and Big Bird (Autoencoders is one of these components in both algorithms): Text Summarizer, Text Generator
2. Image compression
3. Nonlinear version of PCA


### Q2: What is an activation function and discuss the use of an activation function? Explain three different types of activation functions? ###

Answer:

In mathematical terms, the activation function serves as a gate between the current neuron input and its output, going to the next level. Basically, it decides whether neurons should be activated or not.
It is used to introduce non-linearity into a model.

Activation functions are added to introduce non-linearity to the network, it doesn't matter how many layers or how many neurons your net has, the output will be linear combinations of the input in the absence of activation functions. In other words, activation functions are what make a linear regression model different from a neural network. We need non-linearity, to capture more complex features and model more complex variations that simple linear models can not capture.

There are a lot of activation functions:

* Sigmoid function: f(x) = 1/(1+exp(-x))

The output value of it is between 0 and 1, we can use it for classification. It has some problems like the gradient vanishing on the extremes, also it is computationally expensive since it uses exp.

* Relu: f(x) = max(0,x)

it returns 0 if the input is negative and the value of the input if the input is positive. It solves the problem of vanishing gradient for the positive side, however, the problem is still on the negative side. It is fast because we use a linear function in it.

* Leaky ReLU:

F(x)= ax, x<0
F(x)= x, x>=0

It solves the problem of vanishing gradient on both sides by returning a value “a” on the negative side and it does the same thing as ReLU for the positive side.

* Softmax: it is usually used at the last layer for a classification problem because it returns a set of probabilities, where the sum of them is 1. Moreover, it is compatible with cross-entropy loss, which is usually the loss function for classification problems.


### Q3: You are using a deep neural network for a prediction task. After training your model, you notice that it is strongly overfitting the training set and that the performance on the test isn’t good. What can you do to reduce overfitting? ###

To reduce overfitting in a deep neural network changes can be made in three places/stages: The input data to the network, the network architecture, and the training process:

1. The input data to the network:

* Check if all the features are available and reliable
* Check if the training sample distribution is the same as the validation and test set distribution. Because if there is a difference in validation set distribution then it is hard for the model to predict as these complex patterns are unknown to the model.
* Check for train / valid data contamination (or leakage)
* The dataset size is enough, if not try data augmentation to increase the data size
* The dataset is balanced

2. Network architecture:
* Overfitting could be due to model complexity. Question each component:
** can fully connect layers be replaced with convolutional + pooling layers?
** what is the justification for the number of layers and number of neurons chosen? Given how hard it is to tune these, can a pre-trained model be used?
** Add regularization - ridge (l1), lasso (l2), elastic net (both)
* Add dropouts
* Add batch normalization

3. The training process:
* Improvements in validation losses should decide when to stop training. Use callbacks for early stopping when there are no significant changes in the validation loss and restore_best_weights.

