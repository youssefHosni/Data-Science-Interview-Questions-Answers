# Deep Learning Questions #

## Questions ##
* Q1: What are autoencoders? Explain the different layers of autoencoders and mention three practical usages of them?
* Q2: What is an activation function and discuss the use of an activation function? Explain three different types of activation functions?
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

1.) Transformers and Big Bird (Autoencoders is one of these components in both algorithms): Text Summarizer, Text Generator
2.) Image compression
3.) Nonlinear version of PCA


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
