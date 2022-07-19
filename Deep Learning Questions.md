# Deep Learning Questions #

## Questions ##
* [Q1: What are autoencoders? Explain the different layers of autoencoders and mention three practical usages of them?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Deep%20Learning%20Questions.md#:~:text=Questions%20%26%20Answers-,Q1%3A%20What%20are%20autoencoders%3F%20Explain%20the%20different%20layers%20of%20autoencoders%20and%20mention%20three%20practical%20usages%20of%20them%3F,-Answer%3A)
* [Q2: What is an activation function and discuss the use of an activation function? Explain three different types of activation functions?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Deep%20Learning%20Questions.md#:~:text=version%20of%20PCA-,Q2%3A%20What%20is%20an%20activation%20function%20and%20discuss%20the%20use%20of%20an%20activation%20function%3F%20Explain%20three%20different%20types%20of%20activation%20functions%3F,-Answer%3A)
* [Q3: You are using a deep neural network for a prediction task. After training your model, you notice that it is strongly overfitting the training set and that the performance on the test isn’t good. What can you do to reduce overfitting?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Deep%20Learning%20Questions.md#:~:text=Q3%3A%20You%20are%20using%20a%20deep%20neural%20network%20for%20a%20prediction%20task.%20After%20training%20your%20model%2C%20you%20notice%20that%20it%20is%20strongly%20overfitting%20the%20training%20set%20and%20that%20the%20performance%20on%20the%20test%20isn%E2%80%99t%20good.%20What%20can%20you%20do%20to%20reduce%20overfitting%3F)
* [Q4: Why should we use Batch Normalization?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Deep%20Learning%20Questions.md#:~:text=Q4%3A%20Why%20should%20we%20use%20Batch%20Normalization%3F)
* [Q5:How to know whether your model is suffering from the problem of Exploding Gradients?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Deep%20Learning%20Questions.md#:~:text=Q5%3A%20How%20to%20know%20whether%20your%20model%20is%20suffering%20from%20the%20problem%20of%20Exploding%20Gradients%3F)
* [Q6: Can you name and explain a few hyperparameters used for training a neural network?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Deep%20Learning%20Questions.md#:~:text=layer%20during%20training.-,Q6%3A%20Can%20you%20name%20and%20explain%20a%20few%20hyperparameters%20used%20for%20training%20a%20neural%20network%3F,-Answer%3A)
* [Q7: Can you explain the parameter sharing concept in deep learning?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Deep%20Learning%20Questions.md#:~:text=learning%20rate%20right.-,Q7%3A%20Can%20you%20explain%20the%20parameter%20sharing%20concept%20in%20deep%20learning%3F,-Answer%3A)
* [Q8: Describe the architecture of a typical Convolutional Neural Network (CNN)?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Deep%20Learning%20Questions.md#:~:text=Locally%2DConnected%20Layer.-,Q8%3A%20Describe%20the%20architecture%20of%20a%20typical%20Convolutional%20Neural%20Network%20(CNN)%3F,-Footer)
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
  * can fully connect layers be replaced with convolutional + pooling layers?
  * what is the justification for the number of layers and number of neurons chosen? Given how hard it is to tune these, can a pre-trained model be used?
  * Add regularization - ridge (l1), lasso (l2), elastic net (both)
* Add dropouts
* Add batch normalization

3. The training process:
* Improvements in validation losses should decide when to stop training. Use callbacks for early stopping when there are no significant changes in the validation loss and restore_best_weights.

### Q4: Why should we use Batch Normalization? ###

Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch.

Usually, a dataset is fed into the network in the form of batches where the distribution of the data differs for every batch size. By doing this, there might be chances of vanishing gradient or exploding gradient when it tries to backpropagate. In order to combat these issues, we can use BN (with irreducible error) layer mostly on the inputs to the layer before the activation function in the previous layer and after fully connected layers.


Batch Normalisation has the following effects on the Neural Network:

1. Robust Training of the deeper layers of the network.
2. Better covariate-shift proof NN Architecture.
3. Has a slight regularisation effect.
4. Centred and Controlled values of Activation.
5. Tries to Prevent exploding/vanishing gradient.
6. Faster Training/Convergence to the minimum loss function

![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Batch%20normalization.jpg)

### Q5: How to know whether your model is suffering from the problem of Exploding Gradients? ###

By taking incremental steps towards the minimal value, the gradient descent algorithm aims to minimize the error. The weights and biases in a neural network are updated using these processes. However, at times, the steps grow excessively large, resulting in increased updates to weights and bias terms to the point where the weights overflow (or become NaN, that is, Not a Number). An exploding gradient is the result of this, and it is an unstable method.

There are some subtle signs that you may be suffering from exploding gradients during the training of your network, such as:

1. The model is unable to get traction on your training data (e g. poor loss).
2. The model is unstable, resulting in large changes in loss from update to update.
3. The model loss goes to NaN during training.

If you have these types of problems, you can dig deeper to see if you have a problem with exploding gradients. There are some less subtle signs that you can use to confirm that you have exploding gradients:

1. The model weights quickly become very large during training.
2. The model weights go to NaN values during training.
3. The error gradient values are consistently above 1.0 for each node and layer during training.

### Q6: Can you name and explain a few hyperparameters used for training a neural network? ###

Answer:

Hyperparameters are any parameter in the model that affects the performance but is not learned from the data unlike parameters ( weights and biases), the only way to change it is manually by the user.


1. Number of nodes: number of inputs in each layer.

2. Batch normalization: normalization/standardization of inputs in a layer.

3. Learning rate: the rate at which weights are updated.

4. Dropout rate: percent of nodes to drop temporarily during the forward pass.

5. Kernel: matrix to perform dot product of image array with

6. Activation function: defines how the weighted sum of inputs is transformed into outputs (e.g. tanh, sigmoid, softmax, Relu, etc)

7. Number of epochs: number of passes an algorithm has to perform for training

8. Batch size: number of samples to pass through the algorithm individually. E.g. if the dataset has 1000 records and we set a batch size of 100 then the dataset will be divided into 10 batches which will be propagated to the algorithm one after another.

9. Momentum: Momentum can be seen as a learning rate adaptation technique that adds a fraction of the past update vector to the current update vector. This helps damps oscillations and speed up progress towards the minimum.

10. Optimizers: They focus on getting the learning rate right.

* Adagrad optimizer: Adagrad uses a large learning rate for infrequent features and a smaller learning rate for frequent features.

* Other optimizers, like Adadelta, RMSProp, and Adam, make further improvements to fine-tuning the learning rate and momentum to get to the optimal weights and bias. Thus getting the learning rate right is key to well-trained models.

11. Learning Rate: Controls how much to update weights & bias (w+b) terms after training on each batch. Several helpers are used to getting the learning rate right.

### Q7: Can you explain the parameter sharing concept in deep learning? ###
Answer:
Parameter sharing is the method of sharing weights by all neurons in a particular feature map. Therefore helps to reduce the number of parameters in the whole system, making it computationally cheap. It basically means that the same parameters will be used to represent different transformations in the system. This basically means the same matrix elements may be updated multiple times during backpropagation from varied gradients. The same set of elements will facilitate transformations at more than one layer instead of those from a single layer as conventional. This is usually done in architectures like Siamese that tend to have parallel trunks trained simultaneously. In that case, using shared weights in a few layers( usually the bottom layers) helps the model converge better. This behavior, as observed, can be attributed to more diverse feature representations learned by the system. Since neurons corresponding to the same features are triggered in varied scenarios. Helps to model to generalize better.



Note that sometimes the parameter sharing assumption may not make sense. This is especially the case when the input images to a ConvNet have some specific centered structure, where we should expect, for example, that completely different features should be learned on one side of the image than another. 

One practical example is when the input is faces that have been centered in the image. You might expect that different eye-specific or hair-specific features could (and should) be learned in different spatial locations. In that case, it is common to relax the parameter sharing scheme, and instead, simply call the layer a Locally-Connected Layer.

### Q8: Describe the architecture of a typical Convolutional Neural Network (CNN)? ###


* In a typical CNN architecture, a few convolutional layers are connected in a cascade style.
* Each convolutional layer is followed by a Rectified Linear Unit (ReLU) layer, then a pooling layer, then one or more convolutional layers (+ReLU), then another pooling layer.
* The output from each convolution layer is a set of objects called feature maps, generated by a single kernel filter.
* The feature maps are used to define a new input to the next layer.
* At the end, there is one or more fully connected layers.
* Pretty much depending on problem type, the network might be deep though.

![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/CNN_architecture.png)


