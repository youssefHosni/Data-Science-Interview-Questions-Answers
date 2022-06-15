# Machine Learning Interview Questions #

## Questions ##
* [Q1: Mention three ways to make your model robust to outliers?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Machine%20Learning%20Questions.md#:~:text=Questions%20%26%20Answers-,Q1%3A%20Mention%20three%20ways%20to%20make%20your%20model%20robust%20to%20outliers%3F,-Investigating%20the%20outliers)
* [Q2: Describe the motivation behind random forests and mention two reasons why they are better than individual decision trees?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Machine%20Learning%20Questions.md#:~:text=means%20losing%20information.-,Q2%3A%20Describe%20the%20motivation%20behind%20random%20forests%20and%20mention%20two%20reasons%20why%20they%20are%20better%20than%20individual%20decision%20trees%3F,-The%20motivation%20behind)
* [Q3: What are the differences and similarities between gradient boosting and random forest? and what are the advantage and disadvantages of each when compared to each other?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Machine%20Learning%20Questions.md#:~:text=of%20the%20crowd.%22-,Q3%3A%20What%20are%20the%20differences%20and%20similarities%20between%20gradient%20boosting%20and%20random%20forest%3F%20and%20what%20are%20the%20advantage%20and%20disadvantages%20of%20each%20when%20compared%20to%20each%20other%3F,-Similarities%3A)
* [Q4: What are L1 and L2 regularization? What are the differences between the two?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Machine%20Learning%20Questions.md#:~:text=of%20random%20forest.-,Q4%3A%20What%20are%20L1%20and%20L2%20regularization%3F%20What%20are%20the%20differences%20between%20the%20two%3F,-Answer%3A)
* [Q5: What are the Bias and Variance in a Machine Learning Model and explain the bias-variance trade-off?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Machine%20Learning%20Questions.md#:~:text=in%20the%20model.-,Q5%3A%20What%20are%20the%20Bias%20and%20Variance%20in%20a%20Machine%20Learning%20Model%20and%20explain%20the%20bias%2Dvariance%20trade%2Doff%3F,-Answer%3A)
* [Q6: Mention three ways to handle missing or corrupted data in a dataset?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Machine%20Learning%20Questions.md#:~:text=the%20input%20features.-,Q6%3A%20Mention%20three%20ways%20to%20handle%20missing%20or%20corrupted%20data%20in%20a%20dataset%3F,-Answer%3A)
* [Q7: Explain briefly the logistic regression model and state an example of when you have used it recently?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Machine%20Learning%20Questions.md#:~:text=dataset%20is%20large.-,Q7%3A%20Explain%20briefly%20the%20logistic%20regression%20model%20and%20state%20an%20example%20of%20when%20you%20have%20used%20it%20recently%3F,-Logistic%20regression%20is)
* [Q8:Explain briefly batch gradient descent, stochastic gradient descent, and mini-batch gradient descent? and what are the pros and cons for each of them?]()
* [Q9: Explain what is information gain and entropy in the context of decision trees?]()
* [Q10: Explain the linear regression model and discuss its assumption?]()
* [Q11: Explain briefly the K-Means clustering and how can we find the best value of K?]()
* [Q12: Define Precision, recall, and F1 and discuss the trade-off between them?]()



-------------------------------------------------------------------------------------------------------------------------------------------------------------

## Questions & Answers ##

### Q1: Mention three ways to make your model robust to outliers? ###

1. Investigating the outliers is always the first step in understanding how to treat them. After you understand the nature of why the outliers occurred you can apply one of the several methods mentioned below.

2. Add regularization that will reduce variance, for example, L1 or L2 regularization.

3. Use tree-based models (random forest, gradient boosting ) that are generally less affected by outliers.

4. Winsorize the data. Winsorizing or winsorization is the transformation of statistics by limiting extreme values in the statistical data to reduce the effect of possibly spurious outliers. In numerical data, if the distribution is almost normal using the Z-score we can detect the outliers and treat them by either removing or capping them with some value.
If the distribution is skewed using IQR we can detect and treat it by again either removing or capping it with some value. In categorical data check for value_count in the percentage if we have very few records from some category, either we can remove it or can cap it with some categorical value like others.

5. Transform the data, for example, you do a log transformation when the response variable follows an exponential distribution or is right-skewed.

6. Use more robust error metrics such as MAE or Huber loss instead of MSE.

7. Remove the outliers, only do this if you are certain that the outliers are true anomalies that are not worth adding to your model. This should be your last consideration since dropping them means losing information.

### Q2: Describe the motivation behind random forests and mention two reasons why they are better than individual decision trees? ###

The motivation behind random forest or ensemble models in general in layman's terms, Let's say we have a question/problem to solve we bring 100 people and ask each of them the question/problem and record their solution. Next, we prepare a solution which is a combination/ a mixture of all the solutions provided by these 100 people. We will find that the aggregated solution will be close to the actual solution. This is known as the "Wisdom of the crowd" and this is the motivation behind Random Forests. We take weak learners (ML models) specifically, Decision Trees in the case of Random Forest & aggregate their results to get good predictions by removing dependency on a particular set of features.  In regression, we take the mean and for Classification, we take the majority vote of the classifiers.

A random forest is generally better than a decision tree, however, you should note that no algorithm is better than the other it will always depend on the use case & the dataset [Check the No Free Lunch Theorem ](https://machinelearningmastery.com/no-free-lunch-theorem-for-machine-learning/). Reasons why random forests allow for stronger prediction than individual decision trees:
1) Decision trees are prone to overfit whereas random forest generalizes better on unseen data as it is using randomness in feature selection as well as during sampling of the data. Therefore, random forests have lower variance compared to that of the decision tree without substantially increasing the error due to bias.

2) Generally, ensemble models like Random Forest perform better as they are aggregations of various models (Decision Trees in the case of Random Forest), using the concept of the "Wisdom of the crowd."


### Q3: What are the differences and similarities between gradient boosting and random forest? and what are the advantage and disadvantages of each when compared to each other? ###

Similarities:
1. Both these algorithms are decision-tree based algorithms
2. Both these algorithms are ensemble algorithms
3. Both are flexible models and do not need much data preprocessing.

Differences:
1. Random forests (Uses Bagging): Trees are arranged in a parallel fashion where the results of all trees are aggregated at the end through averaging or majority vote. Gradient boosting (Uses Boosting): Trees are arranged in a series sequential fashion where every tree tries to minimize the error of the previous tree.

2. Radnom forests: Every tree is constructed independently of the other trees. Gradient boosting: Every tree is dependent on the previous tree.


Advantages of gradient boosting over random forests:

1. Gradient boosting can be more accurate than Random forests because we train them to minimize the previous tree's error.
2. Gradient boosting is capable of capturing complex patterns in the data.
3. Gradient boosting is better than random forest when used on unbalanced data sets.

Advantages of random forests over gradient boosting :
1. Radnom forest is less prone to overfit as compared to gradient boosting.
2. Random forest has faster training as trees are created parallelly & independent of each other.

The disadvantage of GB over RF:

1. Gradient boosting is more prone to overfitting than random forests due to their focus on mistakes during training iterations and the lack of independence in tree building.

2. If the data is noisy the boosted trees might overfit and start modeling the noise.

3. In GB training might take longer because every tree is created sequentially.

4. Tunning the hyperparameters of gradient boosting is harder than those of random forest.


### Q4: What are L1 and L2 regularization? What are the differences between the two? ###

Answer:

Regularization is a technique used to avoid overfitting by trying to make the model more simple. One way to apply regularization is by adding the weights to the loss function. This is done in order to consider minimizing unimportant weights. In L1 regularization we add the sum of the absolute of the weights to the loss function. In L2 regularization we add the sum of the squares of the weights to the loss function.

So both L1 and L2 regularization are ways to reduce overfitting, but to understand the difference it's better to know how they are calculated:
Loss (L2) : Cost function + L * weights ²
Loss (L1) : Cost function + L * |weights|
Where L is the regularization parameter

1- L2 regularization penalizes huge parameters preventing any of the single parameters to get too large. But weights never become zeros. It adds parameters square to the loss. Preventing the model from overfitting on any single feature.

2 - L1 regularization penalizes weights by adding a term to the loss function which is the absolute value of the loss. This leads to it removing small values of the parameters leading in the end to the parameter hitting zero and staying there for the rest of the epochs. Removing this specific variable completely from our calculation. So, It helps in simplifying our model. It is also helpful for feature selection as it shrinks the coefficient to zero which is not significant in the model.

### Q5: What are the Bias and Variance in a Machine Learning Model and explain the bias-variance trade-off? ###

Answer:

The goal of any supervised machine learning model is to estimate the mapping function (f) that predicts the target variable (y) given input (x). The prediction error can be broken down into three parts:

Bias: The bias is the simplifying assumption made by the model to make the target function easy to learn. Low bias suggests fewer assumptions made about the form of the target function. High bias suggests more assumptions made about the form of the target data. The smaller the bias error the better the model is. If the bias error is high, this means that the model is underfitting the training data. 

Variance: Variance is the amount that the estimate of the target function will change if different training data was used. The target function is estimated from the training data by a machine learning algorithm, so we should expect the algorithm to have some variance. Ideally, it should not change too much from one training dataset to the next, meaning that the algorithm is good at picking out the hidden underlying mapping between the inputs and the output variables. If the variance error is high this indicates that the model overfits the training data.

Irreducible error: It is the error introduced from the chosen framing of the problem and may be caused by factors like unknown variables that influence the mapping of the input variables to the output variable. The irreducible error cannot be reduced regardless of what algorithm is used.

The goal of any supervised machine learning algorithm is to achieve low bias and low variance. In turn, the algorithm should achieve good prediction performance. The parameterization of machine learning algorithms is often a battle to balance out bias and variance.
For example, if you want to predict the housing prices given a large set of potential predictors. A model with high bias but low variance, such as linear regression will be easy to implement, but it will oversimplify the problem resulting in high bias and low variance. This high bias and low variance would mean in this context that the predicted house prices are frequently off from the market value, but the value of the variance of these predicted prices is low.
On the other side, a model with low bias and high variance such as a neural network will lead to predicted house prices closer to the market value, but with predictions varying widely based on the input features. 

### Q6: Mention three ways to handle missing or corrupted data in a dataset? ###

Answer:

In general, real-world data often has a lot of missing values. The cause of missing values can be data corruption or failure to record data. The handling of missing data is very important during the preprocessing of the dataset as many machine learning algorithms do not support missing values. However, you should start by asking the data owner/stakeholder about the missing or corrupted data. It might be at the data entry level, because of file encoding, etc. which if aligned, can be handled without the need to use advanced techniques.

There are different ways to handle missing data, we will discuss only three of them:

1. Deleting the row with missing values

The first method to handle missing values is to delete the rows or columns that have null values. This is an easy and fast method and leads to a robust model, however, it will lead to the loss of a lot of information depending on the amount of missing data and can only be applied if the missing data represent a small percentage of the whole dataset.

2. Using learning algorithms that support missing values

Some machine learning algorithms are robust to missing values in the dataset. The K-NN algorithm can ignore a column from a distance measure when there are missing values. Naive Bayes can also support missing values when making a prediction. Another algorithm that can handle a dataset with missing values or null values is the random forest model and Xgboost (check the post in the first comment), as it can work on non-linear and categorical data. The problem with this method is that these models' implementation in the scikit-learn library does not support handling missing values, so you will have to implement it yourself.


3. Missing value imputation 

Data imputation means the substitution of estimated values for missing or inconsistent data in your dataset. There are different ways to estimate the values that will replace the missing value. The simplest one is to replace the missing value with the most repeated value in the row or the column. Another simple way is to replace it with the mean, median, or mode of the rest of the row or the column. This advantage of this is that it is an easy and fast way to handle the missing data, but it might lead to data leakage and does not factor the covariance between features. A better way is to use a machine learning model to learn the pattern between the data and predict the missing values, this is a very good method to estimate the missing values that will not lead to data leakage and will factor the covariance between the feature, the drawback of this method is the computational complexity especially if your dataset is large.

### Q7: Explain briefly the logistic regression model and state an example of when you have used it recently? ###

Answer:

Logistic regression is used to calculate the probability of occurrence of an event in the form of a dependent output variable based on independent input variables. Logistic regression is commonly used to estimate the probability that an instance belongs to a particular class. If the probability is bigger than 0.5 then it will belong to that class (positive) and if it is below 0.5 it will belong to the other class. This will make it a binary classifier.

It is important to remember that the Logistic regression isn't a classification model, it's an ordinary type of regression algorithm, and it was developed and used before machine learning, but it can be used in classification when we put a threshold to determine specific categories"

There is a lot of classification applications to it:

Classify email as spam or not, To identify whether the patient is healthy or not, and so on.

### Q8:Explain briefly batch gradient descent, stochastic gradient descent, and mini-batch gradient descent? and what are the pros and cons for each of them? ###

Gradient descent is a generic optimization algorithm cable for finding optimal solutions to a wide range of problems. The general idea of gradient descent is to tweak parameters iteratively in order to minimize a cost function.

Batch Gradient Descent:
In Batch Gradient descent the whole training data is used to minimize the loss function by taking a step towards the nearest minimum by calculating the gradient (the direction of descent)

Pros:
Since the whole data set is used to calculate the gradient it will be stable and reach the minimum of the cost function without bouncing (if the learning rate is chosen cooreclty)

Cons:

Since batch gradient descent uses all the training set to compute the gradient at every step, it will be very slow especially if the size of the training data is large.


Stochastic Gradient Descent:

Stochastic Gradient Descent picks up a random instance in the training data set at every step and computes the gradient-based only on that single instance.

Pros:
1. It makes the training much faster as it only works on one instance at a time.
2. It become easier to train large datasets

Cons:

Due to the stochastic (random) nature of this algorithm, this algorithm is much less regular than the batch gradient descent. Instead of gently decreasing until it reaches the minimum, the cost function will bounce up and down, decreasing only on average. Over time it will end up very close to the minimum, but once it gets there it will continue to bounce around, not settling down there. So once the algorithm stops the final parameter are good but not optimal. For this reason, it is important to use a training schedule to overcome this randomness.

Mini-batch Gradient:

At each step instead of computing the gradients on the whole data set as in the Batch Gradient Descent or using one random instance as in the Stochastic Gradient Descent, this algorithm computes the gradients on small random sets of instances called mini-batches.

Pros: 
1. The algorithm's progress space is less erratic than with Stochastic Gradient Descent, especially with large mini-batches.
2. You can get a performance boost from hardware optimization of matrix operations, especially when using GPUs.

Cons: 
1. It might be difficult to escape from local minima.

![alt text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/gradient%20descent%20vs%20batch%20gradient%20descent.png)

### Q9: Explain what is information gain and entropy in the context of decision trees? ###
### Q10: Explain the linear regression model and discuss its assumption? ###
### Q11: Explain briefly the K-Means clustering and how can we find the best value of K? ###
### Q12: Define Precision, recall, and F1 and discuss the trade-off between them? ###
