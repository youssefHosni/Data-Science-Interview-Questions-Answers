# Machine Learning Interview Questions #

**Q1: Mention three ways to make your model robust to outliers?**

1. Investigating the outliers is always the first step in understanding how to treat them. After you understand the nature of why the outliers occurred you can apply one of the several methods mentioned below.

2. Add regularization that will reduce variance, for example, L1 or L2 regularization.

3. Use tree-based models (random forest, gradient boosting ) that are generally less affected by outliers.

4. Winsorize the data. Winsorizing or winsorization is the transformation of statistics by limiting extreme values in the statistical data to reduce the effect of possibly spurious outliers. In numerical data, if the distribution is almost normal using the Z-score we can detect the outliers and treat them by either removing or capping them with some value.
If the distribution is skewed using IQR we can detect and treat it by again either removing or capping it with some value. In categorical data check for value_count in the percentage if we have very few records from some category, either we can remove it or can cap it with some categorical value like others.

5. Transform the data, for example, you do a log transformation when the response variable follows an exponential distribution or is right-skewed.

6. Use more robust error metrics such as MAE or Huber loss instead of MSE.

7. Remove the outliers, only do this if you are certain that the outliers are true anomalies that are not worth adding to your model. This should be your last consideration since dropping them means losing information.

**Q2: Describe the motivation behind random forests and mention two reasons why they are better than individual decision trees?**

The motivation behind random forest or ensemble models in general in layman's terms, Let's say we have a question/problem to solve we bring 100 people and ask each of them the question/problem and record their solution. Next, we prepare a solution which is a combination/ a mixture of all the solutions provided by these 100 people. We will find that the aggregated solution will be close to the actual solution. This is known as the "Wisdom of the crowd" and this is the motivation behind Random Forests. We take weak learners (ML models) specifically, Decision Trees in the case of Random Forest & aggregate their results to get good predictions by removing dependency on a particular set of features.  In regression, we take the mean and for Classification, we take the majority vote of the classifiers.

A random forest is generally better than a decision tree, however, you should note that no algorithm is better than the other it will always depend on the use case & the dataset [Check the No Free Lunch Theorem in the first comment](https://machinelearningmastery.com/no-free-lunch-theorem-for-machine-learning/). Reasons why random forests allow for stronger prediction than individual decision trees:
1) Decision trees are prone to overfit whereas random forest generalizes better on unseen data as it is using randomness in feature selection as well as during sampling of the data. Therefore, random forests have lower variance compared to that of the decision tree without substantially increasing the error due to bias.

2) Generally, ensemble models like Random Forest perform better as they are aggregations of various models (Decision Trees in the case of Random Forest), using the concept of the "Wisdom of the crowd."


**Q3: What are the differences and similarities between gradient boosting and random forest? and what are the advantage and disadvantages of each when compared to each other?**
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


**Q4: What are L1 and L2 regularization? What are the differences between the two?**
Answer: (Summary of all of the answers)

Regularization is a technique used to avoid overfitting by trying to make the model more simple. One way to apply regularization is by adding the weights to the loss function. This is done in order to consider minimizing unimportant weights. In L1 regularization we add the sum of the absolute of the weights to the loss function. In L2 regularization we add the sum of the squares of the weights to the loss function.

So both L1 and L2 regularization are ways to reduce overfitting, but to understand the difference it's better to know how they are calculated:
Loss (L2) : Cost function + L * weights ²
Loss (L1) : Cost function + L * |weights|
Where L is the regularization parameter

1- L2 regularization penalizes huge parameters preventing any of the single parameters to get too large. But weights never become zeros. It adds parameters square to the loss. Preventing the model from overfitting on any single feature.

2 - L1 regularization penalizes weights by adding a term to the loss function which is the absolute value of the loss. This leads to it removing small values of the parameters leading in the end to the parameter hitting zero and staying there for the rest of the epochs. Removing this specific variable completely from our calculation. So, It helps in simplifying our model. It is also helpful for feature selection as it shrinks the coefficient to zero which is not significant in the model.

