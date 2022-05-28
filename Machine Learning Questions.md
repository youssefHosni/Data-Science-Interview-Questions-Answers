# Machine Learning Interview Questions #

Q1: Mention three ways to make your model robust to outliers?

1. Investigating the outliers is always the first step in understanding how to treat them. After you understand the nature of why the outliers occurred you can apply one of the several methods mentioned below.

2. Add regularization that will reduce variance, for example, L1 or L2 regularization.

3. Use tree-based models (random forest, gradient boosting ) that are generally less affected by outliers.

4. Winsorize the data. Winsorizing or winsorization is the transformation of statistics by limiting extreme values in the statistical data to reduce the effect of possibly spurious outliers. In numerical data, if the distribution is almost normal using the Z-score we can detect the outliers and treat them by either removing or capping them with some value.
If the distribution is skewed using IQR we can detect and treat it by again either removing or capping it with some value. In categorical data check for value_count in the percentage if we have very few records from some category, either we can remove it or can cap it with some categorical value like others.

5. Transform the data, for example, you do a log transformation when the response variable follows an exponential distribution or is right-skewed.

6. Use more robust error metrics such as MAE or Huber loss instead of MSE.

7. Remove the outliers, only do this if you are certain that the outliers are true anomalies that are not worth adding to your model. This should be your last consideration since dropping them means losing information.

