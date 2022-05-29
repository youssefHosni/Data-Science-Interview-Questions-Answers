# Statistics Questions # 

## Questions ##
* [Q1: Explain the central limit theorem and give examples of when you can use it in a real-world problem?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Statistics%20Questions.md#:~:text=Questions%20%26%20Answers-,Q1%3A%20Explain%20the%20central%20limit%20theorem%20and%20give%20examples%20of%20when%20you%20can%20use%20it%20in%20a%20real%2Dworld%20problem%3F,-Answer%20(Solution%20(Summary)
* [Q2: Describe briefly the hypothesis testing and p-value in layman’s term? And give a practical application for them ?]()



## Questions & Answers ##

### Q1: Explain the central limit theorem and give examples of when you can use it in a real-world problem? ###

Answer (Solution (Summary of all of the mentioned answers):

The center limit theorem states that if any random variable, regardless of the distribution, is sampled a large enough times, the sample mean will be approximately normally distributed. This allows for studying the properties of any statistical distribution as long as there is a large enough sample size.

Important remark from Adrian Olszewski:
⚠️ we can rely on the CLT with means (because it applies to any unbiased statistic) only if expressing data in this way makes sense. And it makes sense *ONLY* in the case of unimodal and symmetric data, coming from additive processes. So forget skewed, multi-modal data with mixtures of distributions, coming from multiplicative processes, and non-trivial mean-variance relationships. That are the places where arithmetic means is meaningless. Thus, using the CLT of e.g. bootstrap will give some valid answers to an invalid question.
⚠️ the distribution of means isn't enough. Every single kind of inference requires the entire test statistic to follow a certain distribution. And the test statistic consists also of the estimate of variance. Never assume the same sample size sufficient for means will suffice for the entire test statistic. See an excerpt from Rand Wilcox attached. Especially do never believe in magic numbers like N=30.
⚠️ think first about how to sensible describe your data, state the hypothesis of interest and then apply a valid method.


Examples of real-world usage of CLT:

1. The CLT can be used at any company with a large amount of data. Consider companies like Uber/Lyft wants to test whether adding a new feature will increase the booked rides or not using hypothesis testing. So if we have a large number of individual ride X, which in this case is a Bernoulli random variable (since the rider will book a ride or not), we can estimate the statistical properties of the total number of bookings. Understanding and estimating these statistical properties play a significant role in applying hypothesis testing to your data and knowing whether adding a new feature will increase the number of booked riders or not.

2. Manufacturing plants often use the central limit theorem to estimate how many products produced by the plant are defective.



### Q2: Describe briefly the hypothesis testing and p-value in layman’s term? And give a practical application for them ? ###
In Layman's terms:

- Hypothesis test is where you have a current state (null hypothesis) and an alternative state (alternative hypothesis). You assess the results of both of the states and see some differences. You want to decide whether the difference is due to the alternative approach or not.

You use the p-value to decide this, where the p-value is the likelihood of getting the same results the alternative approach achieved if you keep using the existing approach. It's the probability to find the result in the gaussian distribution of the results you may get from the existing approach.

The rule of thumb is to reject the null hypothesis if the p-value < 0.05, which means that the probability to get these results from the existing approach is <95%. But this % changes according to task and domain.

To explain the hypothesis testing in Layman's term with an example, suppose we have two drugs A and B, and we want to determine whether these two drugs are the same or different. This idea of trying to determine whether the drugs are the same or different is called hypothesis testing. The null hypothesis is that the drugs are the same, and the p-value helps us decide whether we should reject the null hypothesis or not.

p-values are numbers between 0 and 1, and in this particular case, it helps us to quantify how confident we should be to conclude that drug A is different from drug B. The closer the p-value is to 0, the more confident we are that the drugs A and B are different.
