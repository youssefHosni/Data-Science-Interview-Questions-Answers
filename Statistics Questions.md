# Statistics Questions # 

## Questions ##
* [Q1: Explain the central limit theorem and give examples of when you can use it in a real-world problem?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Statistics%20Questions.md#:~:text=Questions%20%26%20Answers-,Q1%3A%20Explain%20the%20central%20limit%20theorem%20and%20give%20examples%20of%20when%20you%20can%20use%20it%20in%20a%20real%2Dworld%20problem%3F,-Answers%3A)
* [Q2: Briefly explain the A/B testing and its application? What are some common pitfalls encountered in A/B testing?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Statistics%20Questions.md#:~:text=Q2%3ABriefly%20explain%20the%20A/B%20testing%20and%20its%20application%3F%20What%20are%20some%20common%20pitfalls%20encountered%20in%20A/B%20testing%3F)
* [Q3: Describe briefly the hypothesis testing and p-value in layman’s term? And give a practical application for them ?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Statistics%20Questions.md#:~:text=plant%20are%20defective.-,Q2%3A%20Describe%20briefly%20the%20hypothesis%20testing%20and%20p%2Dvalue%20in%20layman%E2%80%99s%20term%3F%20And%20give%20a%20practical%20application%20for%20them%20%3F,-In%20Layman%27s%20terms)
* [Q4: Given a left-skewed distribution that has a median of 60, what conclusions can we draw about the mean and the mode of the data?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Statistics%20Questions.md#:~:text=B%20are%20different.-,Q4%3A%20Given%20a%20left%2Dskewed%20distribution%20that%20has%20a%20median%20of%2060%2C%20what%20conclusions%20can%20we%20draw%20about%20the%20mean%20and%20the%20mode%20of%20the%20data%3F,-Footer)
* [Q5: What is the meaning of selection bias and how to avoid it?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Statistics%20Questions.md#:~:text=Q5%3A%20What%20is%20the%20meaning%20of%20selection%20bias%20and%20how%20to%20avoid%20it%3F%23%23%23)
* [Q6: Explain the long-tailed distribution and provide three examples of relevant phenomena that have long tails. Why are they important in classification and regression problems?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Statistics%20Questions.md#:~:text=Q6%3A%20Q6%3A%20Explain%20the%20long%2Dtailed%20distribution%20and%20provide%20three%20examples%20of%20relevant%20phenomena%20that%20have%20long%20tails.%20Why%20are%20they%20important%20in%20classification%20and%20regression%20problems%3F)
* [Q7: What is the meaning of KPI in statistics](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Statistics%20Questions.md#:~:text=is%20normally%20distributed.-,Q7%3A%20What%20is%20the%20meaning%20of%20KPI%20in%20statistics,-Answer%3A)
* [Q8: Say you flip a coin 10 times and observe only one head. What would be the null hypothesis and p-value for testing whether the coin is fair or not?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Statistics%20Questions.md#:~:text=what%E2%80%99s%20most%20important.-,Q8%3A%20Say%20you%20flip%20a%20coin%2010%20times%20and%20observe%20only%20one%20head.%20What%20would%20be%20the%20null%20hypothesis%20and%20p%2Dvalue%20for%20testing%20whether%20the%20coin%20is%20fair%20or%20not%3F,-Answer%3A)

## Questions & Answers ##

### Q1: Explain the central limit theorem and give examples of when you can use it in a real-world problem? ###

Answers:

The center limit theorem states that if any random variable, regardless of the distribution, is sampled a large enough times, the sample mean will be approximately normally distributed. This allows for studying the properties of any statistical distribution as long as there is a large enough sample size.

Important remark from Adrian Olszewski:
⚠️ we can rely on the CLT with means (because it applies to any unbiased statistic) only if expressing data in this way makes sense. And it makes sense *ONLY* in the case of unimodal and symmetric data, coming from additive processes. So forget skewed, multi-modal data with mixtures of distributions, coming from multiplicative processes, and non-trivial mean-variance relationships. That are the places where arithmetic means is meaningless. Thus, using the CLT of e.g. bootstrap will give some valid answers to an invalid question.

⚠️ the distribution of means isn't enough. Every single kind of inference requires the entire test statistic to follow a certain distribution. And the test statistic consists also of the estimate of variance. Never assume the same sample size sufficient for means will suffice for the entire test statistic. See an excerpt from Rand Wilcox attached. Especially do never believe in magic numbers like N=30.

⚠️ think first about how to sensible describe your data, state the hypothesis of interest and then apply a valid method.


Examples of real-world usage of CLT:

1. The CLT can be used at any company with a large amount of data. Consider companies like Uber/Lyft wants to test whether adding a new feature will increase the booked rides or not using hypothesis testing. So if we have a large number of individual ride X, which in this case is a Bernoulli random variable (since the rider will book a ride or not), we can estimate the statistical properties of the total number of bookings. Understanding and estimating these statistical properties play a significant role in applying hypothesis testing to your data and knowing whether adding a new feature will increase the number of booked riders or not.

2. Manufacturing plants often use the central limit theorem to estimate how many products produced by the plant are defective.

### Q2:Briefly explain the A/B testing and its application? What are some common pitfalls encountered in A/B testing? ###
A/B testing helps us to determine whether a change in something will cause a change in performance significantly or not. So in other words you aim to statistically estimate the impact of a given change within your digital product (for example). You measure success and counter metrics on at least 1 treatment vs 1 control group (there can be more than 1 XP group for multivariate tests).

Applications:
1. Consider the example of a general store that sells bread packets but not butter, for a year. If we want to check whether its sale depends on the butter or not, then suppose the store also sells butter and sales for next year are observed. Now we can determine whether selling butter can significantly increase/decrease or doesn't affect the sale of bread.

2. While developing the landing page of a website you create 2 different versions of the page. You define a criteria for success eg. conversion rate. Then define your hypothesis
Null hypothesis(H): No difference between the performance of the 2 versions. Alternative hypothesis(H'): version A will perform better than B.

NOTE: You will have to split your traffic randomly(to avoid sample bias) into 2 versions. The split doesn't have to be symmetric, you just need to set the minimum sample size for each version to avoid undersample bias.

Now if version A gives better results than version B, we will still have to statistically prove that results derived from our sample represent the entire population. Now one of the very common tests used to do so is 2 sample t-test where we use values of significance level (alpha) and p-value to see which hypothesis is right. If p-value<alpha, H is rejected.


Common pitfalls:

1. Wrong success metrics inadequate to the business problem
2. Lack of counter metric, as you might add friction to the product regardless along with the positive impact
3. Sample mismatch: heterogeneous control and treatment, unequal variances
4. Underpowered test: too small sample or XP running too short 5. Not accounting for network effects (introduce bias within measurement)


### Q3: Describe briefly the hypothesis testing and p-value in layman’s term? And give a practical application for them ? ###
In Layman's terms:

- Hypothesis test is where you have a current state (null hypothesis) and an alternative state (alternative hypothesis). You assess the results of both of the states and see some differences. You want to decide whether the difference is due to the alternative approach or not.

You use the p-value to decide this, where the p-value is the likelihood of getting the same results the alternative approach achieved if you keep using the existing approach. It's the probability to find the result in the gaussian distribution of the results you may get from the existing approach.

The rule of thumb is to reject the null hypothesis if the p-value < 0.05, which means that the probability to get these results from the existing approach is <95%. But this % changes according to task and domain.

To explain the hypothesis testing in Layman's term with an example, suppose we have two drugs A and B, and we want to determine whether these two drugs are the same or different. This idea of trying to determine whether the drugs are the same or different is called hypothesis testing. The null hypothesis is that the drugs are the same, and the p-value helps us decide whether we should reject the null hypothesis or not.

p-values are numbers between 0 and 1, and in this particular case, it helps us to quantify how confident we should be to conclude that drug A is different from drug B. The closer the p-value is to 0, the more confident we are that the drugs A and B are different.

### Q4: Given a left-skewed distribution that has a median of 60, what conclusions can we draw about the mean and the mode of the data? ###

Answer:
Left skewed distribution means the tail of the distribution is to the left and the tip is to the right. So the mean which tends to be near outliers (very large or small values) will be shifted towards the left or in other words, towards the tail.

While the mode (which represents the most repeated value) will be near the tip and the median is the middle element independent of the distribution skewness, therefore it will be smaller than the mode and more than the mean.

Mean < 60
Mode > 60

![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/1657144303401.jpg)

### Q5: What is the meaning of selection bias and how to avoid it? ###
Answer:

Sampling bias is the phenomenon that occurs when a research study design fails to collect a representative sample of a target population. This typically occurs because the selection criteria for respondents failed to capture a wide enough sampling frame to represent all viewpoints.

The cause of sampling bias almost always owes to one of two conditions.
1. Poor methodology: In most cases, non-representative samples pop up when researchers set improper parameters for survey research. The most accurate and repeatable sampling method is simple random sampling where a large number of respondents are chosen at random. When researchers stray from random sampling (also called probability sampling), they risk injecting their own selection bias into recruiting respondents.

2. Poor execution: Sometimes data researchers craft scientifically sound sampling methods, but their work is undermined when field workers cut corners. By reverting to convenience sampling (where the only people studied are those who are easy to reach) or giving up on reaching non-responders, a field worker can jeopardize the careful methodology set up by data scientists.

The best way to avoid sampling bias is to stick to probability-based sampling methods. These include simple random sampling, systematic sampling, cluster sampling, and stratified sampling. In these methodologies, respondents are only chosen through processes of random selection—even if they are sometimes sorted into demographic groups along the way.
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Sampling%20bias.png)

### Q6: Explain the long-tailed distribution and provide three examples of relevant phenomena that have long tails. Why are they important in classification and regression problems? ###

Answer: 
A long-tailed distribution is a type of heavy-tailed distribution that has a tail (or tails) that drop off gradually and asymptotically.
 
Three examples of relevant phenomena that have long tails:

1. Frequencies of languages spoken
2. Population of cities
3. Pageviews of articles

All of these follow something close to 80-20 rule: 80% of outcomes (or outputs) result from 20% of all causes (or inputs) for any given event. This 20% forms the long tail in the distribution.

It’s important to be mindful of long-tailed distributions in classification and regression problems because the least frequently occurring values make up the majority of the population. This can ultimately change the way that you deal with outliers, and it also conflicts with some machine learning techniques with the assumption that the data is normally distributed.
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/long-tailed%20distribution.jpg)

### Q7: What is the meaning of KPI in statistics ###
Answer:

**KPI** stands for key performance indicator, a quantifiable measure of performance over time for a specific objective. KPIs provide targets for teams to shoot for, milestones to gauge progress, and insights that help people across the organization make better decisions. From finance and HR to marketing and sales, key performance indicators help every area of the business move forward at the strategic level.

KPIs are an important way to ensure your teams are supporting the overall goals of the organization. Here are some of the biggest reasons why you need key performance indicators.

* Keep your teams aligned: Whether measuring project success or employee performance, KPIs keep teams moving in the same direction.
* Provide a health check: Key performance indicators give you a realistic look at the health of your organization, from risk factors to financial indicators.
* Make adjustments: KPIs help you clearly see your successes and failures so you can do more of what’s working, and less of what’s not.
* Hold your teams accountable: Make sure everyone provides value with key performance indicators that help employees track their progress and help managers move things along.

Types of KPIs
Key performance indicators come in many flavors. While some are used to measure monthly progress against a goal, others have a longer-term focus. The one thing all KPIs have in common is that they’re tied to strategic goals. Here’s an overview of some of the most common types of KPIs.

* **Strategic**: These big-picture key performance indicators monitor organizational goals. Executives typically look to one or two strategic KPIs to find out how the organization is doing at any given time. Examples include return on investment, revenue and market share.
* **Operational:** These KPIs typically measure performance in a shorter time frame, and are focused on organizational processes and efficiencies. Some examples include sales by region, average monthly transportation costs and cost per acquisition (CPA).
* **Functional Unit:** Many key performance indicators are tied to specific functions, such finance or IT. While IT might track time to resolution or average uptime, finance KPIs track gross profit margin or return on assets. These functional KPIs can also be classified as strategic or operational.
* **Leading vs Lagging:** Regardless of the type of key performance indicator you define, you should know the difference between leading indicators and lagging indicators. While leading KPIs can help predict outcomes, lagging KPIs track what has already happened. Organizations use a mix of both to ensure they’re tracking what’s most important.


![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/KPI.png)

### Q8: Say you flip a coin 10 times and observe only one head. What would be the null hypothesis and p-value for testing whether the coin is fair or not? ###

Answer:
The null hypothesis is that the coin is fair, and the alternative hypothesis is that the coin is biased. The p-value is the probability of observing the results obtained given that the null hypothesisis true. In total for 10 flips of a coin, there are 2^10 = 1024 possible outcomes and in only 10 of them are there 9 tails and one heads. Hence, the exact probability of the give result is the p-value, which is 10/1024 = 0.0098. Therfore, with a signifcance level set, for example, at 0.05, we can reject the null hypothesis. 
