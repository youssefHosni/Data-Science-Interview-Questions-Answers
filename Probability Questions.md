# Probability Questions #

## Questions ##
* [Q1: You and your friend are playing a game with a fair coin. The two of you will continue to toss the coin until the sequence HH or TH shows up. If HH shows up first, you win, and if TH shows up first your friend win. What is the probability of you winning the game?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Probability%20Questions.md#:~:text=Questions%20%26%20Answers-,Q1%3A%20You%20and%20your%20friend%20are%20playing%20a%20game%20with%20a%20fair%20coin.%20The%20two%20of%20you%20will%20continue%20to%20toss%20the%20coin%20until%20the%20sequence%20HH%20or%20TH%20shows%20up.%20If%20HH%20shows%20up%20first%2C%20you%20win%2C%20and%20if%20TH%20shows%20up%20first%20your%20friend%20win.%20What%20is%20the%20probability%20of%20you%20winning%20the%20game%3F,-Answer%3A)
* [Q2: If you roll a dice three times, what is the probability to get two consecutive threes?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Probability%20Questions.md#:~:text=friend%20(3/4)-,Q2%3A%20If%20you%20roll%20a%20dice%20three%20times%2C%20what%20is%20the%20probability%20to%20get%20two%20consecutive%20threes%3F,-The%20right%20answer)
* [Q3: If you have three draws from a uniformly distributed random variable between 0 and 2, what is the probability that the median of three numbers is greater than 1.5?]()
------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Questions & Answers ##


### Q1: You and your friend are playing a game with a fair coin. The two of you will continue to toss the coin until the sequence HH or TH shows up. If HH shows up first, you win, and if TH shows up first your friend win. What is the probability of you winning the game? ###

Answer:

If T is ever flipped, you cannot then reach HH before your friend reaches TH. Therefore, the probability of you winning this is to flip HH initially. Therefore the sample space will be {HH, HT, TH, TT} and the probability of you winning will be (1/4) and your friend (3/4)


### Q2: If you roll a dice three times, what is the probability to get two consecutive threes? ###

The right answer is 11/216

There are different ways to answer this question:

1. If we roll a dice three times we can get two consecutive 3’s in three ways:
1. The first two rolls are 3s and the third is any other number with a probability of 1/6 * 1/6 * 5/6.

2. The first one is not three while the other two rolls are 3s with a probability of 5/6 * 1/6 * 1/6

3. The last one is that the three rolls are 3s with probability 1/6 ^ 3

So the final result is 2 * (5/6 * (1/6)^2) + (1/6)*3 = 11/216

By Inclusion-Exclusion Principle:

Probability of at least two consecutive threes
= Probability of two consecutive threes in first two rolls + Probability of two consecutive threes in last two rolls - Probability of three consecutive threes

= 2*Probability of two consecutive threes in first two rolls - Probability of three consecutive threes
= 2*1/6*1/6 - 1/6*1/6*1/6 = 11/216

It can be seen also like this:

The sample space is made of (x, y, z) tuples where each letter can take a value from 1 to 6, therefore the sample space has 6x6x6=216 values, and the number of outcomes that are considered two consecutive threes is (3,3, X) or (X, 3, 3), the number of possible outcomes is therefore 6 for the first scenario (3,3,1) till 
(3,3,6) and 6 for the other scenario (1,3,3) till (6,3,3) and subtract the duplicate (3,3,3) which appears in both, and this leaves us with a probability of 11/216.

### Q3: If you have three draws from a uniformly distributed random variable between 0 and 2, what is the probability that the median of three numbers is greater than 1.5? ###
The right answer is 5/32 or 0.156. There are different methods to solve it:

* **Method 1:**

To get a median greater than 1.5 at least two of the three numbers must be greater than 1.5. The probability of one number being greater than 1.5 in this distribution is 0.25. Then, using the binomial distribution with three trials and a success probability of 0.25 we compute the probability of 2 or more successes to get the probability of the median is more than 1.5, which would be about 15.6%.

* **Method2 :**

A median greater than 1.5 will occur when o all three uniformly distributed random numbers are greater than 1.5 or 1 uniform distributed random number between 0 and 1.5 and the other two are greater than 1.5.

So, the probability of the above event is
= {(2 - 1.5) / 2}^3 + (3 choose 1)(1.5/2)(0.5/2)^2
= 10/64 = 5/32

* **Method3:**

Using the Monte Carlo method as shown in the figure below:
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Monte%20Carlo%20Methods.png)
