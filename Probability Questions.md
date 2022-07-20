# Probability Questions #

## Questions ##
* [Q1: You and your friend are playing a game with a fair coin. The two of you will continue to toss the coin until the sequence HH or TH shows up. If HH shows up first, you win, and if TH shows up first your friend win. What is the probability of you winning the game?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Probability%20Questions.md#:~:text=Questions%20%26%20Answers-,Q1%3A%20You%20and%20your%20friend%20are%20playing%20a%20game%20with%20a%20fair%20coin.%20The%20two%20of%20you%20will%20continue%20to%20toss%20the%20coin%20until%20the%20sequence%20HH%20or%20TH%20shows%20up.%20If%20HH%20shows%20up%20first%2C%20you%20win%2C%20and%20if%20TH%20shows%20up%20first%20your%20friend%20win.%20What%20is%20the%20probability%20of%20you%20winning%20the%20game%3F,-Answer%3A)
* [Q2: If you roll a dice three times, what is the probability to get two consecutive threes?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Probability%20Questions.md#:~:text=friend%20(3/4)-,Q2%3A%20If%20you%20roll%20a%20dice%20three%20times%2C%20what%20is%20the%20probability%20to%20get%20two%20consecutive%20threes%3F,-The%20right%20answer)
* [Q3: Suppose you have ten fair dice. If you randomly throw them simultaneously, what is the probability that the sum of all of the top faces is divisible by six?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Probability%20Questions.md#:~:text=of%2011/216.-,Q3%3A%20Suppose%20you%20have%20ten%20fair%20dice.%20If%20you%20randomly%20throw%20them%20simultaneously%2C%20what%20is%20the%20probability%20that%20the%20sum%20of%20all%20of%20the%20top%20faces%20is%20divisible%20by%20six%3F,-Answer%3A%201/6)
* [Q4: If you have three draws from a uniformly distributed random variable between 0 and 2, what is the probability that the median of three numbers is greater than 1.5?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Probability%20Questions.md#:~:text=of%2011/216.-,Q3%3A%20If%20you%20have%20three%20draws%20from%20a%20uniformly%20distributed%20random%20variable%20between%200%20and%202%2C%20what%20is%20the%20probability%20that%20the%20median%20of%20three%20numbers%20is%20greater%20than%201.5%3F,-The%20right%20answer)
* [Q5: Assume you have a deck of 100 cards with values ranging from 1 to 100 and you draw two cards randomly without replacement, what is the probability that the number of one of them is double the other?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Probability%20Questions.md#:~:text=the%20figure%20below%3A-,Q5%3A%20Assume%20you%20have%20a%20deck%20of%20100%20cards%20with%20values%20ranging%20from%201%20to%20100%20and%20you%20draw%20two%20cards%20randomly%20without%20replacement%2C%20what%20is%20the%20probability%20that%20the%20number%20of%20one%20of%20them%20is%20double%20the%20other%3F,-Footer)
* [Q6: What is the difference between the Bernoulli and Binomial distribution?]()
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

= 2 * Probability of two consecutive threes in first two rolls - Probability of three consecutive threes
= 2 * (1/6) * (1/6) - (1/6) * (1/6) * (1/6) = 11/216

It can be seen also like this:

The sample space is made of (x, y, z) tuples where each letter can take a value from 1 to 6, therefore the sample space has 6x6x6=216 values, and the number of outcomes that are considered two consecutive threes is (3,3, X) or (X, 3, 3), the number of possible outcomes is therefore 6 for the first scenario (3,3,1) till 
(3,3,6) and 6 for the other scenario (1,3,3) till (6,3,3) and subtract the duplicate (3,3,3) which appears in both, and this leaves us with a probability of 11/216.

### Q3: Suppose you have ten fair dice. If you randomly throw them simultaneously, what is the probability that the sum of all of the top faces is divisible by six? ###
Answer:
1/6

Explanation:
With 10 dices, the possible sums divisible by 6 are 12, 18, 24, 30, 36, 42, 48, 54, and 60. You don't actually need to calculate the probability of getting each of these numbers as the final sums from 10 dices because no matter what the sum of the first 9 numbers is, you can still choose a number between 1 to 6 on the last die and add to that previous sum to make the final sum divisible by 6. Therefore, we only care about the last die. And the probability to get that number on the last die is 1/6. So the answer is 1/6


### Q4: If you have three draws from a uniformly distributed random variable between 0 and 2, what is the probability that the median of three numbers is greater than 1.5? ###
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

### Q5: Assume you have a deck of 100 cards with values ranging from 1 to 100 and you draw two cards randomly without replacement, what is the probability that the number of one of them is double the other? ###

There are a total of (100 C 2) = 4950 ways to choose two cards at random from the 100 cards and there are only 50 pairs of these 4950 ways that you will get one number and it's double. Therefore the probability that the number of one of them is double the other is 50/4950.

### Q6: What is the difference between the Bernoulli and Binomial distribution? ###
Answer:
