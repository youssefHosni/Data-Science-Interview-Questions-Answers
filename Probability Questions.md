# Probability Questions #

## Questions ##
* Q1: You and your friend are playing a game with a fair coin. The two of you will continue to toss the coin until the sequence HH or TH shows up. If HH shows up first, you win, and if TH shows up first your friend win. What is the probability of you winning the game?
* Q2: If you roll a dice three times, what is the probability to get two consecutive threes?
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

The sample space is made of (x, y, z) tuples where each letter can take a value from 1 to 6, therefore the sample space has 6x6x6=216 values, and the number of outcomes that are considered two consecutive threes is (3,3, X) or (X, 3, 3), the number of possible outcomes is therefore 6 for the first scenario (3,3,1) till (3,3,6) and 6 for the other scenario (1,3,3) till (6,3,3) and subtract the duplicate (3,3,3) which appears in both, and this leaves us with a probability of 11/216.
