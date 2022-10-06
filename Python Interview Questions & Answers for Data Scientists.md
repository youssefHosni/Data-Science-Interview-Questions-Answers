# Python Questions #
## Questions: ##
* [Q1:Given two arrays, write a python function to return the intersection of the two? For example, X = [1,5,9,0] and Y = [3,0,2,9] it should return [9,0]](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=Q1%3A%20Given%20two%20arrays%2C%20write%20a%20python%20function%20to%20return%20the%20intersection%20of%20the%20two%3F%20For%20example%2C%20X%20%3D%20%5B1%2C5%2C9%2C0%5D%20and%20Y%20%3D%20%5B3%2C0%2C2%2C9%5D%20it%20should%20return%20%5B9%2C0%5D)
* [Q2 :Given an array, find all the duplicates in this array? For example: input: [1,2,3,1,3,6,5] output: [1,3]](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=intersect%20(set(Y))-,Q2%3AGiven%20an%20array%2C%20find%20all%20the%20duplicates%20in%20this%20array%3F%20For%20example%3A%20input%3A%20%5B1%2C2%2C3%2C1%2C3%2C6%2C5%5D%20output%3A%20%5B1%2C3%5D,-Answer%3A)
* [Q3: Given an integer array, return the maximum product of any three numbers in the array?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=i)
* [Q4: Given an integer array, find the sum of the largest contiguous subarray within the array. For example, given the array A = [0,-1,-5,-2,3,14] it should return 17 because of [3,14]. Note that if all the elements are negative it should return zero.](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=Q4%3A%20Q4%3A%20Given%20an%20integer%20array%2C%20find%20the%20sum%20of%20the%20largest%20contiguous%20subarray%20within%20the%20array.%20For%20example%2C%20given%20the%20array%20A%20%3D%20%5B0%2C%2D1%2C%2D5%2C%2D2%2C3%2C14%5D%20it%20should%20return%2017%20because%20of%20%5B3%2C14%5D.%20Note%20that%20if%20all%20the%20elements%20are%20negative%20it%20should%20return%20zero.)
* [Q5: Define tuples and lists in Python What are the major differences between them?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=Q5%3A%20Define%20tuples%20and%20lists%20in%20Python%20What%20are%20the%20major%20differences%20between%20them%3F%23%23%23)
* [Q6: Compute the Euclidean Distance Between Two Series?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=and%20modify%20stuff-,Q6%3A%20Compute%20the%20Euclidean%20Distance%20Between%20Two%20Series%3F,-Footer)
* [Q7: Given an integer n and an integer K, output a list of all of the combination of k numbers chosen from 1 to n. For example, if n=3 and k=2, return [1,2][1,3],[2,3]](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=Q7%3A%20Given%20an%20integer%20n%20and%20an%20integer%20K%2C%20output%20a%20list%20of%20all%20of%20the%20combination%20of%20k%20numbers%20chosen%20from%201%20to%20n.%20For%20example%2C%20if%20n%3D3%20and%20k%3D2%2C%20return%20%5B1%2C2%5D%2C%5B1%2C3%5D%2C%5B2%2C3%5D)
* [Q8: Write a function to generate N samples from a normal distribution and plot them on the histogram](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=print(list_num%2C%22%5Cn%22)-,Q8%3A%20Write%20a%20function%20to%20generate%20N%20samples%20from%20a%20normal%20distribution%20and%20plot%20them%20on%20the%20histogram,-Answer%3A)
* [Q9: What is the difference between apply and applymap function in pandas? ](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=From%20scratch%3A-,Q9%3A%20What%20is%20the%20difference%20between%20apply%20and%20applymap%20function%20in%20pandas%3F,-Answer%3A)
* [Q10 Given a string, return the first recurring character in it, or “None” if there is no recurring character. Example: input = "pythoninterviewquestion" , output = "n"](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=Q10%3A%20Given%20a%20string%2C%20return%20the%20first%20recurring%20character%20in%20it%2C%20or%20%E2%80%9CNone%E2%80%9D%20if%20there%20is%20no%20recurring%20character.%20Example%3A%20input%20%3D%20%22pythoninterviewquestion%22%20%2C%20output%20%3D%20%22n%22)

----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Questions & Answers: ##

### Q1: Given two arrays, write a python function to return the intersection of the two? For example, X = [1,5,9,0] and Y = [3,0,2,9] it should return [9,0] ###

Answer:
```
set(X).intersect (set(Y))
```

### Q2: Given an array, find all the duplicates in this array? For example: input: [1,2,3,1,3,6,5] output: [1,3] ###

Answer:
```
set1=set()
res=set()
for i in list:
  if i in set1:
    res.add(i)
  else:
    set1.add(i)
print(res)
```


### Q3: Given an integer array, return the maximum product of any three numbers in the array? ###

Answer:

```
import heapq

def max_three(arr):
    a = heapq.nlargest(3, arr) # largerst 3 numbers for postive case
    b = heapq.nsmallest(2, arr) # for negative case
    return max(a[2]*a[1]*a[0], b[1]*b[0]*a[0])
```

### Q4: Given an integer array, find the sum of the largest contiguous subarray within the array. For example, given the array A = [0,-1,-5,-2,3,14] it should return 17 because of [3,14]. Note that if all the elements are negative it should return zero.

```
def max_subarray(arr):
  n = len(arr)
  max_sum = arr[0] #max
  curr_sum = 0 
  for i in range(n):
    curr_sum += arr[i]
    max_sum = max(max_sum, curr_sum)
    if curr_sum <0:
      curr_sum  = 0
  return max_sum    
      
```

### Q5: Define tuples and lists in Python What are the major differences between them?###
Answer:

Lists:
In Python, a list is created by placing elements inside square brackets [], separated by commas. A list can have any number of items and they may be of different types (integer, float, string, etc.). A list can also have another list as an item. This is called a nested list.

1. Lists are mutable
2. Lists are better for performing operations, such as insertion and deletion.
3. Lists consume more memory
4. Lists have several built-in methods


Tuples:
A tuple is a collection of objects which ordered and immutable. Tuples are sequences, just like lists. The differences between tuples and lists are, the tuples cannot be changed unlike lists and tuples use parentheses, whereas lists use square brackets.

1. Tuples are immutable
2. Tuple data type is appropriate for accessing the elements
3. Tuples consume less memory as compared to the list
4. Tuple does not have many built-in methods.


* Mutable = we can change, add, delete and modify stuff
* Immutable = we cannot change, add, delete and modify stuff

### Q6: Compute the Euclidean Distance Between Two Series? ###
```
```

### Q7: Given an integer n and an integer K, output a list of all of the combination of k numbers chosen from 1 to n. For example, if n=3 and k=2, return [1,2],[1,3],[2,3] ### 

Answer
```
from itertools import combinations
def find_combintaion(k,n):
    list_num = []
    comb = combinations([x for x in range(1, n+1)],k)
    for i in comb:
        list_num.append(i)
    print("(K:{},n:{}):".format(k,n))
    print(list_num,"\n")
```

### Q8: Write a function to generate N samples from a normal distribution and plot them on the histogram ###

Answer:
Using bultin Libraries:
```
import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn((N,))
plt.hist(x)
```
From scratch: 
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Python%20questions%208.png)


### Q9: What is the difference between apply and applymap function in pandas? ###

Answer:

Both the methods only accept callables as arguments but what sets them apart is that applymap is defined on dataframes and works element-wise. While apply can be defined on data frames as well as series and can work row/column-wise as well as element-wise. In terms of use case, applymap is used for transformations while apply is used for more complex operations and aggregations. Applymap only returns a dataframe while apply can return a scalar value, series, or dataframe.


### Q10: Given a string, return the first recurring character in it, or “None” if there is no recurring character. Example: input = "pythoninterviewquestion" , output = "n" ###

Answer:
```
input_string = "pythoninterviewquestion"

def first_recurring(input_str):
  
  a_str = ""
  for letter in input_str:
    a_str = a_str + letter
    if a_str.count(letter) > 1:
      return letter
  return None

first_recurring(input_string)

```

