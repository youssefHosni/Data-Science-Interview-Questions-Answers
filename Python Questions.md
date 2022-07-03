# Python Questions #
## Questions: ##
* [Q1:Given two arrays, write a python function to return the intersection of the two? For example, X = [1,5,9,0] and Y = [3,0,2,9] it should return [9,0]](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=Q1%3A%20Given%20two%20arrays%2C%20write%20a%20python%20function%20to%20return%20the%20intersection%20of%20the%20two%3F%20For%20example%2C%20X%20%3D%20%5B1%2C5%2C9%2C0%5D%20and%20Y%20%3D%20%5B3%2C0%2C2%2C9%5D%20it%20should%20return%20%5B9%2C0%5D)
* [Q2:Given an array, find all the duplicates in this array? For example: input: [1,2,3,1,3,6,5] output: [1,3]](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=intersect%20(set(Y))-,Q2%3AGiven%20an%20array%2C%20find%20all%20the%20duplicates%20in%20this%20array%3F%20For%20example%3A%20input%3A%20%5B1%2C2%2C3%2C1%2C3%2C6%2C5%5D%20output%3A%20%5B1%2C3%5D,-Answer%3A)
* [Q3: Given an integer array, return the maximum product of any three numbers in the array?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=i)%0Aprint(res)-,Q3%3A%20Given%20an%20integer%20array%2C%20return%20the%20maximum%20product%20of%20any%20three%20numbers%20in%20the%20array%3F,-Answer%3A)
* [Q4: Given an integer array, find the sum of the largest contiguous subarray within the array. For example, given the array A = [0,-1,-5,-2,3,14] it should return 17 because of [3,14]. Note that if all the elements are negative it should return zero.](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=Q4%3A%20Q4%3A%20Given%20an%20integer%20array%2C%20find%20the%20sum%20of%20the%20largest%20contiguous%20subarray%20within%20the%20array.%20For%20example%2C%20given%20the%20array%20A%20%3D%20%5B0%2C%2D1%2C%2D5%2C%2D2%2C3%2C14%5D%20it%20should%20return%2017%20because%20of%20%5B3%2C14%5D.%20Note%20that%20if%20all%20the%20elements%20are%20negative%20it%20should%20return%20zero.)
* [Q5: Define tuples and lists in Python What are the major differences between them?]()

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
res=[]
for i in list:
  if i in set1:
    res.append(i)
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

### Q4: Q4: Given an integer array, find the sum of the largest contiguous subarray within the array. For example, given the array A = [0,-1,-5,-2,3,14] it should return 17 because of [3,14]. Note that if all the elements are negative it should return zero.

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


Mutable = we can change, add, delete and modify stuff
Immutable = we cannot change, add, delete and modify stuff


