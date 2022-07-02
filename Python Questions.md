# Python Questions #
## Questions: ##
* [Q1:Given two arrays, write a python function to return the intersection of the two? For example, X = [1,5,9,0] and Y = [3,0,2,9] it should return [9,0]](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=Q1%3A%20Given%20two%20arrays%2C%20write%20a%20python%20function%20to%20return%20the%20intersection%20of%20the%20two%3F%20For%20example%2C%20X%20%3D%20%5B1%2C5%2C9%2C0%5D%20and%20Y%20%3D%20%5B3%2C0%2C2%2C9%5D%20it%20should%20return%20%5B9%2C0%5D)
* [Q2:Given an array, find all the duplicates in this array? For example: input: [1,2,3,1,3,6,5] output: [1,3]](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=intersect%20(set(Y))-,Q2%3AGiven%20an%20array%2C%20find%20all%20the%20duplicates%20in%20this%20array%3F%20For%20example%3A%20input%3A%20%5B1%2C2%2C3%2C1%2C3%2C6%2C5%5D%20output%3A%20%5B1%2C3%5D,-Answer%3A)
* [Q3: Given an integer array, return the maximum product of any three numbers in the array?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=i)%0Aprint(res)-,Q3%3A%20Given%20an%20integer%20array%2C%20return%20the%20maximum%20product%20of%20any%20three%20numbers%20in%20the%20array%3F,-Answer%3A)
* [Q4: Given an integer array, find the sum of the largest contiguous subarray within the array. For example, given the array A = [0,-1,-5,-2,3,14] it should return 17 because of [3,14]. Note that if all the elements are negative it should return zero.]()
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



