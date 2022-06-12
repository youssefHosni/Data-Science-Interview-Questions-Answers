# Python Questions #
## Questions: ##
* [Q1:Given two arrays, write a python function to return the intersection of the two? For example, X = [1,5,9,0] and Y = [3,0,2,9] it should return [9,0]](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Python%20Questions.md#:~:text=Q1%3A%20Given%20two%20arrays%2C%20write%20a%20python%20function%20to%20return%20the%20intersection%20of%20the%20two%3F%20For%20example%2C%20X%20%3D%20%5B1%2C5%2C9%2C0%5D%20and%20Y%20%3D%20%5B3%2C0%2C2%2C9%5D%20it%20should%20return%20%5B9%2C0%5D)
* [Q2:Given an array, find all the duplicates in this array? For example: input: [1,2,3,1,3,6,5] output: [1,3]]()
* [Q3: Given an integer array, return the maximum product of any three numbers in the array?]()
----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Questions & Answers: ##

### Q1: Given two arrays, write a python function to return the intersection of the two? For example, X = [1,5,9,0] and Y = [3,0,2,9] it should return [9,0] ###

Answer:
```
set(X).intersect (set(Y))
```

### Q2:Given an array, find all the duplicates in this array? For example: input: [1,2,3,1,3,6,5] output: [1,3] ###

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

