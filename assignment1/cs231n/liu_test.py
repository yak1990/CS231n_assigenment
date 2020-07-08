import math
a=[0.16799663,0.35189628,0.83474904]
b=[0.48443503,0.62367356,0.22711577]
c=[(i-j)*(i-j) for i,j in zip(a,b)]
print([i-j for  i,j in zip(a,b)])
print(c)
print(math.sqrt(sum(c)))