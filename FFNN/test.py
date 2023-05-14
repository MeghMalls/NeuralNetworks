import numpy as np

kernels = np.array([[[[1,1]],[[2,2]],[[3,3]]],[[[4,4]],[[5,5]],[[6,6]]]])
z=np.random.randint(1,5,(2,3,4,4))
print('...................')
print(z)
print(z[0][0])
print('...................')

a=np.array([0,0,1])
A=a.reshape(3,1)
b=np.array([5,9,3])
B=b.reshape(3,1)
print(a/b)
