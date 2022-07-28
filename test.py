import numpy as np

A = np.array([[1,2],[3,4]])

E = np.array([[10,20, 20],[33,42, 40], [55,66,66]])

D = np.array([[1], [2], [3]])


print(np.sum(D))

print("check shpape : ", np.shape(D))
print("E : ", E)

#print(A*E)

print(D*E)
