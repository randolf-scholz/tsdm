import numpy as np

arr = np.random.randn(10, 2)
arr_as_list_of_tuples = list(map(tuple, arr))

print(arr)
print(list(arr))
print(arr_as_list_of_tuples)
