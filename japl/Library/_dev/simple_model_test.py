from SimpleModel import truth
import simple_model
args = (1, 1,2,3, [1,2,3], [1,2,3,4,5,6,7,8,9])
ret =simple_model.func(*args)

for i, j in zip(truth, ret):
    assert i == j
print("PASS")
