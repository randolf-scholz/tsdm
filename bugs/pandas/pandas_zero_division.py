import pyarrow as pa

pos_one_second = pa.scalar(1, type=pa.duration("us"))  # 1s
pos_two_second = pa.scalar(2, type=pa.duration("us"))  # 2s
neg_one_second = pa.scalar(-1, type=pa.duration("us"))  # -1s
neg_two_second = pa.scalar(-2, type=pa.duration("us"))  # -2s

print(pa.compute.divide(pos_one_second, pos_one_second))  # 1.0 ✔
print(pa.compute.divide(pos_two_second, pos_one_second))  # 2.0 ✔
print(pa.compute.divide(pos_one_second, pos_two_second))  # 0.5 ✔
print(pa.compute.divide(pos_two_second, pos_two_second))  # 1.0 ✔

print(pa.compute.divide(pos_one_second, pos_one_second))  # 1.0 ✔
print(pa.compute.divide(neg_one_second, pos_one_second))  # NAN ✘
print(pa.compute.divide(pos_one_second, neg_one_second))  # NAN ✘
print(pa.compute.divide(neg_one_second, neg_one_second))  # NAN ✘

print(pa.compute.divide(pos_one_second, pos_one_second))  # 1.0 ✔
print(pa.compute.divide(neg_two_second, pos_one_second))  # NAN ✘
print(pa.compute.divide(pos_one_second, neg_two_second))  # NAN ✘
print(pa.compute.divide(neg_two_second, neg_two_second))  # NAN ✘
