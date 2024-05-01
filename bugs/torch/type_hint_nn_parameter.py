from torch import Tensor
from torch.nn import Parameter


class MyTensor(Tensor): ...


t = Parameter(MyTensor([1, 2, 3]))
reveal_type(t)
assert type(t) == MyTensor
assert type(t) != Parameter

t2 = Parameter(Tensor([1, 2, 3]))
reveal_type(t2)
assert type(t2) != Tensor
assert type(t2) == Parameter
