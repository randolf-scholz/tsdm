import torch


class Foo(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


print(Foo.__annotations__)
print(Foo().__annotations__)
torch.jit.script(Foo())
