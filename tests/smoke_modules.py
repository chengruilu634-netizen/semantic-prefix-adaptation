import torch
from semantic_prefix_adaptation import QueryHighResGate, JointMemoryFusion

torch.manual_seed(17)
q = QueryHighResGate()
m = JointMemoryFusion()
assert sum(p.numel() for p in q.parameters()) == 13248
assert sum(p.numel() for p in m.parameters()) == 40288
x = [torch.randn(5,32,16,16), torch.randn(5,64,8,8)]
query = torch.randn(5,1,256)
y = q(x,query)
assert all(torch.equal(a,b) for a,b in zip(x,y))
mem = torch.randn(4,5,64,8,8,requires_grad=True)
out = m(mem)
assert torch.equal(out["fused"],mem[-1])
assert torch.count_nonzero(out["correction"]) == 0
out["fused"].square().mean().backward()
assert mem.grad is not None and torch.isfinite(mem.grad).all()
print("Parameter counts, zero identities and live-anchor gradient: passed")
