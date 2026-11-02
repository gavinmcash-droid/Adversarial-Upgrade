"""Models, independent random streams, metrics, and isolated optimization steps."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

BASE = (2, 4, 4, 1)
WIDE = (4, 8, 8, 1)

class Network(nn.Module):
    def __init__(self, inputs, channels=BASE):
        super().__init__()
        self.fc = nn.Linear(inputs, 32)
        self.convs = nn.ModuleList()
        for before, after, kernel, stride in zip((1, *channels[:-1]), channels, (4,2,1,1), (1,2,1,1)):
            self.convs.append(nn.Conv1d(before, after, kernel, stride=stride))

    def forward(self, x):
        x = torch.sigmoid(self.fc(x)).unsqueeze(1)
        for i, conv in enumerate(self.convs):
            # TensorFlow SAME: extra element goes on the right. Length ceil(L/stride).
            length, kernel, stride = x.shape[-1], conv.kernel_size[0], conv.stride[0]
            total = max(0, ((length + stride - 1)//stride-1)*stride+kernel-length)
            x = conv(F.pad(x, (total//2, total-total//2)))
            x = torch.tanh(x) if i == 3 else torch.sigmoid(x)
        return x.squeeze(1)

class Keyed(Network):
    def __init__(self):
        super().__init__(32, BASE)
    def forward(self, data, key):
        return super().forward(torch.cat((data, key), dim=-1))

class Eve(Network):
    def __init__(self, channels=BASE):
        super().__init__(16, channels)
    # Inherited forward accepts ciphertext only; no key argument exists.

def stream_seed(seed, label):
    return int.from_bytes(hashlib.sha256(f'{seed}:{label}'.encode()).digest()[:8], 'big') % (2**63-1)

def generator(seed, label):
    return torch.Generator().manual_seed(stream_seed(seed, label))

def sample(g, batch):
    p = torch.randint(0, 2, (batch,16), generator=g).float()*2-1
    k = torch.randint(0, 2, (batch,16), generator=g).float()*2-1
    return p, k

def errors(pred, p):
    """Continuous error in bit-equivalent units per message, NOT discrete errors."""
    return (pred-p).abs().sum(dim=-1)/2

def metrics(pred, p):
    wrong = (pred >= 0) != (p >= 0)  # zero is assigned +1
    return {'normalized_l1': errors(pred,p).mean().item()/16,
            'bit_error_rate': wrong.float().mean().item(),
            'message_accuracy': (~wrong.any(dim=-1)).float().mean().item()}

def ab_objective(b, e, p):
    return (errors(b,p) + ((8-errors(e,p))/8).square()).mean()

def freeze(model):
    model.requires_grad_(False)
    model.eval()

def checked_step(loss, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise FloatingPointError('Missing or nonfinite gradients')
    optimizer.step()

def ab_step(a,b,e,opt,p,k):
    flags = [v.requires_grad for v in e.parameters()]
    e.requires_grad_(False)
    try:
        c = a(p,k)
        loss = ab_objective(b(c,k),e(c),p)
        checked_step(loss,opt)
    finally:
        for v, flag in zip(e.parameters(),flags): v.requires_grad_(flag)
    return loss.item()

def eve_step(a,e,opt,p,k):
    with torch.no_grad(): c = a(p,k)
    loss = errors(e(c),p).mean()
    checked_step(loss,opt)
    return loss.item()
