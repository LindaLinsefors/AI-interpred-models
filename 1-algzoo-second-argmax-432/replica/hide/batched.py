"""K independent copies of the 2nd-argmax RNN trained at once on one GPU.

Each copy k has its own weights, init, Adam state and fresh data, so this is
exactly K independent runs of train_replica.py's recipe (Adam is elementwise, and
the loss is a sum of per-copy losses), just without the per-step Python overhead:
the whole train step is captured in a CUDA graph.

    h_t = ReLU(W_ih x_t + W_hh h_{t-1} + b_h),   logits = W_out h_9 + b_out
b_h / b_out exist only for bias='hh' / 'hh_out' (bias='none' = the original model).
"""
import math, time
import torch
import torch.nn.functional as F

H, L, OUT = 16, 10, 10


def init_params(K, bias, gen, device):
    # PyTorch defaults: nn.RNN weights/biases ~ U(-1/sqrt(H), 1/sqrt(H));
    # nn.Linear(16, 10) weight (kaiming_uniform a=sqrt 5) and bias ~ U(-1/sqrt(16), 1/sqrt(16)).
    u = lambda *s: ((torch.rand(*s, generator=gen) * 2 - 1) / math.sqrt(H)).to(device)
    p = {'W_ih': u(K, H, 1), 'W_hh': u(K, H, H), 'W_out': u(K, OUT, H)}
    if bias in ('hh', 'hh_out'):
        p['b_h'] = u(K, H)
    if bias == 'hh_out':
        p['b_out'] = u(K, OUT)
    return {k: v.requires_grad_() for k, v in p.items()}


def forward(p, x):
    """x: (K, B, L) -> logits (K, B, OUT)."""
    K, B, _ = x.shape
    h = x.new_zeros(K, B, H)
    w_ih = p['W_ih'][:, None, :, 0]                  # (K, 1, H)
    b_h = p['b_h'][:, None, :] if 'b_h' in p else 0
    w_hhT = p['W_hh'].transpose(1, 2)
    for t in range(L):
        h = torch.relu(x[:, :, t, None] * w_ih + torch.bmm(h, w_hhT) + b_h)
    logits = torch.bmm(h, p['W_out'].transpose(1, 2))
    if 'b_out' in p:
        logits = logits + p['b_out'][:, None, :]
    return logits


def targets(x):
    return x.topk(2, dim=-1).indices[..., 1]          # position of the 2nd largest


def sample(K, B, dist, device, gen=None):
    if dist == 'gauss':
        return torch.randn(K, B, L, device=device, generator=gen)
    return torch.rand(K, B, L, device=device, generator=gen) * 2 - 1


@torch.no_grad()
def evaluate(p, dist, n=2**20, chunk=2**16, seed=12345):
    """Per-copy accuracy and CE on n fresh inputs (same inputs for every copy)."""
    K = p['W_ih'].shape[0]
    dev = p['W_ih'].device
    g = torch.Generator(device=dev).manual_seed(seed)
    correct = torch.zeros(K, device=dev)
    loss = torch.zeros(K, device=dev)
    for _ in range(n // chunk):
        x1 = sample(1, chunk, dist, dev, g)
        y = targets(x1)[0]
        logits = forward(p, x1.expand(K, -1, -1))
        correct += (logits.argmax(-1) == y).float().sum(1)
        loss += F.cross_entropy(logits.reshape(-1, OUT), y.repeat(K), reduction='none').view(K, -1).sum(1)
    return (correct / n).cpu(), (loss / n).cpu()


def train(K, batch_size, seed, bias='none', epochs=1000, lr=0.005, gamma=0.995, weight_decay=1e-14,
          samples_per_epoch=100_000, device='cuda', log_every=50, log=print):
    """Notebook recipe: Adam + ExponentialLR(gamma) per epoch, 'epochs' x samples_per_epoch fresh Unif(-1,1) samples."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    p = init_params(K, bias, gen, device)
    params = list(p.values())
    lr_t = torch.tensor(lr, device=device)
    opt = torch.optim.Adam(params, lr=lr_t, weight_decay=weight_decay, capturable=True)
    loss_acc = torch.zeros(K, device=device)
    corr_acc = torch.zeros(K, device=device)
    B = batch_size

    def step():
        x = sample(K, B, 'unif', device)
        y = targets(x)
        logits = forward(p, x)
        losses = F.cross_entropy(logits.reshape(-1, OUT), y.reshape(-1), reduction='none').view(K, B).mean(1)
        opt.zero_grad(set_to_none=False)
        losses.sum().backward()
        opt.step()
        with torch.no_grad():
            loss_acc.add_(losses.detach())
            corr_acc.add_((logits.argmax(-1) == y).float().mean(1))

    # warm up on a side stream (initialises Adam state), then capture one step in a CUDA graph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    # undo the warm-up steps so every copy starts from its init with fresh Adam state
    with torch.no_grad():
        p0 = init_params(K, bias, torch.Generator().manual_seed(seed), device)
        for k in p:
            p[k].copy_(p0[k])
        for st in opt.state.values():
            for v in st.values():
                v.zero_()
    g = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=False)
    with torch.cuda.graph(g):
        step()

    steps_per_epoch = samples_per_epoch // B
    t0 = time.time()
    history = []
    for epoch in range(epochs):
        loss_acc.zero_(); corr_acc.zero_()
        for _ in range(steps_per_epoch):
            g.replay()
        lr_t.mul_(gamma)                               # ExponentialLR, stepped once per epoch
        if (epoch + 1) % log_every == 0 or epoch == 0:
            l = (loss_acc / steps_per_epoch).cpu()
            a = (corr_acc / steps_per_epoch).cpu()
            history.append({'epoch': epoch + 1, 'loss': l.tolist(), 'acc': a.tolist()})
            log(f"bs={B} bias={bias} seed={seed} epoch {epoch+1} best loss {l.min():.4f} median {l.median():.4f} "
                f"best acc {a.max():.4f} t={time.time()-t0:.0f}s")
    return {k: v.detach() for k, v in p.items()}, history
