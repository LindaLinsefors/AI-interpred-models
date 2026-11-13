"""Replica training of ARC's example 2nd-argmax RNN (M_16,10, 432 params).

Setup copied from Zihao Chen's notebook that produced
one_layer_16_hidden_94_acc_2nd_argmax.pth (= AlgZoo's example_2nd_argmax()):
  - 1-layer ReLU nn.RNN, hidden 16, no biases; nn.Linear(16, 10, bias=False); default PyTorch init
  - inputs iid Unif(-1, 1), length 10, fresh batch every step; target = argsort[-2]
  - cross-entropy on logits from the final hidden state
  - Adam(lr=0.005, weight_decay=1e-14), ExponentialLR(gamma=0.995) stepped per epoch
  - 1000 epochs x 100,000 samples per epoch (steps/epoch = 100000 // batch_size)
Not recorded in the notebook: batch size and seed (swept here).
"""
import argparse, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN, DIM = 10, 1


class OneLayerRNNNoBias(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size=DIM, hidden_size=hidden_size, batch_first=True, nonlinearity='relu', bias=False)
        self.linear = nn.Linear(hidden_size, SEQ_LEN, bias=False)

    def forward(self, x, init_state=None):
        outputs, h_n = self.rnn(x, init_state)
        return self.linear(outputs), self.linear(h_n.squeeze(0))


def make_data_arg2ndmax(samples, seq_len, dim, device, factor=1):
    src = torch.rand(samples, seq_len, dim, device=device) * 2 * factor - factor
    y = src.sort(dim=1)[1][:, -2].squeeze()
    return src, y


@torch.no_grad()
def evaluate(model, dist, n=2**20, seed=12345):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, SEQ_LEN, DIM, generator=g) if dist == 'gauss' else torch.rand(n, SEQ_LEN, DIM, generator=g) * 2 - 1
    y = x.sort(dim=1)[1][:, -2].squeeze()
    _, logits = model(x)
    return (logits.argmax(1) == y).float().mean().item()


def train(batch_size, seed, epochs=1000, lr=0.005, weight_decay=1e-14, samples_per_epoch=100000, log=None):
    torch.manual_seed(seed)
    model = OneLayerRNNNoBias(16)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    steps_per_epoch = samples_per_epoch // batch_size
    t0 = time.time()
    for epoch in range(epochs):
        tot_loss = correct = total = 0
        for _ in range(steps_per_epoch):
            src, y = make_data_arg2ndmax(batch_size, SEQ_LEN, DIM, 'cpu', 1)
            _, pred = model(src)
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()
        if log and (epoch % 50 == 49 or epoch == 0):
            print(f"bs={batch_size} seed={seed} epoch {epoch+1} loss {tot_loss/steps_per_epoch:.4f} "
                  f"acc {correct/total:.4f} t={time.time()-t0:.0f}s", file=log, flush=True)
    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    import sys
    model = train(a.batch_size, a.seed, a.epochs, log=sys.stdout)
    res = dict(batch_size=a.batch_size, seed=a.seed, epochs=a.epochs,
               acc_uniform=evaluate(model, 'unif'), acc_gauss=evaluate(model, 'gauss'))
    torch.save(model.state_dict(), a.out + '.pth')
    json.dump(res, open(a.out + '.json', 'w'))
    print(res)
