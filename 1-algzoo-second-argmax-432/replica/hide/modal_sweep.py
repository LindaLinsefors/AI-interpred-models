"""Fan batched training runs (batched.py) out over Modal GPUs.

    modal run modal_sweep.py --sweep NAME --configs '[{"batch_size":64,"seed":0,"K":64}, ...]'

Each config trains K independent copies with batched.train and returns their
weights, uniform/Gaussian test accuracy + CE (2^20 samples each), and training history.
Results are saved locally to runs_modal/<sweep>/<bias>_bs<B>_s<seed>_<tag>.pt.
"""
import json, pathlib
import modal

HERE = pathlib.Path(__file__).parent
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
         .add_local_file(HERE / "batched.py", "/root/batched.py"))
app = modal.App("algzoo-2nd-argmax-replica", image=image)


@app.function(gpu="H100", timeout=4 * 3600)
def train_job(cfg):
    import torch, batched
    cfg = dict(cfg)
    K, B, seed = cfg.pop("K"), cfg.pop("batch_size"), cfg.pop("seed")
    tag = cfg.pop("tag", "")
    p, history = batched.train(K, B, seed, log=lambda s: print(s, flush=True), **cfg)
    out = {"K": K, "batch_size": B, "seed": seed, "tag": tag, **cfg,
           "params": {k: v.cpu() for k, v in p.items()}, "history": history}
    for dist in ("unif", "gauss"):
        acc, ce = batched.evaluate(p, dist)
        out[f"acc_{dist}"], out[f"ce_{dist}"] = acc, ce
    print(f"DONE bs={B} seed={seed} {cfg} best unif acc {out['acc_unif'].max():.4f} "
          f"best gauss acc {out['acc_gauss'].max():.4f}", flush=True)
    return out


@app.local_entrypoint()
def main(sweep: str, configs: str):
    import torch
    cfgs = json.loads(configs)
    outdir = HERE / "runs_modal" / sweep
    outdir.mkdir(parents=True, exist_ok=True)
    for out in train_job.map(cfgs, order_outputs=False, return_exceptions=True):
        if isinstance(out, Exception):
            print("FAILED:", repr(out), flush=True)
            continue
        name = f"{out.get('bias', 'none')}_bs{out['batch_size']}_s{out['seed']}{('_' + out['tag']) if out['tag'] else ''}.pt"
        torch.save(out, outdir / name)
        print("saved", name, flush=True)
