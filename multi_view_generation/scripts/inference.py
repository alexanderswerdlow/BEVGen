import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from torch.utils.data import default_collate
import typer
from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig
from hydra import compose, initialize
from hydra.utils import instantiate
import torch
import hydra
import pytorch_lightning as pl
from typing import List, Optional, Tuple
from omegaconf import OmegaConf, DictConfig
import time
from random import randint
from torch.profiler import profile, ProfilerActivity
from pathlib import Path
import os

def move_to(obj, device):
    if torch.is_tensor(obj):
        if isinstance(obj, torch.FloatTensor):
            obj = obj.to(dtype=torch.half)
        return obj.to(device=device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, str):
        return obj
    else:
        raise TypeError("Invalid type for move_to")


def random_bbox(bbox):
    v = [randint(0, v) for v in bbox]
    left = min(v[0], v[2])
    upper = min(v[1], v[3])
    right = max(v[0], v[2])
    lower = max(v[1], v[3])
    return [left, upper, right, lower]


def get_fake_data(cfg, device, dtype, batch_size):
    def get_sample():
        num_cams = 4
        return {'image': torch.randn(([num_cams, cfg.cam_res[0], cfg.cam_res[1], 3]), dtype=dtype),
        'segmentation': torch.randn(([256, 256, 7]), dtype=dtype),
        'angle': torch.pi,
        'dataset': 'argoverse',
        'token': '17c2936e57db48809ba36e21e466aba9',
        'channel': 'CAM_FRONT_LEFT',
        'cam_idx': torch.tensor([0, 1, 2, 3, 4, 5]),
        'intrinsics_inv': torch.randn([num_cams, 3, 3]),
        'extrinsics_inv': torch.randn([num_cams, 4, 4]),
        'view': torch.randn([3, 3]),
        'center': torch.randn([1, 200, 200]),
        'pose': torch.randn([4, 4]),
        'bbx': torch.tensor([[random_bbox([0, 0, cfg.cam_res[0], cfg.cam_res[1]]) for _ in range(5)] for _ in range(6)]),
        }
    batch = default_collate([get_sample() for _ in range(batch_size)])
    batch = move_to(batch, device)
    return batch


def forward_pass(model: pl.LightningModule, batch: dict):
    model.inference_step(batch=batch)


def forward_backward_pass(model: pl.LightningModule, batch: dict):
    loss = model.shared_step(batch=batch, batch_idx=0, inference=True)
    model.zero_grad()
    loss.backward()


def get_model_hydra_compose(overrides: Optional[List[str]], quick_init=False) -> Tuple[pl.LightningModule, DictConfig]:
    with initialize(version_base="1.2", config_path="../../configs", job_name="inference"):
        appends = []
        if quick_init:
            appends.append("model.first_stage.ckpt_path=null")
            appends.append("model.cond_stage.ckpt_path=null")

        cfg = compose(config_name="train.yaml", overrides=[
            "model.transformer.cfg.output_dir=output", 
            *appends,
            *overrides])
        print(OmegaConf.to_yaml(cfg))
        pl.seed_everything(cfg.seed)
        model = instantiate(cfg.model)
        model = model.to('cuda:0')
        model.eval()
        model.half()
        return model, cfg

def benchmark_model(model, data, batch_size, active_runs, inference_function=forward_backward_pass):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inference_function(model, data)
    inference_function(model, data)

    start = time.time()
    for _ in range(active_runs):
        inference_function(model, data)
        torch.cuda.synchronize()

    stop = time.time()
    max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
    step_time = (stop - start) / (active_runs * batch_size)
    print(f"Peak memory use: {max_memory}MB - {step_time * 1e3}ms")
    return step_time, max_memory

def profile_model(model, data, batch_size, active_runs, inference_function=forward_backward_pass):
    inference_function(model, data)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        record_shapes=True,
        with_modules=True,
        with_flops=True,
        profile_memory=True,
        with_stack=True
        ) as prof:
        for _ in range(active_runs):
            inference_function(model, data)
            torch.cuda.synchronize()
        
    save_path = Path("output/trace.json")
    os.makedirs(save_path.parents[0], exist_ok=True)
    prof.export_chrome_trace(str(save_path))
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    overrides: Optional[List[str]] = typer.Argument(None),
    fake_data: bool = False,
    benchmark: bool = False,
    profile: bool = False,
    active_runs: int = 4,
    batch_size: int = 8,
):
    print(f"Got {overrides=}")
    model, cfg = get_model_hydra_compose(overrides, quick_init=fake_data)
    
    if fake_data:
        data = get_fake_data(cfg, model.device, torch.half, batch_size)

    if benchmark:
        # benchmark_model(model, data, batch_size, active_runs, inference_function=forward_pass)
        benchmark_model(model, data, batch_size, active_runs, inference_function=forward_backward_pass)

    if profile:
        profile_model(model, data, batch_size, active_runs)

if __name__ == "__main__":
    app()

# 0.05, Peak memory use: 1767MB - 39.18616473674774ms
# 0.15, Peak memory use: 1980MB - 56.774742901325226ms
# 0.25, Peak memory use: 2363MB - 71.31486386060715ms
# 0.35, Peak memory use: 2744MB - 82.94279873371124ms
# 0.45, Peak memory use: 3127MB - 106.9994568824768ms
# 1.0, Peak memory use: 3507MB - 120.8610013127327ms

# CUDA_VISIBLE_DEVICES=7 python multi_view_generation/scripts/inference.py experiment=benchmark/nuscenes.yaml --fake-data --benchmark --batch-size 8
# 242ms
# 340.93

# CUDA_VISIBLE_DEVICES=7 python multi_view_generation/scripts/inference.py experiment=multi_view_stage_2_full_argoverse.yaml +cfg.plot=True --fake-data --benchmark --batch-size 2

