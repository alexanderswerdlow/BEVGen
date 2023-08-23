import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from multi_view_generation.modules.transformer.mingpt_sparse import GPTConfig
import torch.nn as nn
from multi_view_generation.bev_utils import save_binary_as_image
from hydra import compose, initialize
from hydra.utils import instantiate
import torch
import hydra

def test_gpt(model, num_runs=0, batch_size=2):
    model = model.cuda().half()
    dt = torch.float16
    batch = {'intrinsics_inv': torch.randn([batch_size, 6, 3, 3], device='cuda').to(dt), 'extrinsics_inv': torch.randn([batch_size, 6, 4, 4], device='cuda').to(dt)}
    inputs = [torch.randint(0, 1024, (batch_size, 6, model.cfg.cam_latent_h * model.cfg.cam_latent_w)).to(dtype=torch.int64, device='cuda'), torch.randint(0, 10, (batch_size, 256)).to(dtype=torch.int64, device='cuda'), batch, True]

    output = model(*inputs)
    
    if num_runs == 0:
        return "", output

    loss_fn = torch.nn.MSELoss(reduction='sum')

    # bookeeping
    import time
    start = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(num_runs):
        output = model(*inputs)
        loss = loss_fn(output, torch.rand_like(output))
        model.zero_grad()
        loss.backward()

    torch.cuda.synchronize()
    stop = time.time()

    # now report
    max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
    return f"Peak memory use: {max_memory}MB - {(round((stop - start) * 1e6) / 1e3) / num_runs}ms", output

def test_mask_generation_special():
    import pytorch_lightning as pl
    initialize(version_base="1.2", config_path="../../../configs", job_name="mask_generator")
    cfg = compose(config_name="train.yaml", 
    overrides=[
    "experiment=multi_view_stage_2_full_v2", 
    "model.transformer.cfg.plot=False", 
    "model.transformer.cfg.output_dir=output",
    "model.transformer.cfg.bev_latent_res=[3,3]",
    "model.transformer.cfg.cam_latent_res=[4,3]",
    "model.transformer.cfg.num_cams=3",
    ])
    pl.seed_everything(cfg.seed)
    cfg_: GPTConfig = instantiate(cfg.model.transformer.cfg)
    save_binary_as_image(cfg_.prob_matrix[cfg_.num_cond_tokens:-cfg_.num_pad_tokens, :-cfg_.num_pad_tokens], 'prob_matrix.png')

def test_mask_generation():
    import pytorch_lightning as pl
    initialize(version_base="1.2", config_path="../../../configs", job_name="mask_generator")
    cfg = compose(config_name="train.yaml", 
    overrides=["experiment=ablation/ablation_causal_bias_img_bev_embed_bbox_sparse", "model.transformer.cfg.plot=True", "model.transformer.cfg.output_dir=output"])
    pl.seed_everything(cfg.seed)
    cfg_ = instantiate(cfg.model.transformer.cfg)
    print(cfg_.prob_matrix)

def test_gpt_deepspeed(num_runs=4, batch_size=2, override=[]):
    import pytorch_lightning as pl
    initialize(version_base="1.2", config_path="../../../configs", job_name="mask_generator")
    cfg = compose(config_name="train.yaml", 
    overrides=override)
    pl.seed_everything(cfg.seed)
    model = instantiate(cfg.model.transformer)
    perf, output = test_gpt(model, num_runs=num_runs, batch_size=batch_size)
    print(perf)
    print(output[torch.isfinite(output)].mean())
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    del model

if __name__ == "__main__":
    test_mask_generation()

    