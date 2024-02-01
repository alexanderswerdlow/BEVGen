
# Bird's Eye View Generation (BEVGen)

[**Webpage**](https://metadriverse.github.io/bevgen/) | 
[**Code**](https://github.com/alexanderswerdlow/BEVGen) |
[**Paper**](https://arxiv.org/abs/2301.04634)

This is the official code repository for BEVGen, "Street-View Image Generation from a Bird’s-Eye View Layout, Alexander Swerdlow, Runsheng Xu, Bolei Zhou"

## Dependencies

We provide pinned dependencies in `pinned_requirements.txt` for reproducibility but suggest first attempting to install with unpinned dependencies:

- `pip install -r requirements.txt`

An Nvidia GPU is required for training and inference. This project has been tested on an Nvidia A5000 with Python 3.9.13 and CUDA 11.6 on Ubuntu 20.04.

## Datasets

### nuScenes

The nuScenes dataset can be downloaded [here](https://www.nuscenes.org/nuscenes).
At a minimum you need nuScenes Mini + Map Expansion packs. Uncompress these and put them in the same directory. Define this directory in `configs/paths` as `nuscenes_dir`. We recomend setting `NUSCENES_DATA_DIR=...` as an enviorment variable before running instead of modifying directly as some scripts require the enviorment variable to be defined.

The directory structure should look as follows (optionally with `v1.0-trainval` for full training):
```
nuscenes
└───samples
│   └───basemap
|   └───expansion
|   └───prediction
└───maps
└───sweeps
└───v1.0-mini
```

If you wish to train with the full nuScenes dataset, you must add `extras.mini_dataset=False` as a command-line override for hydra or define it in an experiment config. This is done to speed up debugging since loading the full dataset can take ~30 seconds.

### Argoverse 2

The Argoverse 2 dataset can be downloaded [here](https://www.argoverse.org/av2.html#download-link)

Extract the compressed files and define this directory in `configs/paths` as `argoverse_dir`. We recomend setting `ARGOVERSE_DATA_DIR=...` as an enviorment variable before running instead of modifying directly as some scripts require the enviorment variable to be defined. The structure should look as follows:

```
av2
└───sensor
│   └───test
|   └───train
|   └───val
|       └───0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2
|       └───...
```

In addition, to run 2nd stage training, you must pre-generate the BEV representation:

```
python multi_view_generation/scripts/argoverse_preprocess_all_cam.py --multiprocess --save-dir ${ARGOVERSE_DATA_DIR}/generated_bev_representation
```

Then, you must define `bev_dir_name=generated_bev_representation` as an argument for the Argoverse dataset in the `datamodule` config.

## Pretrained Models

We provide pre-trained weights for the Argoverse dataset. This model uses a bidirectional transformer decoder that outperforms the masking order described in the original paper. The original masking strategies are still available but may require additional configuration before training.

The following command will download the weights from huggingface:

```
mkdir -p pretrained
wget https://huggingface.co/aswerdlow/bevgen/resolve/main/argoverse_rgb.ckpt -P pretrained
wget https://huggingface.co/aswerdlow/bevgen/resolve/main/argoverse_bev.ckpt -P pretrained
wget https://huggingface.co/aswerdlow/bevgen/resolve/main/argoverse_stage_two.tar.gz -P pretrained && tar -xf pretrained/argoverse_stage_two.tar.gz -C pretrained
```

Please download
## Commands

Full Training Example for Stage 1 and 2:

```
CUDA_VISIBLE_DEVICES=0,1 python train.py experiment=multi_view_stage_1_cam_rect_argoverse trainer.devices=2 extras.mini_dataset=False

CUDA_VISIBLE_DEVICES=0,1 python train.py experiment=multi_view_stage_1_bev_argoverse trainer.devices=2 extras.mini_dataset=False

CUDA_VISIBLE_DEVICES=0,1 python train.py experiment=muse_stage_two_multi_view trainer.devices=2 extras.mini_dataset=False
```

To profile code, append `debug=profile`:

```
CUDA_VISIBLE_DEVICES=3 python train.py experiment=multi_view_stage_2 debug=profile
```

More info on hydra can be found [here](https://github.com/facebookresearch/hydra).

## Code Organization

#### Datasets

`multi_view_generation/bev_utils/nuscenes_dataset.py` and `multi_view_generation/bev_utils/argoverse.py` handles all dataloading for nuScenes and Argoverse 2 respectively. Note that not all combinations of configurations were tested and some may [silently] fail. Refer to `configs/datamodule` to see examples of valid configurations used.

#### Image Logging

Note that `multi_view_generation/utils/callback.py` handles most saving/logging of images during training and inference. We save data in 3 places: WandDB, the run directory and, if configured, a separate directory defined by `save_dir` as an argument to the callback.

You also may wish to enable `rand_str` if you generate multiple samples with the same `sample_token` and save them to the same directory.

The `save_nuscenes_fmt` saves the generated data in the original nuScenes data structure to allow for data augmentation applications.

#### Metrics

`multi_view_generation/scripts/metrics*.py` generates metrics as reported in the paper.

## Conventions

We define the angle of the camera at between [0, 2π) going counterclockwise relative to the ego reference frame.

On nuScenes, this means `CAM_FRONT_RIGHT` ≈ 5.2969 rad and `CAM_FRONT_LEFT` ≈ 0.9629 rad

All BEV representations have the ego vehicle frame at the center of the segmented image, pointing upwards.

## Errata

This codebase contains a large number of programs, each with many possible configurations, that were used for development and testing. These are not necessary for the core training and inference but were nonentheless provided to aid future research. However, it is expected that any functionality not directly described in this document will require modification before working as expected.

Specific errata are as follows:

- The DeepSpeed optimizer along with 16-bit precision is required for stage 2 training. DeepSpeed's sparse transformer module only supports fp16 and their optimizer automatically scales the lr and retries in case of NaNs.
- Stage 2 training can become unstable with small LR tweaks. Monitor the gradients and loss carefully.
- The Attention mask passed to the sparse attention module cannot be empty in any rows, even if they will be ignored. This causes NaN outputs.
- When resuming training be careful about which weights are loaded. The 1st stages are loaded first, then the model checkpoint (if available), and then the global checkpoint (if available). The model checkpoint (`+model.ckpt_path=...`) only loads model weights and the global checkpoint (`ckpt_path=...`) loads model weights, optimizer states, etc.
- DeepSpeed does not currently support changing the number of GPUs (referred to as world-size) when resuming from a global checkpoint.
- We use [pyrootutils](https://github.com/ashleve/pyrootutils) so that we can run from anywhere as a script and use either absolute or relative imports. The import must be at the top of the file to work correctly.
- Wandb logs show incorrect step counts for some graphs by default. Switch to `global_step` to get an accurate step count.
- To run 2nd stage training with the camera bias and sparse masking, you must call `save_cam_data` in the respective dataset class. This saves the camera intrinsics/extrinsics so we can initialize the model before any data is passed through.
- Some scripts require the enviorment variable `SAVE_DATA_DIR` to be defined. 
- Both nuScenes and Argoverse datasets will save cached files in `~/.cache/nuscenes` and `~/.cache/av2` respectively. This speeds up instantiating the datasets but if you change your dataset files, you must remember to delete the related cache files.

### Known Bugs
- Some nested hydra experiment configs require changing the paths for their parent configs depending on how they are called. E.g. `../config_name` vs `config_name` and `./config_name` vs `parent/config_name`.
- Generation will sometimes complete sucessfully, despite only running on a portion of the 'test' set. When provided with an `eval_generate` argument, the NuScenes dataloader will detect which instances have already been generated and remove those from the test set mitigate this issue.

## Credits

- [enhancing-transformers](https://github.com/thuanz123/enhancing-transformers)
- [taming-transformers](https://github.com/CompVis/taming-transformers)
- [minGPT](https://github.com/karpathy/minGPT)
- [PatchGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [Learned Perceptual Similarity (LPIPS)](https://github.com/richzhang/PerceptualSimilarity)
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
- [Cross View Transformers](https://github.com/bradyz/cross_view_transformers)
- [muse-maskgit-pytorch](https://github.com/lucidrains/muse-maskgit-pytorch)
- [Deepspeed](https://github.com/microsoft/DeepSpeed)
