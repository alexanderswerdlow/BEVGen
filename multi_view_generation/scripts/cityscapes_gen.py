import torch
from paddleseg.cvlibs import manager, Config
from paddleseg.transforms import Compose, Resize

from multi_view_generation.bev_utils.nuscenes_dataset import NuScenesDataset
from torch.utils.data import DataLoader, Dataset
import os
import math
import torchvision.transforms as T
import torch
import cv2
import numpy as np
import paddle
import typer

from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, visualize
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_show_locals=False)

def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def im_crop_center(img, w, h):
    img_width, img_height = img.size
    left, right = (img_width - w) / 2, (img_width + w) / 2
    top, bottom = (img_height - h) / 2, (img_height + h) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    return img.crop((left, top, right, bottom))

def preprocess(im_path, transforms):
    data = {}
    data['img'] = im_path
    # (900, 1600, 3) -1.0 float32
    # c h w

    # print(data['img'].shape, data['img'].min(), data['img'].max(), data['img'].dtype)
    from paddleseg.transforms import functional
    from PIL import Image
    data['img'] = Image.open(data['img'])
    # data['img'] = im_crop_center(data['img'], 900, 900)
    data['img'] = data['img'].resize((384, 192), Image.Resampling.LANCZOS)
    data['img'] = np.array(data['img']) 
    data['img'] = data['img'][:, :, ::-1].copy()
    data['img'] = (data['img']).astype(np.float32)
    data = transforms(data)
    data['img'] = data['img'][np.newaxis, ...]
    data['img'] = paddle.to_tensor(data['img'])
    return data

@app.command()
def main(
    selected_number: int = 0,
    total_number: int = 1,
    mini: bool = False,
    split: int = 0,
):

    cfg = Config("/home/aswerdlow/github/cityscapes_seg/PaddleSeg/configs/ocrnet/ocrnet_hrnetw48_cityscapes_1024x512_160k.yml")
    model = cfg.model

    model_path = '/home/aswerdlow/github/cityscapes_seg/paddle/model.pdparams'# Path of best model
    if model_path:
        para_state_dict = paddle.load(model_path)  
        model.set_dict(para_state_dict)            # Load parameters
        print('Loaded trained params of model successfully')
    else:
        raise ValueError('The model_path is wrong: {}'.format(model_path))


    transforms = Compose(cfg.val_transforms)
    utils.utils.load_entire_model(model, model_path)
    model.eval()

    CITYSCAPES_CONFIG = {
        'split': split,
        'return_cam_img': True,
        'return_bev_img': True,
        'return_all_cams': True,
        'stage_2_training': True,
        'metadrive_compatible_v2': True,
        'non_square_images': True,
        'mini_dataset': mini,
        'only_keyframes': True,
        'cam_res': (900, 1600),
        'augment_cam_img': False,
        'augment_bev_img': False,
        "square_images_cityscapes": True,
        'generate_split': (total_number, selected_number, 1),
    }
    from pathlib import Path
    root_dir = '/data/datasets/nuscenes/'
    save_dir = Path('/data1/datasets/nuscenes_cityscapes_v3')
    with paddle.no_grad():
        dataset = NuScenesDataset(**CITYSCAPES_CONFIG)
        batch_iter = DataLoader(dataset, batch_size=32, num_workers=16, shuffle=False, pin_memory=True)
        for i, data in tqdm(enumerate(batch_iter)):
            for img_paths in data['image_paths']:
                imgs = []
                for img_path in img_paths:
                    data_ = preprocess(root_dir + img_path, transforms)
                    imgs.append(data_['img'])
                
                img = paddle.concat(imgs)
                pred, _ = infer.inference(model, img, trans_info=data_['trans_info'])
                pred = paddle.squeeze(pred)
                pred = pred.numpy().astype('uint8')
                
                for i in range(len(img_paths)):
                    file_path = (save_dir / img_paths[i]).with_suffix('.npz')
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(file_path, pred=pred[i])

if __name__ == "__main__":
    app()
