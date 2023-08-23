import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np
from nuscenes.utils.geometry_utils import BoxVisibility
from tqdm import tqdm
import random
from multi_view_generation.bev_utils.nuscenes_helper import get_split, parse_scene, NuScenesSingleton
from pathlib import Path
import pickle
import time
from os.path import exists
from multi_view_generation.bev_utils import NUSCENES_DIR

def get_helper():
    helper = None
    dataset_type = "v1.0-trainval"
    pickle_name = Path.home() / ".cache" / "nuscenes" / f"nusc-{dataset_type}.p"

    if exists(pickle_name):
        start_time = time.time()
        with open(pickle_name, "rb") as f:
            helper = pickle.load(f)
        print(f'Took {round(time.time() - start_time, 1)} seconds to load NuScenes {dataset_type}')
    else:
        pickle_name.parent.mkdir(parents=True, exist_ok=True)
        helper = NuScenesSingleton(NUSCENES_DIR, dataset_type)
        with open(pickle_name, "wb") as f:  # "wb" because we want to write in binary mode
            pickle.dump(helper, f)
    return helper


split_scenes = get_split("val")
helper = get_helper()
nusc = helper.nusc

images = [
    s["token"]
    for s in nusc.sample_data
    if (
        s["sensor_modality"] == "camera"
        and s["channel"] == "CAM_FRONT"
        and nusc.get("scene", nusc.get("sample", s["sample_token"])["scene_token"])["name"] in split_scenes
    )
]

images = images[::20]

# nusc.render_sample_data(image_token, with_anns=True, verbose=False, out_path='test.png')

viz_weights = {'1': 0.05, '2': 0.5, '3': 0.7, '4': 2.0}
viz_sums = []

for image_token in tqdm(images):
    boxes = nusc.get_sample_data(image_token, box_vis_level=BoxVisibility.NONE)[1]

    # We perform a very rough weighting of how close the annotation is and how visible it is
    viz_tokens = sum([
        viz_weights[nusc.get('sample_annotation', box.token)['visibility_token']] * ((1 / (np.linalg.norm(box.center))**3) * np.prod(box.wlh))
        for box in boxes
    ])
    viz_sums.append(viz_tokens)


image_tokens = random.choices(images, viz_sums, k=500)

frozen_image_tokens = 'pretrained/test_image_tokens.txt'
with open(frozen_image_tokens, 'w') as f:
    f.write('\n'.join(image_tokens))
