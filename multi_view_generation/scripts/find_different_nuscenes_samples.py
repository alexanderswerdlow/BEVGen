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
    dataset_type = "v1.0-trainval" # "v1.0-mini" # 
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



helper = get_helper()
nusc = helper.nusc

# images = [
#     s["token"]
#     for s in nusc.sample_data
#     if (
#         s["sensor_modality"] == "camera"
#         and s["channel"] == "CAM_FRONT"
#         and nusc.get("scene", nusc.get("sample", s["sample_token"])["scene_token"])["name"] in split_scenes
#     )
# ]

# images = images[::20]

# nusc.render_sample_data(image_token, with_anns=True, verbose=False, out_path='test.png')
# [(nusc.get("scene", scene_token)["name"], nusc.get("scene", scene_token) for scene_token in good_scene_tokens]


good_scene_tokens = []
log_locations = [log['location'] for log in nusc.log]

for log_location in log_locations:
    log_tokens = [log['token'] for log in nusc.log if log['location'] == log_location]
    assert len(log_tokens) > 0, 'Error: This split has 0 scenes for location %s!' % log_location
    T = None

    for split_type in ['train', 'val']:
        split_scenes = get_split(split_type)
        scene_tokens = [nusc.field2token('scene', 'name', scene_name)[0] for scene_name in split_scenes if len(nusc.field2token('scene', 'name', scene_name)) > 0]

        # Filter scenes.
        scene_tokens_location = [e['token'] for e in nusc.scene if e['log_token'] in log_tokens]
        if scene_tokens is not None:
            scene_tokens_location = [t for t in scene_tokens_location if t in scene_tokens]
        if len(scene_tokens_location) == 0:
            continue

        map_poses = []
        map_mask = None

        for scene_token in tqdm(scene_tokens_location):
            # scene = nusc.get('scene', scene_token[0])
            # scene_tokens_location = [scene['first_sample_token'], scene['last_sample_token']]
        
            # Get records from the database.
            scene_record = nusc.get('scene', scene_token)
            log_record = nusc.get('log', scene_record['log_token'])
            map_record = nusc.get('map', log_record['map_token'])
            map_mask = map_record['mask']

            scene_poses = []

            # For each sample in the scene, store the ego pose.
            sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)
            for sample_token in sample_tokens:
                sample_record = nusc.get('sample', sample_token)

                # Poses are associated with the sample_data. Here we use the lidar sample_data.
                sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
                pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])

                if split_type == 'train':
                    map_poses.append(np.array([pose_record['translation'][0], pose_record['translation'][1]]))
                else:
                    scene_poses.append(np.array([pose_record['translation'][0], pose_record['translation'][1]]))

            if split_type == 'val':
                if T is None:
                    continue
                scene_poses = np.vstack(scene_poses)
                num_close = T.query_ball_point(scene_poses, r = 5, return_length=True).sum()
                if num_close < 5:
                    good_scene_tokens.append(scene_token)

        
        if split_type == 'train':
            map_poses = np.vstack(map_poses)
            from scipy.spatial import KDTree
            T = KDTree(map_poses)

print(set(good_scene_tokens))


# good_scene_tokens = {'b789de07180846cc972118ee6d1fb027', '9068766ee9374872a380fe75fcfb299e', '905cfed4f0fc46679e8df8890cca4141', '6741d407b1b44511853e5ec7aaee2992', '36f27b26ef4c423c9b79ac984dc33bae', '9088db17416043e5880a53178bfa461c'}

"""
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
        and nusc.get("sample", s["sample_token"])["scene_token"] in good_scene_tokens
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
"""