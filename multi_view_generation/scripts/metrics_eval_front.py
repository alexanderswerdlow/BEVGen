import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
import typer
from pathlib import Path
from cleanfid import fid
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from multi_view_generation.bev_utils import SAVE_DATA_DIR
import hashlib
from os.path import exists
import pickle
import shutil
import os
class GeneratedDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        mini_dataset: bool = False,
        rec_dataset: bool = False,
        cam_name: str = None,
        **kwargs,
    ):
        self.mini_dataset = mini_dataset
        self.dataset_dir = dataset_dir
        self.comp_dataset = 'rec' if rec_dataset else 'gen'
        gt_samples = list((dataset_dir / 'gt').glob(f'**/{cam_name}/*.jpg'))
        gen_samples = list((dataset_dir / 'gen').glob(f'**/{cam_name}/*.jpg'))
        rec_samples = list((dataset_dir / 'rec').glob(f'**/{cam_name}/*.jpg'))
        max_samples = max(len(gt_samples), len(gen_samples), len(rec_samples))
        
        gt_samples = set([gt_path.relative_to(Path(dataset_dir / 'gt')) for gt_path in gt_samples])
        gen_samples = set([gen_path.relative_to(Path(dataset_dir / 'gen')) for gen_path in gen_samples])
        rec_samples = set([rec_path.relative_to(Path(dataset_dir / 'rec')) for rec_path in rec_samples])
        self.samples = list(gt_samples.intersection(gen_samples).intersection(rec_samples))
        if (min_removed := max_samples - len(self.samples)) > 0:
            print(f'Removed at least {min_removed}')
         
        sha_gt = hashlib.sha1(f"{','.join(sorted(list(map(lambda x: str(x), gt_samples))))}".encode("utf-8")).hexdigest()
        sha_gen = hashlib.sha1(f"{','.join(sorted(list(map(lambda x: str(x), gen_samples))))}".encode("utf-8")).hexdigest()
        sha_rec = hashlib.sha1(f"{','.join(sorted(list(map(lambda x: str(x), rec_samples))))}".encode("utf-8")).hexdigest()
        assert sha_gt == sha_gen == sha_rec

        print(f'Total of {len(self.samples)} samples with hash: {sha_gt}')

        # pickle_name = 'data.pkl'
        # if exists(pickle_name):
        #     with open(pickle_name, "rb") as f:
        #         print(len(self.samples))
        #         self.samples = set(pickle.load(f)).intersection(self.samples)
        #         print(len(self.samples))

        # with open(pickle_name, "wb") as f:
        #     pickle.dump(self.samples, f)

        self.samples = list(self.samples)

    def __getitem__(self, idx):
        gt = to_tensor(Image.open(Path(self.dataset_dir) / 'gt' / self.samples[idx]))
        ret = to_tensor(Image.open(Path(self.dataset_dir) / self.comp_dataset / self.samples[idx]))
        return gt, ret

    def __len__(self):
        if self.mini_dataset:
            return 10
        else:
            return len(self.samples)

    def get_all_image_paths(self):
        return list(map(lambda x: Path(self.dataset_dir) / 'gt' / x, self.samples)), list(map(lambda x: Path(self.dataset_dir) / self.comp_dataset / x, self.samples))
        
def compute_fid(dataset: GeneratedDataset):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)
        gt_paths, ret_paths = dataset.get_all_image_paths()
        gt_dir = Path(tmpdirname) / 'gt'
        comp_dir = Path(tmpdirname) / 'comp'
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(comp_dir, exist_ok=True)
        for gt_path, ret_path in zip(gt_paths, ret_paths):
            shutil.copy(gt_path, gt_dir)
            shutil.copy(ret_path, comp_dir)

        score = fid.compute_fid(str(gt_dir.resolve()), str(comp_dir.resolve()))
        print(f'FID Score for dataset {dataset.dataset_dir} is: {score:.2f}')
    return score

def main(
    dataset_dir: Path = (SAVE_DATA_DIR / "baseline" / "multi_view_simple"),
    mini_dataset: bool = False,
    rec_dataset: bool = False,
):  
    for cam_name in ("CAM_FRONT", "CAM_BACK", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"):
        dataset = GeneratedDataset(dataset_dir=dataset_dir, mini_dataset=mini_dataset, rec_dataset=rec_dataset, cam_name=cam_name)
        dataloader = DataLoader(dataset, batch_size=20, num_workers=10, pin_memory=True)
        print(cam_name)
        compute_fid(dataset)
        print(f'Folder {dataset_dir}')

if __name__ == "__main__":
    typer.run(main)
