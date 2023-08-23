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
import torch
import shutil
import os
from multi_view_generation.scripts.metrics_consistency import GeneratedDataset as ConsistencyGeneratedDataset
from multi_view_generation.scripts.metrics_consistency import compute_overlap
from multi_view_generation.scripts.figure_generator import main as figure_generator_main

class GeneratedDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        mini_dataset: bool = False,
        rec_dataset: bool = False,
        argoverse: bool = False,
        **kwargs,
    ):
        self.mini_dataset = mini_dataset
        self.dataset_dir = dataset_dir

        if argoverse:
            gt_samples = list((dataset_dir / 'sample_gt').glob('**/*.jpg'))
            gen_samples = list((dataset_dir / 'sample').glob('**/*.jpg'))
            max_samples = max(len(gt_samples), len(gen_samples))
            
            gt_samples = set([gt_path.relative_to(Path(dataset_dir / 'sample_gt')) for gt_path in gt_samples])
            gen_samples = set([gen_path.relative_to(Path(dataset_dir / 'sample')) for gen_path in gen_samples])
            self.samples = list(gt_samples.intersection(gen_samples))
            if (min_removed := max_samples - len(self.samples)) > 0:
                print(f'Removed at least {min_removed}')
            
            sha_gt = hashlib.sha1(f"{','.join(sorted(list(map(lambda x: str(x), gt_samples))))}".encode("utf-8")).hexdigest()
            sha_gen = hashlib.sha1(f"{','.join(sorted(list(map(lambda x: str(x), gen_samples))))}".encode("utf-8")).hexdigest()
            assert sha_gt == sha_gen

            print(f'Total of {len(self.samples)} samples with hash: {sha_gt}')
        else:
            self.comp_dataset = 'rec' if rec_dataset else 'gen'
            gt_samples = list((dataset_dir / 'gt').glob('**/*.jpg'))
            gen_samples = list((dataset_dir / 'gen').glob('**/*.jpg'))
            rec_samples = list((dataset_dir / 'rec').glob('**/*.jpg'))
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
        self.argoverse = argoverse

    def __getitem__(self, idx):
        if self.argoverse:
            gt = to_tensor(Image.open(Path(self.dataset_dir) / 'sample_gt' / self.samples[idx]))
            ret = to_tensor(Image.open(Path(self.dataset_dir) / 'sample' / self.samples[idx]))
        else:
            gt = to_tensor(Image.open(Path(self.dataset_dir) / 'gt' / self.samples[idx]))
            ret = to_tensor(Image.open(Path(self.dataset_dir) / self.comp_dataset / self.samples[idx]))
        return gt, ret

    def __len__(self):
        if self.mini_dataset:
            return 10
        else:
            return len(self.samples)

    def get_all_image_paths(self):
        ret_samples = self.samples[:10] if self.mini_dataset else self.samples
        if self.argoverse:
            return list(map(lambda x: Path(self.dataset_dir) / 'sample_gt' / x, ret_samples)), list(map(lambda x: Path(self.dataset_dir) / 'sample' / x, ret_samples))
        else:
            return list(map(lambda x: Path(self.dataset_dir) / 'gt' / x, ret_samples)), list(map(lambda x: Path(self.dataset_dir) / self.comp_dataset / x, ret_samples))
        
        

def compute_metrics(dataloader: DataLoader):
    device = torch.device('cuda:0')
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    for gt, gen in tqdm(dataloader):
        gt, gen = gt.to(device), gen.to(device)
        lpips.update(gen, gt)
        ssim.update(gen, gt)
        psnr.update(gen, gt)
    lpips_value = lpips.compute()
    psnr_value = psnr.compute()
    ssim_value = ssim.compute()

    print(f'PSIM/LPIPS Score for dataset is: {lpips_value:.2f}')
    print(f'SSIM Score for dataset is: {ssim_value:.2f}')
    print(f'PSNR Score for dataset is: {psnr_value:.2f}')
    return lpips_value, ssim_value, psnr_value

def compute_fid(dataset: GeneratedDataset):
    import tempfile
    with tempfile.TemporaryDirectory(dir = Path.home() / 'tmp') as tmpdirname:
        print('created temporary directory', tmpdirname)
        gt_paths, ret_paths = dataset.get_all_image_paths()
        gt_dir = Path(tmpdirname) / 'gt'
        comp_dir = Path(tmpdirname) / 'comp'
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(comp_dir, exist_ok=True)
        for gt_path, ret_path in zip(gt_paths, ret_paths):
            shutil.copy(gt_path, gt_dir / f"{gt_path.name[:-4]}_{gt_path.parent.name}.jpg")
            shutil.copy(ret_path, comp_dir / f"{ret_path.name[:-4]}_{ret_path.parent.name}.jpg")

        score = fid.compute_fid(str(gt_dir.resolve()), str(comp_dir.resolve()))
        print(f'FID Score for dataset {dataset.dataset_dir} is: {score:.2f}')
    return score

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    dataset_dir: Path = (SAVE_DATA_DIR / "baseline" / "multi_view_simple"),
    mini_dataset: bool = False,
    rec_dataset: bool = False,
    argoverse: bool = False
):
    dataset = GeneratedDataset(dataset_dir=dataset_dir, mini_dataset=mini_dataset, rec_dataset=rec_dataset, argoverse=argoverse)
    dataloader = DataLoader(dataset, batch_size=20, num_workers=10, pin_memory=True)
    fid = compute_fid(dataset)
    psim, ssim, psnr = compute_metrics(dataloader)

    consistency_dataset = ConsistencyGeneratedDataset(dataset_dir=dataset_dir, mini_dataset=mini_dataset, rec_dataset=rec_dataset)
    overlap_gt_fid, overlap_gen_fid, overlap_gt_ssim, overlap_gen_ssim = compute_overlap(consistency_dataset)

    print(f'Folder: {dataset_dir}')
    print(f'Overlap : {overlap_gt_fid:.3f}, {overlap_gen_fid:.3f}, SSIM: {overlap_gt_ssim:.3f}, {overlap_gen_ssim:.3f}')
    print(f'Results : {fid:.2f} & {psim:.2f} & {ssim:.2f} & {psnr:.2f}')
    figure_generator_main(dataset_dir, make_site=True)
    

if __name__ == "__main__":
    app()