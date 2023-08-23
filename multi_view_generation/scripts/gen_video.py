import imageio
from image_utils import get_files
import typer
from pathlib import Path
import os
import cv2
import numpy as np
from tqdm import tqdm
def main(
    folder: Path = Path('archive/figures/images'),
    output: Path = Path('output.mp4'),
    ):

 
    output_folder = Path('output')
    os.makedirs(output_folder, exist_ok=True)
    orig_name = output_folder / ("_" + output.name)
    final_name = output_folder / output.name
    writer = imageio.get_writer(orig_name, fps=5)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for file in tqdm(sorted(get_files(folder, allowed_extensions=[".png", ".jpg", ".jpeg"]))[::4]):
        arr = imageio.imread(file)
        rgb_arr = arr[..., :3]
        rgb_arr[arr[..., -1] == 0] = [255, 255, 255]
        rgb_arr = cv2.putText(np.ascontiguousarray(rgb_arr), file.name.split("_")[0], (10,425), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        writer.append_data(rgb_arr)

    writer.close()

    os.system("""ffmpeg -i """ + str(orig_name) + """ -y -vf "drawtext=fontsize=20:text='Top\: Generated Images':x=11:y=h-th-50,drawtext=fontsize=20:text='Bottom\: Source Images':x=11:y=h-th-10" -codec:a copy """ + str(final_name))
    # os.system(f"rm {orig_name}")

if __name__ == "__main__":
    typer.run(main)
