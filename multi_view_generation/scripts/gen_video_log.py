import imageio
from image_utils import get_files, get_file_list
import typer
from pathlib import Path
import os

def main(
    folder: Path = Path('archive/figures/images'),
    output: Path = None,
    logid: str = "0b86f508-5df9-4a46-bc59-5b9536dbde9f"
    ):
    
    if output is None:
        output = Path(logid + ".mp4")
    output_folder = Path('output')
    os.makedirs(output_folder, exist_ok=True)
    orig_name = output_folder / ("_" + output.name)
    final_name = output_folder / output.name
    writer = imageio.get_writer(orig_name, fps=5)
    files = sorted(get_files(folder, allowed_extensions=[".png", ".jpg", ".jpeg"], allowed_in_name=logid))[::4]
    print(logid)
    assert len(files) > 0
    import cv2, numpy as np
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for file in files:
        arr = imageio.imread(file)
        rgb_arr = arr[..., :3]
        rgb_arr[arr[..., -1] == 0] = [255, 255, 255]
        # rgb_arr = cv2.putText(np.ascontiguousarray(rgb_arr), file.name.split("_")[0], (10,425), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        writer.append_data(rgb_arr)
    writer.close()

    os.system("""ffmpeg -i """ + str(orig_name) + """ -y -vf "drawtext=fontsize=20:text='Top\: Generated Images':x=11:y=h-th-50,drawtext=fontsize=20:text='Bottom\: Source Images':x=11:y=h-th-10" -codec:a copy """ + str(final_name))
    os.system(f"rm {orig_name}")

folder = '/data1/datasets/generated/argoverse_interesting_debug/argoverse_interesting_debug_site_compare/gen_images_compare'

# images = get_file_list(path=folder, allowed_extensions=[".png", ".jpg", ".jpeg"])
# rand_log_ids = set()

# for image in images:
#     rand_log_ids.add(image.name.split("_")[0])

# total_logs = 20
# rand_log_ids = list(rand_log_ids)[::8]
log_ids = [
# 'b19f3c1a-a84a-3a2d-8d1b-8a4ae201020b',
'78683234-e6f1-3e4e-af52-6f839254e4c0',
'bd90cd1a-38b6-33b7-adec-ba7d4207a8c0',
'c865c156-0f26-411c-a16c-be985333f675',
'02a00399-3857-444e-8db3-a8f58489c394',
#'76916359-96f4-3274-81fe-bb145d497c11',
# '7e4d67b3-c3cc-3288-afe5-043602ea3c70',
# '29a00842-ead2-3050-b587-c5ef507e4125',
# '11ba4e81-c26f-3cd1-827d-b6913bcef64e', 
# '0fb7276f-ecb5-3e5b-87a8-cc74c709c715',
# '02678d04-cc9f-3148-9f95-1ba66347dff9',
# '20bcd747-ef60-391a-9f4a-ae99f049c260',
# '5f8f4a26-59b1-3f70-bcab-b5e3e615d3bc',
# '6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c',
# '185d3943-dd15-397a-8b2e-69cd86628fb7',
# '6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c',
# '7dbc2eac-5871-3480-b322-246e03d954d2',
# 'b87683ae-14c5-321f-8af3-623e7bafc3a7',
]

# assert(len(fixed_log_ids) < total_logs)
   
# import random
# new_log_ids = random.choices(rand_log_ids, k=(total_logs - len(fixed_log_ids)))
# while set(fixed_log_ids).intersection(new_log_ids) != set():
#     new_log_ids = random.choices(rand_log_ids, k=(total_logs - len(fixed_log_ids)))

# log_ids = fixed_log_ids + new_log_ids

# assert(len(log_ids) == total_logs)

with open('videolist.txt', 'w') as f:
    for log_id in log_ids:
        main(folder, logid=log_id)
        f.write(f"file output/{log_id}.mp4\n")

os.system(f"""ffmpeg -y -f concat -safe 0 -i videolist.txt -c copy continuous_scene_generation.mp4""")
    
# if __name__ == "__main__":
#     typer.run(main)
