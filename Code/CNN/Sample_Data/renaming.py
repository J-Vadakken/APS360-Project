# rename the images in the folder by order
#1.png 2.png, ...
import os
import shutil
import re

def rename_files(folder_path):
    files = os.listdir(folder_path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    for i, file in enumerate(files):
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, str(i+1) + '.png'))

rename_files('training_validation_data/00_Stereo_Madness/00_Stereo_Madness_Normal')
rename_files('training_validation_data/00_Stereo_Madness/00_Stereo_Madness_Hitbox')