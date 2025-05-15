#%%
import os
import sys
os.chdir(os.path.dirname(__file__))
#%%
import glob
import contextlib
from PIL import Image

# filepaths
folder = "./results/250324_0837_dev_step7_train67_test112_fit_admissions/animation/"
fp_in =  folder + "./*.png"
fp_out = folder + "./combined_video.gif"

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:

    # lazily load images
    imgs = (stack.enter_context(Image.open(f))
            for f in sorted(glob.glob(fp_in)))
    
    # replace transparency with white background
    imgs = (Image.composite(img, Image.new('RGB', img.size, (255, 255, 255)), img)
            for img in imgs)

    # extract  first image from iterator
    img = next(imgs)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=500, loop=0)
print("done!")
# %%
