# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Annoation and Segmentation

# %% [markdown]
# ## Sampling 100 images with seed = 42

# %%
import os
import shutil
import pandas as pd

from pathlib import Path

# %% [markdown]
# ### Setup path

# %%
images_dir     = Path("../data/Radiographs/")
csv_path       = Path("../data/image_age.csv")
annotation_dir = Path("../data/Annotation-batch-01/")

# %%
df = pd.read_csv(csv_path)
df.head()

# %%
sample = df.sample(n=100, random_state=42)

# %% [markdown]
# ### Copy sampled files to annotation-batch-01

# %%
count = 0
for _, row in sample.iterrows():
    img_name = row['file_name']
    src_path = images_dir / img_name
    dest_path = annotation_dir / img_name
    
    if src_path.exists():
        shutil.copy2(src_path, dest_path)
        count += 1

sample.to_csv("../data/annotation_batch_01.csv", index=False)

print(f"Successfully copied {count} images to {annotation_dir}")

# %% [markdown]
# # :)

# %%
import cv2

img = cv2.imread('../data/Radiographs/0651-001-02.jpg', 0)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced_img = clahe.apply(img)

cv2.imwrite('./enhanced_forensic_image.png', enhanced_img)

# %%
# !uv run python -c "import cv2; print(cv2.__version__)"
# %% [markdown]
# ## Enhance Batch 01 Images

# %%
enhanced_dir = Path("../data/Annotation-batch-01-enhanced/")
enhanced_dir.mkdir(parents=True, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

enhanced_count = 0
for img_path in annotation_dir.iterdir():
    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            enhanced = clahe.apply(img)
            cv2.imwrite(str(enhanced_dir / img_path.name), enhanced)
            enhanced_count += 1

print(f"Successfully enhanced {enhanced_count} images in {enhanced_dir}")
