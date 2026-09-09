"""Download and extract the preprocessed bedroom dataset."""
import os

from huggingface_hub import hf_hub_url

HF_REPO = os.environ.get("HF_DATASET_REPO", "chenguolin/InstructScene_dataset")
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(DATASET_DIR)

scene_zip = hf_hub_url(HF_REPO, filename="InstructScene.zip", repo_type="dataset")
os.system(f"wget {scene_zip} && unzip -o InstructScene.zip && mv -f InstructScene processed")

front_zip = hf_hub_url(HF_REPO, filename="3D-FRONT.zip", repo_type="dataset")
os.system(f"wget {front_zip} && unzip -o 3D-FRONT.zip")

print("Dataset ready under dataset/processed/ and dataset/3D-FRONT/")
