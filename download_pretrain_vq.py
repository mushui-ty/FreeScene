"""Download pretrained fVQ-VAE weights."""
import os

from huggingface_hub import hf_hub_url

HF_REPO = os.environ.get("HF_DATASET_REPO", "chenguolin/InstructScene_dataset")

os.makedirs("out/threedfront_objfeat_vqvae/checkpoints", exist_ok=True)

files = [
    ("threedfront_objfeat_vqvae_epoch_01999.pth", "out/threedfront_objfeat_vqvae/checkpoints/epoch_01999.pth"),
    ("objfeat_bounds.pkl", "out/threedfront_objfeat_vqvae/objfeat_bounds.pkl"),
]

for remote, local in files:
    url = hf_hub_url(HF_REPO, filename=remote, repo_type="dataset")
    os.system(f"wget {url} -O {local}")
    print(f"Downloaded {local}")
