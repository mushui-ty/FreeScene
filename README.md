# FreeScene

Official implementation of **FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts** (CVPR 2025).

Paper: [CVPR OpenAccess](https://openaccess.thecvf.com/content/CVPR2025/papers/Bai_FreeScene_Mixed_Graph_Diffusion_for_3D_Scene_Synthesis_from_Free_CVPR_2025_paper.pdf) | [Project Page](https://cangmushui.github.io/FreeScene-io/)

## Setup

```bash
cd FreeScene
bash settings/setup.sh
pip install -e .
python3 download_pretrain_vq.py
```

Dataset: we use the preprocessed data from [InstructScene](https://github.com/chenguolin/InstructScene). Run `python3 dataset/download.py` (see [dataset/README.md](dataset/README.md)).

## Training (MGDiT)

```bash
bash scripts/train.sh 0
```

Checkpoints: `out/mgdit/checkpoints/`

## Inference (MGDiT only)

```bash
bash scripts/inference.sh 0 1099 scratch   # text-to-scene
bash scripts/inference.sh 0 1099 graph     # graph-to-scene
```

## Full FreeScene (VLM Graph Designer + MGDiT)

Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`), then:

```bash
export OPENAI_API_KEY=your_key

bash scripts/generate_freescene.sh 0 1099 \
  --text "there is a sofa on the left of the double_bed" \
  --image path/to/reference.jpg
```

Output: `out/mgdit/freescene_outputs/`

## Citation

```bibtex
@InProceedings{Bai_2025_CVPR,
  author    = {Bai, Tongyuan and Bai, Wangyuanfan and Chen, Dong and Wu, Tieru and Li, Manyi and Ma, Rui},
  title     = {FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  pages     = {5893--5903}
}
```

## Acknowledgements

We thank [InstructScene](https://github.com/chenguolin/InstructScene) for providing the preprocessed dataset, fVQ-VAE weights, and codebase that this project builds upon.
