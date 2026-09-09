# Dataset

FreeScene adopts the preprocessed dataset from [InstructScene](https://github.com/chenguolin/InstructScene), which provides 3D-FRONT bedroom scenes with scene graphs and text descriptions. Parts of our data pipeline also reference the InstructScene codebase.

Download and extract:

```bash
python3 dataset/download.py
```

Expected layout:

```
dataset/
├── 3D-FRONT/
│   ├── 3D-FRONT/
│   └── 3D-FUTURE-model/
└── processed/
    ├── threed_front_bedroom/
    └── threed_future_model_bedroom.pkl
```

Raw 3D-FRONT data is available from the [official website](https://tianchi.aliyun.com/dataset/65347).
