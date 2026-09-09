from typing import *
from torch import Tensor
from torch.optim import Optimizer

from functools import partial

from torch import optim

from .objfeat_vqvae import ObjectFeatureVQVAE
from .mgdit import MGDiT


def model_from_config(
    config: Dict[str, Any],
    num_objs=-1, num_preds=-1,
    text_emb_dim=512,
    **kwargs
):
    if "mgdit" in config["model"]["name"]:
        return MGDiT(
            num_objs, num_preds,
            use_objfeat="objfeat" in config["model"]["name"],
        )

    elif "objfeatvqvae" in config["model"]["name"]:
        return ObjectFeatureVQVAE(
            config["model"]["objfeat_type"],
            config["model"]["vq_type"],
            **kwargs,
        )

    else:
        raise NotImplementedError(f"Unknown model name: {config['model']['name']}")


def optimizer_from_config(config: Dict[str, Any], params: Iterable[Tensor]) -> Optimizer:
    name = config["name"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.)

    kwargs = dict(
        lr=lr,
        weight_decay=weight_decay
    )

    momentum = config.get("momentum", 0.)
    nesterov = config.get("nesterov", False)
    betas = config.get("betas", (0.9, 0.999))

    optimizer = {
        "sgd": partial(optim.SGD, momentum=momentum, nesterov=nesterov, **kwargs),
        "adam": partial(optim.Adam, betas=betas, **kwargs),
        "adamw": partial(optim.AdamW, betas=betas, **kwargs),
        "radam": partial(optim.RAdam, betas=betas, **kwargs),
    }[name]

    return optimizer(params)
