import os
import argparse
import time
import torch
import random
#import pickle

import numpy as np
import torch
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr
from torch.utils.data import DataLoader

from src.utils import *
from src.data import get_encoded_dataset, filter_function
from src.data.utils_text import fill_templates
from src.models import model_from_config, optimizer_from_config, ObjectFeatureVQVAE
from src.models.clip_encoders import CLIPTextEncoder
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def main():
    parser = argparse.ArgumentParser(
        description='FreeScene MG-DiT training')
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/bedroom_mgdit.yaml",
        help="Path to the experiment configuration",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="mgdit",
        help="Experiment tag for output directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="Resume from this checkpoint epoch",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable wandb logging",
    )

    args = parser.parse_args()

    # Set the random seed
    if args.seed is not None and args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"You have chosen to seed([{args.seed}]) the experiment")

    # Set the device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print(f"Run code on device [{device}]\n")

    # Check if `output_dir=out` exists and if it doesn't create it
    if args.debug == False and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create an experiment directory using the `tag`
    if args.tag is None:
        tag = time.strftime("%Y-%m-%d_%H:%M") + "_" + \
            os.path.split(args.config_file)[-1].split()[0]  # config file name
    else:
        tag = args.tag
    # Create an experiment directory using the `tag`
    if args.tag is None:
        tag = time.strftime("%Y-%m-%d_%H:%M") + "_" + \
            os.path.split(args.config_file)[-1].split()[0]  # config file name
    else:
        tag = args.tag

    exp_dir = os.path.join(args.output_dir, tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    # if debug == False:
    os.makedirs(ckpt_dir, exist_ok=True)
    # Save the parameters of this run to a file
    save_experiment_params(args, tag, exp_dir)


    # Parse the config file
    config: Dict[str, Dict[str, Any]] = load_config(args.config_file)
    config["tag"] = args.tag

    # Load the training and validation datasets
    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=None,
        split=config["training"].get("splits", ["train", "val"])
    )

    # # Compute the bounds for this experiment, save them to a file in the
    # # experiment directory and pass them to the validation dataset
    # np.savez(
    #     os.path.join(exp_dir, "bounds.npz"),
    #     translations=train_dataset.bounds["translations"],
    #     sizes=train_dataset.bounds["sizes"],
    #     angles=train_dataset.bounds["angles"]
    # )
    # print(f"Training set has bounds: {train_dataset.bounds}")
    # print(f"Load [{len(train_dataset)}] training scenes with [{train_dataset.n_object_types}] object types\n")

    config["data"]["encoding_type"] += "_eval"  # use the evaluation encoding for the validation dataset

    val_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=os.path.join(exp_dir, "bounds.npz"),
        augmentations=None,
        split=config["validation"].get("splits", ["test"])
    )
    print(f"Load [{len(val_dataset)}] validation scenes with [{val_dataset.n_object_types}] object types\n")
    # Make sure that the `train_dataset` and the `val_dataset` have the same number of object categories
    assert train_dataset.object_types == val_dataset.object_types
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        shuffle=False
    )

    # Load pretrained VQ-VAE weights; TODO: make it configurable
    # print("Load pretrained VQ-VAE\n")
    # with open(f"{args.output_dir}/{args.fvqvae_tag}/objfeat_bounds.pkl", "rb") as f:
    #     kwargs = pickle.load(f)
    # vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **kwargs)
    # ckpt_path = f"{args.output_dir}/{args.fvqvae_tag}/checkpoints/epoch_{args.fvqvae_epoch:05d}.pth"
    # vqvae_model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    # vqvae_model = vqvae_model.to(device)
    # vqvae_model.eval()
    vqvae_model = None
    # load text encoder
    print(f"Load pretrained text encoder [{config['model']['text_encoder']}]\n")
    if "clip" in config["model"]["text_encoder"]:
        text_encoder = CLIPTextEncoder(config["model"]["text_encoder"], device=device)
    else:
        raise ValueError(f"Invalid text encoder name: [{config['model']['text_encoder']}]")

    # Initialize the model and optimizer
    model = model_from_config(
        config,
        train_dataset.n_object_types,
        train_dataset.n_predicate_types,
        text_emb_dim=text_encoder.text_emb_dim
    ).to(device)
    optimizer = optimizer_from_config(
        config["training"]["optimizer"],
        filter(lambda p: p.requires_grad, model.parameters())
    )

    # TODO create EMA for the model
    ema_config = config["training"]["ema"]
    if ema_config["use_ema"]:
        print(f"Use exponential moving average (EMA) for model parameters\n")
        ema_states = EMAModel(
            model.parameters(),
            decay=ema_config["max_decay"],
            min_decay=ema_config["min_decay"],
            update_after_step=ema_config["update_after_step"],
            use_ema_warmup=ema_config["use_warmup"],
            inv_gamma=ema_config["inv_gamma"],
            power=ema_config["power"]
        )
        ema_states.to(device)
    else:
        ema_states: EMAModel = None
    # TODO load the weight from a previous run if specified
    start_epoch = 0
    start_epoch = load_checkpoints(model, ckpt_dir, ema_states, optimizer, args.checkpoint_epoch, device) + 1

    use_wandb = HAS_WANDB and not args.debug
    if use_wandb:
        wandb.init(
            project="FreeScene",
            name=tag,
            config=args.__dict__,
        )
    # Log the stats to a log file
    StatsLogger.instance().add_output_file(open(
        os.path.join(exp_dir, "logs.txt"), "w"
    ))

    # TODO: add some parameters
    epochs = config["training"]["epochs"]
    steps_per_epoch = config["training"]["steps_per_epoch"]
    loss_weights = config["training"]["loss_weights"]
    save_freq = config["training"]["save_frequency"]  # in epochs
    log_freq = config["training"]["log_frequency"]    # in iterations
    val_freq = config["validation"]["frequency"]      # in epochs

    # Start training
    for i in range(start_epoch, epochs):
        model.train()
        for b, batch in zip(range(steps_per_epoch), yield_forever(train_loader)):
            # TODO create tensor and move appropriate tensors to GPU
            # Move necessary variables to GPU
            for k, v in batch.items():
                if not isinstance(v, list):
                    batch[k] = v.to(device)
            # Prepare CLIP text embeddings
            descriptions = batch["descriptions"]
            texts = []
            for desc_idx, desc in enumerate(descriptions):
                text, _, _ = fill_templates(desc,
                    train_dataset.object_types,
                    train_dataset.predicate_types,
                    batch["object_descs"][desc_idx],
                    seed=None
                )
                texts.append(text)
            text_last_hidden_state, text_embeds = text_encoder(texts)
            # zero previous gradient
            optimizer.zero_grad()
            # compute loss
            losses = model.compute_losses(batch, text_last_hidden_state, text_embeds)
            total_loss = torch.zeros(1, device=device)
            for k, v in losses.items():
                if k in loss_weights:
                    total_loss += loss_weights[k] * v
                else:   # when weight in config is not specified
                    total_loss += v
            # backpropagate
            total_loss.backward()
            # update
            optimizer.step()
            # update EMA
            if ema_states is not None:
                ema_states.step(model.parameters())
            # update logger
            '''
                statsLogger is a singleton class, when we call StatsLogger.instance()["name"], it will new an AverageAggregator as the value,
                then we call StatsLogger.instance().update_loss(total_per_batch, batch_size) to update the loss_AverageAggregator,
                or we call StatsLogger.instance()["name"].update(total_per_batch, batch_size) to update the AverageAggregat
                finally we call StatsLogger.instance().print_progress(epoch_index，iter_index) to print loss:(total_per_batch/batch_size) and
                other information name:(total_per_batch/batch_size) and write log to specified file
            '''
            StatsLogger.instance().update_loss(total_loss.item() * batch["objs"].shape[0], batch["objs"].shape[0])
            for k, v in losses.items():
                    StatsLogger.instance()[f"train/{k}"].update(v.item() * batch["objs"].shape[0], batch["objs"].shape[0])
            #TODO: write to wandb (total, other, ema)
            if use_wandb:
                wandb.log({"train/total_loss": total_loss.item()})
                if len(losses) > 1:
                    for k, v in losses.items():
                        wandb.log({f"train/{k}": v.item()})
                if ema_states is not None:
                    wandb.log({"training/ema_decay":ema_states.cur_decay_value})
            # print loss information at certain frequency
            if (i * steps_per_epoch + b) % log_freq == 0:
                StatsLogger.instance().print_progress(i, b)
        # save checkpoint
        if (i+1) % save_freq == 0:
            save_checkpoints(model, optimizer, ckpt_dir, i, ema_states)
        StatsLogger.instance().clear()

        if (i+1) % val_freq == 0:
            print("\n================ Validation ================")
            # Evaluate with the EMA parameters if specified
            if ema_states is not None:
                ema_states.store(model.parameters())
                ema_states.copy_to(model.parameters())
            model.eval()
            with torch.no_grad():
                for val_b, val_batch in enumerate(val_loader):
                    # Move everything to the device
                    for k, v in val_batch.items():
                        if not isinstance(v, list):
                            val_batch[k] = v.to(device)
                    # Prepare CLIP text embeddings
                    val_descriptions = val_batch["descriptions"]
                    val_texts = []
                    for val_desc_idx, val_desc in enumerate(val_descriptions):
                        val_text, _, _ = fill_templates(val_desc,
                            train_dataset.object_types,
                            train_dataset.predicate_types,
                            val_batch["object_descs"][val_desc_idx],
                            seed=None
                        )
                        val_texts.append(val_text)
                    val_text_last_hidden_state, val_text_embeds = text_encoder(val_texts)
                    # compute loss
                    val_losses = model.compute_losses(val_batch, val_text_last_hidden_state, val_text_embeds)
                    val_total_loss = torch.zeros(1, device=device)
                    for k, v in val_losses.items():
                        if k in loss_weights:
                            val_total_loss += loss_weights[k] * v
                        else:  # weight is not specified
                            val_total_loss += v
                    StatsLogger.instance().update_loss(val_total_loss.item() *
                                                    val_batch["objs"].shape[0], val_batch["objs"].shape[0])
                    for k, v in val_losses.items():
                        StatsLogger.instance()[f"val/{k}"].update(v.item() * val_batch["objs"].shape[0], val_batch["objs"].shape[0])
                    StatsLogger.instance().print_progress(i, val_b)
            #TODO write to wandb
            if use_wandb:
                wandb.log({"validation/total_loss": val_total_loss.item()})
                if len(val_losses) > 1:
                    for k, v in val_losses.items():
                        wandb.log({f"validation/{k}": v.item()})
            StatsLogger.instance().clear()
             # Restore the model parameters from EMA states
            if ema_states is not None:
                ema_states.restore(model.parameters())
            print("================ Validation ================\n")
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()