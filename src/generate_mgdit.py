import os
import argparse
import random
import pickle
import time
from tqdm import tqdm
import numpy as np
import torch
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers.training_utils import EMAModel

from src.utils.util import *
from src.utils.visualize import *
from src.data import filter_function, get_encoded_dataset, get_dataset_raw_and_encoded
from src.data.threed_future_dataset import ThreedFutureDataset
from src.data.threed_front_dataset_base import trs_to_corners
from src.data.utils_text import compute_loc_rel, reverse_rel
from src.models import model_from_config, ObjectFeatureVQVAE
from src.models.clip_encoders import CLIPTextEncoder
from src.data.utils_text import fill_templates
def main():
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

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
        help="Experiment tag",
    )
    parser.add_argument(
        "--fvqvae_tag",
        type=str,
        default="threedfront_objfeat_vqvae",
        help="fVQ-VAE experiment tag",
    )
    parser.add_argument(
        "--fvqvae_epoch",
        type=int,
        default=1999,
        help="Epoch of the pretrained fVQ-VAE"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="The epoch to load the checkpoint from"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="The number of processed spawned by the batch provider (default=0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG (default=0)"
    )
    parser.add_argument(
        "--n_scenes",
        type=int,
        default=5,
        help="The number of scenes to be generated"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the generated scenes"
    )
    parser.add_argument(
        "--eight_views",
        action="store_true",
        help="Render 8 views of the scene"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution of the rendered image"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more information"
    )
    parser.add_argument(
        "--condition_type",
        type=str,
        default="text",
        choices=["text", "none"],
        help="The type of the CLIP embedded conditioning (choices: `text`, `none`)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use for training"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.,
        help="scale for the classifier-free guidance"
    )
    parser.add_argument(
        "--draw_scene_graph",
        action="store_true",
        help="Draw used scene graphs"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="scratch",
        choices=["scratch", "graph", "allcond"],
        help="Generation mode: scratch (text-to-scene), graph (graph-to-scene), allcond",
    )
    parser.add_argument(
        "--scene_uid",
        type=str,
        default="",
    )
    parser.add_argument(
        "--floor",
        type=str,
        default="",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
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

    # Check if `ckpt_dir` exists
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    assert os.path.exists(
        ckpt_dir), f"Checkpoint directory {ckpt_dir} does not exist."

    # load config
    config = load_config(args.config_file)
    config["tag"] = args.tag

    # load 3D-FUTURE dataset
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        config["data"]["path_to_pickled_3d_future_models"]
    )
    print(f"Load [{len(objects_dataset)}] 3D-FUTURE models")
    # add eval to data_enocoding_type
    if "eval" not in config["data"]["encoding_type"]:
        config["data"]["encoding_type"] += "_eval"
        
    # Load training dataset to load model
    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=os.path.join(exp_dir, "bounds.npz"),
        augmentations=None,
        split=config["validation"].get("splits", ["test"])
    )
    print(
        f"Load [{len(dataset)}] validation scenes with [{dataset.n_object_types}] object types\n")
    # use all scenes if n_scenes == 0 , default is 0 i.e. batch size in config_file
    if args.n_scenes == 0:  # use all scenes
        B = config["validation"]["batch_size"]
    else:
        B = args.n_scenes
    dataloader = DataLoader(
        dataset,
        batch_size=B,
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
        shuffle=False
    )
    # Load pretrained VQ-VAE weights; TODO: make it configurable
    print("Load pretrained VQ-VAE\n")
    with open(f"{args.output_dir}/{args.fvqvae_tag}/objfeat_bounds.pkl", "rb") as f:
        kwargs = pickle.load(f)
    vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **kwargs)
    ckpt_path = f"{args.output_dir}/{args.fvqvae_tag}/checkpoints/epoch_{args.fvqvae_epoch:05d}.pth"
    vqvae_model.load_state_dict(torch.load(
        ckpt_path, map_location="cpu")["model"])
    vqvae_model = vqvae_model.to(device)
    vqvae_model.eval()

    # load text encoder
    print(
        f"Load pretrained text encoder [{config['model']['text_encoder']}]\n")
    if "clip" in config["model"]["text_encoder"]:
        text_encoder = CLIPTextEncoder(
            config["model"]["text_encoder"], device=device)
    else:
        raise ValueError(
            f"Invalid text encoder name: [{config['model']['text_encoder']}]")

    # Initialize the model and optimizer
    model = model_from_config(
        config,
        raw_dataset.n_object_types,
        raw_dataset.n_predicate_types,
        text_emb_dim=text_encoder.text_emb_dim
    ).to(device)
    # Create EMA for the model
    ema_config = config["training"]["ema"]
    if ema_config["use_ema"]:
        ema_states = EMAModel(model.parameters())
        ema_states.to(device)
    else:
        ema_states: EMAModel = None
    # Load the weights from a checkpoint
    load_epoch = load_checkpoints(
        model, ckpt_dir, ema_states, epoch=args.checkpoint_epoch, device=device)
    # Evaluate with the EMA parameters if specified
    if ema_states is not None:
        print(f"Copy EMA parameters to the model\n")
        ema_states.copy_to(model.parameters())
    model.eval()

    # Check if `save_dir` for generated scenes exists and if it doesn't create it
    save_dir = os.path.join(
        exp_dir, "generated_scenes",
        f"epoch_{load_epoch:05d}_{args.mode}_seed{args.seed}",
    )
    os.makedirs(save_dir, exist_ok=True)

    # Generate the boxes from scene graphs
    classes = np.array(dataset.object_types)
    print("Sample boxes from scene graphs with the [diffusion model]")
    for batch_idx, batch in tqdm(
        enumerate(dataloader),
        desc=f"Process each batch",
        total=len(dataloader), ncols=125,
        disable=args.verbose
    ):
        # Move everything to the device
        for k, v in batch.items():
            if not isinstance(v, list):
                batch[k] = v.to(device)

        descriptions = batch["descriptions"]
        texts = []
        for desc_idx, desc in enumerate(descriptions):
            text, _, _ = fill_templates(
                desc,
                raw_dataset.object_types,
                raw_dataset.predicate_types,
                batch["object_descs"][desc_idx],
                seed=None,
            )
            texts.append(text)
        text_last_hidden_state, text_embeds = text_encoder(texts)
        if args.condition_type == "none":
            text_last_hidden_state = torch.zeros_like(text_last_hidden_state)
            text_embeds = torch.zeros_like(text_embeds)
        # # Unpack the batch parameters
        # objs = batch["objs"]
        # edges = batch["edges"]
        # objfeat_vq_indices = batch["objfeat_vq_indices"]
        # obj_masks = batch["obj_masks"]

        # Unpack the batch parameters
        x = batch["objs"]  # (b, n)
        e = batch["edges"]  # (b, n, n)
        o = batch["objfeat_vq_indices"]
        o = o.reshape(x.shape[0], x.shape[1], -1)  # (b, n, 4)
        mask = batch["obj_masks"]  # (b, n)
        boxes = batch["boxes"]  # (b, n, 8)
        B, N = x.shape

        sample_kwargs = dict(
            full_step=True,
            num_diffusion_steps=1000,
            cfg_scale=args.cfg_scale,
        )
        with torch.no_grad():
            if args.mode == "scratch":
                boxes_pred, x_out, e_out, objfeat_vq_indices = model.generate_samples(
                    B, N, text_last_hidden_state, text_embeds, **sample_kwargs,
                )
            elif args.mode == "graph":
                boxes_pred, x_out, e_out, objfeat_vq_indices = model.generate_samples(
                    B, N, text_last_hidden_state, text_embeds,
                    x=x, e=e, not_complete=True, **sample_kwargs,
                )
            else:
                boxes_pred, x_out, e_out, objfeat_vq_indices = model.generate_samples(
                    B, N, text_last_hidden_state, text_embeds,
                    x=x, e=e, o=o, **sample_kwargs,
                )
        if objfeat_vq_indices is not None:
            # Replace the empty token with a random token within the vocabulary
            objfeat_vq_indices_rand = torch.randint_like(
                objfeat_vq_indices, 0, 64)
            objfeat_vq_indices[objfeat_vq_indices ==
                               64] = objfeat_vq_indices_rand[objfeat_vq_indices == 64]
        # Decode objfeat indices to objfeat embeddings
        B, N = objfeat_vq_indices.shape[:2]
        objfeats = vqvae_model.reconstruct_from_indices(
            objfeat_vq_indices.reshape(B*N, -1)
        ).reshape(B, N, -1)
        objfeats = (objfeats * mask[..., None].float()).cpu().numpy()

        objs = x_out.cpu()
        true_objs = batch["objs"].cpu()
        obj_masks = mask.cpu()
        boxes_pred = boxes_pred.cpu()  # (B, N, 8)

        # parse scenes parameters
        bbox_params = {
            # +1 for empty node (not really used)
            "class_labels": F.one_hot(objs, num_classes=raw_dataset.n_object_types+1).float(),
            "translations": boxes_pred[..., :3],
            "sizes": boxes_pred[..., 3:6],
            "angles": boxes_pred[..., 6:]
        }
        '''
        angles: 
        # theta = arctan sin/cos y/x
        print(f"Postprocess [{k}] by [arctan2]")
        sample_params[k] = np.arctan2(v[..., 1:2], v[..., 0:1])
        translate, size
        # bounds record the maximum and minimum of the variables
        print(f"Postprocess [{k}] by bounds {bounds[k][0]}~{bounds[k][1]}")
        sample_params[k] = Scale.descale(
            v, bounds[k][0], bounds[k][1]
        )
        '''
        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).numpy()
        assert bbox_params_t.shape[-1] == 7 + raw_dataset.n_object_types+1

        # Evaluate (and visualize) each scene in the batch
        progress_bar = tqdm(
            total=len(bbox_params_t),
            desc="Visualize each scene",
            ncols=125,
            disable=args.verbose
        )
        for i in range(len(bbox_params_t)):
            if args.scene_uid != "" and args.scene_uid != batch['scene_uids'][i]:
                continue
            # Get the textured objects by retrieving the 3D models
            trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, _ = get_textured_objects(
                bbox_params_t[i],
                objects_dataset, classes,
                objfeats[i] if model.use_objfeat else None,
                "openshape_vitg14",  # TODO: make it configurable
                verbose=args.verbose
            )

            # Evaluation for scene graph condition
            # 1. Get the decoded scene graphs from generated boxes
            obj_class_ids = [
                dataset.object_types.index(c) if c is not None
                else dataset.n_object_types
                for c in obj_classes
            ]
            relations = []
            cls_dim = dataset.n_object_types+1
            for idx in range(len(obj_class_ids)):
                if obj_class_ids[idx] == dataset.n_object_types:  # empty object
                    continue
                c1_id = obj_class_ids[idx]
                t1 = bbox_params_t[i, idx, cls_dim:cls_dim+3]
                r1 = bbox_params_t[i, idx, cls_dim+6]
                # bbox_params_t[i, idx, cls_dim+3:cls_dim+6]; use the retrieved size
                s1 = obj_sizes[idx]
                corners1 = trs_to_corners(t1, r1, s1)
                name1 = dataset.object_types[c1_id]
                for other_idx in range(idx+1, len(obj_class_ids)):
                    if obj_class_ids[other_idx] == dataset.n_object_types:  # empty object
                        continue
                    c2_id = obj_class_ids[other_idx]
                    t2 = bbox_params_t[i, other_idx, cls_dim:cls_dim+3]
                    r2 = bbox_params_t[i, other_idx, cls_dim+6]
                    # bbox_params_t[i, other_idx, cls_dim+3:cls_dim+6]; use the retrieved size
                    s2 = obj_sizes[other_idx]
                    corners2 = trs_to_corners(t2, r2, s2)
                    name2 = dataset.object_types[c2_id]

                    loc_rel_str = compute_loc_rel(
                        corners1, corners2, name1, name2)
                    if loc_rel_str is not None:
                        # print(classes[obj_class_ids[idx]], loc_rel_str, classes[obj_class_ids[other_idx]])
                        relation_id = dataset.predicate_types.index(
                            loc_rel_str)
                        relations.append([idx, relation_id, other_idx])
                        # Add the reverse relation
                        # print(classes[obj_class_ids[other_idx]], reverse_rel(loc_rel_str), classes[obj_class_ids[idx]])
                        rev_relation_id = dataset.predicate_types.index(
                            reverse_rel(loc_rel_str))
                        relations.append([other_idx, rev_relation_id, idx])

            edges = dataset.n_predicate_types * np.ones(
                (len(obj_class_ids), len(obj_class_ids)), dtype=np.int64
            )
            for s, p, o in relations:
                edges[s, o] = p
            true_edges = batch["edges"][i].cpu().numpy()

            progress_bar.update(1)

            # Whether to visualize the scene by blender rendering
            if not args.visualize or args.debug:
                continue

            # To get the manually created floor plan, which includes vertices of all meshes in the scene
            if len(trimesh_meshes) != 0:
                all_vertices = np.concatenate([
                    tr_mesh.vertices for tr_mesh in trimesh_meshes
                ], axis=0)
                x_max, x_min = all_vertices[:, 0].max(), all_vertices[:, 0].min()
                z_max, z_min = all_vertices[:, 2].max(), all_vertices[:, 2].min()
            else:
                x_max, x_min = 1.0, -1.0
                z_max, z_min = 1.0, -1.0
            if args.floor != "":
                floor_texture_path = args.floor
            else:
                floor_texture_path = config["data"]["path_to_floor_plan_textures"]
            tr_floor, _ = floor_plan_from_scene(
                # `raw_dataset[0]` is not really used
                raw_dataset[0], floor_texture_path,
                without_room_mask=True,
                rectangle_floor=True, room_size=[x_min, z_min, x_max, z_max]
            )
            trimesh_meshes.append(tr_floor)

            # Create a trimesh scene and export it to a temporary directory
            ii = batch_idx * B + i
            export_dir = os.path.join(
                save_dir, f"{ii:04d}@{batch['scene_uids'][i]}_cfg{args.cfg_scale:.1f}")
            tmp_dir = os.path.join(export_dir, "tmp")
            os.makedirs(export_dir, exist_ok=True)
            os.makedirs(tmp_dir, exist_ok=True)
            export_scene(tmp_dir, trimesh_meshes, bbox_meshes)

            # Render the exported scene by calling blender
            blender_render_scene(
                tmp_dir,
                export_dir,
                top_down_view=(not args.eight_views),
                resolution_x=args.resolution,
                resolution_y=args.resolution
            )
            # Delete the temporary directory
            #os.system(f"rm -rf {tmp_dir}")
            if args.verbose:
                print(f"Save the scene to {export_dir}\n")

            if args.condition_type == "text":
                    # Save conditioned text
                    if texts is not None:
                        with open(os.path.join(export_dir, "description.txt"), "w") as f:
                            f.write(texts[i])
            # Visualize the used scene graph
            if args.draw_scene_graph:
                obj_ids = objs[i]  # (N,)
                vis_objs, vis_obj_mapping, _vis_count = [], {}, 0
                for idx, obj_id in enumerate(obj_ids):
                    if obj_id != dataset.n_object_types:
                        vis_objs.append(obj_id)
                        vis_obj_mapping[idx] = _vis_count
                        _vis_count += 1
                # edge_ids = edges[i]  # (N, N)
                edge_ids = edges # (N, N)
                length = batch['lengths'][i].item()
                vis_triples = []
                for idx1 in range(length):
                    for idx2 in range(idx1+1, length):
                        if edge_ids[idx1, idx2] != dataset.n_predicate_types:
                            vis_triples.append([
                                vis_obj_mapping[idx1], edge_ids[idx1,
                                                                idx2], vis_obj_mapping[idx2]
                            ])
                vis_objs, vis_triples = torch.tensor(
                    vis_objs), torch.tensor(vis_triples)
                image = draw_scene_graph(
                    vis_objs, vis_triples,
                    object_types=dataset.object_types,
                    predicate_types=dataset.predicate_types
                )
                Image.fromarray(image).save(
                    os.path.join(export_dir, f"output_scene_graph.png")
                )
            if args.mode == "graph":
                obj_ids = true_objs[i]  # (N,)
                vis_objs, vis_obj_mapping, _vis_count = [], {}, 0
                for idx, obj_id in enumerate(obj_ids):
                    if obj_id != dataset.n_object_types:
                        vis_objs.append(obj_id)
                        vis_obj_mapping[idx] = _vis_count
                        _vis_count += 1
                # edge_ids = edges[i]  # (N, N)
                edge_ids = true_edges # (N, N)
                length = batch['lengths'][i].item()
                vis_triples = []
                for idx1 in range(length):
                    for idx2 in range(idx1+1, length):
                        if edge_ids[idx1, idx2] != dataset.n_predicate_types:
                            vis_triples.append([
                                vis_obj_mapping[idx1], edge_ids[idx1,
                                                                idx2], vis_obj_mapping[idx2]
                            ])
                vis_objs, vis_triples = torch.tensor(
                    vis_objs), torch.tensor(vis_triples)
                image = draw_scene_graph(
                    vis_objs, vis_triples,
                    object_types=dataset.object_types,
                    predicate_types=dataset.predicate_types
                )
                Image.fromarray(image).save(
                    os.path.join(export_dir, f"input_scene_graph.png")
                )
        # Not generate all scenes
        if args.n_scenes != 0:  # only generate the first `n_scenes` scenes
            break

   


if __name__ == "__main__":
    main()
