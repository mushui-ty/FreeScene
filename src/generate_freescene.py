"""Full FreeScene pipeline: VLM Graph Designer + MGDiT."""
import argparse
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from PIL import Image

from src.data.threed_future_dataset import ThreedFutureDataset
from src.data.utils_text import reverse_rel
from src.models import model_from_config, ObjectFeatureVQVAE
from src.models.clip_encoders import CLIPTextEncoder
from src.utils.util import load_config, load_checkpoints
from src.utils.visualize import (
    blender_render_scene, draw_scene_graph, export_scene,
    floor_plan_from_scene, get_textured_objects,
)
from src.vlm_graph_designer import design_graph


def main():
    parser = argparse.ArgumentParser(description="FreeScene: VLM Graph Designer + MGDiT")
    parser.add_argument("--config_file", type=str, default="configs/bedroom_mgdit.yaml")
    parser.add_argument("--tag", type=str, default="mgdit")
    parser.add_argument("--checkpoint_epoch", type=int, required=True)
    parser.add_argument("--text", type=str, default="", help="User text prompt")
    parser.add_argument("--image", type=str, default="", help="Reference image path")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--fvqvae_tag", type=str, default="threedfront_objfeat_vqvae")
    parser.add_argument("--fvqvae_epoch", type=int, default=1999)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_nodes", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--vlm_model", type=str, default="gpt-4o")
    parser.add_argument("--draw_scene_graph", action="store_true")
    args = parser.parse_args()

    if not args.text and not args.image:
        parser.error("Provide at least --text or --image.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config_file)
    config["tag"] = args.tag
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    from src.data import get_dataset_raw_and_encoded, filter_function
    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(config["data"], split=["test"]),
        split=["test"],
    )
    object_types = dataset.object_types
    predicate_types = dataset.predicate_types
    n_object_types = dataset.n_object_types
    n_predicate_types = dataset.n_predicate_types

    print("Running VLM Graph Designer...")
    object_list, triplet_list, description = design_graph(
        object_types, predicate_types,
        user_text=args.text,
        image_path=args.image or None,
        model=args.vlm_model,
    )
    print(f"Description: {description}")
    print(f"Objects: {object_list}")
    print(f"Triplets: {triplet_list}")

    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        config["data"]["path_to_pickled_3d_future_models"]
    )

    with open(f"{args.output_dir}/{args.fvqvae_tag}/objfeat_bounds.pkl", "rb") as f:
        vq_kwargs = pickle.load(f)
    vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **vq_kwargs)
    vqvae_model.load_state_dict(torch.load(
        f"{args.output_dir}/{args.fvqvae_tag}/checkpoints/epoch_{args.fvqvae_epoch:05d}.pth",
        map_location="cpu",
    )["model"])
    vqvae_model = vqvae_model.to(device).eval()

    text_encoder = CLIPTextEncoder(config["model"]["text_encoder"], device=device)
    model = model_from_config(
        config, raw_dataset.n_object_types, raw_dataset.n_predicate_types,
        text_emb_dim=text_encoder.text_emb_dim,
    ).to(device)
    ema_states = EMAModel(model.parameters()) if config["training"]["ema"]["use_ema"] else None
    load_epoch = load_checkpoints(model, ckpt_dir, ema_states, epoch=args.checkpoint_epoch, device=device)
    if ema_states is not None:
        ema_states.copy_to(model.parameters())
    model.eval()

    N = args.num_nodes
    x = torch.tensor(object_list, dtype=torch.long).unsqueeze(0).to(device)
    x = F.pad(x, (0, N - x.shape[1], 0, 0), value=n_object_types)
    e = torch.full((1, N, N), n_predicate_types, dtype=torch.long, device=device)
    for s, p, o in triplet_list:
        rev_p = predicate_types.index(reverse_rel(predicate_types[p]))
        e[0, s, o] = p
        e[0, o, s] = rev_p

    text_last_hidden_state, text_embeds = text_encoder([description])
    with torch.no_grad():
        boxes_pred, x_out, e_out, objfeat_vq_indices = model.generate_samples(
            1, N, text_last_hidden_state, text_embeds,
            x=x, e=e, not_complete=True,
            full_step=True, num_diffusion_steps=1000, cfg_scale=args.cfg_scale,
        )

    objfeat_vq_indices_rand = torch.randint_like(objfeat_vq_indices, 0, 64)
    objfeat_vq_indices[objfeat_vq_indices == 64] = objfeat_vq_indices_rand[objfeat_vq_indices == 64]
    objfeats = vqvae_model.reconstruct_from_indices(
        objfeat_vq_indices.reshape(-1, objfeat_vq_indices.shape[-1])
    ).reshape(1, N, -1).cpu().numpy()

    bbox_params = {
        "class_labels": F.one_hot(x_out.cpu(), num_classes=n_object_types + 1).float(),
        "translations": boxes_pred[..., :3].cpu(),
        "sizes": boxes_pred[..., 3:6].cpu(),
        "angles": boxes_pred[..., 6:].cpu(),
    }
    boxes = dataset.post_process(bbox_params)
    bbox_params_t = torch.cat([
        boxes["class_labels"], boxes["translations"],
        boxes["sizes"], boxes["angles"],
    ], dim=-1).numpy()[0]

    trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, _ = get_textured_objects(
        bbox_params_t, objects_dataset, np.array(object_types),
        objfeats[0], "openshape_vitg14",
    )

    if trimesh_meshes:
        all_vertices = np.concatenate([m.vertices for m in trimesh_meshes], axis=0)
        x_max, x_min = all_vertices[:, 0].max(), all_vertices[:, 0].min()
        z_max, z_min = all_vertices[:, 2].max(), all_vertices[:, 2].min()
    else:
        x_max = x_min = z_max = z_min = 1.0

    tr_floor, _ = floor_plan_from_scene(
        raw_dataset[0], config["data"]["path_to_floor_plan_textures"],
        without_room_mask=True, rectangle_floor=True,
        room_size=[x_min, z_min, x_max, z_max],
    )
    trimesh_meshes.append(tr_floor)

    save_dir = os.path.join(exp_dir, "freescene_outputs", f"epoch_{load_epoch:05d}_seed{args.seed}")
    export_dir = os.path.join(save_dir, str(int(time.time())))
    tmp_dir = os.path.join(export_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    export_scene(tmp_dir, trimesh_meshes, bbox_meshes)
    blender_render_scene(tmp_dir, export_dir, top_down_view=True,
                         resolution_x=args.resolution, resolution_y=args.resolution)

    with open(os.path.join(export_dir, "description.txt"), "w") as f:
        f.write(description)

    if args.draw_scene_graph:
        edges = e_out.cpu().numpy()[0]
        vis_objs, vis_map, cnt = [], {}, 0
        for idx, obj_id in enumerate(x_out[0].cpu().tolist()):
            if obj_id != n_object_types:
                vis_objs.append(obj_id)
                vis_map[idx] = cnt
                cnt += 1
        vis_triples = []
        for i in range(N):
            for j in range(i + 1, N):
                if edges[i, j] != n_predicate_types and i in vis_map and j in vis_map:
                    vis_triples.append([vis_map[i], edges[i, j], vis_map[j]])
        image = draw_scene_graph(
            torch.tensor(vis_objs), torch.tensor(vis_triples),
            object_types=object_types, predicate_types=predicate_types,
        )
        Image.fromarray(image).save(os.path.join(export_dir, "scene_graph.png"))

    print(f"Saved to {export_dir}")


if __name__ == "__main__":
    main()
