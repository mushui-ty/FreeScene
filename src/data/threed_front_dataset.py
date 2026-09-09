from .utils_text import reverse_rel
from .threed_front_dataset_base import *


class MGDiTDataset(DatasetDecoratorBase):
    def __init__(self, dataset, objfeat_type=None):
        super().__init__(dataset)
        self.objfeat_type = objfeat_type

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        sample_params_new = {}
        for k, v in sample_params.items():
            if k == "class_labels":
                class_labels = np.copy(v)
                class_ids = np.argmax(class_labels, axis=-1).astype(np.int64)
                sample_params_new["objs"] = class_ids

        if "descriptions" in sample_params:
            sample_params_new["descriptions"] = sample_params["descriptions"]

        with open(sample_params["models_info_path"], "rb") as f:
            models_info = pickle.load(f)
        objfeat_vq_indices = [
            np.array(model_info["objfeat_vq_indices"])
            for model_info in models_info
        ]
        object_descs = [
            model_info["chatgpt_caption"]
            for model_info in models_info
        ]
        if "permutation" in sample_params:
            objfeat_vq_indices = [objfeat_vq_indices[i] for i in sample_params["permutation"]]
            object_descs = [object_descs[i] for i in sample_params["permutation"]]

        sample_params_new["objfeat_vq_indices"] = np.vstack(objfeat_vq_indices)
        sample_params.update(sample_params_new)
        sample_params["length"] = sample_params["class_labels"].shape[0]
        sample_params["object_descs"] = object_descs
        return sample_params

    def collate_fn(self, samples):
        sample_params_pad = {
            "scene_uids": [],
            "lengths": [],
            "boxes": [],
            "objs": [],
            "edges": [],
            "obj_masks": [],
            "descriptions": [],
            "room_masks": [],
            "objfeat_vq_indices": [],
            "object_descs": [],
        }

        max_length = max(sample["length"] for sample in samples)
        if self.objfeat_type is not None:
            sample_params_pad.update({f"{self.objfeat_type}_features": []})

        for sample_params in samples:
            scene_uid = str(sample_params["scene_uid"])
            length = sample_params["length"]
            objs = sample_params["objs"]
            triples = sample_params["relations"]
            boxes = np.concatenate([
                sample_params["translations"],
                sample_params["sizes"],
                sample_params["angles"]
            ], axis=-1)
            room_mask = sample_params["room_layout"]

            sample_params_pad["scene_uids"].append(scene_uid)
            sample_params_pad["room_masks"].append(room_mask)
            sample_params_pad["lengths"].append(length)
            sample_params_pad["objs"].append(np.pad(
                objs, (0, max_length - objs.shape[0]),
                mode="constant", constant_values=self.n_object_types
            ))
            sample_params_pad["boxes"].append(np.pad(
                boxes, ((0, max_length - boxes.shape[0]), (0, 0)),
                mode="constant", constant_values=0.
            ))
            if "descriptions" in sample_params:
                sample_params_pad["descriptions"].append(sample_params["descriptions"])
            if "object_descs" in sample_params:
                sample_params_pad["object_descs"].append(sample_params["object_descs"])
            if self.objfeat_type is not None:
                objfeats = sample_params[f"{self.objfeat_type}_features"]
                key = f"{self.objfeat_type}_features"
                padded = np.pad(
                    objfeats, ((0, max_length - objfeats.shape[0]), (0, 0)),
                    mode="constant", constant_values=0.
                )
                if key in sample_params_pad:
                    sample_params_pad[key].append(padded)
                else:
                    sample_params_pad[key] = [padded]

            edges = self.n_predicate_types * np.ones((max_length, max_length), dtype=np.int64)
            for s, p, o in triples:
                rev_p = self.predicate_types.index(reverse_rel(self.predicate_types[p]))
                edges[s, o] = p
                edges[o, s] = rev_p
            sample_params_pad["edges"].append(edges)

            objfeat_vq_indices = sample_params["objfeat_vq_indices"]
            objfeat_vq_indices_pad = np.random.randint(
                0, 64, size=(max_length, objfeat_vq_indices.shape[1])
            )
            objfeat_vq_indices_pad[:objfeat_vq_indices.shape[0]] = objfeat_vq_indices
            sample_params_pad["objfeat_vq_indices"].append(objfeat_vq_indices_pad)

            obj_mask = np.zeros(max_length, dtype=np.int64)
            obj_mask[:sample_params["length"]] = 1
            sample_params_pad["obj_masks"].append(obj_mask)

        for k, v in sample_params_pad.items():
            if k in ["scene_uids", "descriptions", "object_descs"]:
                sample_params_pad[k] = v
            elif k in ["boxes", "room_masks"]:
                sample_params_pad[k] = torch.from_numpy(np.stack(v, axis=0)).float()
            else:
                sample_params_pad[k] = torch.from_numpy(np.stack(v, axis=0)).long()

        return sample_params_pad

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


def dataset_encoding_factory(
    name,
    dataset,
    augmentations=None,
    box_ordering=None
) -> DatasetDecoratorBase:
    if "cached" in name:
        dataset_collection = OrderedDataset(
            CachedDatasetCollection(dataset),
            ["class_labels", "translations", "sizes", "angles"],
            box_ordering=box_ordering
        )
    else:
        box_ordered_dataset = BoxOrderedDataset(dataset, box_ordering)
        class_labels = ClassLabelsEncoder(box_ordered_dataset)
        translations = TranslationEncoder(box_ordered_dataset)
        sizes = SizeEncoder(box_ordered_dataset)
        angles = AngleEncoder(box_ordered_dataset)
        dataset_collection = DatasetCollection(
            class_labels, translations, sizes, angles
        )

    if name == "basic":
        return DatasetCollection(class_labels, translations, sizes, angles)

    if isinstance(augmentations, list):
        for aug_type in augmentations:
            if aug_type == "rotation":
                dataset_collection = Rotation(dataset_collection)
            elif aug_type == "fixed_rotation":
                dataset_collection = Rotation(dataset_collection, fixed=True)
            elif aug_type == "jitter":
                dataset_collection = Jitter(dataset_collection)

    if "graph" in name or "desc" in name:
        dataset_collection = Add_SceneGraph(dataset_collection)

    if "desc" in name:
        seed = int(name.split("_")[-1]) if "seed" in name else None
        dataset_collection = Add_Description(dataset_collection, seed=seed)

    objfeat_type = None
    if "objfeat" in name:
        if "openshape_vitg14" in name:
            objfeat_type = "openshape_vitg14"
        else:
            raise ValueError(f"Unknown object feature type in [{name}]")
        dataset_collection = Add_Objfeature(dataset_collection, objfeat_type)

    if "sincos_angle" in name:
        dataset_collection = Scale_CosinAngle(dataset_collection)
    else:
        dataset_collection = Scale(dataset_collection)

    permute_keys = [
        "class_labels", "translations", "sizes", "angles",
        "relations", "descriptions", "openshape_vitg14_features"
    ]

    if "mgdit" not in name:
        raise NotImplementedError(f"Unknown encoding type: {name}")

    assert "graph" in name or "desc" in name, "Scene graphs are required."
    if "no_prm" in name or "eval" in name:
        return MGDiTDataset(dataset_collection, objfeat_type)

    dataset_collection = Permutation(dataset_collection, permute_keys)
    return MGDiTDataset(dataset_collection, objfeat_type)
