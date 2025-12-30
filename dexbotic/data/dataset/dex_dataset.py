import copy
import json
import math
import os
import random
import warnings
from typing import Callable, Dict, List

import megfile
import torch
from torch.utils.data import Dataset

from dexbotic.data.data_source.register import CONVERSATION_DATA
from dexbotic.data.dataset.augmentations import PixelAug
from dexbotic.data.dataset.depth_preprocess import PreprocessDepth
from dexbotic.data.dataset.rgb_preprocess import PreprocessRGB
from dexbotic.data.dataset.transform.common import ExtracKeys, ToTensor
from dexbotic.data.dataset.transform.language import ToConversation_Old


class DexDataset(Dataset):
    default_keys = ["input_ids", "labels", "action", "image"]

    def __init__(
        self,
        data_args,
        tokenization_func: Callable[[List[Dict], bool], dict[str, torch.Tensor]],
        action_process_func=None,
        image_process_func=None,
        depth_process_func=None,
        **kwargs,
    ):
        """Args:
        data_args: argparse.Namespace, the arguments for the dataset
        tokenization_func: callable, the function to tokenize the prompt
        action_process_func: callable, the function to process the action
        """
        self._build_dataset_from_name(data_args.dataset_name)

        self.num_images = getattr(data_args, "num_images", 1)
        self.data_keys = getattr(data_args, "data_keys", self.default_keys)
        self.images_keys = getattr(data_args, "images_keys", None)
        self.depths_keys = getattr(data_args, "depths_keys", None)
        self.load_depth = getattr(data_args, "load_depth", False)

        self.action_process_func = action_process_func
        self.tokenization_func = tokenization_func
        if image_process_func is None:
            if isinstance(data_args.aug_policy, str):
                image_process_func = PreprocessRGB(
                    image_processor=data_args.image_processor,
                    image_aspect_ratio=data_args.image_aspect_ratio,
                    augmentations=PixelAug(policy=data_args.aug_policy)
                    if data_args.aug_policy
                    else None,
                    image_pad_mode=getattr(data_args, "image_pad_mode", "mean"),
                )
                self.image_process_func = [
                    image_process_func for _ in range(self.num_images)
                ]
            elif isinstance(data_args.aug_policy, list):
                assert (
                    len(data_args.aug_policy) == self.num_images
                ), f"The length of aug_policy {len(data_args.aug_policy)} must be equal to num_images {self.num_images}"
                self.image_process_func = []
                for policy in data_args.aug_policy:
                    image_process_func = PreprocessRGB(
                        image_processor=data_args.image_processor,
                        image_aspect_ratio=data_args.image_aspect_ratio,
                        augmentations=PixelAug(policy=policy) if policy else None,
                        image_pad_mode=getattr(data_args, "image_pad_mode", "mean"),
                    )
                    self.image_process_func.append(image_process_func)
            else:
                raise ValueError(
                    f"Invalid aug_policy: {data_args.aug_policy}, must be str or list"
                )
        else:
            if isinstance(image_process_func, list):
                assert (
                    len(image_process_func) == self.num_images
                ), f"The length of image_process_func {len(image_process_func)} must be equal to num_images {self.num_images}"
                self.image_process_func = image_process_func
            else:
                self.image_process_func = [
                    image_process_func for _ in range(self.num_images)
                ]
        if depth_process_func is None:
            self.depth_process_func = PreprocessDepth(
                target_size=getattr(
                    data_args.image_processor,
                    "crop_size",
                    data_args.image_processor.size,
                )
            )
        else:
            self.depth_process_func = depth_process_func
        self.key_extract_func = ExtracKeys()

    def _build_dataset_from_name(self, dataset_names):
        datasets_info = []
        for name in dataset_names.split("+"):
            # TODO: support pre-defined mix datasets
            dataset = CONVERSATION_DATA[name]
            datasets_info.append(dataset)

        self.datasets_info = datasets_info
        self._build_dataset_index()

    def _build_dataset_index(self):
        total_samples = 0
        data_indices = []
        global_index = []
        file_name_map = {}
        dataset_map = {}
        file_id = 0
        dataset_id = 0
        for dataset_info in self.datasets_info:
            data_path = dataset_info["annotations"]
            data_path_prefix = dataset_info.get("data_path_prefix", "")
            frequency = dataset_info["frequency"]
            meta_data = dataset_info["meta_data"]

            if data_path not in dataset_map:
                dataset_map[data_path] = {
                    "id": dataset_id,
                    "meta_data": meta_data,
                    "data_path_prefix": data_path_prefix,
                }
                dataset_id += 1
            dataset_index = dataset_map[data_path]["id"]

            data_index = self._get_index_cache(data_path)["data"]
            data_index = list(data_index.items())
            data_index = self._deterministic_shuffle_data_index(data_index)

            sampled_data_index = []
            while frequency > 0:
                if frequency >= 1:
                    sampled_data_index.extend(copy.deepcopy(data_index))
                else:
                    sampled_data_index.extend(
                        copy.deepcopy(
                            data_index[: math.ceil(len(data_index) * frequency)]
                        )
                    )
                frequency -= 1

            for jsonl_file, num_samples in sampled_data_index:
                if jsonl_file not in file_name_map:
                    file_name_map[jsonl_file] = file_id
                    file_id += 1
                file_index = file_name_map[jsonl_file]
                for frame_index in range(num_samples):
                    global_index.append((dataset_index, file_index, frame_index))

            total_samples += sum(num_samples for _, num_samples in sampled_data_index)
            data_indices.extend(sampled_data_index)

        self.global_index = global_index
        self.file_name_map = {v: k for k, v in file_name_map.items()}
        self.dataset_map = {
            v["id"]: {
                "data_path": k,
                "meta_data": v["meta_data"],
                "data_path_prefix": v["data_path_prefix"],
            }
            for k, v in dataset_map.items()
        }
        self.total_samples = total_samples

    def _deterministic_shuffle_data_index(self, data_index):
        data_index.sort(key=lambda x: x[0])
        dataset_seed = 42
        rng = random.Random(dataset_seed)
        rng.shuffle(data_index)
        return data_index

    def unsafe_getitem(self, idx) -> dict:
        dataset_index, file_index, frame_index = self.global_index[idx]
        jsonl_file = self.file_name_map[file_index]
        dataset_info = self.dataset_map[dataset_index]
        dataset = dataset_info["data_path"]
        meta_data = dataset_info["meta_data"]
        data_path_prefix = dataset_info["data_path_prefix"]
        episode_data_list = load_jsonl(jsonl_file, parse=True)

        # NOTE: due to the action shift in AddAction, the length of episode_data_list may be less than frame_index.
        #     In this case, we will use a random frame_index.
        length_decrease = getattr(self.action_process_func, "predict_length", 0)
        if frame_index >= len(episode_data_list) - length_decrease:
            frame_index = random.randint(
                0, len(episode_data_list) - length_decrease - 1
            )

        meta_data.update(
            dict(
                fram_indicies=[frame_index],
                jsonl_file=jsonl_file,
                dataset=dataset,
                num_images=self.num_images,
                images_keys=self.images_keys,
                depths_keys=self.depths_keys,
                load_depth=self.load_depth,
                data_path_prefix=data_path_prefix,
            )
        )

        # 1. process the episode data
        data = self.action_process_func(episode_data_list, meta_data=meta_data)
        # 2. get the frame data
        if isinstance(data, list):
            data = data[frame_index]
        data.update({"meta_data": meta_data})
        return_dict = {}

        # 3. preprocess rgb
        rgb_data = data.pop("rgb_data", [])
        if len(rgb_data) < self.num_images:
            warnings.warn(
                "The length of rgb_data is less than num_images, padding with None"
            )
            rgb_data = rgb_data + [None] * (self.num_images - len(rgb_data))

        pixel_values = [
            image_process_func(data)
            for image_process_func, data in zip(
                self.image_process_func, rgb_data, strict=True
            )
        ]
        return_dict["image"] = (
            pixel_values[0]
            if len(pixel_values) == 1
            else torch.stack(pixel_values, dim=0)
        )

        # 3.1 extract depth data
        if self.load_depth:
            depth_data = data.pop("depth_data", [])
            if len(depth_data) < self.num_images:
                warnings.warn(
                    "The length of depth_data is less than num_images, padding with None"
                )
                depth_data = depth_data + [None] * (self.num_images - len(depth_data))
            depth_values = [self.depth_process_func(_) for _ in depth_data]
            return_dict["depth"] = (
                depth_values[0]
                if len(depth_values) == 1
                else torch.stack(depth_values, dim=0)
            )

        # 4. tokenize the prompt
        if "conversations" not in data:
            data = ToConversation_Old()(data)
        conversations = data["conversations"]
        tokenized_dict = self.tokenization_func(
            conversations=conversations, has_image=True
        )
        return_dict["input_ids"] = tokenized_dict["input_ids"]
        return_dict["labels"] = tokenized_dict["labels"]

        # 5. extract other data and convert to tensor
        other_keys = [_ for _ in self.data_keys if _ not in return_dict]
        return_dict.update(self.key_extract_func(data, other_keys))
        return_dict = ToTensor()(return_dict)

        return return_dict

    def __getitem__(self, idx) -> dict:
        try:
            return self.unsafe_getitem(idx)
        except Exception:
            return self.unsafe_getitem(random.randint(0, len(self) - 1))

    def __len__(self):
        return self.total_samples

    def _get_index_cache(self, data_path):
        """Cache the index of the dataset to speed up the data loading process
        Chache format:
        {
            "meta_data": {
                "total_samples": 1000,
                "total_jsonl_files": 10,
            },
            "data": {
                "jsonl_file_name1": 80, // number of samples in the jsonl file
                "jsonl_file_name2": 1242, // number of samples in the jsonl file
                ...
            }
        }
        """
        index_cache_file = os.path.join(data_path, "index_cache.json")
        if megfile.smart_exists(index_cache_file):
            with megfile.smart_open(index_cache_file, "r") as f:
                index_cache = json.load(f)
            if self._check_index_cache(data_path, index_cache):
                return index_cache
        index_cache = self._build_index_cache(data_path)
        return index_cache

    def _build_index_cache(self, data_path):
        print(f"Building index cache for {data_path} ...")
        jsonl_files = megfile.smart_glob(os.path.join(data_path, "**", "*.jsonl"))
        index_cache = {
            "meta_data": {
                "total_samples": 0,
                "total_jsonl_files": len(jsonl_files),
            },
            "data": {},
        }
        for jsonl_file in jsonl_files:
            samples = load_jsonl(jsonl_file)
            index_cache["data"][jsonl_file] = len(samples)
            index_cache["meta_data"]["total_samples"] += len(samples)
        index_cache_file = os.path.join(data_path, "index_cache.json")
        with megfile.smart_open(index_cache_file, "w") as f:
            json.dump(index_cache, f, indent=2)
        return index_cache

    def _check_index_cache(self, data_path, index_cache):
        # only check the number of jsonl files
        jsonl_files = megfile.smart_glob(os.path.join(data_path, "**", "*.jsonl"))

        return len(jsonl_files) == index_cache["meta_data"]["total_jsonl_files"]


def load_jsonl(file_path, parse=False):
    with megfile.smart_open(file_path, "r") as f:
        if parse:
            return [json.loads(_) for _ in f.readlines() if _.strip()]
        else:
            return [_ for _ in f.readlines() if _.strip()]
