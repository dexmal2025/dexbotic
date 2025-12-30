import numpy as np
import torch


class ToNumpy:
    """Convert all numbers in the episode_data_dict to numpy array, keeping strings unchanged.
    """

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:

        if isinstance(episode_data_dict, dict):
            return {key: self.__call__(value)
                    for key, value in episode_data_dict.items()}
        elif isinstance(episode_data_dict, list):
            if all(isinstance(item, (int, float, bool, complex, np.number))
                   for item in episode_data_dict):
                # Convert list of numbers to numpy array
                return np.array(episode_data_dict)
            else:
                # Recursively process elements
                episode_data_dict = [self.__call__(item) for item in episode_data_dict]
                if all(isinstance(item, np.ndarray) for item in episode_data_dict):
                    # Stack list of numpy arrays into a multi-dimensional array
                    episode_data_dict = np.stack(episode_data_dict)
                return episode_data_dict
        elif isinstance(episode_data_dict, (int, float, bool, complex, np.number)):
            return np.array(episode_data_dict)  # Convert single numbers to numpy array
        elif isinstance(episode_data_dict, str):
            return episode_data_dict  # Keep strings unchanged
        else:
            return episode_data_dict  # Other types remain unchanged


class ToTensor:
    def __call__(self, data):
        if isinstance(data, dict):
            return {key: self.__call__(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.__call__(item) for item in data]
        else:
            return torch.as_tensor(data)


class ToList:
    """Convert episode dict to frame list

    This transform is the inverse of ToDict and should be used in the end of the pipeline.
    """

    def __init__(self, select_frame: bool = False):
        self.select_frame = select_frame

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        meta_data = episode_data_dict.pop("meta_data", None)
        list_length = len(
            episode_data_dict.get("prompt") or episode_data_dict.get("conversations")
        )
        episode_data_list = []
        for i in range(list_length):
            episode_data_list.append({})
            for key, value in episode_data_dict.items():
                episode_data_list[i][key] = value[i]
        if self.select_frame:
            episode_data_list = episode_data_list[meta_data["fram_indicies"][0]]
        return episode_data_list


class ToDict:
    """Convert frame list to episode dict

       This transform is the inverse of ToList and should be used in the begin of the pipeline.
    """

    def __call__(self, episode_data_list: dict, meta_data: dict = {}, **kwargs) -> dict:

        episode_data_dict = {}
        for key in episode_data_list[0].keys():
            episode_data_dict[key] = [frame[key] for frame in episode_data_list]
        episode_data_dict['meta_data'] = meta_data
        return episode_data_dict


class Pipeline:
    def __init__(self, transforms: list):
        self.transforms = []
        for transform in transforms:
            self.add(transform)

    def __call__(self, episode_data_dict: dict, **kwargs):
        for transform in self.transforms:
            episode_data_dict = transform(episode_data_dict, **kwargs)
        return episode_data_dict

    def add(self, transform) -> None:
        if isinstance(transform, list):
            for trans in transform:
                self.transforms.append(trans)
        else:
            self.transforms.append(transform)
            if hasattr(transform, 'predict_length'):
                self.predict_length = transform.predict_length
            if hasattr(transform, 'statistic_mapping'):
                self.statistic_mapping = transform.statistic_mapping


class ExtracKeys:
    """Extract keys from episode_data_dict
    """

    def __call__(self, episode_data_dict: dict, keys: list[str], **kwargs):
        # check all keys are in the episode_data_dict
        for key in keys:
            assert key in episode_data_dict, f'{key} is not in {episode_data_dict["meta_data"]["jsonl_file"]}'

        return {key: episode_data_dict[key] for key in keys}


class AddActionFlag:
    def __init__(self, empty_action_value: np.ndarray, enable=True):
        self.empty_action_value = empty_action_value
        self.enable = enable

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if not self.enable:
            return episode_data_dict
        episode_data_dict["has_action"] = np.ones((1,), dtype=bool)
        if "action" not in episode_data_dict:
            episode_data_dict["action"] = np.zeros_like(self.empty_action_value)
            episode_data_dict["has_action"] = np.zeros((1,), dtype=bool)

        return episode_data_dict


class AddTextFlag:
    def __init__(self, enable=True):
        self.enable = enable

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if not self.enable:
            return episode_data_dict
        if "has_text" not in episode_data_dict:
            episode_data_dict["has_text"] = np.ones((1,), dtype=bool)
        return episode_data_dict
