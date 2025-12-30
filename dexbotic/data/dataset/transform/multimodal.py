import io
import os
from typing import Optional
import megfile
import numpy as np
from PIL import Image
from decord import VideoReader
import av
from collections import defaultdict


# TODO: support history multi-modal data

class LoadMultiModal:
    """Load RGB data from the episode_data_dict with the given fram_indicies.

       All keys in the episode_data_dict that start with 'images' will be loaded
       and concatenated in alphabetical order.

       The parsed data will be moved to 'rgb_data' and the original data will be removed.
    """
    def __init__(self, return_masks: bool = False):
        self.return_masks = return_masks

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        """Use the `fram_indicies`, `num_images`, and `images_keys` in the `meta_data` to load the RGB data.

           frame_indicies: None, int, or list of int, the  (temporal) frame indices to load. Default: None
           None means load all frames.

           num_images: None or int, the number of images (views) to load from the video. Default: None
           None means load all views. If it is an int, The first `num_images` in alphabetical order will be loaded.

           images_keys: None or list of str, the keys of the images to load. Default: None

           depths_keys: None or list of str, the keys of the depth images to load. Default: None

           load_depth: bool, whether to load depth data. Default: False
        """

        fram_indicies = episode_data_dict['meta_data']['fram_indicies']
        num_images = episode_data_dict['meta_data']['num_images']
        image_keys = episode_data_dict['meta_data']['images_keys']
        depths_keys = episode_data_dict['meta_data']['depths_keys']
        load_depth = episode_data_dict['meta_data']['load_depth']
        data_path_prefix = episode_data_dict['meta_data']['data_path_prefix']

        if isinstance(fram_indicies, np.ndarray):
            fram_indicies = fram_indicies.tolist()
        if isinstance(fram_indicies, int):
            fram_indicies = [fram_indicies]

        # Load RGB data
        if image_keys is None:
            keys_to_parse = [
                key for key in episode_data_dict.keys() if key.startswith('images')]
        else:
            keys_to_parse = [key for key in image_keys if key in episode_data_dict.keys()]
        keys_to_parse.sort()
        if self.return_masks:
            image_masks = [True if f"images_{i}" in episode_data_dict.keys() else False for i in range(1, num_images + 1)]
            episode_data_dict["image_masks"] = np.array([image_masks] * len(episode_data_dict['prompt']))
        if num_images is not None:
            keys_to_parse = keys_to_parse[:num_images]

        for key in keys_to_parse:
            episode_data_dict = self._load_rgb(episode_data_dict, key, fram_indicies, data_path_prefix)

        # move data to `rgb_data`
        episode_data_dict['rgb_data'] = []
        for rgb_data in zip(*[episode_data_dict[key] for key in keys_to_parse]):
            rgb_data = [_.get('data', None) for _ in rgb_data]
            episode_data_dict['rgb_data'].append(rgb_data)
        if len(episode_data_dict['rgb_data']) == 0:
            episode_data_dict.pop('rgb_data')

        for key in keys_to_parse:
            episode_data_dict.pop(key)

        # Load depth data
        if load_depth:
            if depths_keys is None:
                keys_to_parse = [
                    key for key in episode_data_dict.keys() if key.startswith('depths')]
            else:
                keys_to_parse = depths_keys
            keys_to_parse.sort()
            if num_images is not None:
                keys_to_parse = keys_to_parse[:num_images]

            for key in keys_to_parse:
                episode_data_dict = self._load_depth(
                    episode_data_dict, key, fram_indicies, data_path_prefix)

            # move data to `depth_data`
            episode_data_dict['depth_data'] = []
            for depth_data in zip(*[episode_data_dict[key] for key in keys_to_parse]):
                depth_data = [_.get('data', None) for _ in depth_data]
                episode_data_dict['depth_data'].append(depth_data)
            if len(episode_data_dict['depth_data']) == 0:
                episode_data_dict.pop('depth_data')
            for key in keys_to_parse:
                episode_data_dict.pop(key)

        return episode_data_dict

    def _load_rgb(self, episode_data_dict, key, fram_indicies=None, data_path_prefix=''):
        images = episode_data_dict[key]
        image_frames = [(idx, _) for idx, _ in enumerate(images) if _[
            'type'] == 'image' if fram_indicies is None or idx in fram_indicies]
        video_frames = [(idx, _) for idx, _ in enumerate(images) if _[
            'type'] == 'video' if fram_indicies is None or idx in fram_indicies]

        # load video frames
        video_cache = {}
        frame_indices = defaultdict(list)

        for _, frame in video_frames:
            video_url = os.path.join(data_path_prefix, frame['url'])
            frame_idx = int(frame['frame_idx'])
            frame_indices[video_url].append(frame_idx)

        for video_url, indices in frame_indices.items():
            video_cache[video_url] = self._load_video(video_url, indices)

        for _, frame in video_frames:
            video_url = os.path.join(data_path_prefix, frame['url'])
            frame_idx = int(frame['frame_idx'])
            frame['data'] = video_cache[video_url][frame_idx]

        # load image frames
        for _, frame in image_frames:
            image_url = os.path.join(data_path_prefix, frame['url'])
            frame['data'] = self._load_image(image_url)

        return episode_data_dict

    def _load_depth(self, episode_data_dict, key, fram_indicies=None, data_path_prefix=''):
        depths = episode_data_dict[key]
        depth_frames = [(idx, _) for idx, _ in enumerate(depths) if _[
            'type'] == 'video' if fram_indicies is None or idx in fram_indicies]

        # load depth frames
        depth_cache = {}
        frame_indices = defaultdict(list)

        for _, frame in depth_frames:
            depth_url = os.path.join(data_path_prefix, frame['url'])
            frame_idx = int(frame['frame_idx'])
            frame_indices[depth_url].append(frame_idx)

        for depth_url, indices in frame_indices.items():
            depth_cache[depth_url] = self._load_depth_video(depth_url, indices)

        for _, frame in depth_frames:
            depth_url = os.path.join(data_path_prefix, frame['url'])
            frame_idx = int(frame['frame_idx'])
            frame['data'] = depth_cache[depth_url][frame_idx]

        return episode_data_dict

    @staticmethod
    def _load_video(video_url, frame_indices):
        with megfile.smart_open(video_url, mode='rb') as f:
            f.seek(0)
            vr = VideoReader(f, num_threads=1)
            frames = vr.get_batch(frame_indices).asnumpy()
            images = {idx: Image.fromarray(frame)
                      for idx, frame in zip(frame_indices, frames)}
            del vr
        return images

    @staticmethod
    def _load_image(image_url):
        with megfile.smart_open(image_url, mode='rb') as f:
            f.seek(0)
            bytes_data = f.read()
            image = Image.open(io.BytesIO(bytes_data), "r").convert('RGB')
        return image

    @staticmethod
    def _load_depth_video(depth_url, frame_indices):
        with megfile.smart_open(depth_url, mode='rb') as f:
            container = av.open(f)
            images = {}
            stream = container.streams.video[0]
            frame_indices_set = set(frame_indices)

            for i, frame in enumerate(container.decode(stream)):
                if i > max(frame_indices):
                    break
                if i in frame_indices_set:
                    img = frame.to_ndarray(format='gray16le')
                    images[i] = img

            container.close()
        return images
