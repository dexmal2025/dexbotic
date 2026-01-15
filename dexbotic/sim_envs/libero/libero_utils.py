"""
Libero utility functions for dexbotic environments.
Adapted from SimpleVLA-RL verl.utils.libero_utils to remove verl dependencies.
"""

import math
import os
import random
from typing import Any, Tuple

import imageio
import numpy as np
from PIL import Image

try:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    LIBERO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: can't import libero: {e}")
    LIBERO_AVAILABLE = False


def get_libero_env(task: Any, resolution: int = 256) -> Tuple[Any, str]:
    """
    Initialize and return the LIBERO environment, along with the task description.

    Args:
        task: LIBERO task object
        resolution: Camera resolution for rendering

    Returns:
        Tuple of (environment, task_description)
    """
    if not LIBERO_AVAILABLE:
        raise ImportError("LIBERO is not installed")

    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        0
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str) -> list:
    """
    Get dummy/no-op action, used to roll out the simulation while the robot does nothing.

    Args:
        model_family: Model family string

    Returns:
        Dummy action as list
    """
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img: np.ndarray, resize_size: Tuple[int, int]) -> np.ndarray:
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE: To make input images in distribution with respect to the inputs seen at training time,
    we follow the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.

    Args:
        img: Input image as numpy array
        resize_size: Target size as (height, width) tuple

    Returns:
        Resized image as numpy array
    """
    assert isinstance(resize_size, tuple)

    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(img)

    # Encode and decode as JPEG to match RLDS dataset processing
    import io

    jpeg_buffer = io.BytesIO()
    pil_img.save(jpeg_buffer, format="JPEG")
    jpeg_buffer.seek(0)
    pil_img = Image.open(jpeg_buffer)

    # Resize using Lanczos3 (LANCZOS) resampling to match TensorFlow's lanczos3
    resized_img = pil_img.resize(
        (resize_size[1], resize_size[0]), resample=Image.Resampling.LANCZOS
    )

    # Convert back to numpy array
    img_array = np.array(resized_img)

    # Clip values to [0, 255] and convert to uint8
    img_array = np.clip(np.round(img_array), 0, 255).astype(np.uint8)

    return img_array


def get_libero_image(obs: dict, resize_size: int) -> np.ndarray:
    """
    Extract image from observations and preprocess it.

    Args:
        obs: Observation dictionary from LIBERO environment
        resize_size: Target resize size (square)

    Returns:
        Preprocessed image as numpy array
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def get_libero_wrist_image(obs: dict, resize_size: int) -> np.ndarray:
    """
    Extract wrist camera image from observations and preprocess it.

    Args:
        obs: Observation dictionary from LIBERO environment
        resize_size: Target resize size (square)

    Returns:
        Preprocessed wrist image as numpy array
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Args:
        quat: (x,y,z,w) vec4 float angles

    Returns:
        (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_image_resize_size(model_family: str) -> int:
    """
    Get image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.

    Args:
        model_family: Model family string

    Returns:
        Resize size as integer
    """
    if model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = (
        2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    )

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] = inverted_action[..., -1] * -1.0

    return inverted_action


def save_rollout_video(
    rollout_images: list, exp_name: str, task_name: str, step_idx: int, success: bool
) -> str:
    """
    Save an MP4 replay of an episode.

    Args:
        rollout_images: List of images from the rollout
        exp_name: Experiment name for directory organization
        task_name: Name of the task
        step_idx: Current step/episode index
        success: Whether the episode was successful

    Returns:
        Path to saved video file
    """
    rollout_dir = f"./rollouts/{exp_name}"
    os.makedirs(rollout_dir, exist_ok=True)
    ran_id = random.randint(1, 10000)
    mp4_path = f"{rollout_dir}/step={step_idx}--task={task_name}--success={success}--ran={ran_id}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path
