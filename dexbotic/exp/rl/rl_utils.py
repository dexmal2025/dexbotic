import json
import math

import megfile
import numpy as np


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
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


def read_normalization_stats(action_norm_file):
    if action_norm_file is None or not megfile.smart_exists(action_norm_file):
        return {"min": -1, "max": 1}
    with megfile.smart_open(action_norm_file, "r") as f:
        norm_stats = json.load(f)
        if "norm_stats" in norm_stats:
            norm_stats = norm_stats["norm_stats"]
        norm_stats = norm_stats["default"]
    return norm_stats
