"""
SO101 Inference Server for lerobot_client

This script provides a gRPC server that integrates the Dexbotic inference engine,
allowing lerobot_client to connect and receive action predictions.

Dependencies: pip install grpcio protobuf

Usage:
    python so101_inference_example.py
    python so101_inference_example.py --task "stack block"
    python so101_inference_example.py --model_path /path/to/checkpoint --port 8080
"""

import argparse
import io
import logging
import math
import os
import pickle
import socket
import sys
import threading
import time
import types
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Queue

import grpc
import numpy as np
import torch
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
from PIL import Image

# ============================================================================
# gRPC Protobuf Definitions (embedded, no lerobot dependency)
# ============================================================================

_sym_db = _symbol_database.Default()

# Serialized protobuf descriptor (generated from lerobot's services.proto)
_DESCRIPTOR_DATA = b'\n lerobot/transport/services.proto\x12\ttransport"L\n\nTransition\x12\x30\n\x0etransfer_state\x18\x01 \x01(\x0e\x32\x18.transport.TransferState\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c"L\n\nParameters\x12\x30\n\x0etransfer_state\x18\x01 \x01(\x0e\x32\x18.transport.TransferState\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c"T\n\x12InteractionMessage\x12\x30\n\x0etransfer_state\x18\x01 \x01(\x0e\x32\x18.transport.TransferState\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c"M\n\x0bObservation\x12\x30\n\x0etransfer_state\x18\x01 \x01(\x0e\x32\x18.transport.TransferState\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c"\x17\n\x07\x41\x63tions\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c"\x1b\n\x0bPolicySetup\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c"\x07\n\x05\x45mpty*`\n\rTransferState\x12\x14\n\x10TRANSFER_UNKNOWN\x10\x00\x12\x12\n\x0eTRANSFER_BEGIN\x10\x01\x12\x13\n\x0fTRANSFER_MIDDLE\x10\x02\x12\x10\n\x0cTRANSFER_END\x10\x03\x32\x81\x02\n\x0eLearnerService\x12=\n\x10StreamParameters\x12\x10.transport.Empty\x1a\x15.transport.Parameters0\x01\x12<\n\x0fSendTransitions\x12\x15.transport.Transition\x1a\x10.transport.Empty(\x01\x12\x45\n\x10SendInteractions\x12\x1d.transport.InteractionMessage\x1a\x10.transport.Empty(\x01\x12+\n\x05Ready\x12\x10.transport.Empty\x1a\x10.transport.Empty2\xf5\x01\n\x0e\x41syncInference\x12>\n\x10SendObservations\x12\x16.transport.Observation\x1a\x10.transport.Empty(\x01\x12\x32\n\nGetActions\x12\x10.transport.Empty\x1a\x12.transport.Actions\x12\x42\n\x16SendPolicyInstructions\x12\x16.transport.PolicySetup\x1a\x10.transport.Empty\x12+\n\x05Ready\x12\x10.transport.Empty\x1a\x10.transport.Emptyb\x06proto3'

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(_DESCRIPTOR_DATA)
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "services_pb2", _globals)

# TransferState enum values
TRANSFER_BEGIN = 1
TRANSFER_MIDDLE = 2
TRANSFER_END = 3


# ============================================================================
# gRPC Service Definition
# ============================================================================


class AsyncInferenceServicer:
    """Base class for gRPC AsyncInference service."""

    def SendObservations(self, request_iterator, context):
        raise NotImplementedError()

    def GetActions(self, request, context):
        raise NotImplementedError()

    def SendPolicyInstructions(self, request, context):
        raise NotImplementedError()

    def Ready(self, request, context):
        raise NotImplementedError()


def add_AsyncInferenceServicer_to_server(servicer, server):
    """Register gRPC service handlers."""
    empty_class = DESCRIPTOR.message_types_by_name["Empty"]._concrete_class
    observation_class = DESCRIPTOR.message_types_by_name["Observation"]._concrete_class
    actions_class = DESCRIPTOR.message_types_by_name["Actions"]._concrete_class
    policy_setup_class = DESCRIPTOR.message_types_by_name["PolicySetup"]._concrete_class

    rpc_method_handlers = {
        "SendObservations": grpc.stream_unary_rpc_method_handler(
            servicer.SendObservations,
            request_deserializer=observation_class.FromString,
            response_serializer=empty_class.SerializeToString,
        ),
        "GetActions": grpc.unary_unary_rpc_method_handler(
            servicer.GetActions,
            request_deserializer=empty_class.FromString,
            response_serializer=actions_class.SerializeToString,
        ),
        "SendPolicyInstructions": grpc.unary_unary_rpc_method_handler(
            servicer.SendPolicyInstructions,
            request_deserializer=policy_setup_class.FromString,
            response_serializer=empty_class.SerializeToString,
        ),
        "Ready": grpc.unary_unary_rpc_method_handler(
            servicer.Ready,
            request_deserializer=empty_class.FromString,
            response_serializer=empty_class.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "transport.AsyncInference", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# ============================================================================
# Utility Functions
# ============================================================================


def receive_bytes_in_chunks(iterator, queue, shutdown_event, log_prefix=""):
    """Receive chunked byte data from gRPC stream."""
    bytes_buffer = io.BytesIO()

    for item in iterator:
        if shutdown_event is not None and shutdown_event.is_set():
            return None

        if item.transfer_state == TRANSFER_BEGIN:
            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            bytes_buffer.write(item.data)
        elif item.transfer_state == TRANSFER_MIDDLE:
            bytes_buffer.write(item.data)
        elif item.transfer_state == TRANSFER_END:
            bytes_buffer.write(item.data)
            if queue is not None:
                queue.put(bytes_buffer.getvalue())
            else:
                return bytes_buffer.getvalue()
            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)

    return None


def get_local_ip():
    """Get local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ============================================================================
# Lerobot-compatible Classes (for pickle deserialization)
# ============================================================================


@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: torch.Tensor

    def get_action(self):
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: dict
    must_go: bool = False

    def get_observation(self):
        return self.observation


# Register fake lerobot module for pickle deserialization
if "lerobot" not in sys.modules:
    sys.modules["lerobot"] = types.ModuleType("lerobot")

if "lerobot.async_inference" not in sys.modules:
    async_inference_module = types.ModuleType("lerobot.async_inference")
    sys.modules["lerobot.async_inference"] = async_inference_module
    sys.modules["lerobot"].async_inference = async_inference_module

if "lerobot.async_inference.helpers" not in sys.modules:
    helpers_module = types.ModuleType("lerobot.async_inference.helpers")
    sys.modules["lerobot.async_inference.helpers"] = helpers_module
    sys.modules["lerobot.async_inference"].helpers = helpers_module

sys.modules["lerobot.async_inference.helpers"].TimedData = TimedData
sys.modules["lerobot.async_inference.helpers"].TimedAction = TimedAction
sys.modules["lerobot.async_inference.helpers"].TimedObservation = TimedObservation


# ============================================================================
# Dexbotic Imports and Configuration
# ============================================================================


import json

import megfile
from transformers import AutoTokenizer

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.model.cogact.cogact_arch import CogACTForCausalLM
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SO101Server")

# Default configuration
DEFAULT_MODEL_PATH = "model_path"
DEFAULT_PORT = 8080
ACTION_DIM = 6
ACTION_REPEAT_TIMES = int(os.environ.get("ACTION_REPEAT_TIMES", "1"))

# Camera key priority (for dual-camera support)
FRONT_KEYS = ("front", "up")
SIDE_KEYS = ("side", "right")


# ============================================================================
# Inference Engine
# ============================================================================


class SO101InferenceEngine:
    """SO101 inference engine using CogACT model."""

    def __init__(self, model_path: str, action_dim: int = 6):
        self.model_path = model_path
        self.action_dim = action_dim
        self.model = None
        self.tokenizer = None
        self.device = None
        self.norm_stats = None
        self.model_config = None

    def load(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model = CogACTForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": "cuda:0"},
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model_config = self.model.config

        norm_stats_file = os.path.join(self.model_path, "norm_stats.json")
        self.norm_stats = self._read_norm_stats(norm_stats_file)

        logger.info("Model loaded successfully")
        logger.info(f"Normalization stats: {self.norm_stats}")

    def _read_norm_stats(self, path):
        """Read normalization statistics from file."""
        if path is None or not megfile.smart_exists(path):
            logger.warning(f"Norm stats file not found: {path}, using default")
            return {"min": -1, "max": 1}
        with megfile.smart_open(path, "r") as f:
            stats = json.load(f)
            if "norm_stats" in stats:
                stats = stats["norm_stats"]
            return stats.get("default", {"min": -1, "max": 1})

    def infer(self, images: list, text: str) -> list:
        """Run inference and return action sequence."""
        t0 = time.monotonic()

        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                pil_images.append(Image.open(img).convert("RGB"))

        if len(pil_images) == 1:
            image_tensor = self.model.process_images(pil_images).to(
                dtype=self.model.dtype
            )
        else:
            image_tensor = (
                self.model.process_images(pil_images)
                .to(dtype=self.model.dtype)
                .unsqueeze(0)
            )

        conv = conversation_lib.conv_templates[self.model_config.chat_template].copy()
        if self.model_config.chat_template == "step":
            conv.append_message(
                conv.roles[0], text + "<im_start>" + DEFAULT_IMAGE_TOKEN + "<im_end>"
            )
        else:
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], " ")
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        inference_args = {
            "cfg_scale": 1.5,
            "num_ddim_steps": 10,
            "action_norms": self.norm_stats,
        }

        outputs = self.model.inference_action(input_ids, image_tensor, inference_args)
        logger.info(
            f"Inference time: {time.monotonic() - t0:.3f}s, actions: {len(outputs)}"
        )

        return outputs


# ============================================================================
# gRPC Server Implementation
# ============================================================================


class GRPCServer(AsyncInferenceServicer):
    """gRPC server for lerobot_client connection."""

    def __init__(
        self,
        engine: SO101InferenceEngine,
        task: str,
        action_dim: int,
        use_delta: bool,
        repeat_times: int,
    ):
        self.engine = engine
        self.task = task
        self.action_dim = action_dim
        self.use_delta = use_delta
        self.repeat_times = repeat_times

        self.action_queue = deque()
        self.last_action = np.zeros(action_dim)
        self.observation_queue = Queue(maxsize=1)
        self.environment_dt = 1 / 30.0
        self.shutdown_event = threading.Event()
        self._logged_keys = False

        self._empty_class = DESCRIPTOR.message_types_by_name["Empty"]._concrete_class
        self._actions_class = DESCRIPTOR.message_types_by_name[
            "Actions"
        ]._concrete_class

    def Ready(self, request, context):
        logger.info("Robot Client Connected")
        self.observation_queue = Queue(maxsize=1)
        self.action_queue.clear()
        self.last_action = np.zeros(self.action_dim)
        self.shutdown_event.clear()
        return self._empty_class()

    def SendPolicyInstructions(self, request, context):
        return self._empty_class()

    def SendObservations(self, request_iterator, context):
        data = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, logger
        )
        if data:
            obs = pickle.loads(data)
            if self.observation_queue.full():
                try:
                    self.observation_queue.get_nowait()
                except QueueEmpty:
                    pass
            self.observation_queue.put(obs)
        return self._empty_class()

    def _process_image(self, img_data):
        """Process image data to PIL Image."""
        if img_data is None:
            return None

        if isinstance(img_data, torch.Tensor):
            img_data = img_data.permute(1, 2, 0).cpu().numpy()
        elif isinstance(img_data, np.ndarray):
            if (
                img_data.ndim == 3
                and img_data.shape[0] in [1, 3]
                and img_data.shape[2] > 3
            ):
                img_data = np.transpose(img_data, (1, 2, 0))

        if not np.issubdtype(img_data.dtype, np.integer):
            if img_data.max() <= 1.0:
                img_data = img_data * 255.0
            img_data = np.clip(img_data, 0, 255)

        return Image.fromarray(img_data.astype(np.uint8))

    def _delta_action(self, last_action, delta):
        """Convert delta action to absolute action."""
        original = np.copy(last_action)
        if len(original) > 6:
            original[6:] = 0

        action = original + delta

        # Normalize angles to [-pi, pi]
        if len(action) >= 6:
            action[3:6] = np.where(
                action[3:6] > math.pi, action[3:6] - 2 * math.pi, action[3:6]
            )
            action[3:6] = np.where(
                action[3:6] < -math.pi, action[3:6] + 2 * math.pi, action[3:6]
            )

        return action

    def _acquire_actions(self, images: list):
        """Get new actions from inference engine."""
        response = self.engine.infer(images, self.task)

        logger.info(f"Model output actions (raw): {len(response)} steps")
        for i, act in enumerate(response):
            logger.info(f"  Step {i}: {act}")

        last_act = self.last_action
        for action in response:
            if self.use_delta:
                action = self._delta_action(last_act, np.array(action))
            else:
                action = np.array(action)

            for _ in range(self.repeat_times):
                self.action_queue.append(action)
                last_act = action

        logger.info(
            f"Processed actions (after delta): queue size = {len(self.action_queue)}"
        )

    def GetActions(self, request, context):
        try:
            timed_obs = self.observation_queue.get(timeout=1.0)
            raw_obs = timed_obs.observation

            if not self._logged_keys:
                logger.info(f"Observation keys: {list(raw_obs.keys())}")
                self._logged_keys = True

            # Extract images from observation
            images = []
            for keys in [FRONT_KEYS, SIDE_KEYS]:
                for key in keys:
                    if key in raw_obs:
                        img = self._process_image(raw_obs[key])
                        if img is not None:
                            images.append(img)
                        break

            # Get actions if queue is empty
            if len(self.action_queue) == 0 and images:
                logger.info(f"Inferring with {len(images)} images...")
                self._acquire_actions(images)

            # Pack response
            response_chunk = []
            chunk_start_ts = timed_obs.timestamp
            chunk_start_step = timed_obs.timestep

            idx = 0
            while len(self.action_queue) > 0:
                act = self.action_queue.popleft()

                # Dimension correction
                if len(act) != self.action_dim:
                    if len(act) > self.action_dim:
                        act = act[: self.action_dim]
                    else:
                        act = np.concatenate(
                            [act, np.zeros(self.action_dim - len(act))]
                        )

                response_chunk.append(
                    TimedAction(
                        timestamp=chunk_start_ts + idx * self.environment_dt,
                        timestep=chunk_start_step + idx,
                        action=torch.from_numpy(act).float(),
                    )
                )
                idx += 1

            if not response_chunk:
                return self._empty_class()

            actions_msg = self._actions_class()
            actions_msg.data = pickle.dumps(response_chunk)
            return actions_msg

        except QueueEmpty:
            return self._empty_class()
        except Exception as e:
            import traceback

            logger.error(f"Error: {e}\n{traceback.format_exc()}")
            return self._empty_class()


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SO101 Inference Server for lerobot_client"
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--task", type=str, default="Pick up the object")
    parser.add_argument("--action_dim", type=int, default=ACTION_DIM)
    parser.add_argument("--use_delta", action="store_true", default=True)
    parser.add_argument("--repeat_times", type=int, default=ACTION_REPEAT_TIMES)

    args = parser.parse_args()
    local_ip = get_local_ip()

    logger.info("=" * 60)
    logger.info("SO101 Inference Server")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Bind: {args.host}:{args.port}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Action dim: {args.action_dim}")
    logger.info("=" * 60)

    engine = SO101InferenceEngine(args.model_path, args.action_dim)
    engine.load()

    servicer = GRPCServer(
        engine, args.task, args.action_dim, args.use_delta, args.repeat_times
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    add_AsyncInferenceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()

    logger.info(f"gRPC server running on {args.host}:{args.port}")
    logger.info(f"lerobot_client URL: {local_ip}:{args.port}")
    logger.info("Waiting for lerobot_client to connect...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
