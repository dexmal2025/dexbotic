
# SO-101  VLA Development Tutorial

This tutorial aims to guide users from scratch in building a robot control system, which is based on both our Dexbotic toolbox and the **[LeRobot framework](https://github.com/huggingface/lerobot)**, completing the full process deployment from data processing to real-world deployment.


## Preparation

Before starting, please ensure the hardware connection is normal and basic calibration is completed.

### Hardware & Basic Environment

Refer to the **[SO-101 Official Tutorial](https://huggingface.co/docs/lerobot/so101)** to complete the following in order:

1. Servo Calibration  
2. Robot Arm Assembly  
3. Robot Arm Calibration  
4. Teleoperation Test  
5. Camera Installation  
6. Dataset Collection (Recording using LeRobot scripts)


## Data Conversion

To adapt to policy training, we need to convert the raw data collected by LeRobot into our generic **DexData** format.

### Conversion Logic

- **Camera Mapping:** Map physical cameras (e.g., Front/Side) to logical views (Head/Wrist).
- **Time Alignment:** Align video frames, robot states (State), actions (Action), and language instructions (Prompt) on a unified timeline.


### Data Directory Standard

Please ensure the input data structure conforms to the LeRobot standard:

```bash
my_input_dataset/                  # Input root directory
├── insert_ring/                   # Task name
│   └── train/                     # Split (train/test/val)
│       ├── meta/tasks.parquet     # Task metadata
│       ├── data/chunk-000/        # State/action data (.parquet)
│       └── videos/                # Video data
```

### Execution

Run the command:

```bash
python hardware/so101/convert_so101_to_dexdata.py \
  --dataset_path /path/to/lerobot_dataset/press_blue_then_green \
  --output_dir /path/to/so101_dexdata \
  --task_prompt "Pick up the object" \
  --task_name push_button
```

The output structure should look like this:

```text
so101_dexdata/
├── dexdata_jsonl/
│   └── push_button/episode_00000.jsonl
└── videos/
    └── push_button/episode_00000_front.mp4
```


## Policy Training

This section follows the official Dexbotic training paradigm. 


### Register the dataset

Create a dataset registration file under `dexbotic/data/data_source`:

```python
from dexbotic.data.data_source.register import register_dataset

SO101_DATASET = {
    "push_button": {
        "data_path_prefix": "./dexbotic/so101_dexdata/videos/push_button",
        "annotations": "./so101_dexdata/dexdata_jsonl/push_button",
        "frequency": 1,
    },
}

meta_data = {
    "non_delta_mask": [-1],
    "periodic_mask": None,
    "periodic_range": None,
}

register_dataset(SO101_DATASET, meta_data=meta_data, prefix="so101")
```

This makes the dataset name `so101_push_button`.

### Create an experiment config

Modify `playground/example_exp.py` and update:

- `CogActDataConfig.dataset_name = "so101_push_button"`
- `CogActTrainerConfig.output_dir` to your checkpoint directory
- `CogActModelConfig.model_name_or_path` to the pretrained model path

### Start training

```bash
cd /path/to/dexbotic
deepspeed playground/example_exp.py --task train
```


## System Launch

This section starts three processes:

- **VLA policy server**: Runs the trained model and exposes an inference endpoint.
- **Bridge Server**: Receives robot images, forwards requests to the VLA policy, and returns actions.
- **Robot Client**: Runs on the robot side, streams observations to the Bridge, and executes actions.

Before you start, put those scripts into your SO101 working directory (replace paths as needed):

```bash
cp hardware/so101/bridge_server.py ~/path/to/SO101/
cp hardware/so101/client.py ~/path/to/SO101/
```

### Start VLA policy server (Terminal 1)

Use the same training script to launch your policy server:

```bash
cd /path/to/dexbotic
python  playground/example_exp.py --task inference
```

### Start Bridge Server (Terminal 2)

Once started, this service waits for the robot connection and is responsible for displaying the video feed.  
**Note:** The `--task` parameter must match the training Prompt.

```bash
# Enter working directory
cd ~/path/to/SO101
conda activate lerobot 

# Please modify --vla_url (VLA policy URL) according to your actual situation
python bridge_server.py \
  --vla_url http://your_ip:7899 \
  --prompt "Press the button"
```

**Parameter Explanation:**
- `--vla_url`: Specifies the API endpoint of the VLA policy; please replace the example IP (`your_ip`) with your actual backend server address.
- `--prompt`: The text prompt sent to the model, which must strictly match the instruction used during training (including punctuation) to ensure correct inference behavior.

**Success Indicators:**

If your configuration is correct, these logs will be shown in the Terminal 2:

```text
Bridge Server started on [::]:8080
Waiting for Dual-Camera Robot Client...
```

### Start Robot Client (Terminal 3)

This service drives the hardware, sends images to the Bridge, and executes received actions.  
**Permission Hint:** It is recommended to run `sudo chmod 666 /dev/ttyACM0` every time USB is plugged in.

```bash
# Start robot client
python -m lerobot.async_inference.robot_client \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, side: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \
  --server_address=127.0.0.1:8080 \
  --actions_per_chunk=32 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --task="Press the button" \
  --policy_type=act \
  --policy_device=mps
```

**Key Parameter Explanation:**
- `--actions_per_chunk=32`: Defines the total number of actions received by the client in a single inference.
- `--server_address`: If the Bridge Server is on the local machine, enter `127.0.0.1:8080`; if on another computer, enter that computer's IP.


## Successful Run Indicators

1. **Terminal 2:** Displays `Robot Client Connected`, and continuously logs `Sending ... images to VLA`.
2. **Terminal 3:** Displays `Robot connected and ready`.
3. **Robot Arm:** Begins to move smoothly following the instructions.

## Demos
Following the above instructions, we conducted a **Push Button** task on our SO-101 robot, and provide a successful demo video below:

**[📹 View Demo Video: Push Button Task](hardware/so101/demo_press_button.mp4)**