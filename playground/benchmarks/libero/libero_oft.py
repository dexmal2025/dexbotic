
import argparse
from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.oft_exp import (OFTDataConfig, OFTExp,
                                  OFTModelConfig, OFTTrainerConfig,
                                  InferenceConfig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        choices=[
            'train',
            'inference'])
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class LiberoObjectOFTTrainerConfig(OFTTrainerConfig):
    output_dir: str = field(
        default=f'./user_checkpoints/dexbotic/libero_all_oft/all-{datetime.now().strftime("%m%d")}')
    wandb_project: str = field(default='dexbotic_libero_oft')
    num_train_epochs: int = field(default=20)
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)
    


@dataclass
class LiberoObjectOFTDataConfig(OFTDataConfig):
    dataset_name: str = field(default='libero_oft_all')

@dataclass
class LiberoObjectOFTModelConfig(OFTModelConfig):
    # You should put the pre-trained model path here
    model_name_or_path: str = field(
        default='./checkpoints/Dexbotic-Base')


@dataclass
class LiberoObjectOFTInferenceConfig(InferenceConfig):
    model_name_or_path: str = field(
        default='./checkpoints/libero/libero_oft')
    port: int = field(default=7891)

    def process_frame(self) -> None:
        results = self._get_response(
            text=request.form.get('text'),
            images=request.files.getlist('image'),
            states=request.form.get('states', None),
        )
        fix_results = []
        for result in results:
            fix_result = result.copy()
            if result[-1] > 0.5:
                fix_result[-1] = -1
            else:
                fix_result[-1] = 1
            fix_results.append(fix_result)
        return jsonify({'response': fix_results})

@dataclass
class LiberoObjectOFTExp(OFTExp):
    model_config: LiberoObjectOFTModelConfig = field(
        default_factory=LiberoObjectOFTModelConfig)
    trainer_config: LiberoObjectOFTTrainerConfig = field(
        default_factory=LiberoObjectOFTTrainerConfig)
    data_config: LiberoObjectOFTDataConfig = field(
        default_factory=LiberoObjectOFTDataConfig)
    inference_config: LiberoObjectOFTInferenceConfig = field(
        default_factory=LiberoObjectOFTInferenceConfig)


if __name__ == "__main__":
    args = parse_args()
    exp = LiberoObjectOFTExp()
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
