from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

from dexbotic.constants import IGNORE_INDEX


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        labels = labels[:, :self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        mapping_keys = {
            'image': 'images',
            'actions': 'actions',
            'action': 'actions',
            'state': 'states',
            'reward': 'reward',
            'image_masks': 'image_masks',
            'has_action': 'has_action',
            'has_text': 'has_text',
        }
        for key in mapping_keys:
            if key in instances[0]:
                values = [instance[key] for instance in instances]
                if all(x is not None and x.shape == values[0].shape for x in values):
                    batch[mapping_keys[key]] = torch.stack(values)
                else:
                    batch[mapping_keys[key]] = values

        return batch
