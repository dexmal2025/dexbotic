import numpy as np
from itertools import zip_longest

defalut_prompt_template = "<image>\nWhat action should the robot take to {prompt}?"


class AddPromptTemplate:
    """Add the prompt template to `prompt` in the episode_data_dict.

       Have no effect if the `is_robot` is not in the episode_data_dict or is_robot is False.
    """

    def __init__(self,
                 prompt_template: str = defalut_prompt_template,
                 ):
        """Args:
            prompt_template: Tuple[str, (str) -> str], the prompt template for the robot. Default: defalut_prompt_template
        """

        self.prompt_template = prompt_template

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        # assume all data in the episode_data_dict has the same value of `is_robot`
        if 'is_robot' in episode_data_dict and episode_data_dict['is_robot'][0]:
            episode_data_dict['prompt'] = [
                self.prompt_template.format(prompt=_) for _ in episode_data_dict['prompt']]
        return episode_data_dict


class ReplaceAnswer:
    """Replace the `answer` in the episode_data_dict with a default string"""

    def __init__(self, default_answer: str = " ", replace_existing: bool = False):
        """Args:
        default_answer: str, the default answer to replace the original answer. Default: ' '
        """
        self.default_answer = default_answer
        self.replace_existing = replace_existing

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        episode_length = episode_data_dict.get("prompt", None) or episode_data_dict.get(
            "conversations", None
        )
        if episode_length is None:
            raise ValueError(
                "Due to the lack of prompt or conversations, the episode length is not determined."
            )
        if self.replace_existing or (
            "conversations" not in episode_data_dict
            and "answer" not in episode_data_dict
        ):
            episode_data_dict["answer"] = [self.default_answer] * len(episode_length)
            episode_data_dict["has_text"] = np.zeros(
                (len(episode_length), 1), dtype=bool
            )
        return episode_data_dict


class ToConversation_Old:
    """Convert the prompt and answer to a conversation format"""

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "conversations" in episode_data_dict:
            return episode_data_dict
        episode_data_dict["conversations"] = [
            {"from": "human", "value": episode_data_dict.pop("prompt", "")},
            {"from": "gpt", "value": episode_data_dict.pop("answer", "")},
        ]
        return episode_data_dict


class ToConversation:
    """Convert the prompt and answer to a conversation format
       if there is no `conversations` key in the episode_data_dict.

    """

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "conversations" in episode_data_dict:
            return episode_data_dict
        prompts = episode_data_dict.pop("prompt", [])
        answers = episode_data_dict.pop("answer", [])

        conversations = []
        for prompt, answer in zip_longest(prompts, answers, fillvalue=""):
            conversations.append(
                [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
            )

        episode_data_dict["conversations"] = conversations
        return episode_data_dict
