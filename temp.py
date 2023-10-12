import transformers

from lmm_multi_token.training import (
    TrainingArguments,
    ModelArguments,
    train_for_modalities,
)
from lmm_multi_token.training_data import (
    DataArguments,
)
from lmm_multi_token.language_models.mistral import MistralLMMForCausalLM
from lmm_multi_token.modalities.vision_clip import CLIPVisionModality

if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )

    training_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    modalities = [CLIPVisionModality()]

    train_for_modalities(
        MistralLMMForCausalLM, training_args, model_args, data_args, modalities
    )
