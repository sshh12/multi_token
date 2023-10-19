import transformers
import logging

from multi_token.training import (
    TrainingArguments,
    ModelArguments,
    train_for_modalities,
)
from multi_token.training_data import (
    DataArguments,
)
from multi_token.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from multi_token.modalities import MODALITY_BUILDERS

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )

    training_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    modalities = MODALITY_BUILDERS[model_args.modality_builder]()
    model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[model_args.model_cls]

    train_for_modalities(model_cls, training_args, model_args, data_args, modalities)
