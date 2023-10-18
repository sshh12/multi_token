from multi_token.language_models.mistral import (
    MistralLMMForCausalLM,
)

LANGUAGE_MODEL_CLASSES = [MistralLMMForCausalLM]

LANGUAGE_MODEL_NAME_TO_CLASS = {cls.__name__: cls for cls in LANGUAGE_MODEL_CLASSES}
