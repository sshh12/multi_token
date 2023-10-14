from dataclasses import dataclass, field

import transformers
import torch

from lmm_multi_token.training import (
    ModelArguments,
)
from lmm_multi_token.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from lmm_multi_token.modalities import MODALITY_BUILDERS
from lmm_multi_token.inference import load_trained_lora_model
from lmm_multi_token.data_tools import encode_chat


@dataclass
class ServeArguments(ModelArguments):
    port: int = field(default=8080)
    host: str = field(default="0.0.0.0")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ServeArguments,))

    serve_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    modalities = MODALITY_BUILDERS[serve_args.modality_builder]()
    model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[serve_args.model_cls]

    model, tokenizer = load_trained_lora_model(
        model_name_or_path=serve_args.model_name_or_path,
        model_lora_path=serve_args.model_lora_path,
        model_cls=model_cls,
        modalities=modalities,
    )

    chat = dict(
        messages=[dict(role="user", content="Describe the image <image>")],
        images=["https://llava-vl.github.io/static/images/view.jpg"],
    )
    encoded_dict = encode_chat(chat, tokenizer, modalities)

    from transformers import TextStreamer

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=encoded_dict["input_ids"].unsqueeze(0).to(model.device),
            max_new_tokens=16,
            # streamer=streamer,
            use_cache=False,
            modality_inputs={
                m.name: encoded_dict[m.name].unsqueeze(0).to(model.device)
                for m in modalities
            },
        )

    outputs = tokenizer.decode(
        output_ids[0, encoded_dict["input_ids"].shape[0] :]
    ).strip()
    print(outputs)

    import IPython

    IPython.embed()
