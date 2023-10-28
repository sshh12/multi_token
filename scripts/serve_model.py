from dataclasses import dataclass, field
import logging

from flask import Flask, request, jsonify
import transformers
import torch

from multi_token.training import (
    ModelArguments,
)
from multi_token.inference import load_trained_lora_model
from multi_token.data_tools import encode_chat


@dataclass
class ServeArguments(ModelArguments):
    port: int = field(default=8080)
    host: str = field(default="0.0.0.0")
    load_bits: int = field(default=16)
    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.01)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser((ServeArguments,))

    serve_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    model, tokenizer = load_trained_lora_model(
        model_name_or_path=serve_args.model_name_or_path,
        model_lora_path=serve_args.model_lora_path,
        load_bits=serve_args.load_bits,
    )

    app = Flask(__name__)

    @app.route("/generate", methods=["POST"])
    def generate():
        req_json = request.get_json()

        encoded_dict = encode_chat(req_json, tokenizer, model.modalities)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=encoded_dict["input_ids"].unsqueeze(0).to(model.device),
                max_new_tokens=serve_args.max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=serve_args.temperature,
                modality_inputs={
                    m.name: [encoded_dict[m.name]] for m in model.modalities
                },
            )

        outputs = tokenizer.decode(
            output_ids[0, encoded_dict["input_ids"].shape[0] :],
            skip_special_tokens=True,
        ).strip()

        return jsonify({"output": outputs})

    app.run(host=serve_args.host, port=serve_args.port)
