# multi_token

> Embed arbitrary modalities (images, audio, documents, etc) into large language models.

This library is designed to be an extension of LLaVA for encoding ✨anything✨ (images, sounds, documents, videos, motion capture, screenshots, voice recordings, ...) into a format that can used in large language models. It's primary contribution is the ability to embed multiple instances and modalities into a single model and a framework for doing so fairly easily.

## Usage

TODO

## Training

### Add a Modality

You can do this by implementing an instance of `multi_token.modalities.base_modality.Modality` (see [CLIP for vision example](https://github.com/sshh12/multi_token/blob/main/multi_token/modalities/vision_clip.py)).

<details>
<summary>See annotated example</summary>

```python
class MyModality(Modality):
    def __init__(
        self,
    ):
        # ...

    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        # a pytorch module that converts a preprocessed item (after `forward`) into a tensor `(batch size x token width x lm_hidden_size)`

    @property
    def name(self) -> str:
        # the name/ID for this modality
        return "my_modality"

    @property
    def token(self) -> str:
        # the token you'll use in text to represent this
        return "<my-modality>"

    @property
    def data_key(self) -> str:
        # the key in your dataset rows for raw instances of this
        return "my_modality_items"

    @property
    def token_width(self) -> int:
        # how many tokens should we use to present instances of this?
        # too small and it's not descriptive enough, too large and you are using up the context window
        return 1

    def preprocess_row(self, row: Dict) -> Optional[torch.Tensor]:
        # convert raw dataset rows into an arbitrary tensor to pass to `forward`

    @torch.no_grad()
    def forward(self, encoded_values: List[torch.Tensor]) -> List[torch.Tensor]:
        # encode `preprocess_row` output values into the format that will be fed into the projector
```

</details>

Register this new modality by adding it to `multi_token.modalities.MODALITY_BUILDERS`.

```python
MODALITY_BUILDERS = {
    ...,
    "my_modality": lambda: [MyModality()],
}
```

### Dataset

You can see some of the existing [scripts](https://github.com/sshh12/multi_token/tree/main/scripts) for putting things into the correct dataset format.

Schema:
```javascript
// LLaVA/CLIP example
{
    "id": "arbitrary-id-123",
    "images": ["/path/to/image.png"],
    "messages": [{"role": "user", "content": "Describe <image>"}, {"role": "assistant", "content": "This is a potato."}],
}

// Custom
{
    "id": "arbitrary-id-123",
    "my_modality_items": ["/path/to/data OR just the full document"],
    "messages": [{"role": "user", "content": "Describe <my-modality>"}, {"role": "assistant", "content": "This is ..."}],
}
```

Then save with `dataset.save_to_disk(output_folder)`.

### Pretraining

TODO

### Finetuning

TODO

### Inference

TODO

## Comparision to LLaVA

> LLaVA: Large Language and Vision Assistant
>
> [[Project Page](https://llava-vl.github.io/)] [[Demo](https://llava.hliu.cc/)]  [[Data](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)] [[Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)]
> 
> **Improved Baselines with Visual Instruction Tuning** [[Paper](https://arxiv.org/abs/2310.03744)] <br>
> [Haotian Liu](https://hliu.cc), [Chunyuan Li](https://chunyuan.li/), [Yuheng Li](https://yuheng-li.github.io/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)
> 
> **Visual Instruction Tuning** (NeurIPS 2023, **Oral**) [[Paper](https://arxiv.org/abs/2304.08485)]<br>
> [Haotian Liu*](https://hliu.cc), [Chunyuan Li*](https://chunyuan.li/), [Qingyang Wu](https://scholar.google.ca/citations?user=HDiw-TsAAAAJ&hl=en/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) (*Equal Contribution)

The inspiration and much of the source code for this project comes from the original [LLaVA](https://github.com/haotian-liu/LLaVA/) implementation (apache 2.0). 

### Core Differences

* This library is designed to be more modular for adding custom encoders/projectors. In some areas, the LLaVA implementation was simplified (e.g. stripped out a lot of the eval, preprocessing code, and non-LLAMA parts) and in others more complex (handling multiple types of modalities).
* The tokenization and injection of projected encodings into the language model's token space are written from scratch, but, _in theory_, do the exact same thing. I like to think this project's `prepare_inputs_labels_for_multimodal` is a bit easier to grok and manipulate than the original.
* You can use multiple instances of tokens from the same or different modalities (where as LLaVA was only for a single image). For example, `Given <image> and <image>, answer the question asked in <audio>`. 

If one were to train a model using this library with the same base model and projection config as LLaVA-1.5, I would expect nearly identical performance (barring any bugs in this implementation).

## TODOs

* Multi-GPU support
* Full (non-LoRA training)
* Quantization (qLoRA)
* Efficient batch preprocessing
* Efficient batch projection
* Efficient batch collation (based on dataset lengths)
* Efficient batch inference
* Allow for non-INST based instruction formats
* Support more base language models

## Windows Docker Dev

My local dev setup is Windows + WSL + Docker + 3090 Ti. `F:/` is configured to be a large data drive that I share among containers.

1. `docker build -t multi-token-dev .`
2. `docker run -it --gpus all -p 7860:7860 --mount type=bind,source=F:/docker-hf-cache,target=/root/.cache/huggingface --mount type=bind,source=F:/docker-data,target=/data --name multi-token-dev multi-token-dev`