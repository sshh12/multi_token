# multi_token

> Embed arbitrary modalities (images, audio, documents, etc) into large language models.

This library is designed to be an extension of LLaVA for encoding ✨anything✨ (images, sounds, documents, videos, motion capture, screenshots, voice recordings, ...) into a format that can used in large language models. Its primary contribution is the ability to embed multiple instances and modalities into a single model and a framework for doing so fairly easily.

Potentially with this you could ask Large Multimodal Models (LMMs):

- > Read \<document\> and give me a summary.

- > Listen to \<audio\> and answer the spoke question.

- > Compare and contrast \<image\> and \<image\>

- > Given \<screenshot\> and \<game-state\>, what key should I press?

Interested in how this works? See this [blog post](https://blog.sshh.io/p/large-multimodal-models-lmms).

## Usage

```bash
git clone https://github.com/sshh12/multi_token \
        && cd multi_token \
        && pip install -r requirements.txt \
        && pip install -e .

pip install flash-attn --no-build-isolation
```

### Model Zoo

| Base Model                                                | Model | Modality | Notes |
| - | - | - | - |
| [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | [sshh12/Mistral-7B-LoRA-VisionCLIP-LLAVA](https://huggingface.co/sshh12/Mistral-7B-LoRA-VisionCLIP-LLAVA) | **Vision** <br/> <br/> Encode images as `<image>` and with `images`. | ⭐🖼️ A model pretrained and finetuned on the LLaVA dataset. This should be comparable to [BakLLaVA](https://github.com/SkunkworksAI/BakLLaVA) and [LLaVA 1.5](https://llava-vl.github.io/). <br/><br/> Compute: ~160 3090 Ti hours|
| [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | [sshh12/Mistral-7B-CLIP-LoRA-captions-only-demo](https://huggingface.co/sshh12/Mistral-7B-CLIP-LoRA-captions-only-demo) | **Vision** <br/> <br/> Encode images as `<image>` and with `images`. | ⚠️🖼️ This is a __very limited__ image model trained on only a few __caption-only__ examples for the sake of demonstrating a proof of concept. <br/><br/> Compute: ~10 3090 Ti hours |

### Vision (LLaVA-equivalent)

```
python scripts/serve_model.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
    --model_lora_path sshh12/Mistral-7B-LoRA-VisionCLIP-LLAVA \
    --load_bits 4 \
    --port 7860
```

```python
requests.post(
    "http://localhost:7860/generate",
    json={
        "messages": [{"role": "user", "content": "What are things I should be cautious about when I visit this place? <image>"}],
        "images": ["https://llava-vl.github.io/static/images/view.jpg"],
    },
).json()
# {'output': 'When visiting this place, which is a lake with a wooden dock, there are a few things to be cautious about. First, be aware of the water depth and the presence of any hidden obstacles, such as rocks or underwater debris, that could pose a risk to your safety. Second, be mindful of the weather conditions, as sudden changes in weather can make the water unpredictable and potentially dangerous. Lastly, be cautious of any wildlife or marine life in the area, as they may pose a threat to your safety or cause damage to the dock.'}
```

### Ideas

Some other ideas I want to try to implement at some point:
* **WhisperTranscriptModality**: encode voice audio so you can do things like `Answer the question asked in <audio>`. Given additional metadata like the speaker tone, accent, etc. this could be trained to personalize the response in a way that chat-based Q&A can't.
* **MultiImage Q&A**: train a version of VisionClipModality (aka LLaVA) that takes multiple images inputs and answers questions

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

    def preprocess_rows(self, row: List[Dict]) -> List[Optional[Any]]:
        # convert raw dataset rows into an arbitrary tensor to pass to `forward`

    @torch.no_grad()
    def forward(self, encoded_values: List[Any]) -> List[torch.Tensor]:
        # encode `preprocess_rows` output values into the format that will be fed into the projector
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

Use this command with standard huggingface training arguments:

```
deepspeed scripts/train_model.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
    --model_cls MistralLMMForCausalLM \
    --modality_builder vision_clip \
    --dataset_path /data/llava-chat-captions \
    --output_dir /data/output/my_lmm_pretrain \
    --pretrain_projectors \
    --lora_enable True \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --model_max_length 2048 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --report_to wandb \
    --deepspeed ./configs/zero2.json
```

The key arguments are:
* `--modality_builder`: the name of the modality builder to use (see `MODALITY_BUILDERS`)
* `--pretrain_projectors`: freeze the language model and only train the projectors
* `--model_cls`: the model class to use (this should match your base model)

### Finetuning

Use this command with standard huggingface training arguments:

```
deepspeed scripts/train_model.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
    --model_cls MistralLMMForCausalLM \
    --modality_builder vision_clip \
    --pretrained_projectors_path /data/output/my_lmm_pretrain/checkpoint-4000/non_lora_trainables.bin \
    --dataset_path /data/llava-chat-captions \
    --output_dir /data/output/my_lmm_pretrain \
    --pretrain_projectors \
    --lora_enable True \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --model_max_length 2048 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --report_to wandb \
    --deepspeed ./configs/zero2.json
```

The key arguments are:
* `--modality_builder`: the name of the modality builder to use (see `MODALITY_BUILDERS`)
* `--pretrained_projectors_path`: the path to the pretrained projectors (from the pretraining step)
* `--model_cls`: the model class to use (this should match your base model)

You can also omit `pretrained_projectors_path` to just train the full model from scratch. According to the LLaVA paper, this is not as good as training the projectors first (but it will work).

### Inference

Use the following to run a local flask server for inference:

```
python scripts/serve_model.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
    --model_lora_path /data/output/lmm_just_trained_folder \
    --port 7860
```

You can use this utility to upload your model to huggingface:

```
python scripts/upload_model.py \
    -r username/my-new-lmm \
    -m /data/output/lmm_just_trained_folder
```

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
* Training quantization (QLoRA)
* Efficient batch preprocessing
* Efficient batch projection
* Efficient batch collation (based on example lengths)
* Efficient batch inference
* Allow for non-`INST` based instruction formats and system tokens
* Support more base language models

## Development

### Windows Docker Dev

My local dev setup is Windows + WSL + Docker + 3090 Ti (24GB VRAM). `F:/` is configured to be a large data drive that I share among containers.

1. `docker build -t multi-token-dev .`
2. `docker run -it --gpus all -p 7860:7860 --mount type=bind,source=F:/docker-hf-cache,target=/root/.cache/huggingface --mount type=bind,source=F:/docker-data,target=/data --name multi-token-dev multi-token-dev`

### Vast.ai Dev

For some models, I'm using cheapish GPU instances on [vast.ai](https://cloud.vast.ai/).

1. `vastai create instance $ID --image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel --disk 512`
2. `ssh -p $PORT root@$HOST`
3. `curl -o- https://raw.githubusercontent.com/sshh12/multi_token/main/scripts/vastai_setup.sh | bash`

While training I run: `source ./scripts/vastai_sync.sh $INSTANCE_ID` to sync the output folder to my local machine.