# multi_token


## Comparision to LLAVA

> LLaVA: Large Language and Vision Assistant
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

If one were to train a model using this library with the same base model, encoder, and training config as LLaVA-1.5, I would expect nearly identical SOTA performance (barring any bugs in either implementation).

## TODOs

* Multi-GPU support
* Full (non-LoRA training)
* Quantization (qLoRa)
* Efficient batch preprocessing
* Efficient batch projection
* Efficient batch collation (based on dataset lengths)
* Efficient batch inference
* Allow for non-INST based instruction formats
* Support more base language models
