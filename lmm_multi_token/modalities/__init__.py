from lmm_multi_token.modalities.vision_clip import CLIPVisionModality

MODALITY_BUILDERS = {
    "vision_clip": lambda: [CLIPVisionModality()],
}
