from multi_token.modalities.vision_clip import CLIPVisionModality
from multi_token.modalities.imagebind import ImageBindModality

MODALITY_BUILDERS = {
    "vision_clip": lambda: [CLIPVisionModality()],
    "imagebind": lambda: [ImageBindModality()],
}
