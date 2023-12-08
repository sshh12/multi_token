from multi_token.modalities.vision_clip import (
    CLIPVisionModality,
    OUTPUT_LAYER as CLIP_POOL_LAYER,
)
from multi_token.modalities.imagebind import ImageBindModality
from multi_token.modalities.document_gte import DocumentGPTModality
from multi_token.modalities.audio_whisper import WhisperAudioModality

MODALITY_BUILDERS = {
    "vision_clip": lambda: [CLIPVisionModality()],
    "vision_clip_pool": lambda: [
        CLIPVisionModality(feature_layer=CLIP_POOL_LAYER, num_tokens_output=10)
    ],
    "audio_whisper": lambda: [WhisperAudioModality(num_tokens_output=10)],
    "imagebind": lambda: [ImageBindModality()],
    "document_gte": lambda: [DocumentGPTModality()],
    "document_gte_x16": lambda: [DocumentGPTModality(num_tokens_output=32)],
}
