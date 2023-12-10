from multi_token.modalities.vision_clip import (
    CLIPVisionModality,
    OUTPUT_LAYER as CLIP_POOL_LAYER,
)
from multi_token.modalities.imagebind import ImageBindModality
from multi_token.modalities.document_gte import DocumentGTEModality
from multi_token.modalities.audio_whisper import WhisperAudioModality
from multi_token.modalities.audio_clap import CLAPAudioModality
from multi_token.modalities.video_xclip import XCLIPVideoModality

MODALITY_BUILDERS = {
    "vision_clip": lambda: [CLIPVisionModality()],
    "vision_clip_pool": lambda: [
        CLIPVisionModality(feature_layer=CLIP_POOL_LAYER, num_tokens_output=10)
    ],
    "audio_whisper": lambda: [
        WhisperAudioModality(
            num_tokens_output=10, model_name_or_path="openai/whisper-small"
        )
    ],
    "audio_clap": lambda: [CLAPAudioModality(num_tokens_output=5)],
    "video_xclip": lambda: [XCLIPVideoModality(num_tokens_output=10)],
    "imagebind": lambda: [ImageBindModality()],
    "document_gte": lambda: [DocumentGTEModality()],
    "document_gte_x16": lambda: [DocumentGTEModality(num_tokens_output=32)],
}
