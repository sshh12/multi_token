from multi_token.modalities.vision_clip import CLIPVisionModality
from multi_token.modalities.imagebind import ImageBindModality
from multi_token.modalities.document_gte import DocumentGPTModality

MODALITY_BUILDERS = {
    "vision_clip": lambda: [CLIPVisionModality()],
    "imagebind": lambda: [ImageBindModality()],
    "document_gte": lambda: [DocumentGPTModality()],
    "document_gte_x8": lambda: [DocumentGPTModality(num_tokens_output=64)],
}
