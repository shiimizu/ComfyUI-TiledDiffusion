from .tiled_diffusion import NODE_CLASS_MAPPINGS as TD_NCM, NODE_DISPLAY_NAME_MAPPINGS as TD_NDCM
from .tiled_vae import NODE_CLASS_MAPPINGS as TV_NCM, NODE_DISPLAY_NAME_MAPPINGS as TV_NDCM
from .utils import store as _
NODE_CLASS_MAPPINGS = {**TD_NCM, **TV_NCM}
NODE_DISPLAY_NAME_MAPPINGS = {**TD_NDCM, **TV_NDCM}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']