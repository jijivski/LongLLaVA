import os
from .clip_encoder import CLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):

    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    print(f'using vision_tower:{vision_tower}')
    # vision_tower='/nesa_data/remote_shome/zch/workspace/LongLLaVA/models/clip_vit_large_patch14_336'
    is_absolute_path_exists = os.path.exists(vision_tower)
    # print(f'vision_tower_cfg:{vision_tower_cfg}')
    # print(f'Attention here vision_tower is changed mutually! vision_tower:{vision_tower},exists:{is_absolute_path_exists}')
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
