store = {}

# ==================== Hook into sampling functions for ControlNet ====================

import comfy.samplers

def patch1(fn_name):
    def calc_cond_batch(*args, **kwargs):
        x_in = kwargs['x_in'] if 'x_in' in kwargs else args[2]
        model_options = kwargs['model_options'] if 'model_options' in kwargs else args[4]
        if not hasattr(x_in, 'model_options'):
            x_in.model_options = model_options
        return store[fn_name](*args, **kwargs)
    return calc_cond_batch

def patch2(fn_name):
    def get_area_and_mult(*args, **kwargs):
        x_in = kwargs['x_in'] if 'x_in' in kwargs else args[1]
        conds = kwargs['conds'] if 'conds' in kwargs else args[0]
        if (model_options:=getattr(x_in, 'model_options', None)) is not None and 'tiled_diffusion' in model_options:
            if 'control' in conds:
                control = conds['control']
                if not hasattr(control, 'get_control_orig'):
                    control.get_control_orig = control.get_control
                control.get_control = lambda *a, **kw: control
        else:
            if 'control' in conds:
                control = conds['control']
                if hasattr(control, 'get_control_orig') and control.get_control != control.get_control_orig:
                    control.get_control = control.get_control_orig
        return store[fn_name](*args, **kwargs)
    return get_area_and_mult

patches = [
    (comfy.samplers, 'calc_cond_batch', patch1),
    (comfy.samplers, 'get_area_and_mult', patch2),
]
for parent, fn_name, create_patch in patches:
    store[fn_name] = getattr(parent, fn_name)
    setattr(parent, fn_name, create_patch(fn_name))

# ==================== Patch pre_run_control ====================

# Is this necessary anymore?
def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        if 'control' in x:
            try: x['control'].cleanup()
            except Exception: ...
            x['control'].pre_run(model, percent_to_timestep_function)
comfy.samplers.pre_run_control = pre_run_control

# ==================== Patch SAG ====================

import math
import torch.nn.functional as F
import comfy_extras.nodes_sag
from comfy_extras.nodes_sag import gaussian_blur_2d 
def create_blur_map(x0, attn, sigma=3.0, threshold=1.0):
    # reshape and GAP the attention map
    _, hw1, hw2 = attn.shape
    b, _, lh, lw = x0.shape
    attn = attn.reshape(b, -1, hw1, hw2)
    # Global Average Pool
    mask = attn.mean(1, keepdim=False).sum(1, keepdim=False) > threshold
    def calc_closest_factors(a):
        for b in range(int(math.sqrt(a)), 0, -1):
            if a % b == 0:
                c = a // b
                return (b,c)
    m = calc_closest_factors(hw1)
    mh = max(m) if lh > lw else min(m)
    mw = m[1] if mh == m[0] else m[0]
    mid_shape = mh, mw

    # Reshape
    mask = (
        mask.reshape(b, *mid_shape)
        .unsqueeze(1)
        .type(attn.dtype)
    )
    # Upsample
    mask = F.interpolate(mask, (lh, lw))

    blurred = gaussian_blur_2d(x0, kernel_size=9, sigma=sigma)
    blurred = blurred * mask + x0 * (1 - mask)
    return blurred
comfy_extras.nodes_sag.create_blur_map = create_blur_map

# ==================== Patch Gligen ====================

def _set_position(self, boxes, masks, positive_embeddings):
    objs = self.position_net(boxes, masks, positive_embeddings)
    def func(x, extra_options):
        key = extra_options["transformer_index"]
        module = self.module_list[key]
        nonlocal objs
        _objs = objs.repeat(-(x.shape[0] // -objs.shape[0]),1,1) if x.shape[0] > objs.shape[0] else objs
        return module(x, _objs.to(device=x.device, dtype=x.dtype))
    return func

import comfy.gligen
comfy.gligen.Gligen._set_position = _set_position
