class Store:
    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

store = Store()

# ==================== Hook into sampling functions for ControlNet ====================

import comfy.samplers

def KSampler_sample(*args, **kwargs):
    orig_fn = store.KSampler_sample
    start_step = get_value_from_args(orig_fn, args, kwargs, 'start_step', 6)
    if start_step is not None:
        store.start_step = start_step
    return orig_fn(*args, **kwargs)

def KSAMPLER_sample(*args, **kwargs):
    orig_fn = store.KSAMPLER_sample
    store.sigmas = get_value_from_args(orig_fn, args, kwargs, 'sigmas', 2)
    store.extra_args = get_value_from_args(orig_fn, args, kwargs, 'extra_args', 3)
    store.model_options = store.extra_args['model_options']
    return orig_fn(*args, **kwargs)

def get_area_and_mult(*args, **kwargs):
    conds = kwargs['conds'] if 'conds' in kwargs else args[0]
    if (model_options:=getattr(store, 'model_options', None)) is not None and 'tiled_diffusion' in model_options:
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
    return store.get_area_and_mult(*args, **kwargs)

def get_value_from_args(fn, args, kwargs, key_to_lookup, idx=None):
    value = None
    if key_to_lookup in kwargs:
        value = kwargs[key_to_lookup]
    else:
        try:
            # Get its position in the formal parameters list and retrieve from args
            arg_names = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            index = arg_names.index(key_to_lookup)
            value = args[index] if index < len(args) else None
        except Exception:
            if idx is not None and idx < len(args):
                value = args[idx]
    return value

def register_hooks():
    patches = [
        (comfy.samplers.KSampler, 'sample', KSampler_sample),
        (comfy.samplers.KSAMPLER, 'sample', KSAMPLER_sample),
        (comfy.samplers, 'get_area_and_mult', get_area_and_mult),
    ]
    for parent, fn_name, fn_patch in patches:
        if not hasattr(parent, f"_{fn_name}"):
            setattr(store, f"_{fn_name}", getattr(parent, fn_name))
        setattr(store, fn_patch.__name__, getattr(parent, fn_name))
        setattr(parent, fn_name, fn_patch)

register_hooks()

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

from math import sqrt
import torch.nn.functional as F
import comfy_extras.nodes_sag
from comfy_extras.nodes_sag import gaussian_blur_2d 

def calc_closest_factors(a):
    for b in range(int(sqrt(a)), 0, -1):
        if a % b == 0:
            c = a // b
            return (b, c)

def create_blur_map(x0, attn, sigma=3.0, threshold=1.0):
    # reshape and GAP the attention map
    _, hw1, hw2 = attn.shape
    b, _, lh, lw = x0.shape
    attn = attn.reshape(b, -1, hw1, hw2)
    # Global Average Pool
    mask = attn.mean(1, keepdim=False).sum(1, keepdim=False) > threshold
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
