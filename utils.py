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

def KSAMPLER_sample(*args, **kwargs):
    orig_fn = store.KSAMPLER_sample
    extra_args = None
    model_options = None
    try:
        extra_args = kwargs['extra_args'] if 'extra_args' in kwargs else args[3]
        model_options = extra_args['model_options']
    except Exception: ...
    if model_options is not None and 'tiled_diffusion' in model_options and extra_args is not None:
        sigmas_ = kwargs['sigmas'] if 'sigmas' in kwargs else args[2]
        sigmas_all = model_options.pop('sigmas', None)
        sigmas = sigmas_all if sigmas_all is not None else sigmas_
        store.sigmas = sigmas
        store.model_options = model_options
        store.extra_args = extra_args
    else:
        for attr in  ['sigmas', 'model_options', 'extra_args']:
            _delattr(store, attr)
    return orig_fn(*args, **kwargs)

def KSampler_sample(*args, **kwargs):
    orig_fn = store.KSampler_sample
    self = args[0]
    model_patcher = getattr(self, 'model', None)
    model_options = getattr(model_patcher, 'model_options', None)
    if model_options is not None and 'tiled_diffusion' in model_options:
        sigmas = None
        try: sigmas = kwargs['sigmas'] if 'sigmas' in kwargs else args[10]
        except Exception: ...
        if sigmas is None:
            sigmas = getattr(self, 'sigmas', None)
        if sigmas is not None:
            model_options = model_options.copy()
            model_options['sigmas'] = sigmas
            self.model.model_options = model_options
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

def _delattr(obj, attr):
    try:
        if hasattr(obj, attr): delattr(obj, attr)
    except Exception: ...

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
