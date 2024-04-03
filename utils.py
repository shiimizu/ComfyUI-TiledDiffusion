import inspect
import importlib
from textwrap import dedent, indent
from copy import copy
import types
import functools
import os
import sys
import binascii
from typing import List, NamedTuple

class Hook(NamedTuple):
    fn: object
    module_name: str
    target: str
    orig_key: str
    module_name_path: str

def gen_id():
    return binascii.hexlify(os.urandom(1024))[64:72].decode("utf-8")

def hook_calc_cond_uncond_batch():
    try:
        from comfy.samplers import calc_cond_batch
        calc_cond_batch_ = calc_cond_batch
    except Exception:
        from comfy.samplers import calc_cond_uncond_batch
        calc_cond_batch_ = calc_cond_uncond_batch
    # this function should only be run by us
    orig_key = f"{calc_cond_batch_.__name__}_original_tiled_diffusion_{gen_id()}"
    payload = [{
        "mode": "replace",
        "target_line": 'control.get_control',
        "code_to_insert": """control if 'tiled_diffusion' in model_options else control.get_control"""
    },
    {
        "dedent": False,
        "target_line": calc_cond_batch_.__name__,
        "code_to_insert": f"""
    if 'tiled_diffusion' not in model_options:
        return {orig_key}{inspect.signature(calc_cond_batch_)}"""
    }]
    fn = inject_code(calc_cond_batch_, payload, 'w')
    return create_hook(fn, 'comfy.samplers', orig_key=orig_key)

def hook_sag_create_blur_map():
    imported = False
    try:
        import comfy_extras
        from comfy_extras import nodes_sag
        imported = True
    except Exception: ...
    if not imported: return
    import comfy_extras
    from comfy_extras import nodes_sag
    import re
    source=inspect.getsource(nodes_sag.create_blur_map)
    replace_str="""
    def calc_closest_factors(a):
        for b in range(int(math.sqrt(a)), 0, -1):
            if a%b == 0:
                c = a // b
                return (b,c)
    m = calc_closest_factors(hw1)
    mh = max(m) if lh > lw else min(m)
    mw = m[1] if mh == m[0] else m[0]
    mid_shape = mh, mw"""
    modified_source = re.sub(r"ratio =.*\s+mid_shape =.*", replace_str, source, flags=re.MULTILINE)
    fn = write_to_file_and_return_fn(nodes_sag.create_blur_map, modified_source)
    return create_hook(fn, 'comfy_extras.nodes_sag')

def hook_samplers_pre_run_control():
    from comfy.samplers import pre_run_control
    payload = [{
        "dedent": False,
        "target_line": "if 'control' in x:",
        "code_to_insert": """    try: x['control'].cleanup()\n    except Exception: ..."""
    },  
    {
    "target_line": "s = model.model_sampling",
    "code_to_insert": """
    def find_outer_instance(target:str, target_type):
        import inspect
        frame = inspect.currentframe()
        i = 0
        while frame and i < 7:
            if (found:=frame.f_locals.get(target, None)) is not None:
                if isinstance(found, target_type):
                    return found
            frame = frame.f_back
            i += 1
        return None
    from comfy.model_patcher import ModelPatcher
    if (_model:=find_outer_instance('model', ModelPatcher)) is not None:
        if (model_function_wrapper:=_model.model_options.get('model_function_wrapper', None)) is not None:
            import sys
            tiled_diffusion = sys.modules.get('ComfyUI-TiledDiffusion.tiled_diffusion', None)
            if tiled_diffusion is None:
                for key in sys.modules:
                    if 'tiled_diffusion' in key:
                        tiled_diffusion = sys.modules[key]
                        break
            if (AbstractDiffusion:=getattr(tiled_diffusion, 'AbstractDiffusion', None)) is not None:
                if isinstance(model_function_wrapper, AbstractDiffusion):
                    model_function_wrapper.reset()
    """}]
    fn = inject_code(pre_run_control, payload)
    return create_hook(fn, 'comfy.samplers')

def hook_gligen__set_position():
    from comfy.gligen import Gligen
    source=inspect.getsource(Gligen._set_position)
    replace_str="""
            nonlocal objs
            if x.shape[0] > objs.shape[0]:
                _objs = objs.repeat(-(x.shape[0] // -objs.shape[0]),1,1)
            else:
                _objs = objs
            return module(x, _objs)"""
    modified_source = dedent(source.replace("    return module(x, objs)", replace_str, 1))
    fn = write_to_file_and_return_fn(Gligen._set_position, modified_source)
    return create_hook(fn, 'comfy.gligen', 'Gligen._set_position')

def create_hook(fn, module_name:str, target = None, orig_key = None):
    if target is None: target = fn.__name__
    if orig_key is None: orig_key = f'{target}_original'
    module_name_path = os.path.normpath(module_name.replace('.', '/'))
    return Hook(fn, module_name, target, orig_key, module_name_path)

def _getattr(obj, name:str, default=None):
    """multi-level getattr"""
    for attr in name.split('.'):
        obj = getattr(obj, attr, default)
    return obj

def _hasattr(obj, name:str):
    """multi-level hasattr"""
    return _getattr(obj, name) is not None

def _setattr(obj, name:str, value=None):
    """multi-level setattr"""
    split = name.split('.')
    if not split[:-1]:
        return setattr(obj, name, value)
    else:
        name = split[-1]
        for attr in split[:-1]:
            obj = getattr(obj, attr, None)
        return setattr(obj, name, value)

def hook_all(restore=False, hooks=None):
    if hooks is None:
        hooks: List[Hook] = [
            hook_calc_cond_uncond_batch(),
            hook_sag_create_blur_map(),
            hook_samplers_pre_run_control(),
            hook_gligen__set_position(),
        ]
    for key, module in sys.modules.items():
        for hook in hooks:
            if key == hook.module_name or key.endswith(hook.module_name_path):
                if _hasattr(module, hook.target):
                    if not _hasattr(module, hook.orig_key):
                        if (orig_fn:=_getattr(module, hook.target, None)) is not None:
                            _setattr(module, hook.orig_key, orig_fn)
                    if restore:
                        _setattr(module, hook.target, _getattr(module, hook.orig_key, None))
                    else:
                        _setattr(module, hook.target, hook.fn)

def inject_code(original_func, data, mode='a'):
    # Get the source code of the original function
    original_source = inspect.getsource(original_func)

    # Split the source code into lines
    lines = original_source.split("\n")

    for item in data:
        # Find the line number of the target line
        target_line_number = None
        for i, line in enumerate(lines):
            if item['target_line'] not in line: continue
            target_line_number = i + 1
            if item.get("mode","insert") == "replace":
                lines[i] = lines[i].replace(item['target_line'], item['code_to_insert'])
                break

            # Find the indentation of the line where the new code will be inserted
            indentation = ''
            for char in line:
                if char == ' ':
                    indentation += char
                else:
                    break
            
            # Indent the new code to match the original
            code_to_insert = item['code_to_insert']
            if item.get("dedent",True):
                code_to_insert = dedent(item['code_to_insert'])
            code_to_insert = indent(code_to_insert, indentation)

            break

        # Insert the code to be injected after the target line
        if item.get("mode","insert") == "insert" and target_line_number is not None:
            lines.insert(target_line_number, code_to_insert)

    # Recreate the modified source code
    modified_source = "\n".join(lines)
    modified_source = dedent(modified_source.strip("\n"))
    return write_to_file_and_return_fn(original_func, modified_source, mode)

def write_to_file_and_return_fn(original_func, source:str, mode='a'):
    # Write the modified source code to a temporary file so the
    # source code and stack traces can still be viewed when debugging.
    custom_name = ".patches.py"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_file_path = os.path.join(current_dir, custom_name)
    with open(temp_file_path, mode) as temp_file:
        temp_file.write(source)
        temp_file.write("\n")
        temp_file.flush()

        MODULE_PATH = temp_file.name
        MODULE_NAME = __name__.split('.')[0].replace('-','_') + "_patch_modules"
        spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Retrieve the modified function from the module
        modified_function = getattr(module, original_func.__name__)

    # Adapted from https://stackoverflow.com/a/49077211
    def copy_func(f, globals=None, module=None, code=None, update_wrapper=True):
        if globals is None: globals = f.__globals__
        if code is None: code = f.__code__
        g = types.FunctionType(code, globals, name=f.__name__,
                            argdefs=f.__defaults__, closure=f.__closure__)
        if update_wrapper: g = functools.update_wrapper(g, f)
        if module is not None: g.__module__ = module
        g.__kwdefaults__ = copy(f.__kwdefaults__)
        return g
        
    return copy_func(original_func, code=modified_function.__code__, update_wrapper=False)

