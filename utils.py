import inspect
import importlib
from textwrap import dedent, indent
from copy import copy
import types
import functools
import os
import sys
import binascii
from collections import namedtuple
from typing import List

Hook = namedtuple('Hook', ['fn', 'orig_key', 'module_name', 'module_name_nt', 'module_name_unix'])

def gen_id():
    return binascii.hexlify(os.urandom(1024))[64:72].decode("utf-8")

def hook_calc_cond_uncond_batch():
    from comfy.samplers import calc_cond_uncond_batch
    # this function should only be run by us
    orig_key = f"calc_cond_uncond_batch_original_tiled_diffusion_{gen_id()}"
    payload = [{
        "mode": "replace",
        "target_line": 'control.get_control',
        "code_to_insert": """control if 'tiled_diffusion' in model_options else control.get_control"""
    },
    {
        "dedent": False,
        "target_line": 'calc_cond_uncond_batch',
        "code_to_insert": f"""
    if 'tiled_diffusion' not in model_options:
        return {orig_key}(model, cond, uncond, x_in, timestep, model_options)"""
    }]
    fn = inject_code(calc_cond_uncond_batch, payload)
    return create_hook(fn, 'comfy.samplers', orig_key)

def hook_sag_create_blur_map():
    imported = False
    try:
        import comfy_extras
        from comfy_extras import nodes_sag
        imported = True
    except: ...
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
    fn = write_to_file_and_return_fn(nodes_sag.create_blur_map, modified_source, 'a')
    return create_hook(fn, 'comfy_extras.nodes_sag')

def hook_samplers_pre_run_control():
    from comfy.samplers import pre_run_control
    payload = [{
        "dedent": False,
        "target_line": "if 'control' in x:",
        "code_to_insert": """    try: x['control'].cleanup()\n    except: ..."""
    }]
    fn = inject_code(pre_run_control, payload, 'a')
    return create_hook(fn, 'comfy.samplers')

def create_hook(fn, module_name, orig_key = None):
    if orig_key is None: orig_key = f'{fn.__name__}_original'
    module_name_nt = '\\'.join(module_name.split('.'))
    module_name_unix = '/'.join(module_name.split('.'))
    return Hook(fn, orig_key, module_name, module_name_nt, module_name_unix)


def hook_all():
    hooks: List[Hook] = [
        hook_calc_cond_uncond_batch(),
        hook_sag_create_blur_map(),
        hook_samplers_pre_run_control(),
    ]

    for m in sys.modules.keys():
        for hook in hooks:
            if hook.module_name == m or (os.name != 'nt' and m.endswith(hook.module_name_unix)) or (os.name == 'nt' and m.endswith(hook.module_name_nt)):
                if hasattr(sys.modules[m], hook.fn.__name__):
                    if not hasattr(sys.modules[m], hook.orig_key):
                        if (orig_fn:=getattr(sys.modules[m], hook.fn.__name__, None)) is not None:
                            setattr(sys.modules[m], hook.orig_key, orig_fn)
                    setattr(sys.modules[m], hook.fn.__name__, hook.fn)


def inject_code(original_func, data, mode='w'):
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

def write_to_file_and_return_fn(original_func, source:str,mode):
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

