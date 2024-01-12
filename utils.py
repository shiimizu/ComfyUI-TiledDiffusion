import inspect
import importlib
from textwrap import dedent, indent
from copy import copy
import types
import functools
import os
import sys
import binascii

def gen_id():
    return binascii.hexlify(os.urandom(1024))[64:72].decode("utf-8")

def hook_calc_cond_uncond_batch():
    import comfy.samplers
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
    fn = inject_code(comfy.samplers.calc_cond_uncond_batch, payload)
    for m in sys.modules.keys():
        if 'samplers' in m and 'extra_samplers' not in m:
            if not hasattr(sys.modules[m], orig_key):
                calc_cond_uncond_batch = getattr(sys.modules[m], 'calc_cond_uncond_batch')
                setattr(sys.modules[m], orig_key, calc_cond_uncond_batch)
            setattr(sys.modules[m], 'calc_cond_uncond_batch', fn)

def hook_sag_create_blur_map():
    imported = False
    try:
        import comfy_extras
        if hasattr(comfy_extras, 'nodes_sag'):
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
    for m in sys.modules.keys():
        if 'nodes_sag' in m:
            setattr(sys.modules[m], 'create_blur_map', fn)


def hook_all():
    hook_calc_cond_uncond_batch()
    hook_sag_create_blur_map()

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

