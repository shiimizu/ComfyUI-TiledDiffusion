import comfy.samplers
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
    # this function should only be run by us
    orig_key = f"calc_cond_uncond_batch_original_tiled_diffusion_{gen_id()}"
    if not hasattr(comfy.samplers, orig_key):
        setattr(comfy.samplers, orig_key, comfy.samplers.calc_cond_uncond_batch)
    payload = [{
        "target_line": 'control.get_control',
        "mode": "replace",
        "code_to_insert": """control if 'tiled_diffusion' in model_options else control.get_control"""
    },
    {
        "target_line": 'calc_cond_uncond_batch',
        "dedent": False,
        "code_to_insert": f"""
    if 'tiled_diffusion' not in model_options:
        return {orig_key}(model, cond, uncond, x_in, timestep, model_options)"""
    }]
    fn = inject_code(comfy.samplers.calc_cond_uncond_batch, payload)
    setattr(comfy.samplers, 'calc_cond_uncond_batch', fn)

def hook_all():
    hook_calc_cond_uncond_batch()

def inject_code(original_func, data):
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

        if target_line_number is None:
            raise FileNotFoundError
            # Target line not found, return the original function
            # return original_func

        # Insert the code to be injected after the target line
        if item.get("mode","insert") == "insert":
            lines.insert(target_line_number, code_to_insert)

    # Recreate the modified source code
    modified_source = "\n".join(lines)
    modified_source = dedent(modified_source.strip("\n"))

    # Write the modified source code to a temporary file so the
    # source code and stack traces can still be viewed when debugging.
    custom_name = ".patches.py"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_file_path = os.path.join(current_dir, custom_name)
    with open(temp_file_path, 'w') as temp_file:
        temp_file.write(modified_source)
        temp_file.flush()

        MODULE_PATH = temp_file.name
        MODULE_NAME = __name__.split('.')[0] + "_patch_modules"
        spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Retrieve the modified function from the module
        modified_function = getattr(module, original_func.__name__)

        # Adapted from https://stackoverflow.com/a/49077211
        def copy_func(f, globals=None, module=None, code=None):
            if globals is None:
                globals = f.__globals__
            g = types.FunctionType(f.__code__ if code is None else code, globals, name=f.__name__,
                                argdefs=f.__defaults__, closure=f.__closure__)
            g = functools.update_wrapper(g, f)
            if module is not None:
                g.__module__ = module
            g.__kwdefaults__ = copy(f.__kwdefaults__)
            return g
        
        modified_function = copy_func(original_func, code=modified_function.__code__)

    return modified_function
