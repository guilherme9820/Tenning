from . import utils
import sys
import os


def get_module_dir(module):
    return os.path.abspath(os.path.join(os.path.dirname(module.__file__)))


_current_module = sys.modules[__name__]

_current_module.__path__ = ([get_module_dir(utils)] + _current_module.__path__)
setattr(_current_module, "utils", utils)

# Creates an alias for some modules
generic_utils = utils.generic_utils
data_utils = utils.data_utils
rotation_utils = utils.rotation_utils
linalg_utils = utils.linalg_utils
setattr(_current_module, "generic_utils", generic_utils)
setattr(_current_module, "data_utils", data_utils)
setattr(_current_module, "rotation_utils", rotation_utils)
setattr(_current_module, "linalg_utils", linalg_utils)
