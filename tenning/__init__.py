import sys
import os
from pkg_resources import get_distribution, DistributionNotFound
from . import utils


def get_version():
    """ Returns the version of the package defined in setup.py.

    Raises:
        DistributionNotFound: It is raised when the package is not installed yet.

    Returns:
        The package version.
    """
    package_name = os.path.basename(os.path.dirname(__file__))

    try:
        _dist = get_distribution(package_name)
        # Normalize case for Windows systems
        dist_loc = os.path.normcase(_dist.location)
        here = os.path.normcase(__file__)
        if not here.startswith(os.path.join(dist_loc, package_name)):
            # not installed, but there is another version that *is*
            raise DistributionNotFound
    except DistributionNotFound:
        return 'Please install this project with setup.py'
    else:
        return _dist.version


__version__ = get_version()


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
