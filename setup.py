try:
    # Try to use setuptools so as to enable support of the special
    # "Microsoft Visual C++ Compiler for Python 2.7" (http://aka.ms/vcpython27)
    # for building under Windows.
    # Note setuptools >= 6.0 is required for this.
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from distutils.command import build
from distutils.spawn import spawn
import sys
import os

import versioneer

CFLAGS = []

module1 = Extension('sljit.spam',
                    include_dirs = ['/usr/local/include'],
                    sources = ['sljit/spam.c'],
                            extra_compile_args=CFLAGS,
                            depends=['sljit/_pymodule.h'])


def find_packages(root_dir, root_name):
    """
    Recursively find packages in *root_dir*.
    """
    packages = []
    def rec(path, pkg_name):
        packages.append(pkg_name)
        for fn in sorted(os.listdir(path)):
            subpath = os.path.join(path, fn)
            if os.path.exists(os.path.join(subpath, "__init__.py")):
                subname = "%s.%s" % (pkg_name, fn)
                rec(subpath, subname)
    rec(root_dir, root_name)
    return packages

packages = find_packages("sljit", "sljit")

setup (name = 'sljit',
       version = '0.01',
       description = 'This is a demo package',
       author = 'Wenzhi Cui',
       author_email = 'wc8348@cs.utexas.edu',
       url = 'https://docs.python.org/extending/building',
       packages=packages,
       long_description = '''
This is really just a demo package.
''',
       ext_modules = [module1])
