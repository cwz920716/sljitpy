from setuptools import setup, Extension

CFLAGS = []

module1 = Extension('sljit.spam',
                    include_dirs = ['/usr/local/include'],
                    sources = ['sljit/spam.c'],
                            extra_compile_args=CFLAGS,
                            depends=['sljit/_pymodule.h'])

setup (name = 'sljit',
       version = '0.01',
       description = 'This is a demo package',
       author = 'Wenzhi Cui',
       author_email = 'wc8348@cs.utexas.edu',
       url = 'https://docs.python.org/extending/building',
       long_description = '''
This is really just a demo package.
''',
       ext_modules = [module1])
