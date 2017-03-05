from collections import namedtuple
from ctypes import CFUNCTYPE, c_double
from enum import Enum

import llvmlite.ir as ir
import llvmlite.binding as llvm

class LLVMCodeGenerator(object):
    def __init__(self):
        self.module = ir.Module()
        # Current IR builder.
        self.builder = None

    def emit_enter(self):
        # emit the prototype

class LirEvaluator(object):
    def __init__(self):
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

        self.codegen = LLVMCodeGenerator()
