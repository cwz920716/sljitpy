from collections import namedtuple
from ctypes import CFUNCTYPE, c_double, c_int
from enum import Enum

import llvmlite.ir as ir
import llvmlite.binding as llvm
import llvmlite.llvmpy.core as lc
import llvmlite.llvmpy.passes as lp

from . import cgutils 

class LLVMCodeGenerator(object):
    def __init__(self):
        self.module = ir.Module()
        # Current IR builder.
        self.builder = None
        self.func = None
        self.func_symtab = {}

    def _emit_prototype(self, name):
        # emit the prototype
        func_name = name
        func_ty = ir.FunctionType(cgutils.int32_t, [])
        func = ir.Function(self.module, func_ty, func_name)
        return func

    def _create_entry_block_alloca(self, name, ty):
        """Create an alloca in the entry BB of the current function."""
        builder = ir.IRBuilder()
        builder.position_at_start(self.builder.function.entry_basic_block)
        return builder.alloca(ty, size=None, name=name)

    def emit_enter(self, name):
        self.func_symtab = {}
        self.func = self._emit_prototype(name)
        bb_entry = self.func.append_basic_block('entry')
        self.builder = ir.IRBuilder(bb_entry)
        return

    def emit_exit(self, retval):
        self.builder.ret(retval)
        return
        

class LirEvaluator(object):
    def __init__(self):
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

        self.codegen = LLVMCodeGenerator()
        self.target = llvm.Target.from_default_triple()

    def jit(self, name, optimize=True, llvmdump=True):
        if llvmdump:
            print('======== Unoptimized LLVM IR')
            print(str(self.codegen.module))
        llvmmod = llvm.parse_assembly(str(self.codegen.module))
        target_machine = self.target.create_target_machine(opt=3)
        if optimize:
            pmb = lp.create_pass_manager_builder(opt=3)
            pm = llvm.create_module_pass_manager()
            pmb.populate(pm)
            pm.run(llvmmod)
            if llvmdump:
                print('======== Optimized LLVM IR')
                print(str(llvmmod))
        with llvm.create_mcjit_compiler(llvmmod, target_machine) as ee:
            ee.finalize_object()
            fptr = CFUNCTYPE(c_int)(ee.get_function_address(name))
            result = fptr()
            return result   
