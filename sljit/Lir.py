from collections import namedtuple
from ctypes import CFUNCTYPE, c_double, c_int
from enum import Enum

import llvmlite.ir as ir
import llvmlite.binding as llvm
import llvmlite.llvmpy.core as lc
import llvmlite.llvmpy.passes as lp

from . import cgutils 

def char(i):
    return ir.Constant(cgutils.int8_t, ord(i))

def int32(i):
    return ir.Constant(cgutils.int32_t, i)

def int64(i):
    return ir.Constant(cgutils.int64_t, i)

class LLVMCodeGenerator(object):
    def __init__(self):
        self.module = lc.Module()
        # Current IR builder.
        self.builder = None
        self.func = None
        self.func_symtab = {}

    def _emit_prototype(self, name):
        # emit the prototype
        func_name = name
        func_ty = ir.FunctionType(cgutils.int32_t, [cgutils.int32_t])
        func = ir.Function(self.module, func_ty, func_name)
        return func

    def _create_entry_block_alloca(self, name, ty):
        """Create an alloca in the entry BB of the current function."""
        builder = ir.IRBuilder()
        builder.position_at_start(self.builder.function.entry_basic_block)
        return builder.alloca(ty, size=None, name=name)

    def alloca_once(self, name, ty, size=None):
        return cgutils.alloca_once(self.builder, ty, size=size, name=name)

    def emit_enter(self, name):
        self.func_symtab = {}
        self.func = self._emit_prototype(name)
        bb_entry = self.func.append_basic_block('entry')
        self.builder = ir.IRBuilder(bb_entry)
        return

    def emit_store(self, dst, src):
        assert cgutils.is_pointer(dst.type)
        self.builder.store(src, dst)
        return

    def emit_gep(self, base, indices, inbounds=False, name=''):
        assert cgutils.is_pointer(base.type)
        return self.builder.gep(base, indices, inbounds=False, name='')

    def emit_printf(self, fmt, *args):
        return cgutils.printf(self.builder, fmt, *args)

    def emit_load(self, src, dst_name=''):
        assert cgutils.is_pointer(src.type)
        return self.builder.load(src, name=dst_name)

    def reg_read(self, vreg):
        return self.emit_load(vreg)

    def reg_write(self, vreg, value):
        self.emit_store(vreg, value)

    def emit_lea(self, base_r, immd_i=0, idx_r=None):
        assert cgutils.is_pointer(base_r.type)
        base = self.reg_read(base_r)
        immd = int32(immd_i)
        assert cgutils.is_pointer(base.type)
        if idx_r is not None:
            assert cgutils.is_pointer(idx_r.type)
            idx = self.reg_read(idx_r)
            assert not cgutils.is_pointer(idx.type)
            offset = self.builder.shl(idx, immd)
        else:
            offset = immd
        ptr = self.builder.gep(base, [offset])
        return ptr

    def mem_read(self, base_r, immd_i=0, idx_r=None):
        ptr = self.emit_lea(base_r, immd_i=immd_i, idx_r=idx_r)
        return self.emit_load(ptr)

    def mem_write(self, value, base_r, immd_i=0, idx_r=None):
        ptr = self.emit_lea(base_r, immd_i=immd_i, idx_r=idx_r)
        return self.emit_store(ptr, value)

    """
        MOV supports following mode:
          MOV R1, immd -- supported by MOVI
          MOV R1, R2
          MOV mem, R1
          MOV mem, immd -- supported by MOVI
          MOV R1, mem
        mem supports following addressing mode:
          [r1], [r1 + immd], [r1 + (r2 << immd)]
    """
    def emit_MOV(self, dst, src, ty1=None, ty2=None):
        """
           even if dst and src are two registers in Lir, they are modeled as allocated stack slot in LLVM, i.e., a pointer
        """
        assert cgutils.is_pointer(dst.type) and cgutils.is_pointer(src.type) 
        assert ty1 is None or ty2 is None
        if ty1 is None and ty2 is None:
            val = self.reg_read(src)
            self.reg_write(dst, val)
            return
        if ty1 is not None:
            # a memory write happens
            immd = ty1[0]
            idx = ty1[1]
            tmp = self.reg_read(src)
            self.mem_write(tmp, dst, immd_i=immd, idx_r=idx)
        else:
            # a reg read from a memory
            immd = ty1[0]
            idx = ty1[1]
            tmp = self.mem_read(src, immd_i=immd, idx_r=idx)
            self.reg_write(dst, tmp)
        return

    def emit_LEA(self, dst, src, ty=None):
        assert cgutils.is_pointer(dst.type) and cgutils.is_pointer(src.type) 
        if ty is None:
            val = self.reg_read(src)
            self.reg_write(dst, val)
            return
        else:
            immd = ty[0]
            idx = ty[1]
            addr = self.emit_lea(src, immd_i=immd, idx_r=idx)
            self.reg_write(dst, addr)
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

    def jit(self, name, *args, optimize=True, llvmdump=True):
        if llvmdump:
            print('======== Unoptimized LLVM IR')
            print(str(self.codegen.module))
        llvmmod = llvm.parse_assembly(str(self.codegen.module))
        target_machine = self.target.create_target_machine(opt=3)
        if optimize:
            pmb = lp.create_pass_manager_builder(opt=2)
            pm = llvm.create_module_pass_manager()
            pmb.populate(pm)
            pm.run(llvmmod)
            if llvmdump:
                print('======== Optimized LLVM IR')
                print(str(llvmmod))
        with llvm.create_mcjit_compiler(llvmmod, target_machine) as ee:
            ee.finalize_object()
            fptr = CFUNCTYPE(c_int)(ee.get_function_address(name))
            result = fptr(*args)
            return result   
