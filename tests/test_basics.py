import sljit.spam as spam
import sljit.Lir as Lir
import llvmlite.ir as ir
import sljit.cgutils as cg

lir = Lir.LirEvaluator()
lir.codegen.emit_enter('basic')
lir.codegen.emit_exit( ir.Constant(cg.int32_t, 12345678) )
assert lir.jit('basic') == 12345678
