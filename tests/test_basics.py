import sljit.spam as spam
import sljit.Lir as Lir
import llvmlite.ir as ir
import sljit.cgutils as cg

lir = Lir.LirEvaluator()
lir.codegen.emit_enter('basic')
r_curr_state = lir.codegen.alloca_once('R_CURR_STATE', cg.int32_t)
r_next_state = lir.codegen.alloca_once('R_NEXT_STATE', cg.int32_t)
lir.codegen.emit_store(r_next_state, lir.codegen.func.args[0] )
lir.codegen.emit_MOV(r_curr_state, r_next_state )
lir.codegen.emit_exit( lir.codegen.emit_load(r_curr_state) )
assert lir.jit('basic', 12345678) == 12345678
