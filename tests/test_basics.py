import sljit.spam as spam
import sljit.Lir as Lir
import llvmlite.ir as ir
import sljit.cgutils as cg

lir = Lir.LirEvaluator()
lir.codegen.emit_enter('basic')
r_curr_state = lir.codegen.alloca_once('R_CURR_STATE', cg.int32_t)
r_next_state = lir.codegen.alloca_once('R_NEXT_STATE', cg.int32_t)

r_string = lir.codegen.alloca_once('R_STRING', ir.ArrayType(cg.int8_t, 256) )
r_next_char = lir.codegen.alloca_once('R_NEXT_CHAR', cg.charptr_t)
r_string_start = lir.codegen.alloca_once('R_STRING_START', cg.charptr_t)
r_tmp = lir.codegen.alloca_once('R_TEMP', cg.int8_t)

lir.codegen.emit_MOVI(r_next_state, lir.codegen.func.args[0] )
tmp = lir.codegen.emit_gep( r_string, (Lir.int64(0), Lir.int32(0)) )
lir.codegen.emit_MOVI(r_string_start, tmp)
lir.codegen.emit_MOV(r_next_char, r_string_start)

for c in 'Hello World!\n':
    lir.codegen.emit_MOVI(r_tmp, Lir.char(c))
    lir.codegen.emit_MOV(r_next_char, r_tmp, ty1=(0, None))
    lir.codegen.emit_MOV(r_tmp, r_next_char, ty2=(0, None))
    lir.codegen.emit_LEA(r_next_char, r_next_char, ty=(1, None))
    
lir.codegen.mem_write(Lir.char('\0'), r_next_char, 0)

s = lir.codegen.reg_read(r_string_start)
lir.codegen.emit_printf('%s\n', s)

lir.codegen.emit_NOT(r_curr_state, r_next_state )
lir.codegen.emit_NEG(r_curr_state, r_next_state )
lir.codegen.emit_exit( lir.codegen.emit_load(r_curr_state) )

assert lir.jit('basic', 12345678) == -12345678
