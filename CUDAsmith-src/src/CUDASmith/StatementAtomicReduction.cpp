#include "CUDASmith/CUDAProgramGenerator.h"
#include "CUDASmith/Globals.h"
#include "CUDASmith/ExpressionID.h"
#include "CUDASmith/StatementAtomicReduction.h"
#include "CUDASmith/StatementBarrier.h"

#include "CVQualifiers.h"
#include "Expression.h"
#include "FactMgr.h"
#include "random.h"
#include "Type.h"
#include "VariableSelector.h"

namespace CUDASmith {
  
namespace {
// Variable to store the value of a modified variable.
Variable* hash_buffer = NULL;

// Variable representing the reduction target (either local or global);
// if global, must be a buffer, indexing at the current linear group id.
MemoryBuffer* local_reduction = NULL;
MemoryBuffer* global_reduction = NULL;
}
  
StatementAtomicReduction* StatementAtomicReduction::make_random(CGContext &cg_context) {
  const Type* type = get_int_type();
  Expression* expr = Expression::make_random(cg_context, type);
  const bool has_global = (const bool) rnd_upto(2);
  AtomicOp op = (AtomicOp) rnd_upto(kXor + 1);
  MemoryLoc mem = (MemoryLoc) rnd_upto(kGlobal + 1);
  return new StatementAtomicReduction(expr, has_global, 
                                      cg_context.get_current_block(), op, mem);
}

Variable* StatementAtomicReduction::get_hash_buffer() {
  if (hash_buffer == NULL)
    hash_buffer = Variable::CreateVariable("v_collective", 
        &Type::get_simple_type(eUInt), Constant::make_int(0), 
        new CVQualifiers(std::vector<bool>({false}), 
        std::vector<bool>({false})));
  return hash_buffer;
}

MemoryBuffer* StatementAtomicReduction::get_local_rvar() {
  if (local_reduction == NULL)
    local_reduction = new MemoryBuffer(MemoryBuffer::kLocal,
        "l_atomic_reduction", &Type::get_simple_type(eUInt), 
        Constant::make_int(0), new CVQualifiers(std::vector<bool> ({false}), 
        std::vector<bool> ({true})), {1});
//      std::vector<bool> ({false})), {1});//add by wxy 2018-05-10
  return local_reduction;
}

MemoryBuffer* StatementAtomicReduction::get_global_rvar() {
  if (global_reduction == NULL)
    global_reduction = new MemoryBuffer(MemoryBuffer::kGlobal,
        "g_atomic_reduction", &Type::get_simple_type(eUInt), 
        Constant::make_int(0), new CVQualifiers(std::vector<bool> ({false}), 
        std::vector<bool> ({true})), {CUDAProgramGenerator::get_groups()});
//      std::vector<bool> ({false})), {CUDAProgramGenerator::get_groups()});//wxy 2018-05-10
  return global_reduction;
}

void StatementAtomicReduction::RecordBuffer() {
  Variable* buf = get_hash_buffer();
  if (buf != NULL)
    VariableSelector::GetGlobalVariables()->push_back(buf);
}


void StatementAtomicReduction::AddVarsToGlobals(Globals* globals) {
  MemoryBuffer* local_red = get_local_rvar();
  MemoryBuffer* global_red = get_global_rvar();
  if (local_red != NULL)
    globals->AddLocalMemoryBuffer(local_red);
  if (global_red != NULL)
    globals->AddGlobalMemoryBuffer(global_red);
}
  
//void StatementAtomicReduction::Output(std::ostream& out, FactMgr* fm, int indent) const {
//  ExpressionID* lgrid = new ExpressionID(ExpressionID::kLinearGroup);
//  MemoryBuffer* itemized = get_global_rvar()->itemize({lgrid}, this->parent);
//  output_tab(out, indent);
//  out << "atomic_";
//  switch (op_) {
//    case (kAdd)  : out << "add" ; break;
//    case (kSub)  : out << "sub" ; break;
//    case (kMin)  : out << "min" ; break;
//    case (kMax)  : out << "max" ; break;
//    case (kAnd)  : out << "and" ; break;
//    case (kOr)   : out << "or"  ; break;
//    case (kXor)  : out << "xor" ; break;
//  }
//wxy2017/05/22
/***start***/
void StatementAtomicReduction::Output(std::ostream& out, FactMgr* fm, int indent) const {
  ExpressionID* lgrid = new ExpressionID(ExpressionID::kLinearGroup);
  MemoryBuffer* itemized = get_global_rvar()->itemize({lgrid}, this->parent);
  output_tab(out, indent);
  out << "myAtomic";
  switch (op_) {
    case (kAdd)  : out << "Add" ; break;
    case (kSub)  : out << "Sub" ; break;
    case (kMin)  : out << "Min" ; break;
    case (kMax)  : out << "Max" ; break;
    case (kAnd)  : out << "And" ; break;
    case (kOr)   : out << "Or"  ; break;
    case (kXor)  : out << "Xor" ; break;
  }
 /***end***/
  out << "(&";
  if (mem_loc_ == kLocal)
    out << get_local_rvar()->to_string();
  else if (mem_loc_ == kGlobal) {
    out << itemized->to_string();
  }
  else assert(0);
  out << ", ";
  expr_->Output(out);
  if (add_global_) {
    out << " + ";
    ExpressionID* lglid = new ExpressionID(ExpressionID::kLinearGlobal);
    lglid->Output(out);
  }
  out << ");" << std::endl;
  output_tab(out, indent);
  StatementBarrier::OutputBarrier(out); out << std::endl;
  output_tab(out, indent);
  out << "if (" ;
  ExpressionID* llid = new ExpressionID(ExpressionID::kLinearLocal);
  llid->Output(out);
  out << " == 0)" << std::endl;
  output_tab(out, indent + 1);
  out << get_hash_buffer()->to_string() <<  " += ";
  if (mem_loc_ == kLocal)
    out << get_local_rvar()->to_string();
  else if (mem_loc_ == kGlobal)
    out << itemized->to_string();
  else assert(0);
  out << ";" << std::endl;
  output_tab(out, indent);
  StatementBarrier::OutputBarrier(out); out << std::endl;
}

}

