#include "CUDASmith/StatementComm.h"

#include <algorithm>
#include <ostream>
#include <random>
#include <vector>

#include "Block.h"
#include "CGContext.h"
#include "CUDASmith/CUDAOptions.h"
#include "CUDASmith/CUDAProgramGenerator.h" // temp
#include "CUDASmith/ExpressionID.h"
#include "CUDASmith/Globals.h"
#include "CUDASmith/MemoryBuffer.h"
#include "CUDASmith/StatementBarrier.h"
#include "Constant.h"
#include "Expression.h"
#include "ExpressionFuncall.h"
#include "ExpressionVariable.h"
#include "Lhs.h"
#include "SafeOpFlags.h"
#include "StatementAssign.h"
#include "Type.h"
#include "Variable.h"
#include "VariableSelector.h"

namespace CUDASmith {
namespace {
// Number of permutations.
const int kPermCount = 10;

// Const buffer that holds the random permutations of [0..31].
MemoryBuffer *permutations;
// Each permutation.
std::vector<int> *permute_values[kPermCount];
// Local buffer that holds the intermediate values.
MemoryBuffer *local_values;
// Global buffer that holds the intermediate values.
MemoryBuffer *global_values;
// Variable that holds the random thread ID.
Variable *tid;
// Local Variable used in random expressions throughout the program.
MemoryBuffer *local_var;
// Global Variable used in random expressions throughout the program.
MemoryBuffer *global_var;
}  // namespace

StatementComm *StatementComm::make_random(CGContext& cg_context) {
  int group_size = CUDAProgramGenerator::get_threads_per_group();
  // rand_expr
  Expression *expr = Expression::make_random(
      cg_context, &Type::get_simple_type(eUInt));
  // rand_expr % 10
  expr = new ExpressionFuncall(*new FunctionInvocationBinary(eMod, expr,
      Constant::make_int(10), new SafeOpFlags(false, true, true, sInt32)));
  // tid % group_size
  Expression *expr_ = new ExpressionFuncall(*new FunctionInvocationBinary(eMod,
      new ExpressionVariable(*tid), Constant::make_int(group_size),
      new SafeOpFlags(false, false, true, sInt32)));
  // permutations[rand_expr % 10][tid % group_size]
  expr = new ExpressionVariable(*permutations->itemize(
      std::vector<const Expression *>({expr, expr_}),
      cg_context.get_current_block()));
  // get_linear_group_id() * group_size
  expr_ = new ExpressionFuncall(*new FunctionInvocationBinary(eMul,
      new ExpressionID(ExpressionID::kLinearGroup),
      Constant::make_int(group_size),
      new SafeOpFlags(false, false, true, sInt32)));
  // (get_linear_group_id() * group_size) + permutations[rand_expr % 10][tid % group_size]
  expr = new ExpressionFuncall(*new FunctionInvocationBinary(eAdd,
      expr_, expr, new SafeOpFlags(false, false, true, sInt32)));
  StatementAssign *st_ass = StatementAssign::make_possible_compound_assign(
      cg_context, *new Lhs(*tid), eSimpleAssign, *expr);
  return new StatementComm(cg_context.get_current_block(),
      new StatementBarrier(cg_context.get_current_block()), st_ass);
}

void StatementComm::InitBuffers() {
  unsigned perm_size = CUDAProgramGenerator::get_threads_per_group();
  for (int idx = 0; idx < kPermCount; ++idx) {
    permute_values[idx] = new std::vector<int>(perm_size);
    for (unsigned id = 0; id < perm_size; ++id) (*permute_values[idx])[id] = id;
    std::shuffle(permute_values[idx]->begin(), permute_values[idx]->end(),
        std::default_random_engine(rnd_upto(65535)));
  }
  permutations = MemoryBuffer::CreateMemoryBuffer(MemoryBuffer::kConst,
      "permutations", &Type::get_simple_type(eUInt), NULL,
      {kPermCount, perm_size});
  local_values = MemoryBuffer::CreateMemoryBuffer(MemoryBuffer::kLocal,
      "l_comm_values", &Type::get_simple_type(eLongLong), Constant::make_int(1),
      {CUDAProgramGenerator::get_threads_per_group()});
  global_values = MemoryBuffer::CreateMemoryBuffer(MemoryBuffer::kGlobal,
      "g_comm_values", &Type::get_simple_type(eLongLong), Constant::make_int(1),
      {CUDAProgramGenerator::get_total_threads()});
  // The initial value of the tid will come from the first permutation.
  // Lots of memory leaks here, probably not worth cleaning.
  // get_linear_group_id() * group_size
  Expression *expr = new ExpressionFuncall(*new FunctionInvocationBinary(eMul,
      new ExpressionID(ExpressionID::kLinearGroup),
      Constant::make_int(perm_size),
      new SafeOpFlags(false, false, true, sInt32)));
  // permutations[0][get_linear_group_id()]
  ExpressionVariable *expr_ = new ExpressionVariable(*permutations->itemize(  // leak
      std::vector<const Expression *>({
      Constant::make_int(0), new ExpressionID(ExpressionID::kLinearLocal)}),
      new Block(NULL, 0)));  // leak
  // get_linear_group_id() * group_size + permutations[0][get_linear_group_id()]
  expr = new ExpressionFuncall(*new FunctionInvocationBinary(eAdd,
      expr, expr_, new SafeOpFlags(false, false, true, sInt32)));
  tid = Variable::CreateVariable("tid", &Type::get_simple_type(eUInt), expr,
      new CVQualifiers(std::vector<bool>({false}), std::vector<bool>({false})));
  // The variable that will be accessed throughout the program randomly.
  global_var = global_values->itemize(std::vector<const Expression *>({
      new ExpressionVariable(*tid)}), new Block(NULL, 0));  // leak
  expr = new ExpressionFuncall(*new FunctionInvocationBinary(eMod,
      new ExpressionVariable(*tid), Constant::make_int(perm_size),
      new SafeOpFlags(false, true, true, sInt32)));
  local_var = local_values->itemize(std::vector<const Expression *>({expr}),
      new Block(NULL, 0));  // leak
  if (CUDAOptions::inter_thread_comm()) {
    VariableSelector::GetGlobalVariables()->push_back(local_var);
    VariableSelector::GetGlobalVariables()->push_back(global_var);
  }
}

void StatementComm::OutputPermutations(std::ostream& out) {
  permutations->OutputDecl(out);
  out << " = {" << std::endl;
  for (int idx = 0; idx < kPermCount; ++idx) {
    out << "{" << (*permute_values[idx])[0];
    for (int id = 1; id < (int)permute_values[idx]->size(); ++id)
      out << "," << (*permute_values[idx])[id];
    out << "}";
    if (idx < kPermCount - 1) out << ",";
    out << " // permutation " << idx << std::endl;
  }
  out << "};" << std::endl;
}

void StatementComm::AddVarsToGlobals(Globals *globals) {
  assert(global_values != NULL && local_values != NULL && tid != NULL);
  globals->AddLocalMemoryBuffer(local_values);
  globals->AddGlobalMemoryBuffer(global_values);
  globals->AddGlobalVariable(tid);
}

void StatementComm::HashCommValues(std::ostream& out) {
  assert(global_values != NULL && local_values != NULL);
  local_values->hash(out);
  HashCommValuesGlobalBuffer(out);
}

void StatementComm::HashCommValuesGlobalBuffer(std::ostream& out) {
  output_tab(out, 1);
  out << "transparent_crc(";
  global_values->Output(out);
  out << "[get_linear_group_id() * " << CUDAProgramGenerator::get_threads_per_group() << " + get_linear_local_id()]"; 
  out << ", \"";
  global_values->Output(out);
  out << "[get_linear_group_id() * " << CUDAProgramGenerator::get_threads_per_group() << " + get_linear_local_id()]"; 
  out << "\", print_hash_value);" << std::endl;
}

void StatementComm::Output(std::ostream& out, FactMgr *fm, int indent) const {
  barrier_->Output(out, fm, indent);
  assign_->Output(out, fm, indent);
}

}  // namespace CUDASmith
