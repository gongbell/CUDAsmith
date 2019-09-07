#include "CUDASmith/ExpressionAtomic.h"
#include "CUDASmith/ExpressionID.h"
#include "CUDASmith/StatementAtomicResult.h"
#include "CUDASmith/StatementBarrier.h"
#include "CUDASmith/Globals.h"

#include <algorithm>
#include <bitset>
#include <vector>
#include <map>
#include <numeric>
#include <stack>

#include "ArrayVariable.h"
#include "Block.h"
#include "CGContext.h"
#include "CUDAProgramGenerator.h"
#include "Constant.h"
#include "Expression.h"
#include "ExpressionFuncall.h"
#include "Function.h"
#include "FunctionInvocation.h"
#include "FunctionInvocationBinary.h"
#include "random.h"
#include "SafeOpFlags.h"
#include "StatementIf.h"
#include "VariableSelector.h"


namespace CUDASmith {

namespace {
// Max number of atomic blocks that the generated program should contains
static int no_atomic_blocks = 0;

// Corresponds to the global array parameter passed to the kernel
static MemoryBuffer* global_in_buf = NULL;
static MemoryBuffer* local_in_buf = NULL;

// Corresponds to the special value array in the global struct
static MemoryBuffer* global_sv_buf = NULL;
static MemoryBuffer* local_sv_buf = NULL;

// Holds the global array parameter reference and all references to it;
// these need to be added to the global struct when it is created.
// (in CUDAProgramGenerator::goGenerator())
static std::vector<MemoryBuffer*>* global_in = NULL;
static std::vector<MemoryBuffer*>* local_in = NULL;

// The global parameter for special values; also to be declared in the global
// struct.
static std::vector<MemoryBuffer*>* global_sv = NULL;
static std::vector<MemoryBuffer*>* local_sv = NULL;

// To be used during code generation; holds the variables generated in an
// atomic block and uses them to compute a special value used for checking
static std::stack<std::map<int, std::vector<Variable *>*>*>* block_vars = NULL;
static std::stack<Block*>* atomic_parent = NULL;

static std::vector<int>* free_counters = NULL;
} // namespace

void ExpressionAtomic::InitAtomics() {
  no_atomic_blocks = rnd_upto(99) + 1; //rnd_upto(CUDASmith::CUDAProgramGenerator::get_threads() / CUDASmith::CUDAProgramGenerator::get_groups()) + 1;
  free_counters = new vector<int>(no_atomic_blocks, 0);
  block_vars = new std::stack<std::map<int, std::vector<Variable*>*>*>();
  atomic_parent = new std::stack<Block*>();
  std::iota(free_counters->begin(), free_counters->end(), 0);
  StatementAtomicResult::InitResults();
}

// TODO make if from switch (+ other functions too)
ExpressionAtomic* ExpressionAtomic::make_random(CGContext &cg_context, const Type *type) {
  assert(type->eType == eSimple && type->simple_type == eInt);
  if (free_counters->empty())
    return NULL;
  const int rand_index = free_counters->at(rnd_upto(free_counters->size()));
  const ExpressionAtomicAccess::AtomicMemAccess mem = (ExpressionAtomicAccess::AtomicMemAccess) rnd_upto(2);
  const ExpressionAtomicAccess* eaa = new ExpressionAtomicAccess(rand_index, mem);
  std::vector<const Expression*> eaa_arr (1, eaa);
  MemoryBuffer* itemized;
  if (eaa->is_global()) {
    itemized = GetGlobalBuffer()->itemize(eaa_arr, Block::make_dummy_block(cg_context));
    GetGlobalMems()->push_back(itemized);
  } else if (eaa->is_local()) {
    itemized = GetLocalBuffer()->itemize(eaa_arr, Block::make_dummy_block(cg_context));
    GetLocalMems()->push_back(itemized);
  } else assert(0);
  AtomicExprType atomic_type = rnd_flipcoin(100) ? kInc : kAdd;
  free_counters->erase(std::remove(free_counters->begin(), free_counters->end(), rand_index), free_counters->end());
  switch(atomic_type) {
//     case kDec:
    case kInc:
      return new ExpressionAtomic(atomic_type, itemized, eaa);
    case kAdd:
//     case kSub:
      return new ExpressionAtomic(atomic_type, itemized, eaa, rnd_upto(10) + 1);
    default: assert(0); // should never get here.
  }
}

Expression *ExpressionAtomic::make_condition(CGContext &cg_context, const Type *type) {
  assert(type->eType == eSimple && type->simple_type == eInt);
  ExpressionAtomic *expr_atomic = make_random(cg_context, type);
  if (!expr_atomic)
    return NULL;
  FunctionInvocationBinary *fi = new FunctionInvocationBinary(eCmpEq, expr_atomic,
                                  Constant::make_int(rnd_upto(10)),
                                  SafeOpFlags::make_dummy_flags());
  MakeBlockVars(cg_context.get_current_block());
  return new ExpressionFuncall(*fi);
}

void ExpressionAtomic::AddVarsToGlobals(Globals* globals) {
  if (HasLocalSVMems()) {
    for (MemoryBuffer *mb : *ExpressionAtomic::GetLocalMems()) {
      globals->AddLocalMemoryBuffer(mb);
    }
    for (MemoryBuffer *mb : *ExpressionAtomic::GetLocalSVMems()) {
      globals->AddLocalMemoryBuffer(mb);
    }
  }
  if (HasSVMems()) {
    for (MemoryBuffer *mb : *ExpressionAtomic::GetGlobalMems()) {
      globals->AddGlobalMemoryBuffer(mb);
    }
    for (MemoryBuffer *mb : *ExpressionAtomic::GetSVMems()) {
      globals->AddGlobalMemoryBuffer(mb);
    }
  }
}

MemoryBuffer* ExpressionAtomic::GetGlobalBuffer() {
  if (global_in_buf == NULL)  {
    global_in_buf = new MemoryBuffer(MemoryBuffer::kGlobal,
      "g_atomic_input", &Type::get_simple_type(eUInt), Constant::make_int(0),
      new CVQualifiers(std::vector<bool> ({false}), std::vector<bool> ({true})),
//    new CVQualifiers(std::vector<bool> ({false}), std::vector<bool> ({false})),//add by wxy 2018-05-10
      {CUDAProgramGenerator::get_groups() * no_atomic_blocks});
    GetGlobalMems()->push_back(global_in_buf);
    GetSVBuffer();
  }
  return global_in_buf;
}

MemoryBuffer* ExpressionAtomic::GetSVBuffer() {
  if (global_sv_buf == NULL) {
    global_sv_buf = new MemoryBuffer(MemoryBuffer::kGlobal,
      "g_special_values", &Type::get_simple_type(eUInt), Constant::make_int(0),
      new CVQualifiers(std::vector<bool> ({false}), std::vector<bool> ({true})),
//    new CVQualifiers(std::vector<bool> ({false}), std::vector<bool> ({false})),//add by wxy 2018-05-10
      {CUDAProgramGenerator::get_groups() * no_atomic_blocks});
    GetSVMems()->push_back(global_sv_buf);
  }
  return global_sv_buf;
}
//wxy2017/05/25
/*MemoryBuffer* ExpressionAtomic::GetLocalBuffer() {
  if (local_in_buf == NULL)  {
    local_in_buf = new MemoryBuffer(MemoryBuffer::kGlobal,
      "l_atomic_input", &Type::get_simple_type(eUInt), Constant::make_int(0),
      new CVQualifiers(std::vector<bool> ({false}), std::vector<bool> ({true})),
      {CUDAProgramGenerator::get_groups() * no_atomic_blocks});
    GetLocalMems()->push_back(local_in_buf);
    GetLocalSVBuffer();
  }
  return local_in_buf;
}*/

MemoryBuffer* ExpressionAtomic::GetLocalBuffer() {
  if (local_in_buf == NULL)  {
    local_in_buf = new MemoryBuffer(MemoryBuffer::kLocal,
      "l_atomic_input", &Type::get_simple_type(eUInt), Constant::make_int(0),
      new CVQualifiers(std::vector<bool> ({false}), std::vector<bool> ({true})),
  //    new CVQualifiers(std::vector<bool> ({false}), std::vector<bool> ({false})),//add by wxy 2018-05-10
      {(unsigned)no_atomic_blocks});
    GetLocalMems()->push_back(local_in_buf);
    GetLocalSVBuffer();
  }
  return local_in_buf;
}

MemoryBuffer* ExpressionAtomic::GetLocalSVBuffer() {
  if (local_sv_buf == NULL) {
    local_sv_buf = new MemoryBuffer(MemoryBuffer::kLocal,
      "l_special_values", &Type::get_simple_type(eUInt), Constant::make_int(0),
      new CVQualifiers(std::vector<bool> ({false}), std::vector<bool> ({true})),
//        new CVQualifiers(std::vector<bool> ({false}), std::vector<bool> ({false})),//add by wxy 2018-05-10
      {(unsigned)no_atomic_blocks});

    GetLocalSVMems()->push_back(local_sv_buf);
  }
  return local_sv_buf;
}

int ExpressionAtomic::get_atomic_blocks_no() {
  return no_atomic_blocks;
}

// TODO delete memory references
std::vector<MemoryBuffer*>* ExpressionAtomic::GetGlobalMems() {
  if (global_in == NULL)
    global_in = new std::vector<MemoryBuffer*>();
  return global_in;
}

std::vector<MemoryBuffer*>* ExpressionAtomic::GetSVMems() {
  if (global_sv == NULL) {
    global_sv = new std::vector<MemoryBuffer*>();
//     GetSVBuffer(); // call once to ensure reference is constructed
  }
  return global_sv;
}

std::vector<MemoryBuffer*>* ExpressionAtomic::GetLocalMems() {
  if (local_in == NULL)
    local_in = new std::vector<MemoryBuffer*>();
  return local_in;
}

std::vector<MemoryBuffer*>* ExpressionAtomic::GetLocalSVMems() {
  if (local_sv == NULL) {
    local_sv = new std::vector<MemoryBuffer*>();
//     GetLocalSVBuffer(); // call once to ensure reference is constructed
  }
  return local_sv;
}

bool ExpressionAtomic::HasSVMems() {
  return global_sv != NULL;
}

bool ExpressionAtomic::HasLocalSVMems() {
  return local_sv != NULL;
}

void ExpressionAtomic::MakeBlockVars(Block* parent) {
  assert(block_vars->size() == atomic_parent->size());
  block_vars->emplace(new std::map<int, std::vector<Variable*>*>());
  atomic_parent->push(parent);
//   if (block_vars != NULL) {
//     block_vars->clear();
//   } else {
//     delete(block_vars);
//     block_vars = new std::map<int, std::vector<Variable*>*>();
//   }
//   atomic_parent = parent;
}

// void ExpressionAtomic::PrintBlockVars() {
//   std::cout << "--- BLOCKVARS ---" << std::endl;
//   for (std::map<int, std::vector<Variable*>*>::iterator it = block_vars->begin(); it != block_vars->end(); it++) {
//     std:: cout << "blk id " << (*it).first << std::endl;
//     for (Variable* v : *((*it).second))
//       std::cout << v->to_string() << std::endl;
//   }
//   std::cout << "--- ENDBLOCKVARS ---" << std::endl;
// }

void ExpressionAtomic::GenBlockVars(Block* curr) {
//   std::cout << "Gening " << curr->stm_id << std::endl;
  std::map<int, std::vector<Variable*>*>* blk_vars = block_vars->top();
  assert(blk_vars->find(curr->stm_id) == blk_vars->end());
  std::vector<Variable*>* new_blk_vars = new std::vector<Variable*> ();
  blk_vars->insert(std::pair<int, std::vector<Variable*>*> (curr->stm_id, new_blk_vars));
  if (curr->parent != atomic_parent->top())
    new_blk_vars->insert(new_blk_vars->begin(), curr->parent->local_vars.begin(), curr->parent->local_vars.end());
  if (blk_vars->find(curr->parent->stm_id) != blk_vars->end()) {
    std::vector<Variable*>* parent_blk_vars = (*blk_vars->find(curr->parent->stm_id)).second;
    new_blk_vars->insert(new_blk_vars->begin(), parent_blk_vars->begin(), parent_blk_vars->end());
  }
}

std::vector<Variable*>* ExpressionAtomic::GetBlockVars(Block* b) {
  // TODO may need check for non-visible variables
//   std::cout << "Getting " << b->stm_id << std::endl;
//   assert(block_vars != NULL);
//   assert(!block_vars->empty()); // MakeBlockVars() must be called before
  assert(block_vars->top()->find(b->stm_id) != block_vars->top()->end());
  return (*block_vars->top()->find(b->stm_id)).second;
}

void ExpressionAtomic::InsertBlockVars(std::vector<Variable*> local_vars, std::vector<Variable*>* blk_vars) {
  for (Variable* v : local_vars) {
    if (std::find(blk_vars->begin(), blk_vars->end(), v) != blk_vars->end()) {
//       std::cout << "Dupe ignore " << v->to_string() << std::endl;
      continue;
    }
    std::cout << "Inserting " << v->to_string() << std::endl;
    ArrayVariable * av = dynamic_cast<ArrayVariable*>(v);
    if (av != NULL && (av->get_collective() != av)) {
      blk_vars->push_back(const_cast<Variable*>(av->get_collective()));
      continue;
    }
    blk_vars->push_back(v);
  }
}

void ExpressionAtomic::RemoveBlockVars(Block* b) {
//   std::cout << "Removing " << b->stm_id << std::endl;
  assert(block_vars->top()->find(b->stm_id) != block_vars->top()->end());
  block_vars->top()->erase(b->stm_id);
}

void ExpressionAtomic::DelBlockVars() {
  block_vars->pop();
  atomic_parent->pop();
//   delete(block_vars);
//   block_vars = NULL;
//   atomic_parent = NULL;
}

void ExpressionAtomic::Output(std::ostream& out) const {
  // Need the '&' as atomic expressions expect a pointer to parameter p.
  out << GetOpName() << "(&";
  global_var_->Output(out);
  switch(atomic_type_) {
    case kDec:
    case kInc: break;
    case kAdd:
    case kSub:
    case kXchg:
    case kMin:
    case kMax:
    case kAnd:
    case kOr:
    case kXor: { out << ", " << val_; break; }
    case kCmpxchg: { out << ", " << cmp_ << ", " << val_; break; }
    default: assert(0); // should never get here.
  }
  out << ")";
}

void ExpressionAtomic::OutputHashing(std::ostream& out) {
  if (ExpressionAtomic::HasSVMems()) {
    output_tab(out, 1);
    StatementBarrier::OutputBarrier(out);
    out << std::endl;
    output_tab(out, 1);
    out << "if (!";
    ExpressionID* eid_g = new ExpressionID(ExpressionID::kLinearGlobal);
    eid_g->Output(out);
    out << ")" << std::endl;
    output_tab(out, 2);
    out << "for (i = 0 ; i < " << CUDAProgramGenerator::get_atomic_blocks_no()
        << "; i++)" << std::endl;
    output_tab(out, 3);
    out << "transparent_crc(";
    ExpressionAtomic::GetSVBuffer()->Output(out);
    out << "[i + " << CUDAProgramGenerator::get_atomic_blocks_no() << " * ";
    ExpressionID* lgrid = new ExpressionID(ExpressionID::kLinearGroup);
    lgrid->Output(out);
    out << "], \"";
    ExpressionAtomic::GetSVBuffer()->Output(out);
    out << "[i + " << CUDAProgramGenerator::get_atomic_blocks_no() << " * ";
    lgrid->Output(out);
    out << "]\", print_hash_value);" << std::endl;
  }
  if (ExpressionAtomic::HasLocalSVMems()) {
    output_tab(out, 1);
    StatementBarrier::OutputBarrier(out);
    out << std::endl;
    output_tab(out, 1);
    out << "if (!";
    ExpressionID* eid_l = new ExpressionID(ExpressionID::kLinearLocal);
    eid_l->Output(out);
    out << ")" << std::endl;
    output_tab(out, 2);
    out << "for (i = 0; i < " << CUDAProgramGenerator::get_atomic_blocks_no() <<
        "; i++)" << std::endl;
    output_tab(out, 3);
    out << "transparent_crc(";
    ExpressionAtomic::GetLocalSVBuffer()->Output(out);
    out << "[i], \"";
    ExpressionAtomic::GetLocalSVBuffer()->Output(out);
    out << "[i]\", print_hash_value);" << std::endl;
  }
}
/*
string ExpressionAtomic::GetOpName() const {
  switch(atomic_type_) {
    case kAdd:     return "atomic_add";
    case kSub:     return "atomic_sub";
    case kXchg:    return "atomic_xchg";
    case kInc:     return "atomic_inc";
    case kDec:     return "atomic_dec";
    case kCmpxchg: return "atomic_cmpxchg";
    case kMin:     return "atomic_min";
    case kMax:     return "atomic_max";
    case kAnd:     return "atomic_and";
    case kOr:      return "atomic_or";
    case kXor:     return "atomic_xor";
    default:       assert(0); // should never get here.
  }
}
*/
//wxy2017/05/21
/***start***/
string ExpressionAtomic::GetOpName() const {
  switch(atomic_type_) {
    case kAdd:     return "myAtomicAdd";
    case kSub:     return "myAtomicSub";
    case kXchg:    return "myAtomicExch";
    case kInc:     return "myAtomicInc";
    case kDec:     return "myAtomicDec";
    case kCmpxchg: return "atomicCAS";
    case kMin:     return "myAtomicMin";
    case kMax:     return "myAtomicMax";
    case kAnd:     return "myAtomicAnd";
    case kOr:      return "myAtomicOr";
    case kXor:     return "myAtomicXor";
    default:       assert(0); // should never get here.
  }
}
/***end***/
void ExpressionAtomicAccess::Output(std::ostream& out) const {
  if (mem_ == kGlobal) {
    out << ExpressionAtomic::get_atomic_blocks_no() << " * ";
    ExpressionID* g_access = new ExpressionID(ExpressionID::kLinearGroup);
    g_access->Output(out);
    out << " + " << constant_;
  } else if (mem_ == kLocal) {
    out << constant_;
  }
}

bool ExpressionAtomicAccess::is_global() const {
  return mem_ == kGlobal;
}

bool ExpressionAtomicAccess::is_local() const {
  return mem_ == kLocal;
}

}

