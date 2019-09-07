// Atomic reduction statement representing a series of instructions used to
// exercise atomic operations, while still keeping the program generated 
// deterministic

#ifndef _CUDASMITH_STATEMENTATOMICREDUCTION_H_
#define _CUDASMITH_STATEMENTATOMICREDUCTION_H_

#include "CUDASmith/CUDAStatement.h"

#include "Block.h"
#include "Constant.h"
#include "Expression.h"
#include "FactMgr.h"
#include "Type.h"
#include "Variable.h"

namespace CUDASmith {

class StatementAtomicReduction : public CUDAStatement {
 public:
  enum AtomicOp {
    kAdd = 0,
    kSub,
//     kInc,
//     kDec,
//     kXchg,
//     kCmpxchg,
    kMin,
    kMax, 
    kAnd,
    kOr,
    kXor
  };
  
  enum MemoryLoc {
    kLocal = 0,
    kGlobal
  };
   
  StatementAtomicReduction(Expression* e, const bool add_global, 
    Block* b, AtomicOp op, MemoryLoc mem) : CUDAStatement(kAtomic, b), expr_(e), 
        op_(op), mem_loc_(mem), add_global_(add_global), 
        type_(Type::get_simple_type(eUInt)) {}
        
  static StatementAtomicReduction* make_random(CGContext& cg_context);
  
  static MemoryBuffer* get_local_rvar();
  static MemoryBuffer* get_global_rvar();
  
  static void AddVarsToGlobals(Globals* globals);
  static void RecordBuffer();
        
  // Pure virtual methods from Statement
  void get_blocks(std::vector<const Block*>& blks) const {};
  void get_exprs(std::vector<const Expression*>& exps) const 
    { exps.push_back(expr_); };
  
  void Output(std::ostream& out, FactMgr* fm = 0, int indent = 0) const;
  
 private:
  Expression* expr_;
  AtomicOp op_;
  MemoryLoc mem_loc_;
  const bool add_global_;
  
  const Type& type_;
  
  static Variable* get_hash_buffer();
  
  DISALLOW_COPY_AND_ASSIGN(StatementAtomicReduction);
};

}

#endif