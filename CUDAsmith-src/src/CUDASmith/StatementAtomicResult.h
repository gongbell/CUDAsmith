/* TODO
    * move into a new StatementAtomic class
*/

#ifndef _CUDASMITH_STATEMENTATOMICRESULT_H_
#define _CUDASMITH_STATEMENTATOMICRESULT_H_

#include "CUDASmith/CUDAStatement.h"
#include "CUDASmith/ExpressionAtomic.h"

#include <vector>

#include "Block.h"
#include "Expression.h"
#include "Type.h"
#include "Variable.h"

namespace CUDASmith {
  
class StatementAtomicResult : public CUDAStatement {
 public:
  enum AtomicResultType {
    kAddVar = 0,
    kAddArrVar,
    kSetSpVal,
    kDecl
  };
  
  StatementAtomicResult (Variable* var, Block* b) : CUDAStatement(kAtomic, b),
    var_(var), av_(NULL), access_(NULL), result_type_(kAddVar), type_(Type::get_simple_type(eInt)) {}
    
  StatementAtomicResult (ArrayVariable* av, Block* b) : CUDAStatement(kAtomic, b),
    var_(NULL), av_(av), access_(NULL), result_type_(kAddArrVar), type_(Type::get_simple_type(eInt)) {}
    
  StatementAtomicResult (const ExpressionAtomicAccess* access, Block* b) : CUDAStatement(kAtomic, b),
    var_(NULL), av_(NULL), access_(access), result_type_(kSetSpVal), type_(Type::get_simple_type(eInt)) {}
    
  StatementAtomicResult (Block* b) : CUDAStatement(kAtomic, b),
    var_(NULL), av_(NULL), access_(NULL), result_type_(kDecl), type_(Type::get_simple_type(eInt)) {}
  
  static void InitResults(void);
  static void DefineLocalResultVar(std::ostream& out);
  static void GenSpecialVals(void);
  static void RecordIfID(int id, Expression* expr);
  
  // Pure virtual methods from Statement (need only Output)
  void get_blocks(std::vector<const Block*>&) const {};
  void get_exprs(std::vector<const Expression*>&) const {};
  
  void Output(std::ostream& out, FactMgr* fm = 0, int indent = 0) const;
  
  
private:
  Variable* var_;
  ArrayVariable* av_;
  const ExpressionAtomicAccess* access_;
  
  AtomicResultType result_type_;
  const Type& type_;    

  void PrintStructArrayVars(std::ostream& out, int indent, Variable* av, std::stringstream& ss) const; 

  DISALLOW_COPY_AND_ASSIGN(StatementAtomicResult);
  
}; 

} // namespace CUDASmith

#endif  //  _CUDASMITH_STATEMENTATOMICRESULT_H_  
    
