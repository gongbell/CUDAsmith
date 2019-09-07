// Root of all OpenCL statements.
// Will handle the creation of random OpenCL statements.
// TODO further implement functions in the Statement class
// (e.g. is_compound(t)).

#ifndef _CUDASMITH_CLSTATEMENT_H_
#define _CUDASMITH_CLSTATEMENT_H_

#include "CommonMacros.h"
#include "Statement.h"

class CGContext;

namespace CUDASmith {

// Will be the root of all OpenCL statements.
class CUDAStatement : public Statement {
 public:
  // Dynamic type info.
  enum CUDAStatementType {
    kNone = 0,  // Sentinel value
    kBarrier,
    kEMI,
    kReduction,
    kFakeDiverge,  // Gross hack alert
    kAtomic,
    kComm,
    kMessage,
    kTG
  };

  CUDAStatement(CUDAStatementType type, Block *block)
      : Statement(eCUDAStatement, block),
      cuda_statement_type_(type) {
  }
  CUDAStatement(CUDAStatement&& other) = default;
  CUDAStatement& operator=(CUDAStatement&& other) = default;
  virtual ~CUDAStatement() {}

  // Factory for creating a random OpenCL statement.
  static CUDAStatement *make_random(CGContext& cg_context,
      enum CUDAStatementType st);

  // Create an if statement with the test expression being uniform, but
  // appearing to diverge.
  static CUDAStatement *CreateFakeDivergentIf(CGContext& cg_context);

  // Initialise the probability table for selecting a random statement. Should
  // be called once on startup.
  static void InitProbabilityTable();

  // Getter the the statement type.
  enum CUDAStatementType GetCUDAStatementType() const { return cuda_statement_type_; }

 private:
  CUDAStatementType cuda_statement_type_;

  DISALLOW_COPY_AND_ASSIGN(CUDAStatement);
};

// Hook method for csmith.
Statement *make_random_st(CGContext& cg_context);

}  // namespace CUDASmith

#endif  // _CUDASMITH_CLSTATEMENT_H_
