// Handles basic inter-thread communication.
// Each time a comm statement is created, each work group is assigned an ID from
// a permutation of [0..31]. This ID is used to access a local buffer, which can
// be used in many other expressions.
// TODO fix the stuff in the .cpp, it was rushed so it looks terrible :<

#ifndef _CUDASMITH_STATEMENTCOMM_H_
#define _CUDASMITH_STATEMENTCOMM_H_

#include <memory>
#include <ostream>
#include <vector>

#include "CUDASmith/CUDAStatement.h"
#include "CUDASmith/StatementBarrier.h"
#include "CommonMacros.h"
#include "StatementAssign.h"

class Block;
class CGContext;
namespace CUDASmith { class Globals; }
class Expression;
class FactMgr;

namespace CUDASmith {

class StatementComm : public CUDAStatement {
 public:
  StatementComm(Block *blk, StatementBarrier *barrier, StatementAssign *assign)
      : CUDAStatement(kComm, blk), barrier_(barrier), assign_(assign) {
  }
  StatementComm(StatementComm&& other) = default;
  StatementComm& operator=(StatementComm&& other) = default;
  virtual ~StatementComm() {}

  // Creates a sequence of statements that, when executed, halts all threads by
  // a barrier, then gives each thread an ID from a random permutation of
  // [0..31].
  static StatementComm *make_random(CGContext& cg_context);

  // Creates the buffers used to hold thread IDs and intermediate values.
  static void InitBuffers();

  // Outputs the memory buffer holding the permutations.
  // TODO move to globals.h.
  static void OutputPermutations(std::ostream& out);

  // Adds the local buffers and tid variable to the global struct.
  static void AddVarsToGlobals(Globals *globals);

  // Hash the comm_values array. This must be done manually, as global arrays
  // are not typically hashed. Special hashing method required for the global 
  // buffer.
  static void HashCommValues(std::ostream& out);
  static void HashCommValuesGlobalBuffer(std::ostream& out);

  // Pure virtual in Statement. Not really needed.
  void get_blocks(std::vector<const Block *>& blks) const {}
  void get_exprs(std::vector<const Expression *>& exps) const {}  // TODO

  // Outputs the barrier followed by the assignment.
  void Output(std::ostream& out, FactMgr *fm, int indent) const;

 private:
  std::unique_ptr<StatementBarrier> barrier_;
  std::unique_ptr<StatementAssign> assign_;

  DISALLOW_COPY_AND_ASSIGN(StatementComm);
};

class CommController {

};

}  // namespace CUDASmith

#endif  // _CUDASMITH_STATEMENTCOMM_H_
