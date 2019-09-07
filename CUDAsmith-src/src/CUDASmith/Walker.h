// Generic walker for traversing the statements of a function.
// Does not do any processing on the statements, it will simply make them
// visible to other functions.
//
// This was all pretty much a waste of time!
// I thought the templates would help >:(
//
// Future possible improvements:
// - Get rid of the templated stuff.
// - const correctness (if nothing needs to modify the block) and change
//   pointers to refs.
// - Use CRTP to pass the derived class down the hierarchy, might allow us to
//   get rid of the AdvanceSelector.
// - FunctionWalker accesses the BlockWalker's stuff manually. Add some accessor
//   functions to BlockWalker and use those instead.

#ifndef _CUDASMITH_WALKER_H_
#define _CUDASMITH_WALKER_H_

#include <iostream>
#include <memory>
#include <stack>
#include <vector>

#include "Block.h"
#include "CommonMacros.h"
#include "Function.h"
#include "Statement.h"

namespace CUDASmith {
namespace Walker {
namespace Internal {

class BlockWalker;

// Simple factory method for creating a BlockWalker at the start of a block.
// Allows the BlockWalkers to recursively create other BlockWalkers.
BlockWalker *CreateBlockWalker(Block *block);

// Create a BlockWalker at a specified position in the block.
BlockWalker *CreateBlockWalkerAtStatement(Block *block, Statement *statement);

// Generic type list, simply holds a compile time list of types.
template <eStatementType... types>
struct TypeList;

// All of the statement types that will be traversed by the Walker.
typedef TypeList<eAssign, eBlock, eFor, eIfElse,
                 eInvoke, eReturn, eContinue, eBreak,
                 eGoto, eArrayOp, eCUDAStatement> StatementList;

// Base of the Walker hierarchy. Everything that the Walker depends on should be
// here.
class WalkerBase {
 public:
  WalkerBase() : block_(NULL), statement_(NULL) {}
  virtual ~WalkerBase() {}
  bool AdvanceOne();
  bool AdvanceToStatement(Statement *statement);
  Block *block_;
  Statement *statement_;
 private:
  std::vector<Statement *>::iterator block_it_;
};

// Basic declaration, templated with our list of statements.
template <class TypeList>
class Walker;

// Basic implementation declaration, templated with a statement.
template <eStatementType Type>
class WalkerImpl;

// End of the recursive Walker hierarchy, represented by the empty TypeList.
template <>
class Walker<TypeList<>> : public virtual WalkerBase {
};

// Recursive structure for the walker.
// Each type in our statement list will appear in 'StatementType', and so
// everything in this class will be defined for each statement.
// This class will also be extended with a specialisation of the WalkerImpl
// class for each statement.
template <eStatementType Type, eStatementType... Types>
class Walker<TypeList<Type, Types...>>
    : public WalkerImpl<Type>,
      public Walker<TypeList<Types...>> {
};

// Specialisation of WalkerImpl for For statements, necessary to keep track of
// the Walker for the body block.
template <>
class WalkerImpl<eFor> : public virtual WalkerBase {
 public:
  bool Advance(Statement *statement);
  std::unique_ptr<BlockWalker> for_body_;
};

// Specialisation for the if-then-else implementation, must keep track of all
// the branches that may occur, there is no limit.
template <>
class WalkerImpl<eIfElse> : public virtual WalkerBase {
 public:
  bool Advance(Statement *statement);
  std::unique_ptr<BlockWalker> if_body_;
  std::unique_ptr<BlockWalker> else_body_;
};

// Goto implementation. A BlockWalker for the destination is not created, as
// unless the source and destination is in the same block, it will not help us;
// a FunctionWalker would be better, allowing us to walk between the two points.
template <>
class WalkerImpl<eGoto> : public virtual WalkerBase {
 public:
  bool Advance(Statement *statement);
  // Is the destination of the goto ahead of the statement.
  Statement *destination_;
  bool goto_destination_is_forward_;
};

// Generic template for the Walker of a given statement. A specialisation is
// only needed if yopu need extra members for the statement.
template <eStatementType S>
class WalkerImpl : public virtual WalkerBase {
 public:
  bool Advance(Statement *statement);
};

// Forward declare block specialisation. Needed as there is no block
// specialisation of WalkerImpl.
template <>
bool WalkerImpl<eBlock>::Advance(Statement *statement);

// Generic template for processing a basic statement. Advance is a misnomer
// here, we always advance on every statement, and then apply the Advance method
// in WalkerImpl to process what we found.
template <eStatementType S>
bool WalkerImpl<S>::Advance(Statement *statement) {
  return true;
}

// Simple Walker that brings the implementations together.
// This is specifically for walking through a block, branches do not sit in the
// same block, but form their own. The implementations provide functionality for
// handling this, but entering the branches must be done manually.
class BlockWalker : public Walker<StatementList> {
 public:
  BlockWalker() {}
  explicit BlockWalker(Block *block) { WalkerBase::block_ = block; }
  bool AdvanceBlock() {
    return WalkerBase::AdvanceOne() && AdvanceSelector(WalkerBase::statement_);
  }
  bool AdvanceSelector(Statement *statement);
};

}  // namespace Internal

// Traverses the blocks in a function. Each branch in the function will start a
// separate block which must be kept track of.
class FunctionWalker {
 public:
  explicit FunctionWalker(Function *function) : function_(function) {
    assert(function_);
    block_walker_.reset(Internal::CreateBlockWalker(function_->body));
    
  }
  FunctionWalker(FunctionWalker&& other) = default;
  FunctionWalker& operator=(FunctionWalker&& other) = default;
  virtual ~FunctionWalker() {}

  // Factory for creating a walker at a specific point in a function.
  // The state of the walker will be the same as if the walker had advanced to
  // the statement from the beginning.
  static FunctionWalker *CreateFunctionWalkerAtStatement(
      Function *function, Statement *statement);

  // General function for advancing. Handles branching and gathering of all the
  // necessary data.
  virtual bool Advance();

  // Advances without entering any branches. Like GDB's 'next'.
  // If we are leaving athe true branch of an if, we will enter the else branch.
  virtual bool Next();

  // Information on the current statement/block.
  Statement *GetCurrentStatement() const {
    return block_walker_->WalkerBase::statement_;
  }
  eStatementType GetCurrentStatementType() const {
    return block_walker_->WalkerBase::statement_->get_type();
  }
  Block *GetCurrentBlock() const {
    return block_walker_->WalkerBase::block_;
  }

  // Gets a FunctionWalker for the destination of a Goto statement.
  // Useful for walking between the goto and the destination:
  //   std::unique_ptr<FunctionWalker> dest(src->GetGotoDestination());
  //   if (!src->GotoDestinationIsForward())
  //     while(dest != src) {
  //       do stuff...
  //       dest->Advance();
  //     }
  //
  // Neither walker has any information on which branch the source/dest is in,
  // and would need access to the nested blocks. A possible extension would be
  // to extend this class with a 'GotoFunctionWalker' that could handle this.
  FunctionWalker *GetGotoDestination() {
    return CreateFunctionWalkerAtStatement(
        function_, block_walker_->Internal::WalkerImpl<eGoto>::destination_);
  }
  bool GotoDestinationIsForward() {
    return block_walker_->Internal::WalkerImpl<eGoto>::goto_destination_is_forward_;
  }

  // Equal if they are both at the same point in the same function.
  bool operator==(const FunctionWalker& other) const {
    return function_ == other.function_ &&
        GetCurrentStatement() == other.GetCurrentStatement();
  }
  bool operator!=(const FunctionWalker& other) const {
    return !(*this == other);
  }

  // Branching information.
  int EnteredBranch() const { return blocks_entered_; }
  int ExitedBranch() const { return blocks_exited_; }
 protected:
  // Simple helper function to inform us that there is a block (on a branch
  // statement) that must be entered before we advance.
  bool MustEnterBranch();
  // block_walker must be a raw, unmanaged pointer.
  void EnterBranch(Internal::BlockWalker *block_walker);

  Function *function_;
  std::unique_ptr<Internal::BlockWalker> block_walker_;
  std::stack<std::unique_ptr<Internal::BlockWalker>> nested_blocks_;
  // Represents how many levels of blocks that were changed on the last advance.
  // Blocks entered should be no more than 1, and if it is 1, then the block is
  // entered after the exited blocks. This assumes that there are no empty
  // blocks.
  int blocks_entered_;
  int blocks_exited_;
 private:
  FunctionWalker() {}
  DISALLOW_COPY_AND_ASSIGN(FunctionWalker);
};

}  // namespace Walker
}  // namespace CUDASmith

#endif  // _CUDASMITH_WALKER_H_
