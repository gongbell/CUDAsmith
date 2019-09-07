#include "CUDASmith/Walker.h"

#include <memory>
#include <stack>
#include <vector>

#include "CFGEdge.h"
#include "FactMgr.h"
#include "Function.h"
#include "Statement.h"
#include "StatementFor.h"
#include "StatementGoto.h"
#include "StatementIf.h"

namespace CUDASmith {
namespace Walker {
namespace Internal {

BlockWalker *CreateBlockWalker(Block *block) {
  return new BlockWalker(block);
}

BlockWalker *CreateBlockWalkerAtStatement(Block *block, Statement *statement) {
  std::unique_ptr<BlockWalker> block_walker(new BlockWalker(block));
  assert(block_walker->AdvanceToStatement(statement));
  block_walker->AdvanceSelector(statement);
  return block_walker.release();
}

bool WalkerBase::AdvanceOne() {
  if (!statement_) block_it_ = block_->stms.begin();
  if (block_it_ == block_->stms.end()) {
    statement_ = NULL;
    return false;
  }
  statement_ = *block_it_;
  ++block_it_;
  return true;
}

bool WalkerBase::AdvanceToStatement(Statement *statement) {
  assert(statement != NULL);
  if (!statement_) block_it_ = block_->stms.begin();
  while (block_it_ != block_->stms.end() && *block_it_ != statement)
    ++block_it_;
  if (block_it_ == block_->stms.end()) return false;
  statement_ = *block_it_;
  ++block_it_;
  return true;
}

// Specialisation for a block statement. Not sure if this makes sense...
template <>
bool WalkerImpl<eBlock>::Advance(Statement *statement) {
  std::cout << "Block wut" << std::endl;
  return true;
}

// Specialisation for advancing on a for statement.
bool WalkerImpl<eFor>::Advance(Statement *statement) {
  // I'll worry about const correctness later ;)
  for_body_.reset(CreateBlockWalker(const_cast<Block *>(
      dynamic_cast<StatementFor *>(statement)->get_body())));
  return true;
}

// Specialisation for if-then-else statements. else-ifs are nested in the else
// block.
bool WalkerImpl<eIfElse>::Advance(Statement *statement) {
  StatementIf *statement_if = dynamic_cast<StatementIf *>(statement);
  assert(statement_if != NULL);
  if_body_.reset(CreateBlockWalker(
      const_cast<Block *>(statement_if->get_true_branch()))); // const lol
  Block *else_block = const_cast<Block *>(statement_if->get_false_branch());
  else_body_.reset(else_block != NULL ? CreateBlockWalker(else_block) : NULL);
  return true;
}

// Specialisation for goto statements. A blockWalker will be created for the
// destination of the goto.
bool WalkerImpl<eGoto>::Advance(Statement *statement) {
  std::cout << "Found goto" << std::endl;
  destination_ = const_cast<Statement *>(
      dynamic_cast<StatementGoto *>(statement)->dest); // const lol
  // Find out if it is a forward edge.
  FactMgr *fact_mgr = get_fact_mgr_for_func(statement->func);
  const CFGEdge *cfg_edge = NULL;
  for (unsigned i = 0; i < fact_mgr->cfg_edges.size(); ++i)
    if (statement == fact_mgr->cfg_edges[i]->src) {
      cfg_edge = fact_mgr->cfg_edges[i];
      break;
    }
  assert(cfg_edge != NULL);
  assert(cfg_edge->dest == destination_);
  goto_destination_is_forward_ = !cfg_edge->back_link;
  return true;
}

// No matter what I do, I cannot get rid of this.
// I have tried funciton pointers, traits and even tags.
bool BlockWalker::AdvanceSelector(Statement *statement) {
  assert(statement != NULL);
  switch (statement->get_type()) {
    case eAssign:   return WalkerImpl<eAssign>::Advance(statement);
    case eBlock:    return WalkerImpl<eBlock>::Advance(statement);
    case eFor:      return WalkerImpl<eFor>::Advance(statement);
    case eIfElse:   return WalkerImpl<eIfElse>::Advance(statement);
    case eInvoke:   return WalkerImpl<eInvoke>::Advance(statement);
    case eReturn:   return WalkerImpl<eReturn>::Advance(statement);
    case eContinue: return WalkerImpl<eContinue>::Advance(statement);
    case eBreak:    return WalkerImpl<eBreak>::Advance(statement);
    case eGoto:     return WalkerImpl<eGoto>::Advance(statement);
    case eArrayOp:  return WalkerImpl<eArrayOp>::Advance(statement);
    case eCUDAStatement: return WalkerImpl<eCUDAStatement>::Advance(statement);
    default: assert(false && "Invalid statement.");
  }
}

}  // namespace Internal

using Internal::BlockWalker;
using Internal::WalkerImpl;

FunctionWalker *FunctionWalker::CreateFunctionWalkerAtStatement(
    Function *function, Statement *statement) {
  assert(function != NULL);
  assert(statement != NULL);

  // Iteratively create the block walkers from the most nested block outward.
  std::stack<BlockWalker *> nested_blocks_rev;
  Block *block = statement->parent;
  assert(block != NULL);
  while (block != NULL) {
    BlockWalker *block_walker = Internal::CreateBlockWalkerAtStatement(
        block, statement);
    // Check which branch the previous block was in, and remove it from the
    // parents BlockWalker. Otherwise, we would reenter it when we advance.
    if (!nested_blocks_rev.empty()) {
      BlockWalker *branch = nested_blocks_rev.top();
      if (statement->get_type() == eFor) {
        block_walker->WalkerImpl<eFor>::for_body_.reset();
      } else if (statement->get_type() == eIfElse) {
        block_walker->WalkerImpl<eIfElse>::if_body_.reset();
        if (branch->WalkerBase::block_ ==
            dynamic_cast<StatementIf *>(statement)->get_false_branch())
          block_walker->WalkerImpl<eIfElse>::else_body_.reset();
      }
    }
    // Setup for next iteration.
    block = block->parent;
    // lol const
    statement = const_cast<Statement *>(statement->find_container_stm());
    nested_blocks_rev.push(block_walker);
  }

  // The stack we have created is in the reverse order.
  std::unique_ptr<FunctionWalker> function_walker(new FunctionWalker());
  function_walker->function_ = function;
  while (!nested_blocks_rev.empty()) {
    function_walker->nested_blocks_.emplace(nested_blocks_rev.top());
    nested_blocks_rev.pop();
  }
  function_walker->block_walker_.reset(
      function_walker->nested_blocks_.top().release());
  function_walker->nested_blocks_.pop();
  return function_walker.release();
}

// This probably needs cleaning.
bool FunctionWalker::Advance() {
  blocks_exited_ = 0;
  blocks_entered_ = 0;

  // Check if we must enter a branch before advancing. (If newly constructed,
  // must first Advance).
  eStatementType type;
  if (GetCurrentStatement() == NULL) goto derp;
  type = GetCurrentStatementType();
  if (type == eFor) {
    BlockWalker *block_walker =
        block_walker_->WalkerImpl<eFor>::for_body_.release();
    if (block_walker != NULL) {
      EnterBranch(block_walker);
      ++blocks_entered_;
    }
  } else if (type == eIfElse) {
    BlockWalker *block_walker =
        block_walker_->WalkerImpl<eIfElse>::if_body_.release();
    if (block_walker != NULL) {
      EnterBranch(block_walker);
      ++blocks_entered_;
    }
  }
 derp:

  if (block_walker_->AdvanceBlock()) return true;

  // Reach the end of the current block. If there are any blocks on the stack
  // (i.e. we are in a branch), pop one off and continue.
  do {
    if (nested_blocks_.empty()) return false;
    block_walker_.reset(nested_blocks_.top().release()); // = seems to not work.
    nested_blocks_.pop();
    ++blocks_exited_;
    // If we are exiting a branch, and the statement has multiple branches.
    if (MustEnterBranch()) break;
    // The stuff after the do loop is specifically for multiple branches, so we
    // must return early.
    if (block_walker_->AdvanceBlock()) return true;
  } while (true);

  // Should only get here if we have exited a branch.
  type = GetCurrentStatementType();
  if (type == eFor)
    // We should have already entered the for body at this point. (Dead code)
    assert(block_walker_->WalkerImpl<eFor>::for_body_.get() == NULL);
  if (type == eIfElse) {
    // We should have already entered the true branch at this point.
    assert(block_walker_->WalkerImpl<eIfElse>::if_body_.get() == NULL);
    BlockWalker *block_walker =
        block_walker_->WalkerImpl<eIfElse>::else_body_.release();
    if (block_walker != NULL) {
      EnterBranch(block_walker);
      ++blocks_entered_;
    }
    // Advance if we enter the else block. As long as there are no empty blocks,
    // we should not have to do anything else.
    assert(block_walker_->AdvanceBlock());
  }

  return true;
}

bool FunctionWalker::Next() {
  blocks_exited_ = 0;
  blocks_entered_ = 0;

  // If newly constructed, just use the standard Advance().
  if (GetCurrentStatement() == NULL) return Advance();

  while (!block_walker_->AdvanceBlock()) {
    if (nested_blocks_.empty()) return false;
    block_walker_.reset(nested_blocks_.top().release());
    nested_blocks_.pop();
    ++blocks_exited_;

    // If we were in an if branch, and the else exists, enter it.
    if (GetCurrentStatementType() == eIfElse &&
        block_walker_->WalkerImpl<eIfElse>::else_body_) {
      assert(block_walker_->WalkerImpl<eIfElse>::if_body_.get() == NULL);
      EnterBranch(block_walker_->WalkerImpl<eIfElse>::else_body_.release());
      ++blocks_entered_;
      assert(block_walker_->AdvanceBlock());
      break;
    }
  }
  return true;
}

bool FunctionWalker::MustEnterBranch() {
  eStatementType type = GetCurrentStatementType();
  if (type == eFor)
    return block_walker_->WalkerImpl<eFor>::for_body_ != NULL;
  if (type == eIfElse)
    return block_walker_->WalkerImpl<eIfElse>::if_body_ ||
           block_walker_->WalkerImpl<eIfElse>::else_body_;
  return false;
}

void FunctionWalker::EnterBranch(BlockWalker *block_walker) {
  nested_blocks_.push(std::move(block_walker_));
  block_walker_.reset(block_walker);
}

}  // namespace Walker
}  // namespace CUDASmith
