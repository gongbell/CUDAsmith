#include "CUDASmith/StatementEMI.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "Block.h"
#include "CGContext.h"
#include "CUDASmith/CUDAOptions.h"
#include "CUDASmith/MemoryBuffer.h"
#include "CUDASmith/Walker.h"
#include "ExpressionFuncall.h"
#include "ExpressionVariable.h"
#include "FunctionInvocationBinary.h"
#include "SafeOpFlags.h"
#include "Statement.h"
#include "StatementFor.h"
#include "StatementIf.h"

bool g_FCBoff = false;

namespace CUDASmith {
namespace {
EMIController *emi_controller_inst = NULL;  // Singleton instance.
}  // namespace

StatementEMI *StatementEMI::make_random(CGContext& cg_context) {
  // TODO, better exprs, for now, just do 0>1, 2>3, etc.
  static int item_count = 0;
  assert(item_count < 1024);
  MemoryBuffer *emi_input = EMIController::GetEMIController()->GetEMIInput();
//   MemoryBuffer *item1 = emi_input->itemize({item_count++});
  MemoryBuffer* item1 = emi_input->itemize(std::vector<int> (1, item_count++));
//   MemoryBuffer *item2 = emi_input->itemize({item_count++});
  MemoryBuffer* item2 = emi_input->itemize(std::vector<int> (1, item_count++));
  SafeOpFlags *flags = SafeOpFlags::make_dummy_flags();
  Expression *test = new ExpressionFuncall(*new FunctionInvocationBinary(eCmpLt,
      new ExpressionVariable(*item1), new ExpressionVariable(*item2), flags));
  // True block is a standard random block (that will never run), no false block.
  // TODO make separate context, and discard effects (it is never executed).
  bool prev_emi_context = cg_context.get_emi_context();
  cg_context.set_emi_context(true);
  Block *true_block = Block::make_random(cg_context);
  Block *false_block = Block::make_dummy_block(cg_context);
  cg_context.set_emi_context(prev_emi_context);
  StatementEMI *emi = new StatementEMI(
      cg_context.get_current_block(), new StatementIf(
      cg_context.get_current_block(), *test, *true_block, *false_block));
  // Record information in the controller.
  EMIController *emi_controller = EMIController::GetEMIController();
  emi_controller->AddStatementEMI(emi);
  emi_controller->AddItemisedEMIInput(item1);
  emi_controller->AddItemisedEMIInput(item2);
  return emi;
}

void StatementEMI::Prune() {
  // 'const' pfffft
  PruneBlock(const_cast<Block *>(if_block_->get_true_branch()));
}

void StatementEMI::PruneBlock(Block *block) {
  std::vector<Statement *> del_stms;
  // Statements to be lifted involve modifying vectors we are iterating over.
  // To retain iterator validity, statements are lifted after iteration.
  std::vector<Statement *> lift_stms;
  // Adjust the p_lift probability to give p_lift_adj.
  // As lifting is checked if compound fails, p = (1 - p_compound) * p_lift.
  // So, p_adj = p_lift / (1 - p_compound).
  const int p_lift_adj =
      CUDAOptions::emi_p_compound() == 100 ? CUDAOptions::emi_p_lift() : 100 * (
      ((float)CUDAOptions::emi_p_lift() / 100) /
      (1.0f - ((float)CUDAOptions::emi_p_compound() / 100)));

  for (Statement *st : block->stms) {
    eStatementType st_type = st->eType;
    // If it is a leaf.
    if (st_type != eIfElse && st_type != eFor) {
      if (rnd_flipcoin(CUDAOptions::emi_p_leaf())) del_stms.push_back(st);
      continue;
    }
    // Nested blocks will be pruned regardless of pruning to ensure the random
    // number generator is called the same number of times, otherwise later code
    // that depends on it will produce a different output.
    // Pruning should still be done when lifting.
    if (st_type == eIfElse) {
      StatementIf *st_if = dynamic_cast<StatementIf *>(st);
      assert(st_if != NULL);
      // 'const' haha yeah right.
      PruneBlock(const_cast<Block *>(st_if->get_true_branch()));
      PruneBlock(const_cast<Block *>(st_if->get_false_branch()));
    } else {
      StatementFor *st_for = dynamic_cast<StatementFor *>(st);
      assert(st_for != NULL);
      PruneBlock(const_cast<Block *>(st_for->get_body()));
    }
    // Call BOTH flip_coins, the prevent the RNG from going out of sync.
    bool do_compound = rnd_flipcoin(CUDAOptions::emi_p_compound());
    bool do_lift = rnd_flipcoin(p_lift_adj);
    // Is a compound statement.
    if (do_compound) {
      del_stms.push_back(st);
      continue;
    }
    // Lift case.
    if (!do_lift) continue;
    lift_stms.push_back(st);
    /*if (st_type != eFor) continue;
    // Go through the block deleting cont/break, unless there is a for loop.
    std::vector<Statement *> jmp_stms;
    std::unique_ptr<Walker::FunctionWalker> begin(
        Walker::FunctionWalker::CreateFunctionWalkerAtStatement(func, st));
    begin->Advance();
    std::unique_ptr<Walker::FunctionWalker> end(
        Walker::FunctionWalker::CreateFunctionWalkerAtStatement(func, st));
    end->Next();
    while (*begin != *end) {
      eStatementType st_type = begin->GetCurrentStatementType();
      if (st_type == eFor) {
        begin->Next();
        continue;
      }
      if (st_type == eContinue || st_type == eBreak)
        jmp_stms.push_back(begin->GetCurrentStatement());
      begin->Advance();
    }
    for (Statement *st : jmp_stms) st->parent->remove_stmt(st);*/
  }

  // Remove all the selected statements from the block.
  for (Statement *st : del_stms) block->remove_stmt(st);
  // First check if we are in a for loop before any lifting.
  bool in_loop = false;
  for (Block *nest = block; nest != NULL && !in_loop; nest = nest->parent)
    if (nest->looping) in_loop = true;
  // Lift marked statements.
  std::vector<Statement *>::iterator position = block->stms.begin();
  for (Statement *st : lift_stms) {
    eStatementType st_type = st->eType;
    assert(st_type == eIfElse || st_type == eFor);
    for (; *position != st; ++position) assert(position != block->stms.end());
    if (st_type == eIfElse) {
      StatementIf *st_if = dynamic_cast<StatementIf *>(st);
      assert(st_if != NULL);
      position = MergeBlock(position, block,
          const_cast<Block *>(st_if->get_true_branch()));
      position = MergeBlock(position, block,
          const_cast<Block *>(st_if->get_false_branch()));
    } else {
      StatementFor *st_for = dynamic_cast<StatementFor *>(st);
      assert(st_for != NULL);
      if (!in_loop) RemoveBreakContinue(const_cast<Block *>(st_for->get_body()));
      position = MergeBlock(position, block,
          const_cast<Block *>(st_for->get_body()));
    }
  }
  // Now remove all lifted statements.
  for (Statement *st : lift_stms) block->remove_stmt(st);
}

std::vector<Statement *>::iterator StatementEMI::MergeBlock(
    std::vector<Statement *>::iterator position, Block *former, Block *merger) {
  former->local_vars.insert(former->local_vars.end(),
      merger->local_vars.begin(), merger->local_vars.end());
  for (Statement *st : merger->deleted_stms) st->parent = former;
  former->deleted_stms.insert(former->deleted_stms.end(),
      merger->deleted_stms.begin(), merger->deleted_stms.end());
  int stmt_count = merger->stms.size();
  for (Statement *st : merger->stms) st->parent = former;
  // GCC Bug 55817
  //position = former->stms.insert(position,
  //    merger->stms.begin(), merger->stms.end());
  {
    stmt_count += std::distance(former->stms.begin(), position);
    former->stms.insert(position, merger->stms.begin(), merger->stms.end());
    position = former->stms.begin();
  }
  position += stmt_count;
  merger->local_vars.clear();
  merger->deleted_stms.clear();
  merger->stms.clear();
  return position;
}

void StatementEMI::RemoveBreakContinue(Block *block) {
  std::vector<Statement *> del_stms;
  for (Statement *st : block->stms) {
    eStatementType st_type = st->eType;
    if (st_type == eContinue || st_type == eBreak) del_stms.push_back(st);
    if (st_type == eIfElse) {
      // This is why we need the walker >:(
      StatementIf *st_if = dynamic_cast<StatementIf *>(st);
      assert(st_if != NULL);
      RemoveBreakContinue(const_cast<Block *>(st_if->get_true_branch()));
      RemoveBreakContinue(const_cast<Block *>(st_if->get_false_branch()));
    }
  }
  for (Statement *st : del_stms) block->remove_stmt(st);
}

EMIController *EMIController::GetEMIController() {
  if (emi_controller_inst == NULL) emi_controller_inst = CreateEMIController();
  return emi_controller_inst;
}

void EMIController::ReleaseEMIController() {
  delete emi_controller_inst;
  emi_controller_inst = NULL;
}

EMIController *EMIController::CreateEMIController() {
  // TODO this may need to be __const
  return new EMIController(MemoryBuffer::CreateMemoryBuffer(
      MemoryBuffer::kGlobal, "emi_input", &Type::get_simple_type(eInt), NULL,
      {1024}));
}

bool EMIController::RemoveStatementEMI(StatementEMI *emi) {
  std::vector<StatementEMI *>::iterator pos =
      std::find(emi_sections_.begin(), emi_sections_.end(), emi);
  if (pos != emi_sections_.end()) {
    emi_sections_.erase(pos);
    return true;
  }
  return false;
}

void EMIController::AddItemisedEMIInput(MemoryBuffer *item) {
  assert(item->collective == emi_input_.get());
  itemised_emi_input_.push_back(item);
}

void EMIController::PruneEMISections() {
  for (StatementEMI *emi : emi_sections_) emi->Prune();
}

}  // namespace CUDASmith
