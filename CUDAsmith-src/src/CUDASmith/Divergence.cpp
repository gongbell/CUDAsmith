#include "CUDASmith/Divergence.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <utility>
#include <vector>

#include "ArrayVariable.h"
#include "CUDASmith/CUDAExpression.h"
#include "CUDASmith/Walker.h"
#include "Expression.h"
#include "ExpressionAssign.h"
#include "ExpressionComma.h"
#include "ExpressionFuncall.h"
#include "ExpressionVariable.h"
#include "Function.h"
#include "FunctionInvocation.h"
#include "FunctionInvocationUser.h"
#include "Statement.h"
#include "StatementArrayOp.h"
#include "StatementAssign.h"
#include "StatementBreak.h"
#include "StatementContinue.h"
#include "StatementExpr.h"
#include "StatementFor.h"
#include "StatementIf.h"
#include "StatementReturn.h"
#include "Variable.h"
#include "VariableSelector.h"

namespace CUDASmith {
namespace Internal {

SubBlock *SubBlock::SplitAt(Statement *statement) {
  assert(statement->parent == block_);
  if (statement == begin_statement_) return NULL;
  
  SubBlock *orig_next_sub_block = next_sub_block_.get();
  next_sub_block_.reset(new SubBlock(block_, statement, end_statement_));
  next_sub_block_->prev_sub_block_ = this;
  next_sub_block_->next_sub_block_.reset(orig_next_sub_block);

  // Finally, need to set the end statement of our original sub block to the
  // statement before the one we split at.
  std::vector<Statement *>::iterator block_it =
      std::find(block_->stms.begin(), block_->stms.end(), statement);
  assert(block_it != block_->stms.begin());
  assert(block_it != block_->stms.end());
  --block_it;
  end_statement_ = *block_it;

  return next_sub_block_.get();
}

}  // namespace Internal

using Internal::SubBlock;
using Internal::SavedState;

void FunctionDivergence::GetDivergentCodeSectionsForBlock(Block *block,
    std::vector<std::pair<Statement *, Statement *>> *divergent_sections) {
  assert(divergent_sections != NULL);
  if (status_ != kDone) return;
  SubBlock *sub_block = block_to_sub_block_final_[block].get();

  for (; sub_block != NULL; sub_block = sub_block->next_sub_block_.get()) {
    if (sub_block_div_final_[sub_block])
      divergent_sections->push_back(std::make_pair(
          sub_block->begin_statement_, sub_block->end_statement_));
    else
      for (const std::pair<Statement *, Block*>& branch :
          scope_levels_final_[sub_block])
        GetDivergentCodeSectionsForBlock(branch.second, divergent_sections);
  }
}

void FunctionDivergence::ProcessWithContext(const std::vector<bool>& parameters,
    const std::map<const Variable *, std::set<const Variable *> *>&
        param_derefs_to,
    const std::map<const Variable *, bool>& param_ref_div, bool divergent) {
  assert(status_ != kMidProcess && "Loopy program.");
  status_ = kMidProcess;

  // Reset the object state.
  sub_block_div_.clear();
  variable_div_.clear();
  divergent_value_ = false;
  var_derefs_to_.clear();
  return_derefs_to_.clear();

  // Put passed parameters into the variable divergence map.
  assert(parameters.size() == function_->param.size());
  for (unsigned param_idx = 0; param_idx < parameters.size(); ++param_idx)
    variable_div_[function_->param[param_idx]] = parameters[param_idx];

  // Put in the extra contextual information.
  for (auto item : param_derefs_to) var_derefs_to_[item.first] = *item.second;
  for (auto item : param_ref_div) variable_div_[item.first] = item.second;

  // Set up state to begin processing.
  sub_block_ = block_to_sub_block_final_[function_->body].get();
  if (sub_block_ == NULL) {
    sub_block_ = new SubBlock(function_->body);
    block_to_sub_block_final_[function_->body].reset(sub_block_);
  }
  function_walker_.reset(new Walker::FunctionWalker(function_));
  ProcessBlockStart(function_->body);
  divergent_ = divergent;
  sub_block_div_[sub_block_] = divergent;

  // Main processing loop.
  while (function_walker_->Advance()) ProcessStep();
  // Post process blocks that end at the end of the function.
  int blocks_exited = function_walker_->ExitedBranch();
  for (; blocks_exited > 0; --blocks_exited) ProcessBlockEnd();

  // Check for inconsistencies.
  assert(nested_blocks_.empty());
  assert(saved_if_.empty());

  ProcessFinalise();
  status_ = kDone;
}

void FunctionDivergence::ProcessStep() {
  Statement *statement = function_walker_->GetCurrentStatement();
  eStatementType type = function_walker_->GetCurrentStatementType();

  // May have to move on to the next sub block.
  if (sub_block_->next_sub_block_ &&
      sub_block_->next_sub_block_->begin_statement_ == statement)
    sub_block_ = sub_block_->next_sub_block_.get();

  // Entering and exiting branches. The walker may exit multiple branches at
  // the same step, and some blocks will need post processing.
  // Only one block can be entered at a time though, we will have updated the
  // context in the previous step when we processed the branch statement.
  int blocks_exited = function_walker_->ExitedBranch();
  for (; blocks_exited > 0; --blocks_exited) ProcessBlockEnd();
  int blocks_entered = function_walker_->EnteredBranch();
  assert(blocks_entered <= 1);
  if (blocks_entered == 1)
    ProcessBlockStart(function_walker_->GetCurrentBlock());

  switch (type) {
    case eAssign:   ProcessStatementAssign(statement); break;
    case eBlock:    std::cout << "wut" << std::endl;   break;
    case eFor:      ProcessStatementFor(statement);    break;
    case eIfElse:   ProcessStatementIf(statement);     break;
    case eInvoke:   ProcessStatementInvoke(statement); break;
    case eReturn:   ProcessStatementReturn(statement); break;
    case eContinue:
    case eBreak:    ProcessStatementJump(statement);   break;
    case eGoto:     ProcessStatementGoto(statement);   break;
    case eCUDAStatement: assert(false); break;
    case eArrayOp:  ProcessStatementArray(statement); assert(false); break;
  }
}

void FunctionDivergence::ProcessBlockStart(Block *block) {
  for (Variable *var : block->local_vars) {
    assert(var->init != NULL);
    IsAssignmentDivergent(Lhs(*var), *var->init);
  }
  // Defer handling array elements to when they are accessed.
}

void FunctionDivergence::ProcessBlockEnd() {
  // End of function, leave for ProcessFinalise().
  if (nested_blocks_.empty()) return;

  // Post-process if for or if.
  std::pair<SubBlock *, Statement *> *prev_level = &nested_blocks_.back();
  eStatementType type = prev_level->second->get_type();
  if (type == eFor) {
    ProcessEndStatementFor(prev_level->second);
  } else if (type == eIfElse) {
    ProcessEndStatementIf(prev_level->second);
    // Let EndIf handle popping off the previous context. Invalidates prev_level
    return;
  }

  // Pop off the previous context.
  sub_block_ = prev_level->first;
  divergent_ = sub_block_div_[sub_block_];
  nested_blocks_.pop_back();
}

void FunctionDivergence::ProcessStatementAssign(Statement *statement) {
  StatementAssign *statement_ass = dynamic_cast<StatementAssign *>(statement);
  assert(statement_ass != NULL);
  IsAssignmentDivergent(*statement_ass);
}

void FunctionDivergence::ProcessStatementFor(Statement *statement) {
  StatementFor *statement_for = dynamic_cast<StatementFor *>(statement);
  assert(statement_for != NULL);

  ProcessStatementAssign(const_cast<StatementAssign *>(statement_for->get_init())); //const again.
  SubBlock *branch_block =
      GetSubBlockForBranch(statement, const_cast<Block *>(statement_for->get_body())); //const again.
  bool *branch_block_div = &sub_block_div_[branch_block];
  *branch_block_div |=
      IsExpressionDivergent(*statement_for->get_test()) || divergent_;
  div_->saved_states_.emplace_back(SaveState(statement));
  nested_blocks_.push_back(std::make_pair(sub_block_, statement));
  sub_block_ = branch_block;
  divergent_ = *branch_block_div;
  // When the block has been fully processed, we must go back and test the incr
  // expression and the test again.
}

void FunctionDivergence::ProcessStatementIf(Statement *statement) {
  StatementIf *statement_if = dynamic_cast<StatementIf *>(statement);
  assert(statement_if != NULL);

  SubBlock *branch_block =
      GetSubBlockForBranch(statement, const_cast<Block *>(statement_if->get_true_branch())); //const again.
  bool expr_div = IsExpressionDivergent(*statement_if->get_test());
  bool *branch_block_div = &sub_block_div_[branch_block];
  *branch_block_div |= expr_div || divergent_;

  // If there is a false branch, set the else block divergence so we don't
  // reevaluate the test.
  if (statement_if->get_false_branch() != NULL) {
    SubBlock *false_branch_block =
        GetSubBlockForBranch(statement, const_cast<Block *>(statement_if->get_false_branch())); //const again.
    bool *false_branch_block_div = &sub_block_div_[false_branch_block];
    *false_branch_block_div |= expr_div || divergent_;
  }

  div_->saved_states_.emplace_back(SaveState(statement));
  nested_blocks_.push_back(std::make_pair(sub_block_, statement));
  sub_block_ = branch_block;
  divergent_ = *branch_block_div;
}

void FunctionDivergence::ProcessStatementInvoke(Statement *statement) {
  StatementExpr *statement_expr = dynamic_cast<StatementExpr *>(statement);
  assert(statement_expr != NULL);
  const FunctionInvocation& invoke = *statement_expr->get_invoke();
  if (invoke.invoke_type == eBinaryPrim) return; // Assume noop.
  assert(invoke.invoke_type == eFuncCall);
  const FunctionInvocationUser& invoke_user =
      dynamic_cast<const FunctionInvocationUser&>(invoke);
  IsFunctionCallDivergent(invoke_user, NULL);
}

void FunctionDivergence::ProcessStatementReturn(Statement *statement) {
  StatementReturn *statement_ret = dynamic_cast<StatementReturn *>(statement);
  assert(statement_ret != NULL);

  if (divergent_) MarkSubBlockDivergentViral(sub_block_);

  if (function_->return_type->get_indirect_level() == 0) {
    divergent_value_ |=
        IsExpressionDivergent(*statement_ret->get_var()) || divergent_;
    return;
  }

  // TODO split at branch point if divergent.

  std::set<const Variable *> var_refs;
  divergent_value_ |= GetExpressionVariableReferences(
      *statement_ret->get_var(), &var_refs);
  divergent_value_ |= divergent_;
  return_derefs_to_.insert(var_refs.begin(), var_refs.end());    
}

void FunctionDivergence::ProcessStatementJump(Statement *statement) {
  eStatementType type = statement->get_type();
  const Expression *expr;
  const Block *block;
  if (type == eContinue) {
    StatementContinue *statement_cont =
        dynamic_cast<StatementContinue *>(statement);
    assert(statement_cont != NULL);
    expr = &statement_cont->test;
    block = &statement_cont->loop_blk;
  } else {
    StatementBreak *statement_break = dynamic_cast<StatementBreak *>(statement);
    assert(statement_break != NULL);
    expr = &statement_break->test;
    block = &statement_break->loop_blk;
  }

  // Both jumps work in the same way. If the statement is in a divergent block,
  // then the corresponding for block must be divergent.
  if (IsExpressionDivergent(*expr) || divergent_) {
    SubBlock *sub_block = block_to_sub_block_final_[const_cast<Block *>(block)].get(); //const again, will todo later.
    assert(sub_block != NULL);
    MarkSubBlockDivergentViral(sub_block);
  }
}

void FunctionDivergence::ProcessStatementGoto(Statement *statement) {
  // TODO
  assert(false && "Gotos not yet implemented.");
}

void FunctionDivergence::ProcessStatementArray(Statement *statement) {
  StatementArrayOp *statement_arr = dynamic_cast<StatementArrayOp *>(statement);
  assert(statement_arr != NULL);
  // Remove any references to array members.
  const ArrayVariable *array_var = statement_arr->array_var;
  bool expr_div = IsExpressionDivergent(*statement_arr->init_value);
  std::map<const Variable *, bool> *div_map = array_var->is_global() ?
      &div_->global_var_div_ : &variable_div_;
  for (auto map_it = div_map->begin(); map_it != div_map->end();) {
    if (map_it->first->get_collective() == array_var) {
      auto map_it_rem = map_it++;
      div_map->erase(map_it_rem);
    } else {
      ++map_it;
    }
  }
  SetVariableDivergence(array_var, expr_div || divergent_);
  assert(array_var->type->get_indirect_level() == 0 && "Not implemented.");
}

void FunctionDivergence::ProcessEndStatementFor(Statement *statement) {
  // When a for block has been finished, we may need to go back and re-process
  // any number of times. In general, we will need to recompute if the program
  // elements we have marked as divergent has changed after processing.
  // We think of processing the for block as a function f(X) = X, where X is the
  // set of program elements that are divergent, and we compute it as a fixed
  // point. This is unfortunately very expensive.
  StatementFor *statement_for = dynamic_cast<StatementFor *>(statement);
  assert(statement_for != NULL);

  // Process for inc and test again.
  ProcessStatementAssign(const_cast<StatementAssign *>(statement_for->get_incr())); //const again.
  sub_block_ = GetSubBlockForBranchFromBlock(nested_blocks_.back().first,
       statement, const_cast<Block *>(statement_for->get_body())); //const again, will todo later.
  bool *branch_block_div = &sub_block_div_[sub_block_];
  *branch_block_div |=
      IsExpressionDivergent(*statement_for->get_test()) || divergent_;

  // Restore, check for changes.
  assert(!div_->saved_states_.empty());
  SavedState *saved_state = div_->saved_states_.back();
  assert(saved_state->function_owner_ == this);
  assert(saved_state->statement_ == statement);
  bool change = RestoreAndMergeSavedState(saved_state);
  div_->saved_states_.pop_back();

  std::unique_ptr<Walker::FunctionWalker> saved_walker;
  if (change) saved_walker.reset(function_walker_.release());
  while (change) {
    function_walker_.reset(
        Walker::FunctionWalker::CreateFunctionWalkerAtStatement(
        function_, statement));
    divergent_ = *branch_block_div;
    div_->saved_states_.emplace_back(SaveState(statement));

    // Loop through the for manually. Do not assert the Advance, as if the for
    // ends at the end of the function, Advance returns false.
    assert(function_walker_->Advance());
    while (*function_walker_ != *saved_walker) {
      ProcessStep();
      function_walker_->Advance();
    }
    // Any nested blocks that also end here should be processed.
    for (; nested_blocks_.back().second != statement;)
      ProcessBlockEnd();

    // Process for inc and test again.
    ProcessStatementAssign(const_cast<StatementAssign *>(statement_for->get_incr())); //const again.
    sub_block_ = GetSubBlockForBranchFromBlock(nested_blocks_.back().first,
        statement, const_cast<Block *>(statement_for->get_body())); //const again, will todo later.
    branch_block_div = &sub_block_div_[sub_block_];
    *branch_block_div |=
        IsExpressionDivergent(*statement_for->get_test()) || divergent_;

    // Restore, check for changes.
    assert(!div_->saved_states_.empty());
    SavedState *saved_state = div_->saved_states_.back();
    assert(saved_state->function_owner_ == this);
    assert(saved_state->statement_ == statement);
    change = RestoreAndMergeSavedState(saved_state);
    div_->saved_states_.pop_back();
  }
}

void FunctionDivergence::ProcessEndStatementIf(Statement *statement) {
  // If it was the simple case of no false branch, nothing to do.
  StatementIf *statement_if = dynamic_cast<StatementIf *>(statement);
  assert(statement_if != NULL);

  // If there is a false branch, this will be called twice (after processing
  // each branch).
  if (statement_if->get_false_branch() != NULL) {
    // True branch.
    if (saved_if_.empty() || saved_if_.top().first != statement) {
      saved_if_.emplace(std::make_pair(
          statement, std::unique_ptr<SavedState>(SaveState(statement))));
      sub_block_ = GetSubBlockForBranchFromBlock(nested_blocks_.back().first,
          statement, const_cast<Block *>(statement_if->get_false_branch())); // lol const
      divergent_ = sub_block_div_[sub_block_];
      return;
    }

    // False branch.
    RestoreAndMergeSavedState(saved_if_.top().second.get());
    saved_if_.pop();
  }

  assert(!div_->saved_states_.empty());
  SavedState *saved_state = div_->saved_states_.back();
  assert(saved_state->function_owner_ == this);
  assert(saved_state->statement_ == statement);
  RestoreAndMergeSavedState(saved_state);
  div_->saved_states_.pop_back();
  // Pop off the previous context.
  std::pair<SubBlock *, Statement *> *prev_level = &nested_blocks_.back();
  sub_block_ = prev_level->first;
  divergent_ = sub_block_div_[sub_block_];
  nested_blocks_.pop_back();
}

void FunctionDivergence::ProcessFinalise() {
  MergeMap(&sub_block_div_final_, &sub_block_div_);
  divergent_value_final_ = divergent_value_;
}

bool FunctionDivergence::IsExpressionDivergent(const Expression& expression) {
  eTermType type = expression.term_type;
  if (type == eConstant) return false;
  if (type == eAssignment)
    return IsAssignmentDivergent(
        *dynamic_cast<const ExpressionAssign&>(expression).get_stm_assign());

  if (type == eVariable) {
    const ExpressionVariable& expr_var =
        dynamic_cast<const ExpressionVariable&>(expression);
    std::set<const Variable *> rhs_var_derefs;
    bool div = DereferencePointerVariable(
        *expr_var.get_var(), expr_var.get_indirect_level(), &rhs_var_derefs);
    // Ewwww lambda (because of *var).
    return div || std::any_of(rhs_var_derefs.begin(), rhs_var_derefs.end(),
        [this](const Variable *var){return IsVariableDivergent(*var);});
  }

  if (type == eFunction) {
    const ExpressionFuncall& expr_fun =
        dynamic_cast<const ExpressionFuncall&>(expression);
    const FunctionInvocation *invoke = expr_fun.get_invoke();
    if (invoke->invoke_type == eFuncCall) {
      const FunctionInvocationUser& invoke_user =
          dynamic_cast<const FunctionInvocationUser&>(*invoke);
      return IsFunctionCallDivergent(invoke_user, NULL);
    }
    // The rest of these invokes are binary and unary ops, the properties of
    // which are unimportant.
    if (invoke->invoke_type == eBinaryPrim) {
      assert(invoke->param_value.size() == 2);
      // Both sides must be evaluated, do not merge into a single statement.
      bool param1_div = IsExpressionDivergent(*invoke->param_value[0]);
      bool param2_div = IsExpressionDivergent(*invoke->param_value[1]);
      return param1_div || param2_div;
    }
    assert(invoke->invoke_type == eUnaryPrim);
    assert(invoke->param_value.size() == 1);
    return IsExpressionDivergent(*invoke->param_value[0]);
  }

  if (type == eCommaExpr) {
    const ExpressionComma& expr_comma =
        dynamic_cast<const ExpressionComma&>(expression);
    IsExpressionDivergent(*expr_comma.get_lhs());
    return IsExpressionDivergent(*expr_comma.get_rhs());
  }

  if (type == eCLExpression)
      return dynamic_cast<const CUDAExpression&>(expression).IsDivergent();

  assert(type == eLhs);
  assert(false && "Expression should not be an Lhs.");
  return false;
}

bool FunctionDivergence::IsVariableDivergent(const Variable& variable) {
  bool global = variable.is_global();
  std::map<const Variable *, bool>& div_map =
      global ? div_->global_var_div_ : variable_div_;
  std::map<const Variable *, bool> SavedState:: *SaveMapPtr =
      global ? &SavedState::global_var_div_ : &SavedState::variable_div_;

  // Special case for arrays. For an itemised array member, only process its
  // initialisation when necessary.
  if (variable.isArray) {
    bool found_entry;
    bool div;
    div = SearchMap(div_map, SaveMapPtr, &variable, global, &found_entry);
    if (found_entry) return div;
    const ArrayVariable& var_arr = dynamic_cast<const ArrayVariable&>(variable);
    const Variable *coll = var_arr.get_collective();
    assert(&variable != coll);
    // We would prefer to call IsAssignmentDivergent, but we are not in the same
    // context as when it was initialised. Array initialiser are simple though
    // (either a constant or taking the address of a var), so we just repeat what
    // is necessary.
    // TODO can't do atm because of how arrays are done.
    assert(false);
  }

  bool unused_b;
  return SearchMap(div_map, SaveMapPtr, &variable, global, &unused_b);
}

bool FunctionDivergence::IsSubBlockDivergent(SubBlock *sub_block) {
  bool unused_b;
  return SearchMap(
      sub_block_div_, &SavedState::sub_block_div_, sub_block, false, &unused_b);
}

bool FunctionDivergence::IsFunctionCallDivergent(
    const FunctionInvocationUser& invoke,
    std::set<const Variable *> *return_refs) {
  Function *callee = const_cast<Function *>(invoke.get_func()); //const again, will todo later.
  const std::vector<Variable *>& param_vars = callee->param;
  std::vector<bool> param_div;

  assert(invoke.param_value.size() == param_vars.size());
  for (unsigned param_idx = 0; param_idx < param_vars.size(); ++param_idx)
    param_div.push_back(IsAssignmentDivergent(
        Lhs(*param_vars[param_idx]), *invoke.param_value[param_idx]));

  // For parameters that are pointers, the dereferencing information must be
  // passed into the function. Must also inform it which of them are divergent.
  // We do it manually, instead of using DereferencePointerVaribale.
  std::map<const Variable *, std::set<const Variable *> *> param_derefs_to;
  std::map<const Variable *, bool> param_ref_div;
  for (Variable *param_var : callee->param) {
    std::set<const Variable *> vars_to_deref({param_var});
    for (int deref_lvl = param_var->type->get_indirect_level(); deref_lvl > 0;
        --deref_lvl) {
      std::set<const Variable *> new_vars_to_deref;
      for (const Variable *deref_var : vars_to_deref) {
        // Don't bother dereferencing global.
        if (deref_var->is_global()) continue;
        auto map_it = var_derefs_to_.find(deref_var);
        if(map_it == var_derefs_to_.end()) continue;
        param_derefs_to[map_it->first] = &map_it->second;
        // csmith will not pass pointers to other parameters, so ignore them.
        for (const Variable *var : map_it->second)
          if (!var->is_argument()) new_vars_to_deref.insert(var);
      }
      for (const Variable *new_deref_var : new_vars_to_deref) {
        if (new_deref_var->is_global()) continue;
        param_ref_div[new_deref_var] = IsVariableDivergent(*new_deref_var);
      }
      vars_to_deref = std::move(new_vars_to_deref);
    }
  }

  // Retrieve or create new FunctionDivergence for the function.
  std::unique_ptr<FunctionDivergence> *func_div = &div_->function_div_[callee];
  if (func_div->get() == NULL)
    func_div->reset(new FunctionDivergence(div_, callee));
  (*func_div)->ProcessWithContext(
    param_div, param_derefs_to, param_ref_div, divergent_);

  // Retrieve any information relevant to the calling context. For local
  // pointers passed by pointers, check whether they may point to extra global
  // vars.
  bool div = (*func_div)->divergent_value_final_;
  if (return_refs != NULL)
    *return_refs = std::move((*func_div)->return_derefs_to_);
  for (auto& it_pair : param_derefs_to) {
    const Variable *passed_var = it_pair.first;
    if (passed_var->is_global() || passed_var->is_argument()) continue;
    const std::set<const Variable *>& passed_var_derefs =
        (*func_div)->var_derefs_to_[passed_var];
    std::set<const Variable *> *our_var_derefs = &var_derefs_to_[passed_var];
    for (const Variable *var_deref : passed_var_derefs)
      if (var_deref->is_global()) our_var_derefs->insert(var_deref);
    SetVariableDivergence(passed_var, (*func_div)->variable_div_[passed_var]);
  }

  // Clean up.
  for (Variable *param_var : callee->param) {
    var_derefs_to_.erase(param_var);
    variable_div_.erase(param_var);
  }

  return div;
}

bool FunctionDivergence::IsAssignmentDivergent(
    const Lhs& lhs, const Expression& expr) {
  // Variables in an Lhs have a variable and a type, the variable is always
  // dereferenced/addressed so the type of the variable matches the type.
  // Pointers may be reseated, so if the dereferenced type is still a pointer,
  // we have to update accordingly.
  std::set<const Variable *> lhs_derefs_to;
  bool deref_div = DereferencePointerVariable(*lhs.get_var(),
      lhs.get_indirect_level(), &lhs_derefs_to);
  
  // At this point, we have a set of vars that could all be updated (or just one
  // if we didn't have a pointer). We do one of two different things depending
  // on whether the expr type is a pointer or not.
  //assert(lhs.get_type().is_equivalent(&expr.get_type()));
  if (lhs.get_type().get_indirect_level() > 0) {
    std::set<const Variable *> rhs_var_derefs;
    bool rhs_div = GetExpressionVariableReferences(expr, &rhs_var_derefs);

    // Update pointer dereference maps.
    for (const Variable *lhs_var : lhs_derefs_to) {
      std::set<const Variable *> *lhs_var_derefs;
      lhs_var_derefs = lhs_var->is_global() ?
          &div_->global_var_derefs_to_[lhs_var] : &var_derefs_to_[lhs_var];
      // For now, do not clear lhs. As we do not properly save pointers.
      // TODO uncomment when pointers properly saved.
      //if (!deref_div && lhs_derefs_to.size() == 1 && !divergent_)
      //  lhs_var_derefs->clear();
      lhs_var_derefs->insert(rhs_var_derefs.begin(), rhs_var_derefs.end());
      SetVariableDivergence(lhs_var, rhs_div || deref_div || divergent_);
    }
    return rhs_div;
  }

  // Do not set variable to convergent if the lhs dereferences to multiple vars.
  bool ass_div = IsExpressionDivergent(expr);
  if (lhs_derefs_to.size() <= 1 || (ass_div || divergent_))
    for (const Variable *var : lhs_derefs_to)
      SetVariableDivergence(var, ass_div || divergent_);
  return ass_div;
}

bool FunctionDivergence::DereferencePointerVariable(
    const Variable& var, int deref_level,
    std::set<const Variable *> *deref_vars) {
  assert(deref_vars != NULL);
  deref_vars->clear();
  assert(var.type->get_indirect_level() >= deref_level);
  deref_vars->insert(&var);
  bool deref_div = false;

  for(; deref_level > 0; --deref_level) {
    assert(!deref_vars->empty() && "Dereferencing NULL.");
    std::set<const Variable *> next_deref_level;
    for (const Variable *deref_var : *deref_vars) {
      deref_div |= IsVariableDivergent(*deref_var);
      std::set<const Variable *> *derefs = deref_var->is_global() ?
          &div_->global_var_derefs_to_[deref_var] : &var_derefs_to_[deref_var];
      next_deref_level.insert(derefs->begin(), derefs->end());
    }
    *deref_vars = std::move(next_deref_level);
  }
  return deref_div;
}

bool FunctionDivergence::GetExpressionVariableReferences(const Expression& expr,
    std::set<const Variable *> *var_refs) {
  assert(var_refs != NULL);
  var_refs->clear();
  bool div = false;

  // Referencing NULL.
  if (expr.term_type == eConstant) return div;

  // Why would there be a comma here :/
  if (expr.term_type == eCommaExpr) {
    const ExpressionComma& expr_comma =
        dynamic_cast<const ExpressionComma&>(expr);
    IsExpressionDivergent(*expr_comma.get_lhs());
    return GetExpressionVariableReferences(*expr_comma.get_rhs(), var_refs);
  }

  // Nested assigns, just forward the params. Would be better to return rhs refs
  // instead of lhs ver refs.
  if (expr.term_type == eAssignment) {
    const ExpressionAssign& expr_ass =
        dynamic_cast<const ExpressionAssign&>(expr);
    const StatementAssign& stm_ass = *expr_ass.get_stm_assign();
    IsAssignmentDivergent(stm_ass);
    return GetExpressionVariableReferences(
        ExpressionVariable(*stm_ass.get_lhs()->get_var()), var_refs);
  }
  
  // Calling function, that returns pointer type.
  if (expr.term_type == eFunction) {
    const ExpressionFuncall& expr_fun =
        dynamic_cast<const ExpressionFuncall&>(expr);
    const FunctionInvocation *invoke = expr_fun.get_invoke();
    assert(invoke->invoke_type == eFuncCall);
    const FunctionInvocationUser& invoke_user =
        dynamic_cast<const FunctionInvocationUser&>(*invoke);
    div = IsFunctionCallDivergent(invoke_user, var_refs);
    return div;
  }

  // Can only be referencing variables.
  assert(expr.term_type == eVariable);
  const ExpressionVariable& expr_var =
      dynamic_cast<const ExpressionVariable&>(expr);
  const Variable *rhs_var = expr_var.get_var();
  int expr_deref_level = expr_var.get_indirect_level();

  if (expr_deref_level == -1)
    // Are we taking the address of another variable.
    var_refs->insert(rhs_var);
  else
    div = DereferencePointerVariable(*rhs_var, expr_deref_level + 1, var_refs);

  return div;
}

void FunctionDivergence::MarkSubBlockDivergentViral(SubBlock *sub_block) {
  // If it is already marked as divergent, assume there is nothing to do.
  bool *sub_block_div = &sub_block_div_[sub_block];
  if (*sub_block_div) return;
  *sub_block_div = true;

  // Keep going until we reach the last linked sub block.
  for (; sub_block != NULL; sub_block = sub_block->next_sub_block_.get())
    for (auto& branch : scope_levels_final_[sub_block])
      MarkSubBlockDivergentViral(
          block_to_sub_block_final_[branch.second].get());
}

// TODO array ops
void FunctionDivergence::SetVariableDivergence(
    const Variable *var, bool divergent) {
  if (var->is_global())
    div_->global_var_div_[var] = divergent;
  else
    variable_div_[var] = divergent;
}

SubBlock *FunctionDivergence::GetSubBlockForBranchFromBlock(
    SubBlock *sub_block, Statement *statement, Block *block) {
  // Make sure the branch is tracked in the scope levels.
  std::vector<std::pair<Statement *, Block *>> *scope_level =
      &scope_levels_final_[sub_block];
  std::pair<Statement *, Block *> branch = std::make_pair(statement, block);
  auto it = std::find(scope_level->begin(), scope_level->end(), branch);
  if (it == scope_level->end()) scope_level->push_back(branch);

  // find/construct the sub block that sits at the start of the block.
  std::unique_ptr<SubBlock> *branch_block = &block_to_sub_block_final_[block];
  if (branch_block->get() == NULL) branch_block->reset(new SubBlock(block));
  return branch_block->get();
}

// TODO Save pointer state.
SavedState *FunctionDivergence::SaveState(Statement *statement) {
  SavedState *saved_state = new SavedState(this, statement);

  // Could put this in the initialiser list, which would also stop default
  // constructing these maps, but this is fine :>
  saved_state->sub_block_div_ = std::move(sub_block_div_);
  saved_state->variable_div_ = std::move(variable_div_);
  sub_block_div_.clear();
  variable_div_.clear();

  saved_state->global_var_div_ = std::move(div_->global_var_div_);
  div_->global_var_div_.clear();

  return saved_state;
}

bool FunctionDivergence::RestoreAndMergeSavedState(SavedState *saved_state) {
  assert(saved_state != NULL);
  bool change = false;
  
  // Merge this functions sub blocks.
  change |= MergeMap(&saved_state->sub_block_div_, &sub_block_div_);
  sub_block_div_ = std::move(saved_state->sub_block_div_);
  // Restore local variables.
  change |= MergeMap(&saved_state->variable_div_, &variable_div_);
  variable_div_ = std::move(saved_state->variable_div_);
  // Restore global variables.
  change |= MergeMap(&saved_state->global_var_div_, &div_->global_var_div_);
  div_->global_var_div_ = std::move(saved_state->global_var_div_);

  return change;
}

// Must only be used in this file, will link fail otherwise.
// Move to the header file if you intend to use elsewhere.
template <typename T>
bool FunctionDivergence::MergeMap(
    std::map<T *, bool> *saved, std::map<T *, bool> *merger) {
  bool change = false;
  for (auto map_it = merger->begin(); map_it != merger->end(); ++map_it) {
    bool *saved_div = &(*saved)[map_it->first];
    if (!*saved_div && map_it->second) {
      change = true;
      *saved_div = true;
    }
  }
  return change;
}

// Same restrictions as above.
template<typename T>
bool FunctionDivergence::SearchMap(const std::map<T *, bool>& mapping,
    std::map<T *, bool> Internal::SavedState:: *SaveMapPtr,
    T *item, bool is_global, bool *found_entry) {
  *found_entry = true;
  auto it = mapping.find(item);
  if (it != mapping.end()) return it->second;
  // Search saved states.
  for (auto save_it = div_->saved_states_.rbegin();
      save_it != div_->saved_states_.rend(); ++save_it) {
    if (!is_global && (*save_it)->function_owner_ != this) continue;
    auto save_map_it = ((*save_it)->*SaveMapPtr).find(item);
    if (save_map_it != ((*save_it)->*SaveMapPtr).end())
      return save_map_it->second;
  }
  *found_entry = false;
  return false;
}

void Divergence::ProcessEntryFunction(Function *function) {
  FunctionDivergence *function_div = new FunctionDivergence(this, function);
  function_div_[function].reset(function_div);
  // Process global inits.
  for (const Variable *var : *VariableSelector::GetGlobalVariables()) {
    if (var->isArray) continue;
    assert(var->init != NULL);
    function_div->IsAssignmentDivergent(Lhs(*var), *var->init);
  }

  function_div->Process({});
}

void Divergence::GetDivergentCodeSectionsForFunction(Function *function,
    std::vector<std::pair<Statement *, Statement *>> *divergent_sections) {
  assert(divergent_sections != NULL);
  divergent_sections->clear();

  auto map_it = function_div_.find(function);
  if (map_it != function_div_.end())
    map_it->second->GetDivergentCodeSections(divergent_sections);
}

}  // namespace CUDASmith
