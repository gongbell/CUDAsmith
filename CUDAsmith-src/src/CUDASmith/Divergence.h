// Classes for finding all the points in an OpenCL program that could be
// divergent. A point in the code is divergent if one or more threads reach that
// point, but not all of them.
// This will 'overestimate' the program's divergence, that is, any point that is
// divergent will be found, but we may also mark points that are not divergent
// as being divergent (not sound but complete).
// Marking is done on a per block/per scope level basis, either a block is all
// convergent or all divergent. Some blocks will be split into sub-blocks to
// help prevent overestimating.
// See the report on what properties we use to do this.
//
// Future possible improvements:
// - Replace code that refers to the AST directly into calls to the walker, and
//   create a walker interface. This allows us to plug this code in to other
//   ASTs as long as the walker interface is implemented.
// - Const correctness.
// - Prevent repeated processing of functions, loops, etc. by storing the
//   initial state of the last time we processed.
// - Use the strict method of pointer analysis, instead of the bounded method.
//   This would be expensive, as pointer sets would have to be copied when the
//   state is saved.
// - Perform full static analysis, after all, there is no random behaviour.
//
// Some limitations:
// - Fails for loopy programs, i.e. if func_1 calls func_2, which eventually
//   calls func_1.
// - Becomes very costly for nested for blocks.
//
// Major TODOs:
// - Save pointers.
// - Split sub-blocks on returns.
// - Implement gotos.
// - When arrays are fixed, implement them.
//
// At the moment, this code is largely unusable, it was created right at the
// beginning of the project and was not properly finished (Sub blocks are not
// used, some bits are missing, arrays must be disabled, no proper testing,
// etc). There is also no interaction with any of the CUDAStatements or
// CUDAExpressions, as they were added after this. A lot of work will be required
// to bring this up to scratch.

#ifndef _CUDASMITH_DIVERGENCE_H_
#define _CUDASMITH_DIVERGENCE_H_

#include <map>
#include <memory>
#include <set>
#include <stack>
#include <utility>
#include <vector>

#include "Block.h"
#include "CUDASmith/Walker.h"
#include "Function.h"
#include "StatementAssign.h"
#include "CommonMacros.h"

class Expression;
class Statement;
class Variable;

// Required for SavedState constructor.
namespace CUDASmith { class FunctionDivergence; }

namespace CUDASmith {
namespace Internal {

// Represent part (or all) of a block. The blocks are linked at the point of
// splitting.
class SubBlock {
 public:
  SubBlock(Block *block) : block_(block) {
    assert(block_ != NULL);
    begin_statement_ = *block_->stms.begin();
    end_statement_ = *block_->stms.rbegin();
  }
  SubBlock(Block *block, Statement *begin, Statement *end)
      : block_(block), begin_statement_(begin), end_statement_(end) {
    assert(block_ != NULL);
    assert(begin_statement_ != NULL);
    assert(end_statement_ != NULL);
  }

  // Splits the sub block in two before the passed statement. Returns the sub
  // block after the statement.
  SubBlock *SplitAt(Statement *statement);

  Block *block_;
  Statement *begin_statement_;
  Statement *end_statement_;
  // Acts as a linked list. The next link is owned by the previous link.
  std::unique_ptr<SubBlock> next_sub_block_;
  SubBlock *prev_sub_block_;

 private:
  friend class FunctionDivergence;
  // Moving could invalidate pointers that refer to this.
  SubBlock(SubBlock&& other) = delete;
  SubBlock& operator=(SubBlock&& other) = delete;
  DISALLOW_COPY_AND_ASSIGN(SubBlock);
};

// At some points during processing, we need use a temporary state, without
// modifying the actual state. But we still need refer to the original state.
// This class will store the state of the whole process, but make it available
// as necessary.
// The saved state 'belongs' to the function the saved it. For the function that
// saved it, all of the non-final, non-context members are moved.
// Also save the state of the global variables.
class SavedState {
 public:
  SavedState(FunctionDivergence *function_owner, Statement *statement)
      : function_owner_(function_owner), statement_(statement) {
  }

  FunctionDivergence *function_owner_;
  // Also mark which statement owns it, for safety.
  Statement *statement_;

  // Saved state of the owner's members (minus the context).
  std::map<SubBlock *, bool> sub_block_div_;
  std::map<const Variable *, bool> variable_div_;

  // Globals
  std::map<const Variable *, bool> global_var_div_;
 private:
  DISALLOW_COPY_AND_ASSIGN(SavedState);
};

}  // namespace Internal

// Forward declaration required for FunctionDivergence constructor.
class Divergence;

// Tracks divergence in a single function.
// Will be processed many times, to account for different call locations,
// parameters and global state.
// Each time it is processed, we start from scratch, with no data from the
// previous process. The result is the union of all sub blocks in the function
// that we have marked as possibly being divergent.
class FunctionDivergence {
 public:
  // We may have to pause mid processing to process something else (or may not
  // have even started).
  enum ProcessStatus {
    kNotDone = 0,
    kMidProcess,
    kDone
  };

  FunctionDivergence(Divergence *div, Function *function)
      : div_(div), function_(function), status_(kNotDone) {
    assert(div_ != NULL);
    assert(function_ != NULL);
    // Allows setting up of global vars.
    divergent_ = false;
  }
  FunctionDivergence(FunctionDivergence&& other) = default;
  FunctionDivergence& operator=(FunctionDivergence&& other) = default;
  virtual ~FunctionDivergence() {}

  // Processes the whole function, marking sections as divergent.
  // This can be done multiple times, to account for different program states
  // (i.e. we may run through it a second time, with more global variables
  // marked as divergent).
  // Takes a vector of parameters, that should match the function signature,
  // each param is paired with a bool indicating if it is divergent.
  void Process(const std::vector<bool>& parameters) {
    ProcessWithContext(parameters, {}, {}, false);
  }

  // Fills the passed vector with ranges of divergent code. We assume that if a
  // block is divergent, then all branches in the block are also divergent, and
  // so are not visited.
  void GetDivergentCodeSections(
      std::vector<std::pair<Statement *, Statement *>> *divergent_sections) {
    GetDivergentCodeSectionsForBlock(function_->body, divergent_sections);
  }
  void GetDivergentCodeSectionsForBlock(Block *block,
      std::vector<std::pair<Statement *, Statement *>> *divergent_sections);

  // Same as Process(), but with a large amount of extra context information as
  // baggage. param_deref_to contains pairs of pointers and the variables they
  // dereference to, param_deref_div contains all the variable divergence
  // information for variable outside this function that it may refer to.
  void ProcessWithContext(const std::vector<bool>& parameters,
      const std::map<const Variable *, std::set<const Variable *> *>&
          param_derefs_to,
      const std::map<const Variable *, bool>& param_ref_div, bool divergent);

  Function *GetFunction() const { return function_; }
  ProcessStatus GetProcessStatus() const { return status_; }

 private:
  // A single iteration of the processing loop. In its own function, as we may
  // want to perform processing outside of the main processing loop.
  // Uses the current context, so this must be saved and updated if this is to
  // be used.
  void ProcessStep();
  // Each block will have its own set of variable declarations to be processed.
  void ProcessBlockStart(Block *block);
  // Some branches need some post-processing applied to them (i.e. condition in
  // a for loop). The block is retrieved from the current context.
  void ProcessBlockEnd();

  // Processing for some of the more complex statement types.
  void ProcessStatementAssign(Statement *statement);
  void ProcessStatementFor(Statement *statement);
  void ProcessStatementIf(Statement *statement);
  void ProcessStatementInvoke(Statement *statement);
  void ProcessStatementReturn(Statement *statement);
  void ProcessStatementJump(Statement *statement);
  void ProcessStatementGoto(Statement *statement);
  void ProcessStatementArray(Statement *statement);

  // for and if require some post processing.
  void ProcessEndStatementFor(Statement *statement);
  void ProcessEndStatementIf(Statement *statement);

  // After the whole function has been processed, merge our findings into the
  // final set of data.
  void ProcessFinalise();

  // Main functions for cathcing the divergent behaviour.
  bool IsExpressionDivergent(const Expression& expression);
  bool IsVariableDivergent(const Variable& variable);
  bool IsSubBlockDivergent(Internal::SubBlock *sub_block);

  // Handles a function call, returning true if the result value is divergent.
  // If the function has pointer type, returns the references it may return.
  bool IsFunctionCallDivergent(const FunctionInvocationUser& invoke,
      std::set<const Variable *> *return_refs);
  // Generalise assignment here, needs to account for globals and such. Returns
  // true if the variable assigned is divergent.
  bool IsAssignmentDivergent(const StatementAssign& ass) {
    return IsAssignmentDivergent(*ass.get_lhs(), *ass.get_rhs());
  }
  bool IsAssignmentDivergent(const Lhs& lhs, const Expression& expr);

  // Dereferences the passed pointer the specified number of levels. The result
  // is put in deref_vars.
  // Returns true if any of the vars we dereference are divergent:
  // i.e. int *l1 = expr ? &l2 : &l3; (With l2 divergent).
  //      div = DereferencePointerVariable(l1, 1, &derefs);
  //   div is false (l1 is not divergent), derefs = {l2, l3}.
  bool DereferencePointerVariable(const Variable& var, int deref_level,
      std::set<const Variable *> *deref_vars);
  // Gets the variables referenced in an expression.
  // Relies on the expression being a constant, variable or funcCall type.
  // For Constant, this should be NULL, (void *)0.
  // For variable, this can be &var, var, or *var, with any amount of derefs.
  // For funcCall, the return type of the function must be of pointer type.
  // Returns true if the references are divergent (i.e. we could be referencing
  // different vars in different threads).
  bool GetExpressionVariableReferences(const Expression& expr,
      std::set<const Variable *> *var_refs);

  // Sets the passed sub block as divergent, then 'infects' all of the branched
  // sub blocks recursively.
  void MarkSubBlockDivergentViral(Internal::SubBlock *sub_block);

  // Helper function for correctly assigning the divergence of a variable.
  void SetVariableDivergence(const Variable *var, bool divergent);

  // Helper functions for accessing this class' ridiculous data structures.
  Internal::SubBlock *GetSubBlockForBranch(Statement *statement, Block *block) {
    return GetSubBlockForBranchFromBlock(sub_block_, statement, block);
  }
  Internal::SubBlock *GetSubBlockForBranchFromBlock(
      Internal::SubBlock *sub_block, Statement *statement, Block *block);

  // Costly function. See SavedState class for what is stored.
  Internal::SavedState *SaveState(Statement *statement);
  // Also costly, restores a previously saved state, merging divergent elements
  // in the current state with those in the saved state.
  // Returns true if any elements that were not divergent in the save state have
  // become divergent.
  // Members of the saved state will be in an unspecified state, so the object
  // should be deleted (or you can call clear() on the maps.
  bool RestoreAndMergeSavedState(Internal::SavedState *saved_state);

  // Applies a general merge on two maps of the same type, returning true if the
  // newer map added anything to the previous map.
  template<typename T>
  static bool MergeMap(std::map<T *, bool> *saved, std::map<T *, bool> *merger);
  // Applies a search on a map, going through saved states if necessary.
  // is_global indicates the item has global scope, so no saved state should be
  // ignored.
  template<typename T>
  bool SearchMap(const std::map<T *, bool>& mapping,
      std::map<T *, bool> Internal::SavedState:: *SaveMapPtr,
      T *item, bool is_global, bool *found_entry);

  // Data for each time we process. It is cleared at the start.
  // Which sub blocks we have marked as divergent.
  std::map<Internal::SubBlock *, bool> sub_block_div_;
  // Which (local) variables are divergent.
  std::map<const Variable *, bool> variable_div_;
  // Is the return value possible divergent.
  bool divergent_value_;

  // Keep track of what pointers may be pointing to. Pointing to multiple vars
  // does not mean it is divergent.
  std::map<const Variable *, std::set<const Variable *>> var_derefs_to_;
  // If the function returns pointers, keep track of what the return value may
  // be referencing.
  std::set<const Variable *> return_derefs_to_;

  // The current context.
  Internal::SubBlock *sub_block_;
  bool divergent_;
  std::unique_ptr<Walker::FunctionWalker> function_walker_;
  // Nested sub blocks. Need to access lower levels, so use vector over stack.
  // The sub block is paired with the statement we branched off with.
  std::vector<std::pair<Internal::SubBlock *, Statement *>> nested_blocks_;
  // For i else statements, after processing the true branch, it must be saved
  // but not accessible, the merged after the false branch is done.
  std::stack<std::pair<Statement *, std::unique_ptr<Internal::SavedState>>>
      saved_if_;

  // Maps each block to the sub block that sits at the beginning of the block,
  // the sub block being one from our final divergence map.
  std::map<Block *, std::unique_ptr<Internal::SubBlock>>
      block_to_sub_block_final_;
  // Scope levels/nested blocks.
  std::map<Internal::SubBlock *, std::vector<std::pair<Statement *, Block *>>>
      scope_levels_final_;
  // The result of processing. Provides an overestimate of which sub blocks
  // could be divergent at any time during runtime.
  std::map<Internal::SubBlock *, bool> sub_block_div_final_;
  // Does the return value of the function have a divergent value.
  bool divergent_value_final_;

  Divergence *div_;
  Function *function_;
  ProcessStatus status_;

  // Allows the Divergence class to process stuff in the context of a function
  // (i.e global inits).
  friend class Divergence;
  DISALLOW_COPY_AND_ASSIGN(FunctionDivergence);
};

// Handles the tracking of divergent code for the whole program.
// This will also hold information on data that has program scope (global
// variables, functions).
class Divergence {
 public:
  Divergence() {}
  virtual ~Divergence() {}

  // Processes the whole program, given the entry function.
  void ProcessEntryFunction(Function *function);

  // Fills the vector with ranges of divergent code. Each pair marks the begin
  // and last statement of a divergent section.
  void GetDivergentCodeSectionsForFunction(Function *function,
      std::vector<std::pair<Statement *, Statement *>> *divergent_sections);
 private:
  // Each function has its own instance of the FunctionDivergence class. 
  std::map<Function *, std::unique_ptr<FunctionDivergence>> function_div_;
  // Tracks divergence of the global variables. This means that the order in
  // which we process the functions affects the outcome.
  std::map<const Variable *, bool> global_var_div_;

  // Keeps track of what pointers global variables may be pointing to.
  std::map<const Variable *, std::set<const Variable *>> global_var_derefs_to_;

  // Saved states that need to be visible to all FunctionDivergence objects.
  // Order matters, acessed from back to front.
  std::vector<Internal::SavedState *> saved_states_;

  friend class FunctionDivergence;
  DISALLOW_COPY_AND_ASSIGN(Divergence);
};

}  // namespace CUDASmith

#endif  // _CUDASMITH_DIVERGENCE_H_
