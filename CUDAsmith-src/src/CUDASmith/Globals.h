// Handles the global variables in the program.
// Actually, there are no global variable in an OpenCL progrma, this handles
// transforming global variables into non-global variables without changing the
// behaviour of the program.
//
// Take the following program:
//
// int g_1 = 0;
// int g_2 = 1;
// void func_1(void) {
//   g_1 = g_2;
// }
//
// This is not valid in OpenCL, due to g_1 and g_2 being global, we transform it
// into:
//
// struct global {
//   int g_1;
//   int g_2;
// };
// void func_1(struct global *g) {
//   g->g_1 = g->g_2;
// }
//
// The global struct will be initialised by the kernel entry function, and
// passed as a parameter to all the functions in the program.
// TODO Proper const/mutable of the created struct (Only allow the singleton?
//   assert no changes after struct created?).
// TODO When memory spaces are done properly, merge vars and local buffers.
//
// This was more or less the first CUDASmith thing created, when I knew almost
// nothing of how csmith worked, so it largely sucks.

#ifndef _CUDASMITH_GLOBALS_H_
#define _CUDASMITH_GLOBALS_H_

#include <algorithm>
#include <memory>
#include <ostream>
#include <sstream>
#include <vector>
#include <string>

#include "CommonMacros.h"

class Function;
namespace CUDASmith
{
class MemoryBuffer;
}
class Type;
class Variable;

namespace CUDASmith
{

// Holds pointers to all of the global variables in the program, does not own
// the data that the pointers point to.
class Globals
{
public:
  Globals() {}
  explicit Globals(const std::vector<Variable *> &vars) : global_vars_(vars) {}

  // Move constructor/assignment.
  Globals(Globals &&other) = default;
  Globals &operator=(Globals &&other) = default;

  ~Globals() {}

  // Adds the global variables specified in the parameter.
  void AddGlobalVariable(Variable *var) { global_vars_.push_back(var); }
  void AddGlobalVariables(const std::vector<Variable *> &vars)
  {
    global_vars_.insert(global_vars_.end(), vars.begin(), vars.end());
  }

  // Adds memory buffers. Needs special handling.
  void AddLocalMemoryBuffer(MemoryBuffer *buffer);
  void AddGlobalMemoryBuffer(MemoryBuffer *buffer);

  // Accessors for a singleton instance of the Globals class.
  // The instance is created lazily, and must be careful not to access it after
  // ReleaseGlobals() is called. ReleaseGlobals() should generally be called by
  // whatever initially called CreateGlobals();.
  static Globals *GetGlobals();
  static void ReleaseGlobals();

  // Output the definition of the global struct, including the types of all the
  // variables that have need added.
  // Must be called before ModifyGlobalVariableReferences().
  void OutputStructDefinition(std::ostream &out);

  // Output an initialisation of the global struct. Assigning the value of each
  // element of the struct.
  // Must be called after ModifyGlobalVariableReferences().
  void OutputStructInit(std::ostream &out);

  // Outputs the declaration and initialisation of the local buffer without the
  // reference to the global struct.
  void OutputBufferInits(std::ostream &out);
  //void OutputBufferInitsForEntry(std::ostream &out); //add and delete by ly

  // Modifies the function such that it no longer refers to global variables,
  // instead, it will refer to members of a local struct. Just adds the name of
  // the local struct to the parameter list of the function.
  void AddGlobalStructToFunction(Function *function, Variable *var);

  // Modifies all functions that have been generated to refer to the local
  // struct instead of global variables.
  void AddGlobalStructToAllFunctions();

  // Modifies all the global variables such that they refer to our global struct
  // variable. Any methods that make use of the names of the globals before
  // they are modified must be called before this.
  //
  // Unfortunately, there is no easy way to change how the variables are
  // referenced. We have to make the local struct we pass around have the same
  // name, as we have to change the name field of the variable directly to add
  // "local_struct->" to the start of every global variable.
  void ModifyGlobalVariableReferences();

  // Output index variables that will cover all of the indexes for all of the
  // arrays in the global struct.
  void OutputArrayControlVars(std::ostream &out) const;

  // Hash the item in all local buffers that belongs to the work item.
  void HashLocalBuffers(std::ostream &out) const;

  // Gets the type of the global struct, as a ptr type. The actual type can be
  // retrieved by calling:
  //   const Type *type = GetGlobalStructPtrType().ptr_type;
  const Type &GetGlobalStructPtrType();

  // Gets the Variable object that all the global variables are a part of.
  const Variable &GetGlobalStructVar();

  // Must be called after csmith has generated the program. Collects all the
  // global variables in the program.
  static Globals *CreateGlobals();

private:
  // Constructs the Type and Variable objects using all the variables that have
  // currently been added. It is assumed that all the global variables in the
  // program have already been created and added.
  void CreateGlobalStruct();

  // Helper for modifying global variable references. Recursively adds the
  // global struct to the members of struct and union variables.
  void ModifyGlobalAggregateVariableReferences(Variable *var);

  // A special step to fix the issue of not adding the global struct to global
  // variable indices for arrays of structs (in CSmith, the name of the variable
  // is set to contain the index, rather than it being an itemized variable)
  void FixStructArrays(Variable *field_var, size_t pos);

private:
  std::vector<Variable *> global_vars_;
  std::vector<MemoryBuffer *> buffers_;
  // Type class generated lazily, needs all the global variables to have been
  // added before creation.
  std::unique_ptr<Type> struct_type_;
  std::unique_ptr<Type> struct_type_ptr_;
  // Also created lazily, does not own the data. Want a single variable for the
  // globals, as we modify all the globals to refer to the same struct variable.
  Variable *struct_var_;
  // Buffer inits are collected in a stringstream, as outputting them is
  // typically done after the reference has been modified.
  std::stringstream buff_init_;
  std::stringstream struct_buff_init_;

  DISALLOW_COPY_AND_ASSIGN(Globals);
};

} // namespace CUDASmith

#endif // _CUDASMITH_GLOBALS_H_
