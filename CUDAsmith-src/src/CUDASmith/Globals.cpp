#include "CUDASmith/Globals.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "ArrayVariable.h"
#include "util.h"
#include "CUDASmith/MemoryBuffer.h"
#include "CVQualifiers.h"
#include "Expression.h"
#include "ExpressionVariable.h"
#include "Function.h"
#include "FunctionInvocationUser.h"
#include "Type.h"
#include "Variable.h"
#include "VariableSelector.h"

//added by liuye. in 2017.12.27
#include "CUDASmith/Vector.h"

namespace CUDASmith
{
namespace
{
Globals *globals_inst = NULL; // Singleton instance.
} // namespace

void Globals::AddLocalMemoryBuffer(MemoryBuffer *buffer)
{
  //if(buffer->GetMemorySpace() != MemoryBuffer::kLocal){
  //  printf("not equal");
  //  exit(-1);
  // }
  assert(buffer->GetMemorySpace() == MemoryBuffer::kLocal);
  buffers_.push_back(buffer);
  if (buffer->collective)
    return;
  buffer->OutputDef(buff_init_, 1);
  output_tab(struct_buff_init_, 2);
  buffer->Output(struct_buff_init_);
  struct_buff_init_ << ", // ";
  buffer->Output(struct_buff_init_);
  struct_buff_init_ << std::endl;
}

void Globals::AddGlobalMemoryBuffer(MemoryBuffer *buffer)
{
  assert(buffer->GetMemorySpace() == MemoryBuffer::kGlobal);
  buffers_.push_back(buffer);
  if (buffer->collective)
    return;
  output_tab(struct_buff_init_, 2);
  buffer->Output(struct_buff_init_);
  struct_buff_init_ << ", // ";
  buffer->Output(struct_buff_init_);
  struct_buff_init_ << std::endl;
}

Globals *Globals::GetGlobals()
{
  if (globals_inst == NULL)
    globals_inst = CreateGlobals();
  return globals_inst;
}

void Globals::ReleaseGlobals()
{
  delete globals_inst;
  globals_inst = NULL;
}
/*
void Globals::OutputStructDefinition(std::ostream& out) {
  if (!struct_type_) CreateGlobalStruct();
  struct_type_->Output(out);
  out << " {" << std::endl;
  for (Variable *var : global_vars_) {
    // Need this check, or we will output arrays twice.
    if (var->isArray && dynamic_cast<ArrayVariable *>(var)->collective)
      continue;
    output_tab(out, 1);
    var->OutputDecl(out);
    out << ";" << std::endl;
  }
  for (MemoryBuffer *buffer : buffers_) {
    if (buffer->collective) continue;
    output_tab(out, 1);

    buffer->OutputAliasDecl(out);  

    out << ";" << std::endl;
    //buffer->OutputAliasDecl(out);
    //out << ";" << std::endl;
  }
  out << "};" << std::endl;
}*/

/*delete by wxy 2018-05-07 --------------------------------------------------------------
//added in 2017.12.24. by ly.
void inline OutputDecl(Variable *v, std::ostream &out);
void inline output_qualified_type(CVQualifiers ql, const Type *t, std::ostream &out);
void inline output_qualified_type(Variable *v, std::ostream &out);
bool inline sanity_check(CVQualifiers ql, const Type *t);
void inline output_qualified_type(CVQualifiers ql, const Type *t, std::ostream &out);
void output_qualified_type_with_deputy_annotation(CVQualifiers ql, const Type *t, std::ostream &out, const vector<string> &annotations);
void ArrayVariable_OutputDecl(ArrayVariable *v, std::ostream &out);
void Vector_OutputDecl(Vector *v, std::ostream &out);
void CVQualifiers_OutputFirstQuals(CVQualifiers ql, std::ostream &out);
*/
/*delete by wxy 2018-05-07
void CVQualifiers_OutputFirstQuals(CVQualifiers ql, std::ostream &out)
{
  if (ql.is_consts.size() > 0 && ql.is_consts[0])
  {
    if (!CGOptions::consts())
      assert(0);
    out << "const ";
  }

  if (ql.is_volatiles.size() > 0 && ql.is_volatiles[0])
  {
    if (!CGOptions::volatiles())
      assert(0);
    //Guai 20160906 Start
    // out << "volatile ";
    out << "";
    //Guai 20160906 End
  }
}

void Vector_OutputDecl(Vector *v, std::ostream &out)
{
  // Trying to print all qualifiers prints the type as well. We don't allow
  // vector pointers regardless.
  // v->qfer.OutputFirstQuals(out);
  CVQualifiers_OutputFirstQuals(v->qfer, out);
  v->OutputVectorType(out, v->type, v->sizes[0]);
  out << ' ' << v->get_actual_name();
}
void ArrayVariable_OutputDecl(ArrayVariable *v, std::ostream &out)
{
  // force global variables to be static if necessary
  if (CGOptions::force_globals_static() && v->is_global())
  {
    out << "static ";
  }
  output_qualified_type(v, out);
  out << v->get_actual_name();
  size_t i;
  for (i = 0; i < v->sizes.size(); i++)
  {
    out << "[" << v->sizes[i] << "]";
  }
}
void inline OutputDecl(Variable *v, std::ostream &out)
{
  // force global variables to be static if necessary
  if (CGOptions::force_globals_static() && v->is_global())
  {
    out << "static ";
  }
  output_qualified_type(v, out);
  out << v->get_actual_name();
}
void inline output_qualified_type(Variable *v, std::ostream &out)
{
  if (v->type->eType == ePointer && CGOptions::deputy())
  {
    vector<string> annotations = v->deputy_annotation();
    output_qualified_type_with_deputy_annotation(v->qfer, v->type, out, annotations);
  }
  else
  {
    output_qualified_type(v->qfer, v->type, out);
  }
}*/
/*delete by wxy 2018-05-07
void output_qualified_type_with_deputy_annotation(CVQualifiers ql, const Type *t, std::ostream &out, const vector<string> &annotations)
{
  assert(t);
  assert(sanity_check(ql, t));
  assert(ql.is_consts.size() == annotations.size() + 1);
  size_t i;
  const Type *base = t->get_base_type();
  for (i = 0; i < ql.is_consts.size(); i++)
  {
    if (i > 0)
    {
      out << "* ";
      out << annotations[i - 1] << " ";
    }
    if (ql.is_consts[i])
    {
      if (!CGOptions::consts())
        assert(0);
      out << "const ";
    }
    if (ql.is_volatiles[i])
    {
      if (!CGOptions::volatiles())
        assert(0);
      //Guai 20160912 Start
      if (!((base->eType == eStruct) || (base->eType == eUnion)))
        // out << "volatile ";
        out << "";
      //Guai 20160912 End
    }
    if (i == 0)
    {
      base->Output(out);
      out << " ";
    }
  }
}*/
/*delete by wxy 2018-05-07
//added in 2017.12.24. by ly.
bool inline sanity_check(CVQualifiers ql, const Type *t)
{
  assert(t);
  int level = t->get_indirect_level();
  assert(level >= 0);
  return ql.wildcard || (ql.is_consts.size() == ql.is_volatiles.size() && (static_cast<size_t>(level) + 1) == ql.is_consts.size());
}*/
/* delete by wxy 2018-05-07
//added in 2017.12.24. by ly.
void inline output_qualified_type(CVQualifiers ql, const Type *t, std::ostream &out)
{
  assert(t);
  assert(sanity_check(ql, t));
  size_t i;
  const Type *base = t->get_base_type();
  for (i = 0; i < ql.is_consts.size(); i++)
  {
    if (i > 0)
    {
      out << "*";
    }
    if (ql.is_consts[i])
    {
      if (!CGOptions::consts())
        assert(0);
      if (i > 0)
        out << " ";
      out << "const ";
    }
    if (ql.is_volatiles[i])
    {
      if (!CGOptions::volatiles())
        assert(0);
      if (i > 0)
        out << " ";
      //Guai 20160912 Start
      if (!((base->eType == eStruct) || (base->eType == eUnion)))
        out << "";
      //Guai 20160912 End
    }
    if (i == 0)
    {
      base->Output(out);
      out << " ";
    }
  }
}*/

void Globals::OutputStructDefinition(std::ostream &out)
{
  if (!struct_type_)
    CreateGlobalStruct();
  struct_type_->Output(out);
  out << " {" << std::endl;

  /**
		 * wxy2017-12-21
		*/
  // if (struct_type_->eType == eStruct)
  // {
  //   out << "  "
  //       << "__device__ void operator=(const S" << struct_type_->sid << "& s) volatile{";
  //   really_outputln(out);
  //   out << "  "
  //       << "// assignment by memory copy operation";
  //   really_outputln(out);
  //   out << "  "
  //       << "memcpy((void*)this,(void*)&s,(size_t)sizeof(struct S" << struct_type_->sid << "));";
  //   really_outputln(out);
  //   out << "  "
  //       << "}";
  //   really_outputln(out);

  //   out << "  "
  //       << "__device__ void operator=(volatile const S" << struct_type_->sid << "& s) volatile{";
  //   really_outputln(out);
  //   out << "  "
  //       << "// assignment by memory copy operation";
  //   really_outputln(out);
  //   out << "  "
  //       << "memcpy((void*)this,(void*)&s,(size_t)sizeof(struct S" << struct_type_->sid << "));";
  //   really_outputln(out);
  //   out << "  "
  //       << "}";
  //   really_outputln(out);
  // }

  for (Variable *var : global_vars_)
  {
    // Need this check, or we will output arrays twice.
    if (var->isArray && dynamic_cast<ArrayVariable *>(var)->collective)
      continue;
    output_tab(out, 1);
     var->OutputDecl(out);
     //delete by wxy 2018-05-05
    /*if (NULL != dynamic_cast<Vector *>(var))
    {
      // var->OutputDecl(out);
      Vector_OutputDecl(dynamic_cast<Vector *>(var), out);
    }
    else
    {
      if (NULL != dynamic_cast<ArrayVariable *>(var))
      {
        // var->OutputDecl(out);
        ArrayVariable_OutputDecl(dynamic_cast<ArrayVariable *>(var), out);
      }
      else
      {
        OutputDecl(var, out);
      }
    }*/
    out << ";" << std::endl;
  }
  for (MemoryBuffer *buffer : buffers_)
  {
    if (buffer->collective)
      continue;
    output_tab(out, 1);
    //printf("buffer.outputAliasDecl!");
    buffer->OutputAliasDecl(out);
    out << ";" << std::endl;
  }
  out << "};" << std::endl;
}

void Globals::OutputStructInit(std::ostream &out)
{
  output_tab(out, 1);
  std::string local_name1 = gensym("c_");
  std::string local_name2 = gensym("c_");
  struct_type_->Output(out);
  out << " " << local_name1 << ";" << std::endl;
  output_tab(out, 1);
  struct_type_ptr_->Output(out);
  out << " ";
  struct_var_->Output(out);
  out << " = &" << local_name1 << ";" << std::endl;

  output_tab(out, 1);
  struct_type_->Output(out);
  out << " " << local_name2 << " = {" << std::endl;

  for (Variable *var : global_vars_)
  {
    if (var->isArray)
    {
      ArrayVariable *var_array = dynamic_cast<ArrayVariable *>(var);
      if (var_array->collective)
        continue;
      output_tab(out, 2);
      std::vector<std::string> init_strings;
      init_strings.push_back(var_array->init->to_string());
      for (const Expression *init : var_array->get_more_init_values())
        init_strings.push_back(init->to_string());
      out << var_array->build_initializer_str(init_strings);
    }
    else
    {
      output_tab(out, 2);
      var->init->Output(out);
    }
    out << ", // ";
    var->Output(out);
    out << std::endl;
  }
  out << struct_buff_init_.str();
  struct_buff_init_.clear();

  output_tab(out, 1);
  out << "};" << std::endl;
  output_tab(out, 1);
  out << local_name1 << " = " << local_name2 << ";" << std::endl;
}

void Globals::OutputBufferInits(std::ostream &out)
{
  out << buff_init_.str();
  buff_init_.clear();
}
/*
void Globals::OutputBufferInitsForEntry(std::ostream &out)
{
  
  out << buff_init_.str();
  buff_init_.clear();
}*/ // add and delete by wxy 2017/06/11

void Globals::AddGlobalStructToFunction(Function *function, Variable *var)
{
  function->param.push_back(var);
}

void Globals::AddGlobalStructToAllFunctions()
{
  if (!struct_type_)
    CreateGlobalStruct();
  const std::vector<Function *> &functions = get_all_functions();
  for (Function *function : functions)
    AddGlobalStructToFunction(function, struct_var_);

  // Also need to add the parameter to all invocations of the functions.
  for (FunctionInvocationUser *invocation :
       *FunctionInvocationUser::GetAllFunctionInvocations())
    invocation->param_value.push_back(new ExpressionVariable(*struct_var_));
}

void Globals::ModifyGlobalVariableReferences()
{
  if (!struct_type_)
    CreateGlobalStruct();
  // Append the name of our newly created struct to the front of every global
  // variable (eugghhh).
  std::vector<std::string> names;
  for (Variable *var : global_vars_)
  {
    names.push_back(var->name);
    *const_cast<std::string *>(&var->name) =
        struct_var_->name + "->" + var->name;
    if (var->is_aggregate())
      ModifyGlobalAggregateVariableReferences(var);
  }

  // At some points during generation, a global array variable will be
  // 'itemized'. It will not be put in the list of global variables, so as a
  // fix, we add a method in the VariableSelector that give us the list of all
  // variables.
  // Variable::is_global() is unreliable.
  for (Variable *var : *VariableSelector::GetAllVariables())
  {
    if (var->name.find("g_") == 0 && std::find(buffers_.begin(), buffers_.end(), var) == buffers_.end())
    {
      *const_cast<std::string *>(&var->name) =
          struct_var_->name + "->" + var->name;
      if (var->is_aggregate())
        ModifyGlobalAggregateVariableReferences(var);
    }
    if (var->name.find("l_") == 0 && var->is_aggregate())
    {
      for (Variable *field_var : var->field_vars)
      {
        if (field_var->name.find("[") != std::string::npos)
        {
          FixStructArrays(field_var, field_var->name.find("["));
        }
      }
    }
  }

  // Now add to the buffers.
  for (MemoryBuffer *buffer : buffers_)
  {
    *const_cast<std::string *>(&buffer->name) =
        struct_var_->name + "->" + buffer->name;
    if (buffer->is_aggregate())
      ModifyGlobalAggregateVariableReferences(buffer);
  }
}

void Globals::OutputArrayControlVars(std::ostream &out) const
{
  size_t max_dim = Variable::GetMaxArrayDimension(global_vars_);
  for (MemoryBuffer *buf : buffers_)
    max_dim = std::max(max_dim, buf->get_dimension());
  std::vector<const Variable *> &ctrl_vars = Variable::get_new_ctrl_vars();
  OutputArrayCtrlVars(ctrl_vars, out, max_dim, 1);
}

void Globals::HashLocalBuffers(std::ostream &out) const
{
  for (MemoryBuffer *buffer : buffers_)
    if (buffer->GetMemorySpace() == MemoryBuffer::kLocal)
      buffer->hash(out);
}

const Type &Globals::GetGlobalStructPtrType()
{
  return *struct_type_ptr_.get();
}

const Variable &Globals::GetGlobalStructVar()
{
  return *struct_var_;
}

Globals *Globals::CreateGlobals()
{
  return new Globals(*VariableSelector::GetGlobalVariables());
}

void Globals::CreateGlobalStruct()
{
  std::vector<const Type *> types;
  std::vector<CVQualifiers> qfers;
  std::vector<int> bitfield_len;
  for (Variable *var : global_vars_)
  {
    types.push_back(var->type);
    qfers.push_back(var->qfer);
    bitfield_len.push_back(-1);
  }
  struct_type_.reset(new Type(types, true, false, qfers, bitfield_len));
  assert(struct_type_);
  struct_type_ptr_.reset(new Type(struct_type_.get()));
  assert(struct_type_ptr_);

  // Now create the Variable object. It will be a parameter variable that points
  // to our global struct that is non-const and non-volatile.
  vector<bool> const_qfer({false, false});
  vector<bool> volatile_qfer({false, false});
  CVQualifiers qfer(const_qfer, volatile_qfer);
  struct_var_ = VariableSelector::GenerateParameterVariable(
      struct_type_ptr_.get(), &qfer);
  assert(struct_var_);
}

void Globals::ModifyGlobalAggregateVariableReferences(Variable *var)
{
  assert(var->is_aggregate());
  for (Variable *field_var : var->field_vars)
  {
    *const_cast<std::string *>(&field_var->name) =
        struct_var_->name + "->" + field_var->name;
    if (field_var->is_aggregate())
      ModifyGlobalAggregateVariableReferences(field_var);
    // TODO: Use methods in Variable to detect this.
    if (field_var->name.find("[") != std::string::npos)
      FixStructArrays(field_var, field_var->name.find("["));
  }
}

void Globals::FixStructArrays(Variable *field_var, size_t pos)
{
  while (pos < field_var->name.length())
  {
    pos = field_var->name.find("g_", pos);
    if (pos == std::string::npos)
      break;
    if (field_var->name.compare(pos - 2, 2, "->"))
    {
      string to_insert = struct_var_->name + "->";
      const_cast<std::string *>(&field_var->name)->insert(pos, to_insert);
      pos += to_insert.length();
    }
    else
      pos++;
  }
}

} // namespace CUDASmith
