#include "CUDASmith/MemoryBuffer.h"

#include <iostream>
#include <string>
#include <vector>

#include "ArrayVariable.h"
#include "Block.h"
#include "CUDASmith/ExpressionID.h"
#include "Constant.h"
#include "CVQualifiers.h"
#include "StatementArrayOp.h"
#include "util.h"
#include "Variable.h"
#include "VariableSelector.h"

namespace CUDASmith
{

MemoryBuffer *MemoryBuffer::CreateMemoryBuffer(MemorySpace memory_space,
                                               const std::string &name, const Type *type, const Expression *init,
                                               std::vector<unsigned> sizes)
{
  bool cnst = false; //memory_space == kConst;
  bool vol = false;
  CVQualifiers qfer(std::vector<bool>({cnst}), std::vector<bool>({vol}));
  return new MemoryBuffer(memory_space, name, type, init, &qfer, sizes);
}

MemoryBuffer *MemoryBuffer::itemize(const std::vector<int> &const_indices)
    const
{
  assert(collective == 0);
  assert(const_indices.size() == sizes.size());
  MemoryBuffer *mb = new MemoryBuffer(*this);
  VariableSelector::GetAllVariables()->push_back(mb);
  for (size_t i = 0; i < sizes.size(); i++)
  {
    int index = const_indices[i];
    mb->add_index(new Constant(get_int_type(), StringUtils::int2str(index)));
  }
  mb->collective = this;
  if (type->is_aggregate())
    mb->create_field_vars(type);
  return mb;
}

MemoryBuffer *MemoryBuffer::itemize(
    const std::vector<const Expression *> &expr_indices, Block *blk) const
{
  assert(collective == 0);
  assert(expr_indices.size() == sizes.size());
  MemoryBuffer *mb = new MemoryBuffer(*this);
  VariableSelector::GetAllVariables()->push_back(mb);
  for (const Expression *expr : expr_indices)
    mb->add_index(expr);
  mb->collective = this;
  mb->parent = blk;
  if (type->is_aggregate())
    mb->create_field_vars(type);
  blk->local_vars.push_back(mb);
  return mb;
}

void MemoryBuffer::OutputMemorySpace(
    std::ostream &out, MemorySpace memory_space)
{
  switch (memory_space)
  {
  //wxy 20170517 Start
  //case kGlobal:  out << "__global";  break;
  //case kLocal:   out << "__local";   break;
  case kGlobal:
    out << "";
    break;
  case kLocal:
    out << "__shared__";
    break;
  case kPrivate:
    out << "__private";
    break;
  case kConst:
    out << "__constant__";
    break;
    //wxy 20170517 End
  }
}

//wxy   2017/06/11
void MemoryBuffer::OutputMemorySpaceInStruct(
    std::ostream &out, MemorySpace memory_space)
{
  //printf("OutputMemorySpaceInstruct\n");
  switch (memory_space)
  {
  //wxy 20170517 Start
  //case kGlobal:  out << "__global";  break;
  //case kLocal:   out << "__local";   break;
  case kGlobal:
    out << "";
    break;
  case kLocal:
    out << "";
    break;
  case kPrivate:
    out << "__private";
    break;
  case kConst:
    out << "__constant__";
    break;
  }
}

void MemoryBuffer::OutputWithOwnedItem(std::ostream &out) const
{
  Output(out);
  out << (memory_space_ == kGlobal ? "[get_linear_global_id()]" : memory_space_ == kLocal ? "[get_linear_local_id()]" : "");
}

void MemoryBuffer::hash(std::ostream &out) const
{
  if (collective != NULL)
    return;
  if (memory_space_ == kConst)
    return;
  if (memory_space_ == kPrivate)
  {
    ArrayVariable::hash(out);
    return;
  }

  output_tab(out, 1);
  out << "transparent_crc(";
  OutputWithOwnedItem(out);
  out << ", \"";
  OutputWithOwnedItem(out);
  out << "\", print_hash_value);" << std::endl;
}

void MemoryBuffer::OutputDef(std::ostream &out, int indent) const
{
  if (collective != NULL)
    return;
  if (memory_space_ == kPrivate || memory_space_ == kConst)
  {
    ArrayVariable::OutputDef(out, indent);
    return;
  }

  output_tab(out, indent);
  //OutputDecl(out); //change
  //out << ';' << std::endl;
  if (memory_space_ == kLocal)
  {
    //added by ly.   !!!key for different Local buffer declaration between in struct  and in entryFunction
    OutputDeclForEntry(out);
    out << ';' << std::endl;
    //

    output_tab(out, indent);
    out << "if (";
    ExpressionID(ExpressionID::kLinearLocal).Output(out);
    out << " == 0)" << std::endl;
    std::vector<const Variable *> &ctrl_vars = Variable::get_new_ctrl_vars();
    // TODO Vector params need to match the buffer indices, use init.
    if (type->is_aggregate())
      StatementArrayOp(NULL, this, ctrl_vars, std::vector<int>({0}),
                       std::vector<int>({1}), Constant::make_int(0))
          .Output(out, NULL, indent + 1);
    else
      output_init(out, init, ctrl_vars, indent + 1);
    return;
  }
  else
  {
    OutputDecl(out);
    out << ';' << std::endl;
  }
  output_tab(out, indent);
  OutputWithOwnedItem(out);
  out << " = ";
  init->Output(out);
  out << ";" << std::endl;
}

void MemoryBuffer::OutputDecl(std::ostream &out) const
{
  ArrayVariable::OutputDecl(out);
}
//add wxy 2017/06/11
void MemoryBuffer::output_qualified_typeForEntry(std::ostream &out) const
{
  OutputMemorySpace(out, memory_space_);
  out << " ";
  Variable::output_qualified_type(out);
}

void MemoryBuffer::output_qualified_type(std::ostream &out) const
{
  OutputMemorySpaceInStruct(out, memory_space_);
  out << " ";
  Variable::output_qualified_type(out);
}
void MemoryBuffer::OutputAliasDecl(std::ostream &out) const
{
  output_qualified_type(out);
  out << "*";
  Output(out);
}

} // namespace CUDASmith
