#include "CUDASmith/Vector.h"

#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Block.h"
#include "CGContext.h"
#include "CUDASmith/CUDAOptions.h"
#include "Constant.h"
#include "random.h"
#include "Type.h"
#include "VariableSelector.h"

class ArrayVariable;
class CVQualifiers;
class Expression;
class Variable;

namespace CUDASmith {
namespace {
// Valid vector lengths.
//update by wxy 2018-04-10
//const size_t kSizes[2] = {2,4};
const size_t kSizes[2] = {2,4};
const size_t kSizesCount = 2;

// Outputs for component accesses.
const char kCompSmall[4] = {'x', 'y', 'z', 'w'};
const char kCompBig[16] = {'0', '1', '2', '3', '4', '5', '6', '7',
                           '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
// Suffix outputs.
/*
const char *const kHiStr = "hi";
const char *const kLoStr = "lo";
const char *const kEvenStr = "even";
const char *const kOddStr = "odd";
*/
const char *const kHiStr = "x";
const char *const kLoStr = "y";
const char *const kEvenStr = "x";
const char *const kOddStr = "y";

}  // namespace

std::map<std::pair<enum eSimpleType, unsigned>, const Type *>
    Vector::vector_types_;

Vector *Vector::CreateVectorVariable(const CGContext& cg_context, Block *blk,
    const std::string& name, const Type *type, const Expression *init,
    const CVQualifiers *qfer, const Variable *isFieldVarOf) {
  // This is probably not the best place to put it.
  assert(type != NULL);
  assert((type->eType == eSimple || type->eType == eVector) &&
      type->simple_type != eVoid);
  // The passed type may be a vector type or a simple type.
  int size;
  if (type->eType == eVector) {
    size = type->vector_length_;
    type = &Vector::DemoteVectorTypeToType(type);
  } else {
    size = GetRandomVectorLength(0);
  }
  // Create vector and push to the back of the appropriate variable list.
  Vector *vector = new Vector(
      blk, name, type, init, qfer, size, isFieldVarOf);
  vector->add_init_value(Constant::make_random(type));
  blk ? blk->local_vars.push_back(vector) :
      VariableSelector::GetGlobalVariables()->push_back(vector);
  return vector;
}

Vector *Vector::itemize(void) const {
  return itemize({(int)rnd_upto(sizes[0])});
}

Vector *Vector::itemize(const std::vector<int>& const_indices) const {
  assert(collective == NULL);
  assert(const_indices.size() == 1);
  Vector *vec = new Vector(*this);
  VariableSelector::GetAllVariables()->push_back(vec);
  vec->comp_access_.push_back(const_indices[0]);
  vec->collective = this;
  return vec;
}

Vector *Vector::itemize_simd(void) const {
  size_t num = rnd_upto(5);
  return num == 4 ? itemize() : itemize_simd(kSizes[num]);
}

Vector *Vector::itemize_simd(int access_count) const {
  assert (access_count <= 16 && access_count > 0 &&
      !(access_count & (access_count - 1)));  
 

  std::vector<int> access;
  for (int idx = 0; idx < access_count; ++idx)
    access.push_back(rnd_upto(sizes[0]));
  return itemize_simd(access);
}

Vector *Vector::itemize_simd(const std::vector<int>& const_indices) const {
  assert(collective == NULL);
  assert(const_indices.size() != 0);
  Vector *vec = new Vector(*this);
  VariableSelector::GetAllVariables()->push_back(vec);
  vec->comp_access_ = const_indices;
  vec->collective = this;
  return vec;
}

void Vector::Output(std::ostream& out) const {
  out << get_actual_name();
  if (collective == NULL) return;
  // We have an itemised vector, output all accesses.

  out << '.';
  if (sizes[0] > 4) out << 's';
  // The components must be constant, the list of expressions used by
  // ArrayVariable cannot be used for vectors. No components means we will
  // access the whole vector, so no components need to be printed.
  for (int comp : comp_access_) out << GetComponentChar(sizes[0], comp);
}

void Vector::Output_Self(std::ostream& out) const {
  out << get_actual_name();
  return;
}

void Vector::OutputDef(std::ostream& out, int indent) const {
  if (collective != NULL) return;
  output_tab(out, indent);
  OutputDecl(out);
  if (!no_loop_initializer()) {
    out << ';';
    outputln(out);
    return;
  }
  std::vector<std::string> init_strs;
  assert(init);
  init_strs.push_back(init->to_string());
  for (const auto& expr : init_values) init_strs.push_back(expr->to_string());
  std::string init_str = build_initializer_str(init_strs);
  out << " = " << build_initializer_str(init_strs) << ';';
  outputln(out);
}

void Vector::OutputDecl(std::ostream& out) const {
  // Trying to print all qualifiers prints the type as well. We don't allow
  // vector pointers regardless.
  qfer.OutputFirstQuals(out);
  OutputVectorType(out, type, sizes[0]);
  out << ' ' << get_actual_name();
}

void Vector::hash(std::ostream& out) const {
  if (collective != NULL || !CGOptions::compute_hash()) return;
  // Create temporary itemised vector for printing.
  Vector *vec = itemize({0});
  for (unsigned comp = 0; comp < sizes[0]; ++comp) {
    vec->comp_access_[0] = comp;
    output_tab(out, 1);
    out << "transparent_crc(";
    vec->Output(out);
    out << ", \"";
    vec->Output(out);
    out << "\", print_hash_value);";
    outputln(out);
  }
}

std::string Vector::build_initializer_str(
    const std::vector<std::string>& init_strings) const {
  /*static*/ unsigned long seed = 0xAB;
  std::string ret;
  ret.reserve(1000);
  // Print the raw vector type without qualifiers.
  stringstream ss_type;
  OutputVectorType_Make(ss_type, type, sizes[0]);
  ret.append("(").append(ss_type.str()).append(")");
  // Build a nested set of vector initialisers
  ret.append("(");
  for (unsigned idx = 0; idx < sizes[0]; ++idx) {
    unsigned long rnd_index = ((seed * seed + (idx + 7) * (idx + 13)) * 487);
    if (/*++*/seed >= 0x7AB) seed = 0xAB;
    unsigned items_left = sizes[0] - idx;
    // Print a nested vector with 50% probability.
    /*
    if (items_left > kSizes[0] && rnd_index % 500 >= 250) {
      int max_size_idx = 3;
      while (kSizes[max_size_idx] > items_left - 1) --max_size_idx;
      int size = kSizes[rnd_index % (max_size_idx + 1)];
      Vector vec(NULL, "", type, NULL, &qfer, size, NULL);
      ret.append(vec.build_initializer_str(init_strings));
      idx += size - 1;
    } else {
      ret.append(init_strings[rnd_index % init_strings.size()]);
    }
    */
    ret.append(init_strings[rnd_index % init_strings.size()]);
    if (items_left > 1) ret.append(", ");
  }
  ret.append(")");
  return ret;
}

int Vector::GetRandomVectorLength(int max) {
  if (!max) max = kSizes[kSizesCount - 1];
  int size_idx = kSizesCount - 1;
  while (size_idx >= 0 && kSizes[size_idx] > (unsigned)max) --size_idx;
  return size_idx >= 0 ? kSizes[rnd_upto(size_idx + 1)] : 0;
}

const Type *Vector::PromoteTypeToVectorType(const Type *type, int size) {
  if (!size) size = GetRandomVectorLength(0);
  if (type->eType == eVector && type->vector_length_ == size) return type;
  auto it = vector_types_.find(std::make_pair(type->simple_type, size));
  assert(it != vector_types_.end() && "Unknown vector type.");
  return it->second;
}

const Type &Vector::DemoteVectorTypeToType(const Type *type) {
  return type->eType == eVector ? Type::get_simple_type(type->simple_type) :
      *type;
}


char Vector::GetComponentChar(int vector_size, int index) {
  return vector_size <= 4 ? kCompSmall[index] : kCompBig[index];
}

const char *Vector::GetSuffixString(/*enum*/ int suffix) {
  switch (suffix) {
    case 0: return kHiStr;
    case 1: return kLoStr;
    case 2: return kEvenStr;
    case 3: return kOddStr;
  }
  return "";
}

void Vector::OutputVectorType(std::ostream& out, const Type *type,
    int vector_size) {
    out << "VECTOR(";
  DemoteVectorTypeToType(type).Output(out);
  out << ", " << vector_size << ')';
}
void Vector::OutputVectorType_Make(std::ostream& out, const Type *type,
    int vector_size) {
    out << "VECTOR_MAKE(";
  DemoteVectorTypeToType(type).Output(out);
  out << ", " << vector_size << ')';
}

void Vector::GenerateVectorTypes() {
  for (enum eSimpleType simple = eChar; simple < MAX_SIMPLE_TYPES;
      simple = (eSimpleType)(simple + 1)) {
    for (unsigned size_idx = 0; size_idx < kSizesCount; ++size_idx) {
      Type *t = new Type(simple);
      t->eType = eVector;
      t->vector_length_ = kSizes[size_idx];
      vector_types_[std::make_pair(simple, kSizes[size_idx])] = t;
    }
  }
}

ArrayVariable *itemize_simd_hook(const ArrayVariable *av, int access_count) {
  const Vector *vec = dynamic_cast<const Vector *>(av);
  assert(vec != NULL);
  return vec->itemize_simd(access_count);
}

}  // namespace CUDASmith

