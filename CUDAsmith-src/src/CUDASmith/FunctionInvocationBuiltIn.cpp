#include "CUDASmith/FunctionInvocationBuiltIn.h"

#include <map>
#include <ostream>
#include <vector>

#include "CUDASmith/CUDAOptions.h"
#include "CUDASmith/Vector.h"
#include "ProbabilityTable.h"
#include "Type.h"
#include "VectorFilter.h"

namespace CUDASmith {
namespace {
// Integer function tables.
DistributionTable *integer_func_table = NULL;
// Internal::ParameterType::ParameterTypeMap<enum FunctionInvocationIntegerBuiltIn::BuiltIn> *integer_param_map;
std::map<enum FunctionInvocationIntegerBuiltIn::BuiltIn,
         std::vector<Internal::ParameterType::TypeConversion>> *
    integer_param_map = NULL;

// Integer function names. The array indices line up with the enum value.
const char *const kIntegerNames[20] = {
    ""     ,  "abs"    , "abs_diff", "add_sat" , "hadd" , "rhadd", "clamp" ,
    "clz"   , "ctz"    , "mad_hi"  , "mad_sat" , "max"  , "min"  , "mul_hi",
    "rotate", "sub_sat", "upsample", "popcount", "mad24", "mul24"
};
}  // namespace

namespace Internal {
namespace ParameterType {

const Type& ConvertParameterType(
    const Type& type, enum TypeConversion conversion) {
  // Only convert scalar and vector types. Other types should never get here.
  assert(type.eType == eSimple || type.eType == eVector);
  // For kFlipSignChance2.
  static int flip_next = 0;
  // For kDemoteChance2.
  static int demote_next = 0;

  const Type *t = NULL;
  switch (conversion) {
    case kExact: return type;
    case kUnsignToSign:
      t = type.to_signed(); break;
    case kSignToUnsign:
      t = type.to_unsigned(); break;
    case kFlipSign:
      t = type.is_signed() ? type.to_unsigned() : type.to_signed(); break;
    case kFlipSignChance:
      t = rnd_flipcoin(50) ? &ConvertParameterType(type, kFlipSign) : &type;
      break;
    case kFlipSignChance2: {
      bool reset_flip_next = flip_next != 0;
      if (flip_next == 0) flip_next = rnd_flipcoin(50) ? 1 : -1;
      t = flip_next == 1 ? &ConvertParameterType(type, kFlipSign) : &type;
      if (reset_flip_next) flip_next = 0;
      break;
    }
    case kDemote:
      return Vector::DemoteVectorTypeToType(&type);
    case kDemoteChance:
      return rnd_flipcoin(50) ? Vector::DemoteVectorTypeToType(&type) : type;
    case kDemoteChance2: {
      bool reset_demote_next = demote_next != 0;
      if (demote_next == 0) demote_next = rnd_flipcoin(50) ? 1 : -1;
      t = demote_next == 1 ? &Vector::DemoteVectorTypeToType(&type) : &type;
      if (reset_demote_next) demote_next = 0;
      return *t;
    }
    case kPromote:
      return *Vector::PromoteTypeToVectorType(&type, 0);
    case kNarrow: {
      eSimpleType simple =
          (eSimpleType)(type.simple_type - (type.SizeInBytes() > 1 ? 1 : 0));
      if (simple == eInt || simple == eUInt)
        simple = (eSimpleType)(simple - 1);
      t = &Type::get_simple_type(simple);
      break;
    }
    case kFlipNarrow:
      t = &ConvertParameterType(ConvertParameterType(type,
          kFlipSign), kNarrow);
      break;
    case kToUnsignNarrow:
      t = &ConvertParameterType(ConvertParameterType(type,
          kSignToUnsign), kNarrow);
      break;
    default: assert(false && "Unknown type conversion");
  }

  // For type conversions where the original type is a vector, and so the new
  // type must be promoted, break out of the above switch instead of returning.
  // For nested type conversions, must be careful this does not promote when we
  // don't want to (e.g. Conv(Conv(type, Demote), Flip), outer Conv will
  // promote).
  assert(t != NULL);
  if (type.eType == eVector && t->eType != eVector)
    t = Vector::PromoteTypeToVectorType(t, type.vector_length_);
  return *t;
}

}  // namespace ParameterType
}  // namespace Internal

FunctionInvocationBuiltIn *FunctionInvocationBuiltIn::make_random(
    CGContext& cg_context, const Type& type) {
  return FunctionInvocationIntegerBuiltIn::make_random(cg_context, type);
}

void FunctionInvocationBuiltIn::InitTables() {
  FunctionInvocationIntegerBuiltIn::InitTables();
}

void FunctionInvocationBuiltIn::Output(std::ostream& out) const {
  // del comments by wxy 2018-01-19
  if (CUDAOptions::safe_math()) {
    OutputSafeMacro(out);
  } else {
    OutputFuncName(out);
    out << '(';
  }
  for (size_t idx = 0; idx < param_value.size(); ++idx) {
    // To prevent ambiguities for functions that take both vector and scalars,
    // output a cast for scalar parameters. To prevent ambiguity, the type must
    // be the exact expected type, but csmith may change it to a compatible
    // type, so we must convert the return type.
    if (param_value[idx]->get_type().eType == eSimple) {
      const Type& cast_type = GetParameterType(idx);
      const Type& simple_cast_type = Vector::DemoteVectorTypeToType(&cast_type);
      out << "(";
      simple_cast_type.Output(out);
      out << ")";
    }
    param_value[idx]->Output(out);
    if (idx < param_value.size() - 1) out << ", ";
  }
  out << ')';
  //end
}

void FunctionInvocationBuiltIn::indented_output(std::ostream &out, int indent)
    const {
  output_tab(out, indent);
  Output(out);
}

FunctionInvocationIntegerBuiltIn *FunctionInvocationIntegerBuiltIn::make_random(
    CGContext& cg_context, const Type& type) {
  std::vector<const Type *> param_types;
  enum BuiltIn func = FunctionSelector(type, &param_types);
  FunctionInvocationIntegerBuiltIn *fi =
      new FunctionInvocationIntegerBuiltIn(func, type);
  for (const Type *param_type : param_types)
    fi->param_value.push_back(Expression::make_random(cg_context, param_type));
  return fi;
}

enum FunctionInvocationIntegerBuiltIn::BuiltIn
    FunctionInvocationIntegerBuiltIn::FunctionSelector(
    const Type& type, std::vector<const Type *> *params) {
  assert(params != NULL);
  params->clear();
  if (type.eType != eSimple && type.eType != eVector) {
    params->push_back(&type);
    return kIdentity;
  }

  assert(integer_func_table != NULL);
  VectorFilter filter(integer_func_table);
  // Not available in OpenCL 1.0.
  filter.add(kCtz).add(kPopCount);
  // Abs will not return a signed value.
  if (type.is_signed())
    filter.add(kAbs).add(kAbsDiff);
  // Cannot upsample to a char.
  if (type.simple_type == eChar || type.simple_type == eUChar)
    filter.add(kUpSample);
  // Can only do mad24 and mul24 on 32 bit types.
  if (type.simple_type != eInt && type.simple_type != eUInt)
    filter.add(kMad24).add(kMul24);
  int rnd = rnd_upto(filter.get_max_prob(), &filter);
  enum BuiltIn func = (enum BuiltIn)filter.lookup(rnd);

  // Populate params based on the selected function.
  assert(integer_param_map != NULL);
  auto it = integer_param_map->find(func);
  assert(it != integer_param_map->end() && "Cannot find params for int func");
  for (enum Internal::ParameterType::TypeConversion conversion : it->second)
    params->push_back(
        &Internal::ParameterType::ConvertParameterType(type, conversion));
  return func;
}

void FunctionInvocationIntegerBuiltIn::InitTables() {
  integer_func_table = new DistributionTable();
  for (int func = kAbs; func <= kMul24; ++func)
    integer_func_table->add_entry(func, 10);

  integer_param_map = new std::map<
      enum BuiltIn, std::vector<Internal::ParameterType::TypeConversion>>();
  using Internal::ParameterType::kExact;
  using Internal::ParameterType::kFlipSignChance;
  using Internal::ParameterType::kFlipSignChance2;
  using Internal::ParameterType::kDemoteChance;
  using Internal::ParameterType::kDemoteChance2;
  using Internal::ParameterType::kNarrow;
  using Internal::ParameterType::kToUnsignNarrow;
  (*integer_param_map)[kIdentity] = { kExact };
  (*integer_param_map)[kAbs]      = { kFlipSignChance };
  (*integer_param_map)[kAbsDiff]  = { kFlipSignChance2, kFlipSignChance2 };
  (*integer_param_map)[kAddSat]   = { kExact, kExact };
  (*integer_param_map)[kHAdd]     = { kExact, kExact };
  (*integer_param_map)[kRHAdd]    = { kExact, kExact };
  (*integer_param_map)[kClamp]    = { kExact, kDemoteChance2, kDemoteChance2 };
  (*integer_param_map)[kClz]      = { kExact };
  (*integer_param_map)[kCtz]      = { kExact };
  (*integer_param_map)[kMadHi]    = { kExact, kExact, kExact };
  (*integer_param_map)[kMadSat]   = { kExact, kExact, kExact };
  (*integer_param_map)[kMax]      = { kExact, kDemoteChance };
  (*integer_param_map)[kMin]      = { kExact, kDemoteChance };
  (*integer_param_map)[kMulHi]    = { kExact, kExact };
  (*integer_param_map)[kRotate]   = { kExact, kExact };
  (*integer_param_map)[kSubSat]   = { kExact, kExact };
  (*integer_param_map)[kUpSample] = { kNarrow, kToUnsignNarrow };
  (*integer_param_map)[kPopCount] = { kExact };
  (*integer_param_map)[kMad24]    = { kExact, kExact, kExact };
  (*integer_param_map)[kMul24]    = { kExact, kExact };
}

FunctionInvocationIntegerBuiltIn *FunctionInvocationIntegerBuiltIn::clone()
    const {
  FunctionInvocationIntegerBuiltIn *fi =
      new FunctionInvocationIntegerBuiltIn(built_in_, type_);
  for (const Expression *expr : param_value)
    fi->param_value.push_back(expr->clone());
  return fi;
}

void FunctionInvocationIntegerBuiltIn::OutputFuncName(std::ostream& out) const {
  out << kIntegerNames[built_in_];
}

const Type& FunctionInvocationIntegerBuiltIn::GetParameterType(size_t idx)
    const {
  auto it = integer_param_map->find(built_in_);
  assert(it != integer_param_map->end());
  assert(it->second.size() > idx);
  return Internal::ParameterType::ConvertParameterType(type_, it->second[idx]);
}

void FunctionInvocationIntegerBuiltIn::OutputSafeMacro(std::ostream& out)
    const {
  bool requires_macro = (built_in_ == kMadHi && type_.is_signed()) ||
      built_in_ == kMul24 || built_in_ == kMad24 || built_in_ == kClamp;
  if (requires_macro) {
    out << "safe_";
  }
  OutputFuncName(out);
  if (requires_macro) {
    out << "_func";
    if (built_in_ != kClamp) {
      out << '_';
      Vector::DemoteVectorTypeToType(&type_).Output(out);
      for (const Expression *expr : param_value)
        out << '_' << (expr->get_type().is_signed() ? 's' : 'u');
    }
  }
  out << '(';
  if (requires_macro) {
    // Conveniently, the first type for all of these macros is the same as the
    // return type. For clamp, it may be a vector or scalar of the return type.
    type_.Output(out);
    out << ',';
    if (built_in_ == kClamp) {
      const Type& param_type = param_value[1]->get_type();
      if (param_type.eType == eVector) type_.Output(out);
      else Vector::DemoteVectorTypeToType(&type_).Output(out);
      out << ',';
    }
  }
}

}  // namespace CUDASmith
