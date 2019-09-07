// Functions that are specific to OpenCL.
// Such functions include built-ins (as described in section 6.13 of OpenCL
// specification 2.0), and possibly more.
// TODO make some kind of DEFAULT_MOVE_AND_DESTRUCTOR macro.

#ifndef _CUDASMITH_CLFUNCTIONINVOCATION_H_
#define _CUDASMITH_CLFUNCTIONINVOCATION_H_

#include "CommonMacros.h"
#include "CVQualifiers.h"
#include "FunctionInvocation.h"
#include "SafeOpFlags.h"

#include <map>
#include <ostream>
#include <vector>

class CGContext;
class Expression;
class Type;
class Variable;

namespace CUDASmith {
namespace Internal {
// Helper namespace for handling the parameter types required for built-in
// functions.
namespace ParameterType {

// List of all required type conversions.
// These indicate how the return type of the function needs to be transformed
// to give the type of a given parameter.
// Possible improvement, remove compund conversions, and allow us to chain them.
enum TypeConversion {
  kExact = 0,        // No conversion.
  kUnsignToSign,     // Convert to unsigned iff a signed type.
  kSignToUnsign,     // Convert to signed iff a unsigned type.
  kFlipSign,         // Flip from signed to unsigned and vice versa.
  kFlipSignChance,   // Flip signedness with 50% probability.
  kFlipSignChance2,  // Flip chance, where the next depends on the previous.
  kDemote,           // Demote from vector to scalar.
  kDemoteChance,     // Demote from vector to scalar with 50% probability.
  kDemoteChance2,    // Demote chance, where the next depends on the previous.
  kPromote,          // Promote from scalar to vector.
  kNarrow,           // Reduce the size of the type (e.g. int -> short).
  kFlipNarrow,       // FlipSign and Narrow.
  kToUnsignNarrow,   // SignToUnsign and Narrow.
};

// Useful typedefs for handling parameters. Need GCC 4.7
//template <class BuiltIn>
//using ParameterTypeMap = std::map<BuiltIn, std::vector<enum TypeConversion>>;

// Takes the return type and a conversion style and returns the converted type.
const Type& ConvertParameterType(
    const Type& type, enum TypeConversion conversion);

}  // namespace ParameterType
}  // namespace Internal

// Set of built-in OpenCL function.
// These have been separated into their own subclasses based on their behaviour.
// Each sub-class should provide a static FunctionSelector method producing a
// valid function for the specified return type, and the parameters required.
class FunctionInvocationBuiltIn : public FunctionInvocation {
 public:
  enum BuiltInType { kInteger = 0 };
  FunctionInvocationBuiltIn(enum BuiltInType built_in_type, const Type& type)
    : FunctionInvocation(eBuiltIn, SafeOpFlags::make_dummy_flags()),
      type_(type), built_in_type_(built_in_type) {
  }
  FunctionInvocationBuiltIn(FunctionInvocationBuiltIn&& other) = default;
  FunctionInvocationBuiltIn& operator=(
      FunctionInvocationBuiltIn&& other) = default;
  virtual ~FunctionInvocationBuiltIn() {}

  // Top level factory for a random function.
  // It is unlikely that this will be called, rather the make random in the sub
  // classes should be called directly.
  static FunctionInvocationBuiltIn *make_random(
      CGContext& cg_context, const Type& type);

  // Initialises the probability tables and type mappings.
  static void InitTables();

  // Pure virtual in FunctionInvocation.
  const Type &get_type() const { return type_; }
  void Output(std::ostream &) const;
  void indented_output(std::ostream &out, int indent) const;
  bool safe_invocation() const { return true; }

  // Output only the name of the built-in function (no brackets or params).
  virtual void OutputFuncName(std::ostream& out) const = 0;

  // Get the expected type of the function paramter at index idx.
  virtual const Type& GetParameterType(size_t idx) const = 0;

  enum BuiltInType GetBuiltInType() const { return built_in_type_; }
 protected:
  // Outputs the function wrapped in a safe math macro.
  virtual void OutputSafeMacro(std::ostream& out) const = 0;

  const Type& type_;
 private:
  enum BuiltInType built_in_type_;
  DISALLOW_COPY_AND_ASSIGN(FunctionInvocationBuiltIn);
};

// OpenCL built-in integer functions as described in section 6.13.3 of the
// OpenCL specification 2.0.
class FunctionInvocationIntegerBuiltIn : public FunctionInvocationBuiltIn {
 public:
  // List of all built-ins.
  enum BuiltIn {
    kIdentity = 0,  // Sentinel.
    kAbs, kAbsDiff, kAddSat, kHAdd, kRHAdd, kClamp, kClz, kCtz, kMadHi, kMadSat,
    kMax, kMin, kMulHi, kRotate, kSubSat, kUpSample, kPopCount, kMad24, kMul24
  };

  FunctionInvocationIntegerBuiltIn(enum BuiltIn built_in, const Type& type)
    : FunctionInvocationBuiltIn(kInteger, type), built_in_(built_in) {
  }
  FunctionInvocationIntegerBuiltIn(
      FunctionInvocationIntegerBuiltIn&& other) = default;
  FunctionInvocationIntegerBuiltIn& operator=(
      FunctionInvocationIntegerBuiltIn&& other) = default;
  virtual ~FunctionInvocationIntegerBuiltIn() {}

  // Factory for creating a random invocation for integer built-ins.
  static FunctionInvocationIntegerBuiltIn *make_random(
      CGContext& cg_context, const Type& type);

  // Factory for selecting a random built in integer function for vectors.
  // The return type of the selected function will be valid for the passed type.
  // params will be filled with the types expected for the function parameters.
  // If no function is valid, the identity function will be returned.
  static enum BuiltIn FunctionSelector(
      const Type& type, std::vector<const Type *> *params);

  // Initialises the probability tables and type mappings.
  static void InitTables();

  // Pure virtual in FunctionInvocation.
  FunctionInvocationIntegerBuiltIn *clone() const;

  // Pure virtual in FunctionInvocationBuiltIn.
  void OutputFuncName(std::ostream& out) const;
  const Type& GetParameterType(size_t idx) const;

  enum BuiltIn GetBuiltIn() const { return built_in_; }
 protected:
  void OutputSafeMacro(std::ostream& out) const;
 private:
  enum BuiltIn built_in_;

  DISALLOW_COPY_AND_ASSIGN(FunctionInvocationIntegerBuiltIn);
};

}  // namespace CUDASmith

#endif  // _CUDASMITH_CLFUNCTIONINVOCATION_H_
