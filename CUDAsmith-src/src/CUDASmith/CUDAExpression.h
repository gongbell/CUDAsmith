// Root of expressions specific to OpenCL.
// Acts as a hook by which CUDASmith can inject behaviour into csmith.

#ifndef _CUDASMITH_CLEXPRESSION_H_
#define _CUDASMITH_CLEXPRESSION_H_

#include "CommonMacros.h"
#include "Expression.h"

class CGContext;
class CVQualifiers;
class DistributionTable;
class Type;
class VectorFilter;

namespace CUDASmith {

// All OpenCL related expressions derive from this.
// The type of OpenCL expression to use is handled here, instead of in
// Expression. So we have our own probability table.
class CUDAExpression : public Expression {
 public:
  // Dynamic type information.
  enum CUDAExpressionType {
    kNone = 0,  // Sentinel value.
    kID,
    kVector,
    kAtomic
  };

  explicit CUDAExpression(CUDAExpressionType type) : Expression(eCLExpression),
      cl_expression_type_(type) {
  }
  // Must be careful with the move constructors. These will call the copy constructor
  // in Expression.
  CUDAExpression(CUDAExpression&& other) = default;
  CUDAExpression& operator=(CUDAExpression&& other) = default;
  virtual ~CUDAExpression() {}

  // Factory for creating a random OpenCL expression. This will typically be
  // called by Expression::make_random.
  static /*CL*/Expression *make_random(CGContext &cg_context, const Type *type,
      const CVQualifiers* qfer, enum CUDAExpressionType tt);

  // Create the random probability table, should be called once on startup.
  static void InitProbabilityTable();

  // When evaluated at runtime, will this produce a divergent value.
  virtual bool IsDivergent() const { return false; }//; = 0;
  
  // Getter for cl_expression_type_
  enum CUDAExpressionType GetCLExpressionType() const {
    return cl_expression_type_;
  }

 private:
  CUDAExpressionType cl_expression_type_;
  // Table used for deciding on which OpenCL expression to generate.
  //static DistributionTable *cl_expr_table_;

  DISALLOW_COPY_AND_ASSIGN(CUDAExpression);
};

// Hook method for csmith.
Expression *make_random(CGContext& cg_context, const Type *type,
    const CVQualifiers *qfer);

}  // namespace CUDASmith

#endif  // _CUDASMITH_CLEXPRESSION_H_
