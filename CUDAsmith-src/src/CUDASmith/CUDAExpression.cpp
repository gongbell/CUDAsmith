#include "CUDASmith/CUDAExpression.h"

#include "CGContext.h"
#include "CGOptions.h"
#include "CUDASmith/CUDAOptions.h"
#include "CUDASmith/ExpressionAtomic.h"
#include "CUDASmith/ExpressionID.h"
#include "CUDASmith/ExpressionVector.h"
#include "ProbabilityTable.h"
#include "random.h"
#include "Type.h"
#include "VectorFilter.h"

class CVQualifiers;

namespace CUDASmith {
namespace {
DistributionTable *cuda_expr_table = NULL;
}  // namespace

/*CL*/Expression *CUDAExpression::make_random(CGContext &cg_context, const Type *type,
    const CVQualifiers *qfer, enum CUDAExpressionType tt) {
  // kNone is used for not specifying an expression type, as it does not require
  // recursive generation.
  // If the expected type is a vector type, then automatically set the term type
  // to a vector expression.
  if (tt == kNone && type->eType == eVector) tt = kVector;
  if (tt == kNone) {
    // The probability for Expression picking a CUDAExpression is the sum of the
    // maximum probabilities of all derived CLExpressions. If CUDAExpression is
    // picked, a second lookup is performed, possibly returning NULL.
    // Instead of filtering expression types, we decide when they are chosen if
    // they are eligible, if they are not, then return NULL. This prevents
    // overly inflating the probability of being picked.
    assert(cuda_expr_table != NULL);
    int num = rnd_upto(15);
    tt = type->eType == eVector ? kVector :
        (CUDAExpressionType)VectorFilter(cuda_expr_table).lookup(num);
    // kID will be one of divergence, fake divergence or group divergence.
    // Probability of calling get_global_id(0) without faking divergence is
    // dependent on block and expression depth.
    if (tt == kID) {
      if (!CUDAOptions::divergence() && !CUDAOptions::fake_divergence() &&
          !CUDAOptions::group_divergence())
        return NULL;
      if (type->eType != eSimple || type->is_signed())
        return NULL;
      // TODO find a way to limit div like this.
      //if ((unsigned)ExpressionID::GetGenerationProbability(cg_context, *type) <=
      //    rnd_upto(5))
    }
    // Not many restrictions on vector types.
    if (tt == kVector) {
      if (!CUDAOptions::vectors() ||
          (type->eType != eSimple && type->eType != eVector) ||
          (cg_context.expr_depth + 2 > CGOptions::max_expr_depth()))
        return NULL;
    }
  }

  /*CL*/Expression *expr = NULL;
  switch (tt) {
    case kID:
      expr = ExpressionID::make_random(cg_context, type); break;
    case kVector:
      expr = ExpressionVector::make_random(cg_context, type, qfer, 0); break;
    default: assert(false);
  }
  return expr;
}

void CUDAExpression::InitProbabilityTable() {
  // All probabilities are added, even if the expression type is disabled. This
  // is because the probability of picking a CUDAExpression is fixed in
  // Expression, so not adding them would artificially increase the
  // probabilities of other CLExpressions much more than desired.
  cuda_expr_table = new DistributionTable();
  cuda_expr_table->add_entry(kID, 5);
  cuda_expr_table->add_entry(kVector, 10);
  ExpressionVector::InitProbabilityTable();
}

Expression *make_random(CGContext &cg_context, const Type *type,
    const CVQualifiers *qfer) {
  return CUDAExpression::make_random(cg_context, type, qfer, CUDAExpression::kNone);
}

}  // namespace CUDASmith
