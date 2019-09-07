#include "CUDAVariable.h"

#include "Block.h"
#include "Expression.h"
#include "ExpressionAssign.h"
#include "ExpressionComma.h"
#include "ExpressionFuncall.h"
#include "ExpressionVariable.h"
#include "CUDASmith/ExpressionVector.h"
#include "CUDASmith/Walker.h"
#include "Function.h"
#include "FunctionInvocation.h"
#include "FunctionInvocationUnary.h"
#include "Statement.h"
#include "StatementAssign.h"
#include "StatementArrayOp.h"
#include "StatementBreak.h"
#include "StatementContinue.h"
#include "StatementExpr.h"
#include "StatementFor.h"
#include "StatementGoto.h"
#include "StatementIf.h"
#include "StatementReturn.h"
#include "Variable.h"

#include <set>
#include <vector>
#include <map>
#include <algorithm>

namespace CUDASmith {

void CUDAVariable::ParseUnusedVars() {
  std::set<const Variable*> accesses;
  std::set<const Variable*> expr_vars;

  for (Function * f : get_all_functions()) {
    for (Block * b : f->blocks) {
      Helpers::RecordVarInitialisation(&accesses, b);
      for (Statement * s : b->stms) {
        switch(s->eType) {
          case eAssign: 
            { StatementAssign* as = dynamic_cast<StatementAssign*>(s);
              const Variable* lhsv = as->get_lhs()->get_var()->get_collective();
              Helpers::GetVarsFromExpr(&expr_vars, as->get_expr());
              Helpers::RecordVar(&accesses, lhsv);
              Helpers::RecordMultipleVars(&accesses, &expr_vars);
              break; }
          case eReturn: 
            { StatementReturn* rs = dynamic_cast<StatementReturn*>(s);
              const Variable* rsv = rs->get_var()->get_var()->get_collective();
              Helpers::RecordVar(&accesses, rsv);
              break; }
          case eFor: 
            { StatementFor* sf = dynamic_cast<StatementFor*>(s);
              const Variable* initlhsv = 
                sf->get_init()->get_lhs()->get_var()->get_collective();
              const Variable* incrlhsv = 
                sf->get_init()->get_lhs()->get_var()->get_collective();
              Helpers::GetVarsFromExpr(&expr_vars, sf->get_init()->get_expr());
              Helpers::RecordMultipleVars(&accesses, &expr_vars);
              Helpers::GetVarsFromExpr(&expr_vars, sf->get_test());
              Helpers::RecordMultipleVars(&accesses, &expr_vars);
              Helpers::GetVarsFromExpr(&expr_vars, sf->get_incr()->get_expr());
              Helpers::RecordMultipleVars(&accesses, &expr_vars);
              Helpers::RecordVar(&accesses, initlhsv);
              Helpers::RecordVar(&accesses, incrlhsv);
              break; }
          case eIfElse: 
            { StatementIf* si = dynamic_cast<StatementIf*>(s);
              Helpers::GetVarsFromExpr(&expr_vars, si->get_test());
              Helpers::RecordMultipleVars(&accesses, &expr_vars);
              break; }
          case eInvoke: 
            { StatementExpr* se = dynamic_cast<StatementExpr*>(s);
              Helpers::GetVarsFromExpr(&expr_vars, se->get_call());
              Helpers::RecordMultipleVars(&accesses, &expr_vars);
              break; }
          case eContinue: 
            { StatementContinue* sc = dynamic_cast<StatementContinue*>(s);
              Helpers::GetVarsFromExpr(&expr_vars, &sc->test);
              Helpers::RecordMultipleVars(&accesses, &expr_vars);
              break; }
          case eBreak: 
            { StatementBreak* sb = dynamic_cast<StatementBreak*>(s);
              Helpers::GetVarsFromExpr(&expr_vars, &sb->test);
              Helpers::RecordMultipleVars(&accesses, &expr_vars);
              break; }
          case eGoto: 
            { StatementGoto* sg = dynamic_cast<StatementGoto*>(s);
              Helpers::GetVarsFromExpr(&expr_vars, &sg->test);
              Helpers::RecordMultipleVars(&accesses, &expr_vars);
              break; }
          case eArrayOp: 
            { StatementArrayOp* sao = dynamic_cast<StatementArrayOp*>(s);
              const Variable* arrv = sao->array_var->get_collective();
              std::set<const Variable*> ctrlv 
                (sao->ctrl_vars.begin(), sao->ctrl_vars.end());
              Helpers::GetVarsFromExpr(&expr_vars, sao->init_value);
              Helpers::RecordMultipleVars(&accesses, &expr_vars);
              Helpers::RecordVar(&accesses, arrv);
              Helpers::RecordMultipleVars(&accesses, &ctrlv);
              break; }
          case eCUDAStatement:
          case eBlock: 
            break;
          default:
            assert(0);
        }
      }
    }
  }
  
  for (Function* f : get_all_functions()) {
    for (Block* b : f->blocks) {
      for (Variable* v : b->local_vars) {
        if (std::find(accesses.begin(), accesses.end(), v) == accesses.end()) {
          b->local_vars.erase(std::remove(b->local_vars.begin(), b->local_vars.end(), v), b->local_vars.end());
        }
      }
    }
  }
}

void CUDAVariable::Helpers::GetVarsFromExpr(std::set<const Variable*>* expr_vars, const Expression* expr) {
  assert(expr_vars != NULL);
  expr_vars->clear();
  std::set<const Variable*> inner_expr_vars;
  switch (expr->term_type) {
    case eVariable: 
      { const ExpressionVariable* ev = 
          dynamic_cast<const ExpressionVariable*>(expr);
        expr_vars->insert(ev->get_var()->get_collective()); break; }
    case eFunction: 
      { const ExpressionFuncall* efun = 
          dynamic_cast<const ExpressionFuncall*>(expr); 
        for (const Expression* e : efun->get_invoke()->param_value) {
          GetVarsFromExpr(&inner_expr_vars, e);
          expr_vars->insert(inner_expr_vars.begin(), inner_expr_vars.end());
        }
        break; }
    case eAssignment: 
      { const ExpressionAssign* ea = dynamic_cast<const ExpressionAssign*>(expr);
        expr_vars->insert(ea->get_lhs()->get_var()->get_collective());
        GetVarsFromExpr(&inner_expr_vars, ea->get_rhs());
        expr_vars->insert(inner_expr_vars.begin(), inner_expr_vars.end());
        break; }
    case eCLExpression: 
      { const CUDAExpression* cle = dynamic_cast<const CUDAExpression*>(expr);
        if (cle->GetCLExpressionType() != CUDAExpression::kVector)
          break;
        const ExpressionVector* ev = dynamic_cast<const ExpressionVector*>(expr);
        for (const std::unique_ptr<const Expression>& e : ev->GetExpressions()) {
          const Expression* e_ptr = e.get();
          GetVarsFromExpr(&inner_expr_vars, e_ptr);
          expr_vars->insert(inner_expr_vars.begin(), inner_expr_vars.end());
        }
        break; }
    case eCommaExpr: 
      { const ExpressionComma* ec = dynamic_cast<const ExpressionComma*>(expr);
        GetVarsFromExpr(&inner_expr_vars, ec->get_lhs());
        expr_vars->insert(inner_expr_vars.begin(), inner_expr_vars.end());
        GetVarsFromExpr(&inner_expr_vars, ec->get_rhs());
        expr_vars->insert(inner_expr_vars.begin(), inner_expr_vars.end());
        break;
      }
    case eConstant:
    case eLhs:  // NOTE seems to be very particular case for something 
      break; 
    default:
      assert(0);
  }
}

void CUDAVariable::Helpers::RecordVarInitialisation(
std::set<const Variable*>* accesses, Block* b) {
  std::set<const Variable*> init_vars;
  for (const Variable* v : b->local_vars) {
    GetVarsFromExpr(&init_vars, v->init);
    RecordMultipleVars(accesses, &init_vars);
  }
}

void CUDAVariable::Helpers::RecordVar(
std::set<const Variable*>* accesses, const Variable* var) {
  // ignore global and parameter variables
  if (!var->to_string().compare(0, 2, "g_") 
   || !var->to_string().compare(0, 2, "p_")) {
    return;
  }
  accesses->insert(var);
}

void CUDAVariable::Helpers::RecordMultipleVars(
std::set<const Variable*>* accesses, 
std::set<const Variable*>* vars) {
  for (const Variable* v : *vars) {
    RecordVar(accesses, v);
  }
}

}
