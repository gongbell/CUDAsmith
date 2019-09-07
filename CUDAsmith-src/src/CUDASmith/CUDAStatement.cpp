#include "CUDASmith/CUDAStatement.h"

#include "Block.h"
#include "CGContext.h"
#include "CUDASmith/CUDAOptions.h"
#include "CUDASmith/ExpressionID.h"
#include "CUDASmith/StatementComm.h"
#include "CUDASmith/StatementAtomicReduction.h"
#include "CUDASmith/StatementEMI.h"
#include "CUDASmith/StatementMessage.h"
#include "ProbabilityTable.h"
#include "Type.h"
#include "VectorFilter.h"

#include "CUDASmith/StatementTG.h"
const int Total_Prob = 5+15+5+5+5+10+50;
namespace CUDASmith {
namespace {
DistributionTable *cl_stmt_table = NULL;
}  // namespace

CUDAStatement *CUDAStatement::make_random(CGContext& cg_context,
    enum CUDAStatementType st) {
//   cout<<"make random for clStatement"<<endl;
  //  cout<<(st==kNone)<<endl;
  // kNone means no statement type has been specified. A statement type will be
  // randomly chosen.
  if (st == kNone) {
   
    // Statement generation works mostly the same as expressions. The exception
    // is that if NULL is returned to Statement::make_random(), it will
    // recursively call itself, instead of specifying a non CUDAStatement.
    assert (cl_stmt_table != NULL);
    int num = rnd_upto(Total_Prob);
    st = (CUDAStatementType)VectorFilter(cl_stmt_table).lookup(num);
    // st = kTG;
    // cout<<(st==kTG)<<endl;
    // Must only include barriers if they are enabled and no divergence.
    if (st == kBarrier) {
      if (!CUDAOptions::barriers() || CUDAOptions::divergence()) return NULL;
    }
    // Only use EMI blocks if theey are enabled and we are not already in one.
    if (st == kEMI) {
      if (!CUDAOptions::emi() || cg_context.get_emi_context()) return NULL;
    }
    //add by wxy 2018-03-20
    if (st == kTG){
      
  //      add by wxy 2018-06-03
  //       为了随机生成tg语句的数目  
  /*    long int randnum = ((long int)time(NULL))%100;
      if(randnum < 50){
        if (!CUDAOptions::TG() || cg_context.get_tg_context()) return NULL;
        }
      else{
        return NULL;
      } 
    */  
      if (!CUDAOptions::TG() || cg_context.get_tg_context()) return NULL;
    }
    //end
    // Only generate atomic reductions if they are set.
    if (st == kReduction) {
      if (!CUDAOptions::atomic_reductions() || cg_context.get_atomic_context())
        return NULL;
    }
    // Fake divergence must also be set.
    // Not currently used, as fake divergence is handled by ExpressionID, which
    // will sometimes produce an expression using get_global_id. Would like to
    // move that functionality here for more control.
    if (st == kFakeDiverge) {
      /*if (!CUDAOptions::fake_divergence())*/ return NULL;
    }
    // Inter-thread communication must be set.
    if (st == kComm) {
      if (!CUDAOptions::inter_thread_comm() || cg_context.get_atomic_context())
        return NULL;
    }
    // MP must be set. // TODO check some context.
    if (st == kMessage) {
      if (!CUDAOptions::message_passing())
        return NULL;
    }
  }

  CUDAStatement *stmt = NULL;
  switch (st) {
    case kBarrier:
      assert(false);
    case kEMI:
      stmt = StatementEMI::make_random(cg_context); break;
    case kReduction:
      stmt = StatementAtomicReduction::make_random(cg_context); break;

    case kComm:
      stmt = StatementComm::make_random(cg_context);break;
    case kTG:
      stmt = StatementTG::make_random(cg_context); break;
      //    case kFakeDiverge:
//      stmt = CreateFakeDivergentIf(cg_context); break;
    case kMessage:
      stmt = StatementMessage::make_random(cg_context); break;
    default: assert(false);
  }
  return stmt;
}

/*CUDAStatement *CUDAStatement::CreateFakeDivergentIf(CGContext& cg_context) {
  const Type& type = Type::get_simple_type(eULongLong);
  StatementIf *st_if = new StatementIf(cg_context.get_current_block(),
      *ExpressionID::CreateFakeDivergentExpression(cg_context, type),
      *Block::make_random(cg_context), *Block::make_random(cg_context));
  // For now, bundle it in EMI but do not register with the controller.
  StatementEMI *cl_st = new StatementEMI(cg_context.get_current_block(), st_if);
  return cl_st;
}*/

void CUDAStatement::InitProbabilityTable() {
  cl_stmt_table = new DistributionTable();
  cl_stmt_table->add_entry(kBarrier, 5);
  cl_stmt_table->add_entry(kEMI, 15);
  cl_stmt_table->add_entry(kReduction, 5);
  cl_stmt_table->add_entry(kFakeDiverge, 5);
  cl_stmt_table->add_entry(kComm, 5);
  cl_stmt_table->add_entry(kMessage, 10);

  //add by wxy 2018-03-20
  cl_stmt_table->add_entry(kTG, 50);
}

Statement *make_random_st(CGContext& cg_context) {
  return CUDAStatement::make_random(cg_context, CUDAStatement::kNone);
}

}  // namespace CUDASmith
