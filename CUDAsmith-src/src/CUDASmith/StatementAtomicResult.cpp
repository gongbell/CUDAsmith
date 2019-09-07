#include "CUDASmith/StatementAtomicResult.h"
#include "CUDASmith/ExpressionAtomic.h"
#include "CUDASmith/CUDAStatement.h"

#include "Block.h"
#include "Expression.h"
#include "ExpressionFuncall.h"
#include "Function.h"
#include "Type.h"
#include "Variable.h"

#include <map>
#include <sstream>
#include <iterator>

namespace CUDASmith {
  
namespace {
static std::map<int, const ExpressionAtomicAccess*>* atomic_blocks = NULL;
}

void StatementAtomicResult::InitResults() {
  atomic_blocks = new std::map<int, const ExpressionAtomicAccess*>();
}
  
void StatementAtomicResult::GenSpecialVals() {
  for (Function* f : get_all_functions()) {
    for (Block* b : f->blocks) {
      if (atomic_blocks->find(b->stm_id) != atomic_blocks->end()) {
        StatementAtomicResult* sar_decl = new StatementAtomicResult(b);
        b->stms.push_back(sar_decl);
        for (Variable* v : b->local_vars) {
          if (v->get_collective() != v || v->type->eType == ePointer)
            continue;
          ArrayVariable* av = dynamic_cast<ArrayVariable*>(v);
          if (av != 0) {
            if (av->isVector) {
              std::vector<unsigned int> accesses = av->get_sizes();
              std::vector<int> curr_index (accesses.size(), 0);
              for (unsigned int i = 0; i < accesses.size(); i++) {
                accesses[i]--;
                curr_index[i] = accesses[i];
              }
              int index = accesses.size() - 1;
              Variable* itemized;
              while (true) {
                while (curr_index[index] >= 0) {
                  itemized = av->itemize(curr_index);
                  if (!itemized->field_vars.empty()) {
                    for (Variable * fv : itemized->field_vars) {
                        StatementAtomicResult* sar_res = new StatementAtomicResult(fv, b);
                        b->stms.push_back(sar_res);
                    }
                  }
                  else {
                    StatementAtomicResult* sar_res = new StatementAtomicResult(itemized, b);
                    b->stms.push_back(sar_res);
                  }
                  curr_index[index]--;
                }
                do {
                  curr_index[index] = accesses[index];
                  index--;
                } while (curr_index[index] == 0);
                if (index < 0) break;
                curr_index[index]--;
                index = curr_index.size() - 1;
              }
            } 
            else {
              StatementAtomicResult* sar_res = new StatementAtomicResult(av, b);
              b->stms.push_back(sar_res);
            }
          }
          else {
            StatementAtomicResult* sar_res = new StatementAtomicResult(v, b);
            b->stms.push_back(sar_res);
          }
        }
        StatementAtomicResult* sar_sv = new StatementAtomicResult((*atomic_blocks->find(b->stm_id)).second, b);
        b->stms.push_back(sar_sv);
      }
    }
  }
}

void StatementAtomicResult::RecordIfID(int id, Expression* expr) {
  ExpressionFuncall* ef = dynamic_cast<ExpressionFuncall*>(expr);
  const ExpressionAtomic* ea = NULL;
  for (const Expression* ef_param : ef->get_invoke()->param_value)
    if ((ea = dynamic_cast<const ExpressionAtomic*>(ef_param)))
      break;
  assert(ea);
  atomic_blocks->insert(std::pair<int, const ExpressionAtomicAccess*>(id, ea->get_access()));
}

void StatementAtomicResult::PrintStructArrayVars(std::ostream& out, int indent, Variable* v, std::stringstream& ss) const {
  for (std::vector<Variable*>::iterator it = v->field_vars.begin(); it != v->field_vars.end(); it++) {
    if (!(*it)->field_vars.empty()) {
      PrintStructArrayVars(out, indent, (*it), ss);
      if (std::next(it,1) != v->field_vars.end()) {
        out << std::endl;
        output_tab(out, indent);
      }
      continue;
    }
    out << ss.str() << (*it)->name.substr((*it)->name.find("."), (*it)->name.length()) << ";";
    if (std::next(it,1) != v->field_vars.end()) {
      out << std::endl;
      output_tab(out, indent);
    }
  }
}

void StatementAtomicResult::Output(std::ostream& out, FactMgr* fm, int indent) const {
  output_tab(out, indent);
  switch(result_type_) {
    case kAddVar  : { 
      if (!var_->field_vars.empty()) {
        std::stringstream res_str;
        res_str << "result += " << var_->name;
        PrintStructArrayVars(out, indent, var_, res_str);
      }
      else {
        out << "result += " << var_->to_string() << ";"; 
      }
      break;
    }
    case kAddArrVar : {
      std::vector<unsigned int> accesses = av_->get_sizes();
      out << "int " << av_->name << "_i0";
      for (unsigned int i = 1; i < accesses.size(); i++)
        out << ", " << av_->name << "_i" << i;
      out << ";" << std::endl;
      std::string curr_idx_name;
      for (unsigned int i = 0; i < accesses.size(); i++) {
        output_tab(out, indent);
        curr_idx_name = av_->name + "_i" + std::to_string(i);
        out << "for (" << curr_idx_name << " = 0;"; 
        out << " " << curr_idx_name << " < " << accesses[i] << ";";
        out << " " << curr_idx_name << "++) {";
        out << std::endl;
        indent++;
      }
      output_tab(out, indent);
      std::stringstream add_stream;
      add_stream << "result += " << av_->name;
      for (unsigned int i = 0; i < accesses.size(); i++) {
        curr_idx_name = av_->name + "_i" + std::to_string(i);
        add_stream << "[" << curr_idx_name << "]";
      }
      if (av_->field_vars.empty()) {
        out << add_stream.str() << ";";
      } else {
        PrintStructArrayVars(out, indent, av_, add_stream);
      }
      out << std::endl;
      for (unsigned int i = 0; i < accesses.size(); i++) {
        output_tab(out, --indent);
        out << "}";
        if ((i + 1) != accesses.size())
            out << std::endl;
      }
      break;
    }
    case kSetSpVal: { 
      assert(access_ != NULL);
      //out << "atomic_add(&";
	  /***start***/
	  out << "myAtomicAdd(&";
	  /***end***/
      if (this->access_->is_global()) {
	ExpressionAtomic::GetSVBuffer()->Output(out);
      } else if (this->access_->is_local()) {
	ExpressionAtomic::GetLocalSVBuffer()->Output(out);
      } else assert(0);
      out << "[";
      this->access_->Output(out);
      out << "], result);"; break; }
    case kDecl    : { out << "unsigned int result = 0;"; break; }
    default       : assert(0);
  }
  out << std::endl;
}

void StatementAtomicResult::DefineLocalResultVar(std::ostream& out) {
  out << "unsigned int result = 0";
}

} // namespace CUDASmith

