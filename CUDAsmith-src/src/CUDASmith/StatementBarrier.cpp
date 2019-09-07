#include "CUDASmith/StatementBarrier.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "CUDASmith/Divergence.h"
#include "CUDASmith/Globals.h"
#include "CUDASmith/MemoryBuffer.h"
#include "CUDASmith/Walker.h"
#include "Constant.h"
#include "CVQualifiers.h"
#include "Function.h"
#include "Type.h"
#include "util.h"
#include "Variable.h"
#include "VariableSelector.h"

namespace CUDASmith {

StatementBarrier *StatementBarrier::make_random(Block *block) {
  CVQualifiers gate_qfer(std::vector<bool>({false}), std::vector<bool>({false}));
  Variable *gate = VariableSelector::new_variable(gensym("gate_"),
      &Type::get_simple_type(eInt), Constant::make_int(0), &gate_qfer);
  MemoryBuffer *buffer = MemoryBuffer::CreateMemoryBuffer(MemoryBuffer::kLocal,
      gensym("lb_"), &Type::get_simple_type(eUInt), Constant::make_int(1), {32});
  return new StatementBarrier(block, gate, buffer);
}

void StatementBarrier::Output(std::ostream& out, FactMgr */*fm*/, int indent) const {
  if (gate_ == NULL) {
    output_tab(out, indent);
    OutputBarrier(out);
    out << std::endl;
    return;
  }
  // Output gate check.
  output_tab(out, indent);
  out << "if (!";
  gate_->Output(out);
  out << ") {" << std::endl;
  // Retireve our item, put it in temp storage.
  output_tab(out, indent + 1);
  buffer_->type->Output(out);
  out << " temp = ";
  buffer_->OutputWithOwnedItem(out);
  out << ";" << std::endl;
  // Mutate the item.
  output_tab(out, indent + 1);
  out << "temp *= get_local_id(0);" << std::endl;
  // Block all threads with a barrier.
  output_tab(out, indent + 1);
  OutputBarrier(out);
  out << std::endl;
  // Store result in paired thread's item.
  output_tab(out, indent + 1);
  buffer_->Output(out);
  out << "[get_local_id(0) ^ 1] = temp;" << std::endl;
  // Block all threads with a barrier.
  output_tab(out, indent + 1);
  OutputBarrier(out);
  out << std::endl;
  // Close the gate and exit.
  output_tab(out, indent + 1);
  gate_->Output(out);
  out << " = 1;" << std::endl;
  output_tab(out, indent);
  out << "}" << std::endl;
}

void GenerateBarriers(Divergence *divergence, Globals *globals) {
  assert(globals != NULL);
  const std::vector<Function *>& functions = get_all_functions();
  for (Function *function : functions) {
    std::vector<std::pair<Statement *, Statement *>> divergent_sections;
    divergence->GetDivergentCodeSectionsForFunction(
        function, &divergent_sections);
    if (divergent_sections.empty()) continue;

    StatementBarrier *barrier = NULL;
    auto div_it = divergent_sections.begin();
    auto div_it_end = divergent_sections.end();
    while (div_it != div_it_end) {
      std::unique_ptr<Walker::FunctionWalker> pos(
          Walker::FunctionWalker::CreateFunctionWalkerAtStatement(
          function, div_it->second));
      ++div_it;

      // If it was at the end of the function, or the next divergent section
      // starts where this section ends then skip.
      bool at_end = !pos->Next();
      if (at_end) break;
      Statement *st = pos->GetCurrentStatement();
      Block *block = pos->GetCurrentBlock();
      if (div_it != div_it_end && div_it->first == st) continue;
      barrier = StatementBarrier::make_random(block);

      // Insert at position, may cause multiple moves in vector.
      std::vector<Statement *> *stms = &block->stms;
      auto st_it = std::find(stms->begin(), stms->end(), st);
      assert (st_it != stms->end());
      stms->insert(st_it, barrier);
      break;
    }

    // If barrier isn't NULL, must add the gate and buffer to globals
    // TODO barrier init.
    if (barrier == NULL) continue;
    globals->AddGlobalVariable(barrier->GetGate());
    globals->AddLocalMemoryBuffer(barrier->GetBuffer());
  }
}

}  // namespace CUDASmith
