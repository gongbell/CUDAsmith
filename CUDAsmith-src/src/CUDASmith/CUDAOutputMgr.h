// Outputs to the designated stream such that the resulting file can be compiled
// as OpenCL C.
// Relies on the caller to disable parts of the program that produce invalid
// OpenCL C, as this class will call the output function of the standard csmith
// output managers.

#ifndef _CUDASMITH_CLOUTPUTMGR_H_
#define _CUDASMITH_CLOUTPUTMGR_H_

#include <fstream>
#include <string>

#include "CommonMacros.h"
#include "OutputMgr.h"

namespace CUDASmith {

class Globals;

class CUDAOutputMgr : public OutputMgr {
 public:
  CUDAOutputMgr();
  explicit CUDAOutputMgr(const std::string& filename) : out_(filename.c_str()) {}
  explicit CUDAOutputMgr(const char *filename) : out_(filename) {}
  ~CUDAOutputMgr() { out_.close(); }
  
  // Outputs information regarding the runtime to be read by the host code
  void OutputRuntimeInfo(const std::vector<unsigned int>& threads,
                         const std::vector<unsigned int>& groups);

  // Inherited from OutputMgr. Outputs comments, #defines and forward
  // declarations.
  void OutputHeader(int argc, char *argv[], unsigned long seed);

  // Inherited from OutputMgr. Outputs all the definitions.
  void Output();

  // Inherited from OutputMgr. Gets the stream used for printing the output.
  std::ostream &get_main_out();

  // Outputs the kernel entry function. OutputMain in OutputMgr isn't virtual,
  // so we can't override it.
  void OutputEntryFunction(Globals& globals);

 private:
  std::ofstream out_;

  DISALLOW_COPY_AND_ASSIGN(CUDAOutputMgr);
};

}  // namespace CUDASmith

#endif  // _CUDASMITH_CLOUTPUTMGR_H_
