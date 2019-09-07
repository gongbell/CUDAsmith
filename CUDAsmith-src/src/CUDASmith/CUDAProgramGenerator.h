// Mostly a duplicate of AbsProgramGenerator, however, there is a tight coupling
// between it and the DFS and default program generators, which also have a
// tight coupling between the output managers. We redefine it here so we can use
// our own output manager.

#ifndef _CUDASMITH_CLPROGRAMGENERATOR_H_
#define _CUDASMITH_CLPROGRAMGENERATOR_H_

#include "AbsProgramGenerator.h"
#include "CUDASmith/CUDAOutputMgr.h"
#include "CommonMacros.h"

#include <memory>
#include <string>

namespace CUDASmith {

class CUDAProgramGenerator : public AbsProgramGenerator {
 public:
  explicit CUDAProgramGenerator(unsigned long seed)
      : output_mgr_(new CUDAOutputMgr()), seed_(seed) {
  }
  // Transfer pointer ownership.
  CUDAProgramGenerator(unsigned long seed, OutputMgr *output_mgr)
      : output_mgr_(output_mgr), seed_(seed) {
  }

  // Inherited from AbsProgramGenerator. Creates the random program.
  void goGenerator();

  // Inherited from AbsProgramGenerator. Would ideally return const OutputMgr&,
  // but this is inherited from a pure virtual function, we also want to accept
  // other managers as a parameter.
  OutputMgr *getOutputMgr();

  // Inherited from AbsProgramGenerator. ?
  std::string get_count_prefix(const std::string& name);
  
  // Returns information regarding threads and groups that the generated program
  // decided upon
  static const unsigned int get_threads(void);
  static const unsigned int get_groups(void); 
  static const unsigned int get_total_threads(void);
  static const unsigned int get_threads_per_group(void);
  static const std::vector<unsigned int>& get_global_dims(void);
  static const std::vector<unsigned int>& get_local_dims(void);
  static const unsigned int get_atomic_blocks_no(void);

  // Inherited from AbsProgramGenerator.This one doesn't generally do much, as
  // we assume that the initialise method of the csmith program generators has
  // been called.
  void initialize();

 private:
  std::unique_ptr<OutputMgr> output_mgr_;
  unsigned long seed_;
  
  // To be called at the beginning of the program generation; sets the 
  // runtime parameters of the program, such as number of groups or threads
  // the program should be running with
  void InitRuntimeParameters(void);
  
  // Used as a helper function for calculating the dimensions of the global
  // work size; gets a list of the divisors of the given argument
  void get_divisors(unsigned int val, std::vector<unsigned int> *divisors);

  DISALLOW_COPY_AND_ASSIGN(CUDAProgramGenerator);
};

}  // namespace CUDASmith

#endif  // _CUDASMITH_CLPROGRAMGENERATOR_H_
