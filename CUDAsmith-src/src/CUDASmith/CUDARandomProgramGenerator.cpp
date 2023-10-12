// Entry point to the program.

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "AbsProgramGenerator.h"
#include "CGOptions.h"
#include "CUDASmith/CUDAOptions.h"
#include "CUDASmith/CUDAOutputMgr.h"
#include "CUDASmith/CUDAProgramGenerator.h"
#include "platform.h"

extern bool g_Tgoff;
extern bool g_FCBoff;

// Generator seed.
// static unsigned long g_Seed = 0;
extern unsigned long g_Seed;
bool CheckArgExists(int idx, int argc) {
  if (idx >= argc) std::cout << "Expected another argument" << std::endl;
  return idx < argc;
}

bool ParseIntArg(const char *arg, unsigned long *value) {
  bool res = sscanf(arg, "%lu", value);
  if (!res) std::cout << "Expected integer arg for " << arg << std::endl;
  return res;
}

int main(int argc, char **argv) {
  g_Seed = platform_gen_seed();
  CGOptions::set_default_settings();
  CUDASmith::CUDAOptions::set_default_settings();
  std::string output_filename = "";

  // Parse command line arguments.
  for (int idx = 1; idx < argc; ++idx) {
    if (!strcmp(argv[idx], "--seed") ||
        !strcmp(argv[idx], "-s")) {
      ++idx;
      if (!CheckArgExists(idx, argc)) return -1;
      if (!ParseIntArg(argv[idx], &g_Seed)) return -1;
      continue;
    }

    if (!strcmp(argv[idx], "--atomic_reductions")) {
      CUDASmith::CUDAOptions::atomic_reductions(true);
      continue;
    }

    if (!strcmp(argv[idx], "--atomics")) {
      CUDASmith::CUDAOptions::atomics(true);
      continue;
    }

    if (!strcmp(argv[idx], "--barriers")) {
      CUDASmith::CUDAOptions::barriers(true);
      continue;
    }

    if (!strcmp(argv[idx], "--divergence")) {
      CUDASmith::CUDAOptions::divergence(true);
      continue;
    }

    if (!strcmp(argv[idx], "--embedded")) {
      CUDASmith::CUDAOptions::embedded(true);
      continue;
    }

    if (!strcmp(argv[idx], "--emi")) {
        CUDASmith::CUDAOptions::emi(true);
        long unsigned int fcb_off = 0;
        idx++;
        if (!CheckArgExists(idx, argc)) return -1;
        if (!ParseIntArg(argv[idx], &fcb_off)) return -1;
        if (fcb_off > 1 || fcb_off < 0) {
            std::cout << "Invalid FCB argument, accept 1 or 0" << std::endl;
            return -1;
        }
        g_FCBoff = !(fcb_off == 1);
        continue;
    }

    if (!strcmp(argv[idx], "--emi_p_compound")) {
      ++idx;
      if (!CheckArgExists(idx, argc)) return -1;
      unsigned long value;
      if (!ParseIntArg(argv[idx], &value)) return -1;
      CUDASmith::CUDAOptions::emi_p_compound(value);
      continue;
    }

    if (!strcmp(argv[idx], "--emi_p_leaf")) {
      ++idx;
      if (!CheckArgExists(idx, argc)) return -1;
      unsigned long value;
      if (!ParseIntArg(argv[idx], &value)) return -1;
      CUDASmith::CUDAOptions::emi_p_leaf(value);
      continue;
    }

    if (!strcmp(argv[idx], "--emi_p_lift")) {
      ++idx;
      if (!CheckArgExists(idx, argc)) return -1;
      unsigned long value;
      if (!ParseIntArg(argv[idx], &value)) return -1;
      CUDASmith::CUDAOptions::emi_p_lift(value);
      continue;
    }

    if (!strcmp(argv[idx], "--fake_divergence")) {
      CUDASmith::CUDAOptions::fake_divergence(true);
      continue;
    }

    if (!strcmp(argv[idx], "--group_divergence")) {
      CUDASmith::CUDAOptions::group_divergence(true);
      continue;
    }

    if (!strcmp(argv[idx], "--inter_thread_comm")) {
      CUDASmith::CUDAOptions::inter_thread_comm(true);
      continue;
    }

    if (!strcmp(argv[idx], "--message_passing")) {
      CUDASmith::CUDAOptions::message_passing(true);
      continue;
    }

    if (!strcmp(argv[idx], "--output_file") ||
        !strcmp(argv[idx], "-o")) {
      ++idx;
      if (!CheckArgExists(idx, argc)) return -1;
      CUDASmith::CUDAOptions::output(argv[idx]);
      continue;
    }

    if (!strcmp(argv[idx], "--no-safe_math")) {
      CUDASmith::CUDAOptions::safe_math(false);
      continue;
    }

    if (!strcmp(argv[idx], "--small")) {
      CUDASmith::CUDAOptions::small(true);
      continue;
    }

    if (!strcmp(argv[idx], "--track_divergence")) {
      CUDASmith::CUDAOptions::track_divergence(true);
      continue;
    }

    if (!strcmp(argv[idx], "--vectors")) {
      CUDASmith::CUDAOptions::vectors(true);
      continue;
    }
     if (!strcmp(argv[idx], "--TG")) {
      CUDASmith::CUDAOptions::TG(true);
      long unsigned int tg_off = 0;
      idx++;
      if (!CheckArgExists(idx, argc)) return -1;
      if (!ParseIntArg(argv[idx], &tg_off)) return -1;
      if (tg_off > 1 || tg_off < 0) {
        std::cout << "Invalid TG argument, accept 1 or 0" << std::endl;
        return -1;
      }
      g_Tgoff = !(tg_off == 1);
      continue;
    }

    std::cout << "Invalid option \"" << argv[idx] << '"' << std::endl;
    return -1;
  }
  // End parsing.

  // Resolve any options in CGOptions that must change as a result of options
  // that the user has set.
  CUDASmith::CUDAOptions::ResolveCGOptions();
  // Check for conflicting options
  if (CUDASmith::CUDAOptions::Conflict()) return -1;

  // AbsProgramGenerator does other initialisation stuff, besides itself. So we
  // call it, disregarding the returned object. Still need to delete it.
  AbsProgramGenerator *generator =
      AbsProgramGenerator::CreateInstance(argc, argv, g_Seed);
  if (!generator) {
    cout << "error: can't create AbsProgramGenerator. csmith init failed!"
         << std::endl;
    return -1;
  }

  // Now create our program generator for OpenCL.
  CUDASmith::CUDAProgramGenerator cl_generator(g_Seed);
  cl_generator.goGenerator();

  // Calls Finalization::doFinalization(), which deletes everything, so must be
  // called after program generation.
  delete generator;

  return 0;
}
