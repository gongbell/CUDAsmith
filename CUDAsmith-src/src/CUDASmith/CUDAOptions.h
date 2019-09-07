// Options for CUDASmith. Like CGOptions, but specific to OpenCL C generation.

#ifndef _CUDASMITH_CLOPTIONS_H_
#define _CUDASMITH_CLOPTIONS_H_

namespace CUDASmith {

// Encapsulates all the options used throughout the program such that they can
// easily be accessed anywhere.
class CUDAOptions {
 public:
  CUDAOptions() = delete;
//private: static type name##_;\
  // Eww macro, used here just to make the flags easier to write.
  #define DEFINE_CUDAFLAG(name, type) \
    private: static type name##_; \
    public: \
    static type name(); \
    static void name(type x);
  DEFINE_CUDAFLAG(atomic_reductions, bool)
  DEFINE_CUDAFLAG(atomics, bool)
  DEFINE_CUDAFLAG(barriers, bool)
  DEFINE_CUDAFLAG(divergence, bool)
  DEFINE_CUDAFLAG(embedded, bool)
  DEFINE_CUDAFLAG(emi, bool)
  DEFINE_CUDAFLAG(emi_p_compound, int)
  DEFINE_CUDAFLAG(emi_p_leaf, int)
  DEFINE_CUDAFLAG(emi_p_lift, int)
  DEFINE_CUDAFLAG(fake_divergence, bool)
  DEFINE_CUDAFLAG(group_divergence, bool)
  DEFINE_CUDAFLAG(inter_thread_comm, bool)
  DEFINE_CUDAFLAG(message_passing, bool)
  DEFINE_CUDAFLAG(output, const char*)
  DEFINE_CUDAFLAG(safe_math, bool)
  DEFINE_CUDAFLAG(small, bool)
  DEFINE_CUDAFLAG(track_divergence, bool)
  DEFINE_CUDAFLAG(vectors, bool)
  //add by wxy 2018-03-15
  DEFINE_CUDAFLAG(TG, bool)
  DEFINE_CUDAFLAG(tg_p_leaf, int)
  DEFINE_CUDAFLAG(tg_p_compound, int)
  DEFINE_CUDAFLAG(tg_p_lift, int)   
  //end 
  #undef DEFINE_CUDAFLAG

  // Reset any option changes, even those specified by the user.
  static void set_default_settings();

  // Automagically sets flags in CGOptions to prevent producing invalid OpenCL C
  // programs.
  static void ResolveCGOptions();

  // Checks for conflicts in user arguments.
  static bool Conflict();
};

}  // namespace CUDASmith

#endif  // _CUDASMITH_CLOPTIONS_H_
