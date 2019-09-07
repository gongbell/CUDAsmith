# CUDAsmith
A CUDA compiler fuzzer

The current generator of CUDA kernel is built and tested for Ubuntu 16.04.2. 


To Build the CUDASmith generator from source code, please do it as follows at the terminal:

1. run ‘cd CUDAsmith-src’

2. run ‘mkdir build’

3. run ‘cd build’

4. In build dir , run ‘cmake ..’

5. Then run “make” 

And CUDAsmith will be generated.


The genkernel.sh is a script to generate 100 CUDA kernel in ALL mode.

The generator has many features for generating interesting CUDA programs. The generator is used as follows:
./CUDASmith [--seed <seed>] [flags]
  
There are six modes. The following explains the flags every mode needs when generate the cases.

- BASIC mode: Uses ‘--fake_divergence’ and ‘--group_divergence’ flags

- VECTOR mode: Uses ‘--fake_divergence  --group_divergence --vectors’

- BARRIER	mode: Uses ‘--fake_divergence --group_divergence --vectors --inter_thread_comm’

- ATOMIC mode: Uses ‘--fake_divergence –group_divergence --vectors --atomics’

- ATOMIC REDUCTION mode: Uses ‘--fake_divergence --group_divergence --vectors --atomic_reductons’

- ALL mode: Uses ‘ --fake_divergence --group_divergence --vectors --inter_thread_comm --atomics –atomic_reductions’


Generating the EMI case

-	tg mode

For tg mode , we choose the ALL mode to inject the true guard.

Uses ‘--fake_divergence --group_divergence --vectors --inter_thread_comm -- atomics --atomic_reductions –tg 1’ to generate the tg cases,

Uses ‘--fake_divergence --group_divergence --vectors --inter_thread_comm -- atomics --atomic_reductions –tg 0’ to generate the tg_off cases, these cases are the base cases which have no true guard.


-	fg mode

For fg mode, we alse choose the ALL mode to construct false block

Uses ‘--fake_divergence --group_divergence --vectors --inter_thread_comm --atomics --atomic_reductions --emi’ to generate fg cases
