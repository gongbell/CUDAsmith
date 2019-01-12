# CUDAsmith
A CUDA compiler fuzzer

Using the generator
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

for tg mode , we choose the ALL mode to inject the true guard.
Uses ‘--fake_divergence --group_divergence --vectors --inter_thread_comm -- atomics --atomic_reductions –tg 1’ to generate the tg cases,
Uses ‘--fake_divergence --group_divergence --vectors --inter_thread_comm -- atomics --atomic_reductions –tg 0’ to generate the tg_off cases, these cases are the base cases which have no true guard.


-	fg mode

for fg mode, we alse choose the ALL mode to construct false block
Uses ‘--fake_divergence --group_divergence --vectors --inter_thread_comm --atomics --atomic_reductions --emi’ to generate fg cases
