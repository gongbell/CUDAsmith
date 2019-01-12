#!/bin/bash

echo "CUDA Kernel Generation in All Mode:"
for i in $(seq 1 100)
do
	./CUDASmith --seed $i --fake_divergence --group_divergence --vectors --inter_thread_comm --atomics --atomic_reductions

	mv CUDAProg.cu $i.cu
	echo "$i.cu generation succeed..."
done
