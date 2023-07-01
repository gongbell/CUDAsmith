
#ifndef RANDOM_RUNTIME_H
#define RANDOM_RUNTIME_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_35_atomic_functions.h"
#include "cuda.h"
#include <stdio.h>
typedef  unsigned char uchar;
#define N 1024
//#include <inttype.h>
#define int64_t long
#define uint64_t long
#define int_least64_t long
#define uint_least64_t long 
#define int_fast64_t long
#define uint_fast64_t long
#define intmax_t long
#define uintmax_t long
#define int32_t int
#define uint32_t int
#define int16_t short
#define uint16_t short
#define int8_t char
#define uint8_t char
#define uint int
//typedef ushort unsigned short;
//typedef uint unsigned int;
//typedef ulong unsigned long;
/*
#define ulong unsigned long
#define uint unsigned int
#define ushort unsigned short
#define uchar unsigned char
*/
#define INT64_MIN LONG_MIN
#define INT64_MAX LONG_MAX
#define INT32_MIN INT_MIN
#define INT32_MAX INT_MAX
#define INT16_MIN SHRT_MIN
#define INT16_MAX SHRT_MAX
#define INT8_MIN CHAR_MIN
#define INT8_MAX CHAR_MAX
#define UINT64_MIN ULONG_MIN
#define UINT64_MAX ULONG_MAX
#define UINT32_MIN UINT_MIN
#define UINT32_MAX UINT_MAX
#define UINT16_MIN USHRT_MIN
#define UINT16_MAX USHRT_MAX
#define UINT8_MIN UCHAR_MIN
#define UINT8_MAX UCHAR_MAX

#define transparent_crc(X, Y, Z) transparent_crc_(&crc64_context, X, Y, Z)

#define VECTOR(X , Y) VECTOR_(X, Y)
#define VECTOR_(X, Y) X##Y
#define VECTOR_MAKE(X , Y) VECTOR_MAKE_(X, Y)
#define VECTOR_MAKE_(X, Y) make_##X##Y


#include "cl_safe_math_macros.h"
#include "safe_math_macros.h"

#ifdef NO_ATOMICS
#define atomic_inc(x) -1
#define atomic_add(x,y) (1+1)
#define atomic_sub(x,y) (1+1)
#define atomic_min(x,y) (1+1)
#define atomic_max(x,y) (1+1)
#define atomic_and(x,y) (1+1)
#define atomic_or(x,y)  (1+1)
#define atomic_xor(x,y) (1+1)
#define atomic_noop() /* for sanity checking */
#endif
__device__ unsigned int myAtomicInc(volatile unsigned int *address)
{
      return atomicAdd((unsigned int*)address,1);
}
__device__ int myAtomicInc(volatile int *address)
{
      return  atomicAdd((int*)address,1);
}
__device__ unsigned int myAtomicDec(volatile unsigned int *address)
{
       return  atomicSub((unsigned int*)address,1);
}

__device__ int myAtomicDec(volatile int *address)
{
       return  atomicSub((int*)address,1);
}

__device__ int myAtomicMin(volatile int *address,int val)
{
        return atomicMin((int*)address,val);
         
}
__device__ unsigned int myAtomicMin(volatile unsigned int *address,unsigned int val)
{
        return atomicMin((unsigned int*)address,val);
}
__device__ int myAtomicMax(volatile int *address,int val)
{
        return atomicMax((int*)address,val);
}

__device__ unsigned int myAtomicMax(volatile unsigned int *address,unsigned int val)
{
        return atomicMax((unsigned int*)address,val);
}
__device__ int myAtomicAdd(volatile int *address,int val)
{
        return atomicAdd((int*)address,val);
}
__device__ unsigned int myAtomicAdd(volatile unsigned int *address,unsigned int val)
{
        return atomicAdd((unsigned int *)address,val);
}

__device__ int myAtomicSub(volatile int *address,int val)
{
        return atomicSub((int*)address,val);
}
__device__ unsigned int myAtomicSub(volatile unsigned int *address,unsigned int val)
{
        return atomicSub((unsigned int*)address,val);
}
__device__ int myAtomicExch(volatile int *address,int val)
{
        return atomicExch((unsigned int*)address,val);
}
__device__ int myAtomicAnd(volatile int *address,int val)
{
        return atomicAnd((int *)address,val);
}
__device__ unsigned int myAtomicAnd(volatile unsigned int *address,unsigned int val)
{
        return atomicAnd((unsigned int*)address,val);
}

__device__ int myAtomicOr(volatile int *address,int val)
{
        return atomicOr((int*)address,val);
}
__device__ unsigned int myAtomicOr(volatile unsigned int *address,unsigned int val)
{
        return atomicOr((unsigned int*)address,val);
}
__device__ int myAtomicXor(volatile int *address,int val)
{
        return atomicXor((int *)address,val);
}
__device__ unsigned int myAtomicXor(volatile unsigned int *address,unsigned int val)
{
        return atomicXor((unsigned int*)address,val);
}
/*

__device__ unsigned int myAtomicInc(volatile unsigned int *address)
{
	unsigned int* address_as_ull = (unsigned int *)address;
	unsigned int old = *address_as_ull,assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull,assumed,old+1);
	
	}while(assumed != old);
	return old;
}
__device__ int myAtomicInc(volatile int *address)
{
	int* address_as_ull = (int *)address;
	int old = *address_as_ull,assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull,assumed,old+1);
	
	}while(assumed != old);
	return old;
}
__device__ unsigned int myAtomicDec(volatile unsigned int *address)
{
	unsigned int* address_as_ull = (unsigned int *)address;
	unsigned int old = *address_as_ull,assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull,assumed,old-1);
	
	}while(assumed != old);
	return old;
}
__device__ int myAtomicDec(volatile int *address)
{
	int* address_as_ull = (int *)address;
	int old = *address_as_ull,assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull,assumed,old-1);
	
	}while(assumed != old);
	return old;
}
__device__ int myAtomicMin(volatile int *address,int val)
{
	int* address_ = (int *)address;
	return atomicMin(address_,val);
	
}
__device__ unsigned int myAtomicMin(volatile unsigned int *address,unsigned int val)
{
	unsigned int* address_ = (unsigned int *)address;
	return atomicMin(address_,val);
	
}
__device__ int myAtomicMax(volatile int *address,int val)
{
	int* address_ = (int *)address;
	return atomicMax(address_,val);
}
__device__ unsigned int myAtomicMax(volatile unsigned int *address,unsigned int val)
{
	unsigned int* address_ = (unsigned int *)address;
	return atomicMax(address_,val);
}
__device__ int myAtomicAdd(volatile int *address,int val)
{
	int* address_ = (int *)address;
	return atomicAdd(address_,val);
}
__device__ unsigned int myAtomicAdd(volatile unsigned int *address,unsigned int val)
{
	unsigned int* address_ = (unsigned int *)address;
	return atomicAdd(address_,val);
}

__device__ int myAtomicSub(volatile int *address,int val)
{
	int* address_ = (int *)address;
	return atomicSub(address_,val);
}
__device__ unsigned int myAtomicSub(volatile unsigned int *address,unsigned int val)
{
	unsigned int* address_ = (unsigned int *)address;
	return atomicSub(address_,val);
}
__device__ int myAtomicExch(volatile int *address,int val)
{
	int* address_ = (int *)address;
	return atomicExch(address_,val);
}
__device__ unsigned int myAtomicExch(volatile unsigned int *address,unsigned int val)
{
	unsigned int* address_ = (unsigned int *)address;
	return atomicExch(address_,val);
}
__device__ int myAtomicAnd(volatile int *address,int val)
{
	int* address_ = (int *)address;
	return atomicAnd(address_,val);
}
__device__ unsigned int myAtomicAnd(volatile unsigned int *address,unsigned int val)
{
	unsigned int* address_ = (unsigned int *)address;
	return atomicAnd(address_,val);
}

__device__ int myAtomicOr(volatile int *address,int val)
{
	int* address_ = (int *)address;
	return atomicOr(address_,val);
}
__device__ unsigned int myAtomicOr(volatile unsigned int *address,unsigned int val)
{
	unsigned int* address_ = (unsigned int *)address;
	return atomicOr(address_,val);
}
__device__ int myAtomicXor(volatile int *address,int val)
{
	int* address_ = (int *)address;
	return atomicXor(address_,val);
}
__device__ unsigned int myAtomicXor(volatile unsigned int *address,unsigned int val)
{
	unsigned int* address_ = (unsigned int *)address;
	return atomicXor(address_,val);
}

*/
__device__ int get_block_id(int a)
{
	switch (a)
	{
	case (0) : return blockIdx.x; break;
	case (1) : return blockIdx.y; break;
	case (2) : return blockIdx.z; break;
	}
}

__device__ inline void transparent_crc_no_string (uint64_t *crc64_context, uint64_t val)
{
  *crc64_context += val;
}

#define transparent_crc_(A, B, C, D) transparent_crc_no_string(A, B)
__device__ inline uint32_t
get_global_size(int index){
      if (index==0)
                 return gridDim.x*blockDim.x;
        else if (index==1)
                   return gridDim.y*blockDim.y;
          else if (index==2)
                     return gridDim.z*blockDim.z;
}
__device__ inline uint32_t
get_global_id(int index){
      if (index==0)
                 return blockIdx.x*blockDim.x+threadIdx.x;
        else if (index==1)
                   return blockIdx.y*blockDim.y+threadIdx.y;
          else if (index==2)
                     return blockIdx.z*blockDim.z+threadIdx.z;
}
__device__ inline uint32_t
get_local_size(int index){
      if (index==0)
                 return blockDim.x;
        else if (index==1)
                   return blockDim.y;
          else if (index==2)
                     return blockDim.z;
}
__device__ inline uint32_t
get_local_id (int index)
{
      if (index==0)
                  return threadIdx.x;
        else if (index==1)
                   return threadIdx.y;
          else if (index==2)
                     return threadIdx.z;
}

__device__  inline uint32_t
get_num_groups(int index){
      if (index==0)
                  return gridDim.x;
        else if (index==1)
                   return gridDim.y;
          else if (index==2)
                     return gridDim.z;
}
__device__ inline uint32_t
get_group_id(int index){
      if (index==0)
                  return blockIdx.x;
        else if (index==1)
                   return blockIdx.y;
          else if (index==2)
                     return blockIdx.z;
}
__device__ inline uint32_t
get_linear_group_id (void)
{
          return (get_group_id(2) * get_num_groups(1) + get_group_id(1)) *
                            get_num_groups(0) + get_group_id(0);
}

__device__ inline uint32_t
get_linear_global_id (void)
{
          return (get_global_id(2) * get_global_size(1) + get_global_id(1)) *
                            get_global_size(0) + get_global_id(0);
}

__device__ inline uint32_t
get_linear_local_id (void)
{
          return (get_local_id(2) * get_local_size(1) + get_local_id(1)) *
                            get_local_size(0) + get_local_id(0);
}
/*
__device__ inline uint32_t
get_linear_group_id (void)
{
  return ((blockIdx.z * gridDim.y + blockIdx.y) * 
    gridDim.x + blockIdx.x);
}

__device__ inline uint32_t 
get_linear_global_id (void)
{
  return ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*(blockDim.x*blockDim.y*blockDim.z) + ((threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x); 
//  return ((blockIdx.z*blockDim.z + threadIdx.z) * gridDim.y*blockDim.y + (blockIdx.y*blockDim.y + threadIdx.y) *
  //      gridDim.x*blockDim.x + (blockIdx.x*blockDim.x + threadIdx.x));
}

__device__ inline uint32_t
get_linear_local_id (void)
{
  return ((threadIdx.z * blockDim.y + threadIdx.y) * 
    blockDim.x + threadIdx.x);
}
*/
#endif /* RANDOM_RUNTIME_H */
