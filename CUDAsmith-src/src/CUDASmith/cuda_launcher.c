#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <stdbool.h>
#include <builtin_types.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#define checkCuErrors(err) __checkCuErrors (err,__FILE__,__LINE__)
#define checkCudaErrors(err) __checkCudaErrors (err,__FILE__,__LINE__)
#define uint unsigned int 
inline void __checkCuErrors(CUresult err, const char *file,const int line)
{
	if(CUDA_SUCCESS != err){
		fprintf(stderr,
				"CUDA Driver API error = %04d from file <%s>, line %i.\n",
				err,file,line);
		exit(-1);
	}
}
inline void __checkCudaErrors(cudaError_t err, const char *file,const int line)
{
	if(CUDA_SUCCESS != err){
		fprintf(stderr,
				"CUDA Runtime API error = %04d from file <%s>, line %i.\n",
				err,file,line);
		cudaGetErrorString(cudaGetLastError());
		exit(EXIT_FAILURE);
	}
}

#ifdef EMBEDDED
  typedef unsigned int  RES_TYPE;
#else
  typedef unsigned long RES_TYPE;
#endif

#define DEF_LOCAL_SIZE 32
#define DEF_GLOBAL_SIZE 1024
#define REQ_ARG_COUNT 1

//user input
const char *file;
const char *cubinFile;
const char *args_file = NULL;
size_t binary_size = 0;
char* include_path = ".";
bool debug_build = false;
bool disable_opts = true;
bool disable_fake = false;
bool disable_atomics = false;
bool output_binary = false; 
bool set_device_from_name = false;


// Kernel parameters.
bool atomics = false;
int atomic_counter_no = 0;
bool atomic_reductions = false;
bool emi = false;
bool fake_divergence = false;
bool inter_thread_comm = false;

void *args[6];
// Data to free.
char *source_text = NULL;
char *buf = NULL;
RES_TYPE * h_init_result = NULL;
unsigned int *h_init_atomic_vals = NULL;
unsigned int *h_init_special_vals = NULL;
int *global_reduction_target = NULL;
size_t *local_size = NULL;
size_t *global_size = NULL;
int grid_dim[3] = {1,1,1};//给定的每个维度grid包含的block数=global_size/local_size
char* local_dims = "";
char* global_dims = "";
int *h_sequence_input = NULL;
long *h_comm_vals = NULL;

CUdeviceptr d_init_atomic_vals, d_init_special_vals;
CUdeviceptr d_init_result;
CUdeviceptr atomic_reduction_vals;
CUdeviceptr emi_input;
CUdeviceptr d_sequence_input;
CUdeviceptr d_comm_vals;
// Other parameters
int total_threads = 1;
int no_blocks = 1;
int l_dim = 1;
int g_dim = 1;

// --- global variables ----------------------------------------------------
CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;
size_t     totalGlobalMem; 
size_t 	   freeGlobalMem;

char       *kernel_name = (char*) "entry";

int parse_arg(char* arg, char* val);
int parse_file_args(const char* filename);
int device_index = 0;
// --- functions -----------------------------------------------------------
void print_help() {
  printf("Usage: ./cuda_launcher -f <cl_program>  -d <device_idx> [flags...]\n");
  printf("\n");
  printf("Required flags are:\n");
  printf("  -f FILE --filename FILE                   Test file\n");
  printf("  -f2 FILE --filename2 FILE                 cubin file\n");
  printf("  -p IDX  --platform_idx IDX                Target platform\n");
  printf("  -d IDX  --device_idx IDX                  Target device\n");
  printf("\n");
  printf("Optional flags are:\n");
  printf("  -i PATH --include_path PATH               Include path for kernels (. by default)\n"); //FGG
  printf("  -b N    --binary N                        Compiles the kernel to binary, allocating N bytes\n");
  printf("  -l N    --locals N                        A string with comma-separated values representing the number of work-units per group per dimension\n");
  printf("  -g N    --groups N                        Same as -l, but representing the total number of work-units per dimension\n");
  printf("  -a FILE --args FILE                       Look for file-defined arguments in this file, rather than the test file\n");
  printf("          --atomics                         Test uses atomic sections\n");
  printf("                      ---atomic_reductions  Test uses atomic reductions\n");
  printf("                      ---emi                Test uses EMI\n");
  printf("                      ---fake_divergence    Test uses fake divergence\n");
  printf("                      ---inter_thread_comm  Test uses inter-thread communication\n");
  printf("                      ---debug              Print debug info\n");
  printf("                      ---bin                Output disassembly of kernel in out.bin\n");
  printf("                      ---disable_opts       Disable OpenCL compile optimisations\n");
  printf("                      ---disable_fake       Disable fake divergence feature\n");
  printf("                      ---disable_atomics    Disable atomic sections and reductions\n");
  printf("                      ---set_device_from_name\n");
  printf("                                            Ignore target platform -p and device -d\n");
  printf("                                            Instead try to find a matching platform/device based on the device name\n");
}


void initCUDA()
{
	int deviceCount = 0;
	CUresult err = cuInit(0);
	int major = 0, minor = 0;
//	int deviceNum[10];
	if(err == CUDA_SUCCESS)
		checkCuErrors(cuDeviceGetCount(&deviceCount));
	
	if(deviceCount == 0){
		fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
	}
/*	else{
		for(int i = 0; i < deviceCount; i++)
			deviceNum[i] = i;
	}
*/	
	// get first CUDA device
	checkCuErrors(cuDeviceGet(&device, 0));
//	checkCudaErrors(cudaGetDevice(deviceNum[0]));
	
	
	// get compute capabilities and the devicename
	char name[100];
	cuDeviceGetName(name,100,device);
	printf("> Using device 0: %s\n", name);
	checkCuErrors(cuDeviceComputeCapability(&major, &minor, device));
	printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
	
	checkCuErrors( cuDeviceTotalMem(&totalGlobalMem, device) );
//	checkCudaErrors(cudaMemGetInfo(&freeGlobalMem,&totalGlobalMem));
    printf("  Total amount of global memory:   %llu bytes\n",
           (unsigned long long)totalGlobalMem);
    printf("  64-bit Memory Address:           %s\n",
           (totalGlobalMem > (unsigned long long)4*1024*1024*1024L)?
           "YES" : "NO");
	
	err = cuCtxCreate(&context, 0, device);
	if(err != CUDA_SUCCESS){
		fprintf(stderr, "* Error initializing the CUDA context.\n");
		cuCtxDetach(context);
		exit(-1);
	}
	
	err = cuModuleLoad(&module, cubinFile);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error loading the module %s\n", cubinFile);
        cuCtxDetach(context);
        exit(-1);
    }
	
	err = cuModuleGetFunction(&function, module, kernel_name);
	if(err != CUDA_SUCCESS){
		fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
		cuCtxDetach(context);
        exit(-1);
	}

}
void finalizeCUDA()
{
	cuCtxDetach(context);
}

void setupMemory()
{
	h_init_result = (RES_TYPE*)malloc(sizeof(RES_TYPE) * total_threads);
	int counter;
	for(counter = 0; counter < total_threads; counter++)
		h_init_result[counter] = 0;
	
	checkCuErrors(cuMemAlloc(&d_init_result,sizeof(RES_TYPE) * total_threads));
	checkCuErrors(cuMemcpyHtoD(d_init_result,h_init_result,sizeof(RES_TYPE)*total_threads));
	int kernel_arg = 0;
	args[kernel_arg++] = &d_init_result;
	if(atomics){
		// Create buffer to store counters for the atomic blocks
		int total_counters = atomic_counter_no * no_blocks;
		h_init_atomic_vals = (uint*)malloc(sizeof(uint)*total_counters);
		h_init_special_vals = (uint*)malloc(sizeof(uint)*total_counters);
		
		//init
		int i;
		for(i = 0; i < total_counters; i++){
			h_init_atomic_vals[i] = 0;
			h_init_special_vals[i] = 0;
		}
		
		
		checkCuErrors(cuMemAlloc(&d_init_atomic_vals,sizeof(uint)*total_counters));
		checkCuErrors(cuMemAlloc(&d_init_special_vals,sizeof(uint)*total_counters));
		checkCuErrors(cuMemcpyHtoD(d_init_atomic_vals,h_init_atomic_vals,sizeof(uint)*total_counters));
		checkCuErrors(cuMemcpyHtoD(d_init_special_vals,h_init_special_vals,sizeof(uint)*total_counters));
		args[kernel_arg++] = &d_init_atomic_vals;
		args[kernel_arg++] = &d_init_special_vals;
	}
	
	if(atomic_reductions){
		global_reduction_target = (int*)malloc(sizeof(int)*no_blocks);
		int i;
		for(i = 0; i < no_blocks; i++)
			global_reduction_target[i] = 0;
		
		checkCuErrors(cuMemAlloc(&atomic_reduction_vals,sizeof(int)*no_blocks));
		checkCuErrors(cuMemcpyHtoD(atomic_reduction_vals,global_reduction_target,sizeof(int)*no_blocks));
		args[kernel_arg++] = &atomic_reduction_vals;
	}
	
	if(emi){
		// Create input buffer for EMI.
		int emi_values[1024];
		int i;
		for(i = 0; i < 1024; ++i) emi_values[i] = 1024 - i;
		checkCuErrors(cuMemcpyHtoD(emi_input,emi_values,sizeof(int)*1024));
		args[kernel_arg++] = &emi_input;
	}
	
	if(fake_divergence){
		int max_dimen = global_size[0];
		int i;
		for(i = 1; i < g_dim; i++)
			if(global_size[i] > max_dimen) max_dimen = global_size[i];
		h_sequence_input = (int *)malloc(sizeof(int) * max_dimen);
		for(i = 0; i < max_dimen; i++) h_sequence_input[i] = 10 + i;
		
		checkCuErrors(cuMemAlloc(&d_sequence_input,sizeof(int) * max_dimen));
		checkCuErrors(cuMemcpyHtoD(d_sequence_input,h_sequence_input,sizeof(int) * max_dimen));
		args[kernel_arg++] = &d_sequence_input;
	}
	
	if(inter_thread_comm){
		// Create input for inter thread communication.
		h_comm_vals = (long*)malloc(sizeof(long) * total_threads);
		int i;
		for(i = 0; i < total_threads; ++i) h_comm_vals[i] = 1;
		
		checkCuErrors(cuMemAlloc(&d_comm_vals,sizeof(long) * total_threads));
		checkCuErrors(cuMemcpyHtoD(d_comm_vals,h_comm_vals,sizeof(long) * total_threads));
		args[kernel_arg++] = &d_comm_vals;
	}
}

void releaseMemory()
{
	free(global_size);
	free(local_size);
	if(atomics){
		free(h_init_atomic_vals);
		free(h_init_special_vals);
		checkCuErrors(cuMemFree(d_init_atomic_vals));
		checkCuErrors(cuMemFree(d_init_special_vals));
	}
	if(atomic_reductions){
		free(global_reduction_target);
		checkCuErrors(cuMemFree(atomic_reduction_vals));
	}
	if(fake_divergence){
		free(h_sequence_input);
		checkCuErrors(cuMemFree(d_sequence_input));
	}
	if(inter_thread_comm){
		free(h_comm_vals);
		checkCuErrors(cuMemFree(d_comm_vals));
	}
}

void runKernel()
{
	//initCUDA();
	checkCuErrors(cuLaunchKernel(function, grid_dim[0], grid_dim[1], grid_dim[2],
								 local_size[0], local_size[1], local_size[2],
								 0, 0, args, 0));
}

int main(int argc, char **argv)
{
	// Parse the input. Expect two parameters.(--device_idx,--filename)
	if(argc < 4){
		printf("Expected at least three arguments\n");
		print_help();
		return 1;
	}
	
	int req_arg = 0;
	
	// Parsing arguments
	int arg_no = 0;
	int parse_ret;
	char* curr_arg;
	char* next_arg;
	while (++arg_no < argc) {
		curr_arg = argv[arg_no];
		if (!strcmp(curr_arg, "-h") || !strcmp(curr_arg, "--help")) {
			print_help();
			exit(0);
		}
		if (!strcmp(curr_arg, "-f") || !strcmp(curr_arg, "--filename")) {
			file = argv[++arg_no];
		}
		if (!strcmp(curr_arg, "-f2") || !strcmp(curr_arg, "--filename2")) {
			cubinFile = argv[++arg_no];
		}
		if (!strcmp(curr_arg, "-a") || !strcmp(curr_arg, "--args")) {
			args_file = argv[++arg_no];
		}
	}

	if (!file) {
		printf("Require file (-f) argument!\n");
		return 1;
	}

  // Parse arguments found in the given source file
  if (args_file == NULL) {
    if (!parse_file_args(file)) {
      printf("Failed parsing file for arguments.\n");
      return 1;
    }
  }
  // Parse arguments in defined args file
  else {
    if (!parse_file_args(args_file)) {
      printf("Failed parsing given arguments file.\n");
      return 1;
    }
  }

  arg_no = 0;
  while (++arg_no < argc) {
    curr_arg = argv[arg_no];
    if (strncmp(curr_arg, "---", 3)) {
      if (++arg_no >= argc) {
        printf("Found option %s with no value.\n", curr_arg);
        return 1;
      }
      next_arg = argv[arg_no];
    }
    parse_ret = parse_arg(curr_arg, next_arg);
    if (!parse_ret)
      return 1;
    req_arg += parse_ret - 1;
  }

  if (req_arg < REQ_ARG_COUNT) {
    printf("Require device index (-d) arguments, or device name (-n)!\n");
    return 1;
  }	
	
  // TODO function this
  // Parsing thread and group dimension information
  if (strcmp(local_dims, "") == 0) {
    local_size = (size_t*)malloc(sizeof(size_t));
    local_size[0] = DEF_LOCAL_SIZE;
  } else {
    int i = 0;
    while (local_dims[i] != '\0')
      if (local_dims[i++] == ',')
        l_dim++;
    i = 0;
    local_size = (size_t*)malloc(l_dim * sizeof(size_t));
    char* tok = strtok(local_dims, ",");
    while (tok) {
      local_size[i++] = (size_t) atoi(tok);
      tok = strtok(NULL, ",");
    }
  }
  if (strcmp(global_dims, "") == 0) {
    global_size = (size_t*)malloc(sizeof(size_t));
    global_size[0] = DEF_GLOBAL_SIZE;
  } else {
    int i = 0;
    while (global_dims[i] != '\0')
      if (global_dims[i++] == ',')
        g_dim++;
    i = 0;
    global_size = (size_t*)malloc(g_dim * sizeof(size_t));
    char* tok = strtok(global_dims, ",");
    while (tok) {
      global_size[i++] = atoi(tok);
      tok = strtok(NULL, ",");
    }
  }

 
 
 
  if (g_dim != l_dim) {
    printf("Local and global sizes must have same number of dimensions!\n");
    return 1;
  }
  if (l_dim > 3) {
    printf("Cannot have more than 3 dimensions!\n");
    return 1;
  }
  int d;
  for (d = 1; d < l_dim; d++)
    if (local_size[d] > global_size[d]) {
      printf("Local dimension %d greater than global dimension!\n", d);
      return 1;
    }

  // Calculating total number of work-units for future use
  int i;
  for (i = 0; i < l_dim; i++) {
    total_threads *= global_size[i];
    no_blocks *= global_size[i] / local_size[i];
  }	
  
  // Device ID, not used atm.
  if (device_index < 0) {
    printf("Could not parse device id \"%s\"\n", argv[3]);
    return 1;
  }
  initCUDA();
/*   // Find all the devices for the platform.
  int deviceCount = 0;
  int deviceNum[10];
  checkCuErrors(cuDeviceGetCount(&deviceCount));
  
  if (deviceCount == 0) {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }
  else{
	  for( i = 0; i < deviceCount; i++)
		  deviceNum[i] = i;
  } */
/*   if (deviceCount <= device_idx) {
    printf("No device for id %d\n", device_idx);
    return 1;
  } */
  
  int max_dimensions = 3;
  // Checking that number of threads in each dimension per block is OK
  int * max_block_dim_x = (int*)malloc(sizeof(int));
  int * max_block_dim_y = (int*)malloc(sizeof(int));
  int * max_block_dim_z = (int*)malloc(sizeof(int));
  checkCuErrors(cuDeviceGetAttribute(max_block_dim_x,CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,device));
  checkCuErrors(cuDeviceGetAttribute(max_block_dim_y,CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,device));
  checkCuErrors(cuDeviceGetAttribute(max_block_dim_z,CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,device));
  
  int max_block_dim[3] = {*max_block_dim_x,*max_block_dim_y,*max_block_dim_z};
  int curr_dim;
  for(curr_dim = 0;curr_dim < max_dimensions; curr_dim++){
	  if(max_block_dim[curr_dim] < local_size[curr_dim]){
		  printf("Local work size in dimension %d is %d, which exceeds maximum of %d for this device\n", curr_dim, local_size[curr_dim], max_block_dim[curr_dim]);
		  return 1;
	  }
  }
  free(max_block_dim_x);        
  free(max_block_dim_y);
  free(max_block_dim_z);
  
  for(curr_dim = 0; curr_dim < max_dimensions; curr_dim++){
	  printf("global_size[%d] is %d\n",curr_dim,global_size[curr_dim]);
	  printf("local_size[%d] is %d\n",curr_dim,local_size[curr_dim]);
	  grid_dim[curr_dim] = global_size[curr_dim] / local_size[curr_dim];
  }
  // Checking that block size is not too large(检查每个grid包含的block数目不超过最大值)
  int * max_grid_dim_x = (int*)malloc(sizeof(int));
  int * max_grid_dim_y = (int*)malloc(sizeof(int));
  int * max_grid_dim_z = (int*)malloc(sizeof(int));
  checkCuErrors(cuDeviceGetAttribute(max_grid_dim_x,CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,device));
  checkCuErrors(cuDeviceGetAttribute(max_grid_dim_y,CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,device));
  checkCuErrors(cuDeviceGetAttribute(max_grid_dim_z,CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,device));
  
  int max_grid_dim[3] = {*max_grid_dim_x,*max_grid_dim_y,*max_grid_dim_z};
  for(curr_dim = 0;curr_dim < max_dimensions; curr_dim++){
	  if(max_grid_dim[curr_dim] < grid_dim[curr_dim]){
		  printf("grid size in dimension %d is %d, which exceeds maximum of %d for this device\n", curr_dim, global_size[curr_dim]/local_size[curr_dim], max_grid_dim[curr_dim]);
		  return 1;
	  }
  }
  free(max_grid_dim_x);
  free(max_grid_dim_y);
  free(max_grid_dim_z);

  setupMemory();
  runKernel();
  RES_TYPE * c = (RES_TYPE*)malloc(sizeof(RES_TYPE) * total_threads);
  checkCuErrors(cuMemcpyDtoH(c,d_init_result,sizeof(RES_TYPE)*total_threads));
  //freopen("Result.txt","w",stdout);
  char *result;
  result = strtok(file,".");
  freopen(result,"w",stdout);
  for(i = 0; i < total_threads; ++i)
	  printf("%016x\n",c[i]);
  releaseMemory();
  free(c);
  return 0;
}

int parse_file_args(const char* filename){
	
	FILE* source = fopen(filename,"r");
	if(source == NULL){
		printf("Could not open file %s for argument parsing.\n", filename);
		return 0;
	}
	
	char arg_buf[256];
	fgets(arg_buf,250,source);
	char* new_line;
	if((new_line = strchr(arg_buf,'\n')))
		arg_buf[(int) (new_line - arg_buf)] = '\0';
	
	if(!strncmp(arg_buf,"//",2)){
		char* tok = strtok(arg_buf," ");
		while(tok){
			if(!strncmp(tok,"---",3))
				parse_arg(tok,NULL);
			else if(!strncmp(tok,"-",1))
				parse_arg(tok,strtok(NULL," "));
			tok = strtok(NULL," ");
		}
	}
	fclose(source);
	
	return 1;
} 
/* Function used to parse given arguments. All optional arguments must have a
 * return value of 1. The total return value of required arguments must be
 * equal to the value of REQ_ARG_COUNT.
 */
int parse_arg(char* arg, char* val) {
  if (!strcmp(arg, "-f") || !strcmp(arg, "--filename")) {
    return 1;
  }
  if (!strcmp(arg, "-f2") || !strcmp(arg, "--filename2")) {
    return 1;
  }
  if (!strcmp(arg, "-a") || !strcmp(arg, "--args")) {
    return 1;
  }
  if (!strcmp(arg, "-d") || !strcmp(arg, "--device_idx")) {
    device_index = atoi(val);
    return 2;
  }
  if (!strcmp(arg, "-l") || !strcmp(arg, "--locals")) {
    local_dims = (char*)malloc((strlen(val)+1)*sizeof(char));
    strcpy(local_dims, val);
    return 1;
  }
  if (!strcmp(arg, "-g") || !strcmp(arg, "--groups")) {
    global_dims = (char*)malloc((strlen(val)+1)*sizeof(char));
    strcpy(global_dims, val);
    return 1;
  }
  if (!strcmp(arg, "-i") || !strcmp(arg, "--include_path")) {
    int ii;
    include_path = val;
    for (ii=0; ii<strlen(include_path); ii++)
      if (include_path[ii]=='\\') include_path[ii]='/';

    return 1;
  }
  if (!strcmp(arg, "--atomics")) {
    atomics = true;
    atomic_counter_no = atoi(val);
    return 1;
  }
  if (!strcmp(arg, "---atomic_reductions")) {
    atomic_reductions = true;
    return 1;
  }
  if (!strcmp(arg, "---emi")) {
    emi = true;
    return 1;
  }
  if (!strcmp(arg, "---fake_divergence")) {
    fake_divergence = true;
    return 1;
  }
  if (!strcmp(arg, "---inter_thread_comm")) {
    inter_thread_comm = true;
    return 1;
  }
  if (!strcmp(arg, "---debug")) {
    debug_build = true;
    return 1;
  }
  if (!strcmp(arg, "---disable_fake")) {
    disable_fake = true;
    return 1;
  }
  if (!strcmp(arg, "---disable_atomics")) {
    disable_atomics = true;
    return 1;
  }
  printf("Failed parsing arg %s.", arg);
  return 0;
}


