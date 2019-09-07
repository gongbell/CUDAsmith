// Runs a randomly generated program on a device via OpenCL.
// Usage: cl_launcher <cl_program> <platform_id> <device_id> [flags...]

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#if defined(_MSC_VER) || defined(WINDOWS)
#include <windows.h>
#include <rtcapi.h>

#ifdef XOPENME
#include <xopenme.h>
#endif

bool build_in_progress = false;
bool execution_in_progress = false;

int exception_handler(LPEXCEPTION_POINTERS p)
{
    fprintf(stderr, "Exception detected during test execution!\n");
    if(build_in_progress) {
        fprintf(stderr, "Exception occurred during build\n");
    }
    if(execution_in_progress) {
        fprintf(stderr, "Exception occurred during kernel execution\n");
    }
    if(!build_in_progress && !execution_in_progress) {
        fprintf(stderr, "Unknown source of exception\n");
    }
    exit(1);
}
int runtime_check_handler(int errorType, const char *filename, int linenumber, const char *moduleName, const char *format, ...)
{
    fprintf(stderr, "Error type %d at %s line %d in %s", errorType, filename, linenumber, moduleName);
    exit(1);
}
#endif

#ifdef EMBEDDED
  typedef cl_uint RES_TYPE;
#else
  typedef cl_ulong RES_TYPE;
#endif

#define DEF_LOCAL_SIZE 32
#define DEF_GLOBAL_SIZE 1024
#define REQ_ARG_COUNT 2

// User input.
const char *file;
const char *args_file = NULL;
size_t binary_size = 0;
int device_index = 0;
int platform_index = 0;
char* device_name_given = "";
char* include_path = ".";
bool debug_build = false;
bool disable_opts = false;
bool disable_fake = false;
bool disable_group = false;
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

// Data to free.
char *source_text = NULL;
char *buf = NULL;
RES_TYPE * init_result = NULL;
cl_uint *init_atomic_vals = NULL;
cl_uint *init_special_vals = NULL;
cl_int *global_reduction_target = NULL;
size_t *local_size = NULL;
size_t *global_size = NULL;
char* local_dims = "";
char* global_dims = "";
int *sequence_input = NULL;
cl_long *comm_vals = NULL;

// Other parameters
cl_platform_id *platform;
cl_device_id *device;
int total_threads = 1;
int no_groups = 1;
int l_dim = 1;
int g_dim = 1;

char platformName[256];
char deviceName[256];
int compute_units=0;

int run_on_platform_device(cl_platform_id *, cl_device_id *, cl_uint);
void
#ifdef _MSC_VER
  __stdcall
#endif
  error_callback(const char *, const void *, size_t, void *);
int cl_error_check(cl_int, const char *);
int parse_arg(char* arg, char* val);
int parse_file_args(const char* filename);

void print_help() {
  fprintf(stderr, "Usage: ./cl_launcher -f <cl_program> -p <platform_idx> -d <device_idx> [flags...]\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Required flags are:\n");
  fprintf(stderr, "  -f FILE --filename FILE                   Test file\n");
  fprintf(stderr, "  -p IDX  --platform_idx IDX                Target platform\n");
  fprintf(stderr, "  -d IDX  --device_idx IDX                  Target device\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Optional flags are:\n");
  fprintf(stderr, "  -i PATH --include_path PATH               Include path for kernels (. by default)\n"); //FGG
  fprintf(stderr, "  -b N    --binary N                        Compiles the kernel to binary, allocating N bytes\n");
  fprintf(stderr, "  -l N    --locals N                        A string with comma-separated values representing the number of work-units per group per dimension\n");
  fprintf(stderr, "  -g N    --groups N                        Same as -l, but representing the total number of work-units per dimension\n");
  fprintf(stderr, "  -n NAME --name NAME                       Ensure the device name contains this string\n");
  fprintf(stderr, "  -a FILE --args FILE                       Look for file-defined arguments in this file, rather than the test file\n");
  fprintf(stderr, "          --atomics                         Test uses atomic sections\n");
  fprintf(stderr, "                      ---atomic_reductions  Test uses atomic reductions\n");
  fprintf(stderr, "                      ---emi                Test uses EMI\n");
  fprintf(stderr, "                      ---fake_divergence    Test uses fake divergence\n");
  fprintf(stderr, "                      ---inter_thread_comm  Test uses inter-thread communication\n");
  fprintf(stderr, "                      ---debug              Print debug info\n");
  fprintf(stderr, "                      ---bin                Output disassembly of kernel in out.bin\n");
  fprintf(stderr, "                      ---disable_opts       Disable OpenCL compile optimisations\n");
  fprintf(stderr, "                      ---disable_group      Disable group divergence feature\n");
  fprintf(stderr, "                      ---disable_fake       Disable fake divergence feature\n");
  fprintf(stderr, "                      ---disable_atomics    Disable atomic sections and reductions\n");
  fprintf(stderr, "                      ---set_device_from_name\n");
  fprintf(stderr, "                                            Ignore target platform -p and device -d\n");
  fprintf(stderr, "                                            Instead try to find a matching platform/device based on the device name\n");
}

/*
 * Try to set platform-id and device-id based on the device name.
 * Returns bool if successful match is found or not.
 */
bool setPlatformDeviceFromDeviceName() {
  cl_int err;
  cl_uint num_platforms;
  unsigned p, d;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (cl_error_check(err, "clGetPlatformIDs (num platforms) error"))
    exit(1);
  cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
  assert(platforms);
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (cl_error_check(err, "clGetPlatformIDs (platforms) error"))
    exit(1);
  bool match = false;
  for (p=0; !match && p<num_platforms; ++p) {
    cl_platform_id platform = platforms[p];
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (cl_error_check(err, "clGetDeviceIDs (num devices) error"))
      exit(1);
    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    assert(devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (cl_error_check(err, "clGetDeviceIDs (devices) error"))
      exit(1);
    for (d=0; !match && d<num_devices; ++d) {
      cl_device_id device = devices[d];
      char name[65536];
      size_t size;
      err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, &size);
      if (cl_error_check(err, "clGetDeviceInfo (name) error"))
        exit(1);
      assert(size < sizeof(name));
      if (debug_build) {
        fprintf(stderr, "At platform %d and device %d with name [%s]]\n", p, d, name);
      }
      if (strstr(name, device_name_given) != NULL) {
        match = true;
        if ((platform_index != p) || (device_index != d)) {
          if (debug_build) {
            fprintf(stderr, "Set platform %d and device %d to match %s\n", p, d, device_name_given);
          }
          platform_index = p;
          device_index = d;
        }
      }
    }
    free(devices);
  }
  free(platforms);
  return match;
}

int main(int argc, char **argv) {

#ifdef XOPENME
  xopenme_init(1,5);
#endif

#ifdef _MSC_VER
  DWORD dwMode = SetErrorMode(SEM_NOGPFAULTERRORBOX);
  SetErrorMode(dwMode | SEM_NOGPFAULTERRORBOX);
  SetUnhandledExceptionFilter((LPTOP_LEVEL_EXCEPTION_FILTER)&exception_handler);
  _RTC_SetErrorFunc(&runtime_check_handler);
#endif

  // Parse the input. Expect three parameters.
  if (argc < 4) {
    fprintf(stderr, "Expected at least three arguments\n");
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
    if (!strcmp(curr_arg, "-a") || !strcmp(curr_arg, "--args")) {
      args_file = argv[++arg_no];
    }
  }

  if (!file) {
    fprintf(stderr, "Require file (-f) argument!\n");
    return 1;
  }

  // Parse arguments found in the given source file
  if (args_file == NULL) {
    if (!parse_file_args(file)) {
      fprintf(stderr, "Failed parsing file for arguments.\n");
      return 1;
    }
  }
  // Parse arguments in defined args file
  else {
    if (!parse_file_args(args_file)) {
      fprintf(stderr, "Failed parsing given arguments file.\n");
      return 1;
    }
  }

  arg_no = 0;
  while (++arg_no < argc) {
    curr_arg = argv[arg_no];
    if (strncmp(curr_arg, "---", 3)) {
      if (++arg_no >= argc) {
        fprintf(stderr, "Found option %s with no value.\n", curr_arg);
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
    fprintf(stderr, "Require device index (-d) and platform index (-p) arguments, or device name (-n)!\n");
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
  	free(local_dims);
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
  	free(global_dims);
  }

  // print global and local sizes in debug mode
  if (debug_build) {
    int i;
    size_t global_size_total = 1;
    for (i = 0; i < g_dim; i++)
      global_size_total *= global_size[i];

    fprintf(stderr, "%d-D global size %zu = [%zu",
            g_dim, global_size_total, global_size[0]);
    for (i = 1; i < g_dim; i++)
      fprintf(stderr, ", %zu", global_size[i]);
    fprintf(stderr, "]\n");

    size_t local_size_total = 1;
    for (i = 0; i < l_dim; i++)
      local_size_total *= local_size[i];

    fprintf(stderr, "%d-D local size %zu = [%zu",
            l_dim, local_size_total, local_size[0]);
    for (i = 1; i < l_dim; i++)
      fprintf(stderr, ", %zu", local_size[i]);
    fprintf(stderr, "]\n");
  }

  if (g_dim != l_dim) {
    fprintf(stderr, "Local and global sizes must have same number of dimensions!\n");
    return 1;
  }
  if (l_dim > 3) {
    fprintf(stderr, "Cannot have more than 3 dimensions!\n");
    return 1;
  }
  int d;
  for (d = 1; d < l_dim; d++)
    if (local_size[d] > global_size[d]) {
      fprintf(stderr, "Local dimension %d greater than global dimension!\n", d);
      return 1;
    }

  // Calculating total number of work-units for future use
  int i;
  for (i = 0; i < l_dim; i++) {
    total_threads *= global_size[i];
    no_groups *= global_size[i] / local_size[i];
  }

  // Platform ID, the index in the array of platforms.
  if (platform_index < 0) {
    fprintf(stderr, "Could not parse platform id \"%s\"\n", argv[2]);
    return 1;
  }

  // Device ID, not used atm.
  if (device_index < 0) {
    fprintf(stderr, "Could not parse device id \"%s\"\n", argv[3]);
    return 1;
  }

  if (set_device_from_name) {
    if (strcmp(device_name_given, "") == 0) {
      fprintf(stderr, "Must give '-n NAME' to use --set_device_from_name\n");
      return 1;
    }
    if (!setPlatformDeviceFromDeviceName()) {
      fprintf(stderr, "No matching platform or device found for name %s\n", device_name_given);
      return 1;
    }
  }

  // Query the OpenCL API for the given platform ID.
  cl_int err;
  cl_platform_id * platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*(platform_index + 1));
  cl_uint platform_count;
  err = clGetPlatformIDs(platform_index + 1, platforms, &platform_count);
  if (cl_error_check(err, "clGetPlatformIDs error"))
    return 1;
  if (platform_count <= platform_index) {
    fprintf(stderr, "No platform for id %d\n", platform_index);
    return 1;
  }
  platform = &platforms[platform_index];

  if (debug_build) {
    if (disable_opts)
      fprintf(stderr, "OpenCL optimizations: off\n");
    else
      fprintf(stderr, "OpenCL optimizations: on\n");

    err = clGetPlatformInfo(*platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
    if (cl_error_check(err, "Get Platform Info error")) return 1;
    fprintf(stderr, "Platform: %s\n", platformName);
  }

#ifdef XOPENME
  err = clGetPlatformInfo(*platform, CL_PLATFORM_VENDOR, sizeof(platformName), platformName, NULL);
  if (cl_error_check(err, "Get Platform Info error")) return 1;

  xopenme_add_var_s(0, (char*) "  \"opencl_platform\":\"%s\"", platformName);
#endif

  // Find all the devices for the platform.
  cl_device_id * devices = (cl_device_id*)malloc(sizeof(cl_device_id)*(device_index + 1));
  cl_uint device_count;
  err = clGetDeviceIDs(
      *platform, CL_DEVICE_TYPE_ALL, device_index + 1, devices, &device_count);
  if (cl_error_check(err, "clGetDeviceIDs error"))
    return 1;
  if (device_count <= device_index) {
    fprintf(stderr, "No device for id %d\n", device_index);
    return 1;
  }
  device = &devices[device_index];

  // Check to see if name of device corresponds to given name, if any
  if (device_name_given) {
    size_t name_size = 128, actual_size;
    char* device_name_get = (char*)malloc(name_size*sizeof(char));
    err = clGetDeviceInfo(
      *device, CL_DEVICE_NAME, name_size, device_name_get, &actual_size);
    if (cl_error_check(err, "clGetDeviceInfo error"))
      return 1;
    char* device_name = (char*)malloc(actual_size*sizeof(char));
    strncpy(device_name, device_name_get, actual_size);
    if (!strstr(device_name, device_name_given)) {
      fprintf(stderr, "Given name, %s, not found in device name, %s.\n",
          device_name_given, device_name);
      return 1;
    }
  }

  // Checking device supports given number of dimensions
  cl_uint max_dimensions;
  err = clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_dimensions, NULL);
  if (cl_error_check(err, "Error querying CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS"))
    return 1;
  if(max_dimensions < g_dim) {
    fprintf(stderr, "Kernel uses %d dimensions, exceeds the maximum of %d dimensions for this device\n", g_dim, max_dimensions);
    return 1;
  }

  size_t curr_dim;

  // Checking that number of work items in each dimension is OK
  size_t * max_work_items = (size_t*)malloc(sizeof(size_t) * max_dimensions);
  err = clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_dimensions, max_work_items, NULL);
  if (cl_error_check(err, "Error querying CL_DEVICE_MAX_WORK_ITEM_SIZES"))
    return 1;
  for (curr_dim = 0; curr_dim < l_dim; curr_dim++) {
    if(max_work_items[curr_dim] < local_size[curr_dim]) {
      fprintf(stderr, "Local work size in dimension %zd is %zd, which exceeds maximum of %zd for this device\n", curr_dim, global_size[curr_dim], max_work_items[curr_dim]);
      return 1;
    }
  }
  free(max_work_items);

  // Checking that work group size is not too large
  size_t max_work_group_size, given_work_group_size = 1;
  err = clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
  if (cl_error_check(err, "Error querying CL_DEVICE_MAX_WORK_GROUP_SIZE"))
    return 1;
  for (curr_dim = 0; curr_dim < l_dim; curr_dim++)
    given_work_group_size *= local_size[curr_dim];
  if(max_work_group_size < given_work_group_size) {
    fprintf(stderr, "Kernel work group size is %zd, which exceeds the maximum work group size of %zd for this device\n", given_work_group_size, max_work_group_size);
    return 1;
  }

  if (debug_build) {
    err = clGetDeviceInfo(*device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    if (cl_error_check(err, "Get Device Info error")) return 1;
    fprintf(stderr, "Device: %s\n", deviceName);
  }

#ifdef XOPENME
  err = clGetDeviceInfo(*device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
  if (cl_error_check(err, "Get Device Info error")) return 1;

  err = clGetDeviceInfo(*device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
  if (cl_error_check(err, "Get Device compute units error")) return 1;

  xopenme_add_var_s(1, (char*) "  \"opencl_device\":\"%s\"", deviceName);
  xopenme_add_var_i(2, (char*) "  \"opencl_device_units\":%u", compute_units);
#endif


  int run_err = run_on_platform_device(platform, device, (cl_uint) l_dim);
  free(source_text);
  free(buf);
  free(local_size);
  free(global_size);
  free(platforms);
  free(devices);
  free(init_result);

  if (atomics) {
    free(init_atomic_vals);
    free(init_special_vals);
  }
  if (atomic_reductions) {
    free(global_reduction_target);
  }
  if (fake_divergence) {
    free(sequence_input);
  }
  if (inter_thread_comm) {
    free(comm_vals);
  }

#ifdef XOPENME
  xopenme_dump_state();
  xopenme_finish();
#endif

  return run_err;
}

int run_on_platform_device(cl_platform_id *platform, cl_device_id *device, cl_uint work_dim) {

  // Try to read source file into a binary buffer
  FILE *source = fopen(file, "rb");
  if (source == NULL) {
    fprintf(stderr, "Could not open %s.\n", file);
    return 1;
  }

  size_t source_size;
  if (!binary_size) {
    char temp[1024];
    while (!feof(source)) fread(temp, 1, 1024, source);
    source_size = ftell(source);
    rewind(source);

    source_text = (char*)calloc(1, source_size + 1);
    if (source_text == NULL) {
      fprintf(stderr, "Failed to calloc %ld bytes.\n", source_size);
      return 1;
    }
    fread(source_text, 1, source_size, source);
    fclose(source);
  }
  else {
    buf = (char *) malloc (binary_size);
    source_size = fread(buf, 1, binary_size, source);
    fclose(source);
  }

  // Create a context, that uses our specified platform and device.
  cl_int err;
  cl_context_properties properties[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)*platform, 0 };
  cl_context context =
      clCreateContext(properties, 1, device, error_callback, NULL, &err);
  if (cl_error_check(err, "Error creating context"))
    return 1;

  // Create a command queue for the device in the context just created.
  // CHANGE when cl 2.0 is released.
  //cl_command_queue com_queue =
  //    clCreateCommandQueueWithProperties(context, *device, NULL, &err);
  cl_command_queue com_queue = clCreateCommandQueue(context, *device, 0, &err);
  if (cl_error_check(err, "Error creating command queue"))
    return 1;

  // Create a kernel from the source program. This involves turning the source
  // into a program object, compiling it and creating a kernel object from it.
  cl_program program;
  if (!binary_size) {
    const char *const_source = source_text;
    program =
      clCreateProgramWithSource(context, 1, &const_source, NULL, &err);
  }
  else {
    program =
        clCreateProgramWithBinary(context, 1, device, (const size_t *)&source_size,
                                  (const unsigned char **)&buf, NULL, &err);
  }
  if (cl_error_check(err, "Error creating program"))
    return 1;

  // Add optimisation to options later.
  char* options = (char*)malloc(sizeof(char)*256);
  sprintf(options, "-w -I%s", include_path);
  if (disable_opts)
    sprintf(options, "%s -cl-opt-disable", options);
  if (disable_group)
    sprintf(options, "%s -D NO_GROUP_DIVERGENCE", options);
  if (disable_fake)
    sprintf(options, "%s -D NO_FAKE_DIVERGENCE", options);
  if (disable_atomics)
    sprintf(options, "%s -D NO_ATOMICS", options);

#ifdef _MSC_VER
  build_in_progress = true;
#endif
  err = clBuildProgram(program, 0, NULL, options, NULL, NULL);

#ifdef XOPENME
  xopenme_add_var_s(3, (char*) "  \"opencl_options\":\"%s\"", options);
#endif

#ifdef _MSC_VER
  build_in_progress = false;
#endif
  if (cl_error_check(err, "Error building program")) {
    if (debug_build) {
      size_t err_size;
      err = clGetProgramBuildInfo(
          program, *device, CL_PROGRAM_BUILD_LOG, 0, NULL, &err_size);
      if (cl_error_check(err, "Error getting build info"))
        return 1;
      char *err_code = (char*)malloc(err_size);
      if (err_code == NULL) {
        fprintf(stderr, "Failed to malloc %ld bytes\n", err_size);
        return 1;
      }
      err = clGetProgramBuildInfo(
          program, *device, CL_PROGRAM_BUILD_LOG, err_size, err_code, &err_size);
      if (!cl_error_check(err, "Error getting build info"))
        fprintf(stderr, "%s", err_code);
      free(err_code);
    }
    return 1;
  }
  free(options);

  fprintf(stderr, "Compilation terminated successfully...\n");
  fflush(stdout);

  cl_build_status status;
  err = clGetProgramBuildInfo(
      program, *device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);
  if (cl_error_check(err, "Error getting build info"))
    return 1;

  // If specified, output the binary.
  if (output_binary) {
    size_t bin_size;
    err = clGetProgramInfo(
        program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, NULL);
    if (cl_error_check(err, "Error getting binary info"))
      return 1;
    unsigned char *bin = (unsigned char*)malloc(bin_size);
    if (bin == NULL) {
      fprintf(stderr, "Failed to malloc %ld bytes\n", bin_size);
      return 1;
    }
    err = clGetProgramInfo(
        program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);
    if (cl_error_check(err, "Error getting binary"))
      return 1;
    FILE *bin_out = fopen("out.bin", "wb");
    if (bin_out == NULL) {
      fprintf(stderr, "Could not open output file \"out.bin\"");
      return 1;
    }
    fwrite(bin, sizeof(unsigned char *), bin_size, bin_out);
    fclose(bin_out);
    free(bin);
    return 0;
  }

  // Create the kernel
  cl_kernel kernel = clCreateKernel(program, "entry", &err);
  if (cl_error_check(err, "Error creating kernel"))
    return 1;

  // Create the buffer that will have the results.
  init_result = (RES_TYPE*)malloc(sizeof(RES_TYPE) * total_threads);
  int counter;
  for (counter = 0; counter < total_threads; counter++)
    init_result[counter] = 0;
  cl_mem result = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, total_threads * sizeof(RES_TYPE), init_result, &err);
  if (cl_error_check(err, "Error creating output buffer"))
    return 1;

  // Set the buffers as arguments
  unsigned kernel_arg = 0;
  err = clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &result);
  if (cl_error_check(err, "Error setting kernel argument 0"))
    return 1;

  if (atomics) {
    // Create buffer to store counters for the atomic blocks
    int total_counters = atomic_counter_no * no_groups;
    init_atomic_vals = (cl_uint*)malloc(sizeof(cl_uint) * total_counters);
    init_special_vals = (cl_uint*)malloc(sizeof(cl_uint) * total_counters);

    // Fill the created buffers in host memory
    int i;
    for (i = 0; i < total_counters; i++) {
      init_atomic_vals[i] = 0;
      init_special_vals[i] = 0;
    }

    cl_mem atomic_input = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total_counters * sizeof(cl_uint),
        init_atomic_vals, &err);
    if (cl_error_check(err, "Error creating atomic input buffer"))
      return 1;

    // Create buffer to store special values for the atomic blocks
    cl_mem special_values = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total_counters * sizeof(cl_uint),
        init_special_vals, &err);
    if (cl_error_check(err, "Error creating special values input buffer"))
      return 1;

    err = clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &atomic_input);
    if (cl_error_check(err, "Error setting atomic input array argument"))
      return 1;
    err = clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &special_values);
    if (cl_error_check(err, "Error setting special values array argument"))
      return 1;
  }

  if (atomic_reductions) {
    global_reduction_target = (cl_int*)malloc(sizeof(cl_int) * no_groups);
    int i;
    for (i = 0; i < no_groups; i++)
      global_reduction_target[i] = 0;

    cl_mem atomic_reduction_vars = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, no_groups * sizeof(cl_int),
        global_reduction_target, &err);
    if (cl_error_check(err, "Error creating atomic reduction variable input buffer"))
      return 1;
    err = clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &atomic_reduction_vars);
    if (cl_error_check(err, "Error setting atomic reduction input argument"))
      return 1;
  }

  if (emi) {
    // Create input buffer for EMI.
    int emi_values[1024];
    int i;
    for (i = 0; i < 1024; ++i) emi_values[i] = 1024 - i;
    cl_mem emi_input = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 1024 * sizeof(cl_int), &emi_values, &err);
    if (cl_error_check(err, "Error creating emi buffer"))
      return 1;
    err = clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &emi_input);
    if (cl_error_check(err, "Error setting kernel argument for emi"))
      return 1;
  }

  if (fake_divergence) {
    // Create input for fake divergence.
    size_t max_dimen = global_size[0];
    int i;
    for (i = 1; i < g_dim; ++i)
      if (global_size[i] > max_dimen) max_dimen = global_size[i];
    sequence_input = (int *)malloc(sizeof(int) * max_dimen);
    for (i = 0; i < max_dimen; ++i) sequence_input[i] = 10 + i;
    cl_mem seq_input = clCreateBuffer(
        context, CL_MEM_READ_ONLY, max_dimen * sizeof(cl_int), NULL, &err);
    if (cl_error_check(err, "Error creating fake divergence buffer"))
      return 1;

    err = clEnqueueWriteBuffer(com_queue, seq_input, CL_TRUE, 0, max_dimen * sizeof(cl_int), sequence_input, 0, NULL, NULL);
    if (cl_error_check(err, "Error copying input to fake divergence buffer"))
      return 1;

    err = clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &seq_input);
    if (cl_error_check(err, "Error setting kernel argument for fake divergence"))
      return 1;
  }

  if (inter_thread_comm) {
    // Create input for inter thread communication.
    comm_vals = (cl_long*)malloc(sizeof(cl_long) * total_threads);
    int i;
    for (i = 0; i < total_threads; ++i) comm_vals[i] = 1;
    cl_mem inter_thread = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total_threads * sizeof(cl_long), comm_vals, &err);
    if (cl_error_check(err, "Error creating fake inter thread comm buffer"))
      return 1;
    err = clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &inter_thread);
    if (cl_error_check(err, "Error setting kernel argument for inter thread comm"))
      return 1;
  }


  // Create command to launch the kernel.
#ifdef _MSC_VER
  execution_in_progress = true;
#endif
  err = clEnqueueNDRangeKernel(
      com_queue, kernel, work_dim, NULL, global_size, local_size, 0, NULL, NULL);
  if (cl_error_check(err, "Error enqueueing kernel"))
    return 1;

  // Send a finish command, forcing the kernel to run.
  err = clFinish(com_queue);
  if (cl_error_check(err, "Error sending finish command"))
    return 1;
#ifdef _MSC_VER
  execution_in_progress = false;
#endif

  // Read back the reults of each thread.
  RES_TYPE * c = (RES_TYPE*)malloc(sizeof(RES_TYPE)*total_threads);
  err = clEnqueueReadBuffer(
      com_queue, result, CL_TRUE, 0, total_threads * sizeof(RES_TYPE), c, 0, NULL, NULL);
  if (cl_error_check(err, "Error reading output buffer"))
    return 1;

  ////
  int i;
  for (i = 0; i < total_threads; ++i)
    printf(
#ifdef _MSC_VER
    "%I64x,"
#elif EMBEDDED
    "%#"PRIx32","
#else
    "%#"PRIx64","
#endif
    , c[i]);
////

  free(c);

  return 0;
}

int parse_file_args(const char* filename) {

  FILE* source = fopen(filename, "r");
  if (source == NULL) {
    fprintf(stderr, "Could not open file %s for argument parsing.\n", filename);
    return 0;
  }

  char arg_buf[128];
  fgets(arg_buf, 128, source);
  char* new_line;
  if ((new_line = strchr(arg_buf, '\n')))
    arg_buf[(int) (new_line - arg_buf)] = '\0';

  if (!strncmp(arg_buf, "//", 2)) {
    char* tok = strtok(arg_buf, " ");
    while (tok) {
      if (!strncmp(tok, "---", 3))
        parse_arg(tok, NULL);
      else if (!strncmp(tok, "-", 1))
        parse_arg(tok, strtok(NULL, " "));
      tok = strtok(NULL, " ");
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
  if (!strcmp(arg, "-a") || !strcmp(arg, "--args")) {
    return 1;
  }
  if (!strcmp(arg, "-d") || !strcmp(arg, "--device_idx")) {
    device_index = atoi(val);
    return 2;
  }
  if (!strcmp(arg, "-p") || !strcmp(arg, "--platform_idx")) {
    platform_index = atoi(val);
    return 2;
  }
  if (!strcmp(arg, "-b") || !strcmp(arg, "--binary")) {
    binary_size = atoi(val);
    return 1;
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
  if (!strcmp(arg, "-n") || !strcmp(arg, "--name")) {
    device_name_given = val;
    return 3;
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
  if (!strcmp(arg, "---set_device_from_name")) {
    set_device_from_name = true;
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
  if (!strcmp(arg, "---bin")) {
    output_binary = true;
    return 1;
  }
  if (!strcmp(arg, "---disable_opts")) {
    disable_opts = true;
    return 1;
  }
  if (!strcmp(arg, "---disable_fake")) {
    disable_fake = true;
    return 1;
  }
  if (!strcmp(arg, "---disable_group")) {
    disable_group = true;
    return 1;
  }
  if (!strcmp(arg, "---disable_atomics")) {
    disable_atomics = true;
    return 1;
  }
  fprintf(stderr, "Failed parsing arg %s.", arg);
  return 0;
}

// Called if any error occurs during context creation or at kernel runtime.
// This can be called many time asynchronously, so it must be thread safe.
void
#ifdef _MSC_VER
__stdcall
#endif
error_callback(
    const char *errinfo, const void *private_info, size_t cb, void *user_data) {
  fprintf(stderr, "Error found (callback):\n%s\n", errinfo);
}

// Error checker, useful to help remove some of the boiler plate.
// Return 0 on success, 1 on error.
int cl_error_check(cl_int err, const char *err_string) {
  if (err == CL_SUCCESS)
    return 0;
  fprintf(stderr, "%s: %d\n", err_string, err);
  return 1;
}
