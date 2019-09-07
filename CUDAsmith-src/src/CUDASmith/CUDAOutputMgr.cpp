#include "CUDASmith/CUDAOutputMgr.h"

#include <cstdio>
#include <fstream>
#include <sstream>

#include "CUDASmith/CUDAOptions.h"
#include "CUDASmith/CUDAProgramGenerator.h"
#include "CUDASmith/ExpressionAtomic.h"
#include "CUDASmith/ExpressionID.h"
#include "CUDASmith/Globals.h"
#include "CUDASmith/StatementBarrier.h"
#include "CUDASmith/StatementComm.h"
#include "CUDASmith/StatementMessage.h"
#include "Function.h"
#include "OutputMgr.h"
#include "Type.h"
#include "VariableSelector.h"

namespace CUDASmith
{
int atomic_ID, g_ID[3], l_ID[3];
CUDAOutputMgr::CUDAOutputMgr() : out_(CUDAOptions::output())
{
}

void CUDAOutputMgr::OutputRuntimeInfo(
    const std::vector<unsigned int> &global_dims,
    const std::vector<unsigned int> &local_dims)
{
    std::ostream &out = get_main_out();
    out << "//";
    if (CUDAOptions::atomics())
    {
	out << " --atomics " << CUDAProgramGenerator::get_atomic_blocks_no();
	atomic_ID = CUDAProgramGenerator::get_atomic_blocks_no();
    }
    if (CUDAOptions::atomic_reductions())
	out << " ---atomic_reductions";
    if (CUDAOptions::fake_divergence())
	out << " ---fake_divergence";
    if (CUDAOptions::inter_thread_comm())
	out << " ---inter_thread_comm";
    if (CUDAOptions::emi())
	out << " ---emi";
	//add by wxy 2018-03-20
	if (CUDAOptions::TG())
	out << " ---tg";
	//end
    int i = 0;
    out << " -g ";
    for (std::vector<unsigned int>::const_iterator it = global_dims.begin();
	 it < global_dims.end(); it++)
    {
	out << *it;
	g_ID[i++] = *it;
	if (it + 1 != global_dims.end())
	    out << ",";
    }
    i = 0;
    out << " -l ";
    for (std::vector<unsigned int>::const_iterator it = local_dims.begin();
	 it < local_dims.end(); it++)
    {
	out << *it;
	l_ID[i++] = *it;
	if (it + 1 != local_dims.end())
	    out << ",";
    }
    out << std::endl;
}

void CUDAOutputMgr::OutputHeader(int argc, char *argv[], unsigned long seed)
{
    // Redefine platform independent scalar C types to platform independent scalar
    // OpenCL types.
    std::ostream &out = get_main_out();
    OutputRuntimeInfo(CUDAProgramGenerator::get_global_dims(),
		      CUDAProgramGenerator::get_local_dims());
    //Guai 20160905 Start
    /*
		out <<
		"#define int64_t long\n"
		"#define uint64_t ulong\n"
		"#define int_least64_t long\n"
		"#define uint_least64_t ulong\n"
		"#define int_fast64_t long\n"
		"#define uint_fast64_t ulong\n"
		"#define intmax_t long\n"
		"#define uintmax_t ulong\n"
		"#define int32_t int\n"
		"#define uint32_t uint\n"
		"#define int16_t short\n"
		"#define uint16_t ushort\n"
		"#define int8_t char\n"
		"#define uint8_t uchar\n"
		"\n"
		"#define INT64_MIN LONG_MIN\n"
		"#define INT64_MAX LONG_MAX\n"
		"#define INT32_MIN INT_MIN\n"
		"#define INT32_MAX INT_MAX\n"
		"#define INT16_MIN SHRT_MIN\n"
		"#define INT16_MAX SHRT_MAX\n"
		"#define INT8_MIN CHAR_MIN\n"
		"#define INT8_MAX CHAR_MAX\n"
		"#define UINT64_MIN ULONG_MIN\n"
		"#define UINT64_MAX ULONG_MAX\n"
		"#define UINT32_MIN UINT_MIN\n"
		"#define UINT32_MAX UINT_MAX\n"
		"#define UINT16_MIN USHRT_MIN\n"
		"#define UINT16_MAX USHRT_MAX\n"
		"#define UINT8_MIN UCHAR_MIN\n"
		"#define UINT8_MAX UCHAR_MAX\n"
		"\n"
		"#define transparent_crc(X, Y, Z) "
		"transparent_crc_(&crc64_context, X, Y, Z)\n"
		"\n"
		"#define VECTOR(X , Y) VECTOR_(X, Y)\n"
		"#define VECTOR_(X, Y) X##Y\n"
		<< std::endl;
		*/
    //Guai 20160905 End

    // Macro for expanding GROUP_DIVERGE
    ExpressionIDGroupDiverge::OutputGroupDivergenceMacro(out);
    out << std::endl;
    // Macro for expanding FAKE_DIVERGE.
    ExpressionIDFakeDiverge::OutputFakeDivergenceMacro(out);
    out << std::endl;

    out << std::endl;
    out << "// Seed: " << seed << std::endl;
    out << std::endl;
    //Guai 20160905 Start
    out << "#include \"CUDA.h\"" << std::endl;
    out << std::endl;
    /*ifstream fin0("/home/wxy/jc/main_0.txt");
    if (!fin0)
	out << "NO";
    const int LINE_LENGTH0 = 200;
    char str0[LINE_LENGTH0];
    while (fin0.getline(str0, LINE_LENGTH0))
    {
	out << str0 << std::endl;
    }*/
    //Guai 20160905 End
    // Permuation buffers for inter-thread comm.

    if (CUDAOptions::inter_thread_comm())
	StatementComm::OutputPermutations(out);
}

void CUDAOutputMgr::Output()
{
    std::ostream &out = get_main_out();
    OutputStructUnionDeclarations(out);
    //printf("CUDAOutputMgr::Output:___LINE___");
    // Type of message_t, for message passing.
    if (CUDAOptions::message_passing())
	MessagePassing::OutputMessageType(out);

    Globals *globals = Globals::GetGlobals();
    //globals->OutputStructDefinition(out);
    globals->OutputStructDefinition(out);
    globals->ModifyGlobalVariableReferences();
    globals->AddGlobalStructToAllFunctions();

    OutputForwardDeclarations(out);
    OutputFunctions(out);
    OutputEntryFunction(*globals);
}

std::ostream &CUDAOutputMgr::get_main_out()
{
    return out_;
}

void CUDAOutputMgr::OutputEntryFunction(Globals &globals)
{
    // Would ideally use the ExtensionMgr, but there is no way to set it to our
    // own custom made one (without modifying the code).
    std::ostream &out = get_main_out();
    //Guai 20160901 Begin
    out << "extern \"C\" __global__ void entry( long *result";
    //Guai 20160901 End
    //out << "__kernel void entry(__global ulong *result";
    if (CUDAOptions::atomics())
    {
	out << ",  volatile uint *g_atomic_input";
	out << ",  volatile uint *g_special_values";
    }
    if (CUDAOptions::atomic_reductions())
    {
	out << ",  volatile int *g_atomic_reduction";
    }
	//add by wxy 2018-03-20
	if (CUDAOptions::TG())
	out << ",  int *tg_input";
	//end
    if (CUDAOptions::emi())
	out << ",  int *emi_input";
    if (CUDAOptions::fake_divergence())
	//Guai 20160901 Begin
	out << " , int *sequence_input";
    //Guai 20160901 End
    //out << ", __global int *sequence_input";
    if (CUDAOptions::inter_thread_comm())
	out << ",  long *g_comm_values";
    out << ") {" << std::endl;
    globals.OutputArrayControlVars(out);
    globals.OutputBufferInits(out);
    //globals.OutputBufferInitsForEntry(out);
    //globals.OutputBufferInits(out);
    globals.OutputStructInit(out);

    // Block all threads before entering to ensure the local buffers have been
    // initialised.
    output_tab(out, 1);
    StatementBarrier::OutputBarrier(out);
    out << std::endl;

    output_tab(out, 1);
    out << "func_1(";
    globals.GetGlobalStructVar().Output(out);
    out << ");" << std::endl;
    // If using message passing, check and update constraints to prevent deadlock.
    if (CUDAOptions::message_passing())
	MessagePassing::OutputMessageEndChecks(out);

    // Block all threads after, to prevent hashing stale values.
    output_tab(out, 1);
    StatementBarrier::OutputBarrier(out);
    out << std::endl;

    // Handle hashing and outputting.
    output_tab(out, 1);
    out << "uint64_t crc64_context = 0xFFFFFFFFFFFFFFFFUL;" << std::endl;
    output_tab(out, 1);
    out << "int print_hash_value = 0;" << std::endl;
    HashGlobalVariables(out);
    if (CUDAOptions::atomics())
	ExpressionAtomic::OutputHashing(out);
    if (CUDAOptions::inter_thread_comm())
	StatementComm::HashCommValues(out);
    output_tab(out, 1);
    //Guai 20160901 Begin
    //out << "result[(blockIdx.z*blockDim.z + threadIdx.z) * gridDim.y*blockDim.y + (blockIdx.y*blockDim.y + threadIdx.y) * gridDim.x*blockDim.x + (blockIdx.x*blockDim.x + threadIdx.x)] = crc64_context ^ 0xFFFFFFFFFFFFFFFFUL;"
    //   << std::endl;
    //Guai 20160901 End
    //Guai 20160905 Start
    out << "   result[get_linear_global_id()] = crc64_context ^ 0xFFFFFFFFFFFFFFFFUL;" << std::endl;
    out << "}" << std::endl;
} // namespace CUDASmith
}
