cmake_minimum_required(VERSION 2.8.12)

project(CUDASmith)

include(CheckIncludeFile)

set(CMAKE_BUILD_TYPE Debug )
set(CLSmith_PACKAGE "CLSmith")
set(CLSmith_PACKAGE_BUGREPORT "mail@example.org")
set(CLSmith_PACKAGE_NAME "CLSmith")
set(CLSmith_PACKAGE_STRING "CLSmith x.y.z")
set(CLSmith_PACKAGE_TARNAME "CLSmith")
set(CLSmith_PACKAGE_URL "http://multicore.doc.ic.ac.uk/tools/CLsmith/index.php")
set(CLSmith_PACKAGE_VERSION "x.y.z")
set(CLSmith_VERSION "x.y.z")

execute_process(COMMAND "git" "show" "-s" "--format=%h" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_VARIABLE CLSmith_GIT_HASH)
string(STRIP "${CLSmith_GIT_HASH}" CLSmith_GIT_HASH)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-long-long -Wall -std=c++11")
endif()

SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_definitions(-DGIT_VERSION=\"${CLSmith_GIT_HASH}\")
add_definitions(-DTARGET_CPU_${CMAKE_SYSTEM_PROCESSOR}=1)

add_definitions(-DPACKAGE_NAME=\"${CLSmith_PACKAGE_NAME}\")
add_definitions(-DPACKAGE_TARNAME=\"${CLSmith_PACKAGE_TARNAME}\")
add_definitions(-DPACKAGE_VERSION=\"${CLSmith_PACKAGE_VERSION}\")
add_definitions(-DPACKAGE_STRING=\"${CLSmith_PACKAGE_STRING}\")
add_definitions(-DPACKAGE_BUGREPORT=\"${CLSmith_PACKAGE_BUGREPORT}\")
add_definitions(-DPACKAGE_URL=\"${CLSmith_PACKAGE_URL}\")
add_definitions(-DPACKAGE=\"${CLSmith_PACKAGE}\")
add_definitions(-DVERSION=\"${CLSmith_VERSION}\")

check_include_file("dlfcn.h" HAVE_DLFCN_H)
check_include_file("inttypes.h" HAVE_INTTYPES_H)
check_include_file("memory.h" HAVE_MEMORY_H)
check_include_file("stdint.h" HAVE_STDINT_H)
check_include_file("stdlib.h" HAVE_STDLIB_H)
check_include_file("strings.h" HAVE_STRINGS_H)
check_include_file("string.h" HAVE_STRING_H)
check_include_file("sys/stat.h" HAVE_SYS_STAT_H)
check_include_file("sys/types.h" HAVE_SYS_TYPES_H)
check_include_file("unistd.h" HAVE_UNISTD_H)

if(WIN32)
    set(CLSmith_WINDOWS_SOURCES
        runtime/windows/rand48.h
        runtime/windows/_rand48.c
        runtime/windows/lrand48.c
        runtime/windows/srand48.c
    )
endif()

include_directories(src)

add_executable(CUDASmith
    #CSmith files
    ${CLSmith_WINDOWS_SOURCES}
    src/AbsExtension.cpp
    src/AbsExtension.h
    src/AbsProgramGenerator.cpp
    src/AbsProgramGenerator.h
    src/AbsRndNumGenerator.cpp
    src/AbsRndNumGenerator.h
    src/ArrayVariable.cpp
    src/ArrayVariable.h
    src/Block.cpp
    src/Block.h
    src/Bookkeeper.cpp
    src/Bookkeeper.h
    src/CFGEdge.cpp
    src/CFGEdge.h
    src/CGContext.cpp
    src/CGContext.h
    src/CGOptions.cpp
    src/CGOptions.h
    src/CVQualifiers.cpp
    src/CVQualifiers.h
    src/Common.h
    src/CommonMacros.h
    src/CompatibleChecker.cpp
    src/CompatibleChecker.h
    src/Constant.cpp
    src/Constant.h
    src/CoverageTestExtension.cpp
    src/CoverageTestExtension.h
    src/CrestExtension.cpp
    src/CrestExtension.h
    src/DFSOutputMgr.cpp
    src/DFSOutputMgr.h
    src/DFSProgramGenerator.cpp
    src/DFSProgramGenerator.h
    src/DFSRndNumGenerator.cpp
    src/DFSRndNumGenerator.h
    src/DefaultOutputMgr.cpp
    src/DefaultOutputMgr.h
    src/DefaultProgramGenerator.cpp
    src/DefaultProgramGenerator.h
    src/DefaultRndNumGenerator.cpp
    src/DefaultRndNumGenerator.h
    src/DeltaMonitor.cpp
    src/DeltaMonitor.h
    src/DepthSpec.cpp
    src/DepthSpec.h
    src/Effect.cpp
    src/Effect.h
    src/Enumerator.h
    src/Error.cpp
    src/Error.h
    src/Expression.cpp
    src/Expression.h
    src/ExpressionAssign.cpp
    src/ExpressionAssign.h
    src/ExpressionComma.cpp
    src/ExpressionComma.h
    src/ExpressionFuncall.cpp
    src/ExpressionFuncall.h
    src/ExpressionVariable.cpp
    src/ExpressionVariable.h
    src/ExtensionMgr.cpp
    src/ExtensionMgr.h
    src/ExtensionValue.cpp
    src/ExtensionValue.h
    src/Fact.cpp
    src/Fact.h
    src/FactMgr.cpp
    src/FactMgr.h
    src/FactPointTo.cpp
    src/FactPointTo.h
    src/FactUnion.cpp
    src/FactUnion.h
    src/Filter.cpp
    src/Filter.h
    src/Finalization.cpp
    src/Finalization.h
    src/Function.cpp
    src/Function.h
    src/FunctionInvocation.cpp
    src/FunctionInvocation.h
    src/FunctionInvocationBinary.cpp
    src/FunctionInvocationBinary.h
    src/FunctionInvocationUnary.cpp
    src/FunctionInvocationUnary.h
    src/FunctionInvocationUser.cpp
    src/FunctionInvocationUser.h
    src/KleeExtension.cpp
    src/KleeExtension.h
    src/Lhs.cpp
    src/Lhs.h
    src/LinearSequence.cpp
    src/LinearSequence.h
    src/MspFilters.cpp
    src/MspFilters.h
    src/OutputMgr.cpp
    src/OutputMgr.h
    src/PartialExpander.cpp
    src/PartialExpander.h
    src/Probabilities.cpp
    src/Probabilities.h
    src/ProbabilityTable.h
    src/RandomNumber.cpp
    src/RandomNumber.h
    #src/RandomProgramGenerator.cpp
    src/Reducer.cpp
    src/Reducer.h
    src/ReducerOutputMgr.cpp
    src/ReducerOutputMgr.h
    src/SafeOpFlags.cpp
    src/SafeOpFlags.h
    src/Sequence.cpp
    src/Sequence.h
    src/SequenceFactory.cpp
    src/SequenceFactory.h
    src/SequenceLineParser.h
    src/SimpleDeltaRndNumGenerator.cpp
    src/SimpleDeltaRndNumGenerator.h
    src/SimpleDeltaSequence.cpp
    src/SimpleDeltaSequence.h
    src/SplatExtension.cpp
    src/SplatExtension.h
    src/Statement.cpp
    src/Statement.h
    src/StatementArrayOp.cpp
    src/StatementArrayOp.h
    src/StatementAssign.cpp
    src/StatementAssign.h
    src/StatementBreak.cpp
    src/StatementBreak.h
    src/StatementContinue.cpp
    src/StatementContinue.h
    src/StatementExpr.cpp
    src/StatementExpr.h
    src/StatementFor.cpp
    src/StatementFor.h
    src/StatementGoto.cpp
    src/StatementGoto.h
    src/StatementIf.cpp
    src/StatementIf.h
    src/StatementReturn.cpp
    src/StatementReturn.h
    src/StringUtils.cpp
    src/StringUtils.h
    src/Type.cpp
    src/Type.h
    src/Variable.cpp
    src/Variable.h
    src/VariableSelector.cpp
    src/VariableSelector.h
    src/VectorFilter.cpp
    src/VectorFilter.h
    src/platform.cpp
    src/platform.h
    src/random.cpp
    src/random.h
    src/util.cpp
    src/util.h
    #CUDASmith files
        src/CUDASmith/CUDAOutputMgr.cpp
        src/CUDASmith/CUDAOutputMgr.h
        src/CUDASmith/CUDAProgramGenerator.cpp
        src/CUDASmith/CUDAProgramGenerator.h
        src/CUDASmith/Globals.cpp
        src/CUDASmith/Globals.h
        src/CUDASmith/CUDARandomProgramGenerator.cpp
        src/CUDASmith/Walker.cpp
        src/CUDASmith/Walker.h
        src/CUDASmith/Divergence.cpp
        src/CUDASmith/Divergence.h
        src/CUDASmith/CUDAExpression.cpp
        src/CUDASmith/CUDAExpression.h
        src/CUDASmith/CUDAStatement.cpp
        src/CUDASmith/CUDAStatement.h
        src/CUDASmith/CUDAVariable.cpp
        src/CUDASmith/CUDAVariable.h
        src/CUDASmith/StatementBarrier.cpp
        src/CUDASmith/StatementBarrier.h
        src/CUDASmith/MemoryBuffer.cpp
        src/CUDASmith/MemoryBuffer.h
        src/CUDASmith/Vector.cpp
        src/CUDASmith/Vector.h
        src/CUDASmith/CUDAOptions.cpp
        src/CUDASmith/CUDAOptions.h
        src/CUDASmith/ExpressionVector.cpp
        src/CUDASmith/ExpressionVector.h
        src/CUDASmith/ExpressionAtomic.cpp
        src/CUDASmith/ExpressionAtomic.h
        src/CUDASmith/StatementEMI.cpp
        src/CUDASmith/StatementEMI.h
        src/CUDASmith/StatementTG.h
        src/CUDASmith/StatementTG.cpp
        src/CUDASmith/StatementAtomicResult.cpp

        src/CUDASmith/StatementAtomicResult.h
        src/CUDASmith/FunctionInvocationBuiltIn.cpp
        src/CUDASmith/FunctionInvocationBuiltIn.h
        src/CUDASmith/ExpressionID.cpp
        src/CUDASmith/ExpressionID.h
        src/CUDASmith/StatementComm.cpp
        src/CUDASmith/StatementComm.h
        src/CUDASmith/StatementAtomicReduction.cpp
        src/CUDASmith/StatementAtomicReduction.h
        src/CUDASmith/StatementMessage.cpp
        src/CUDASmith/StatementMessage.h
)

find_program(M4_EXECUTABLE m4 DOC "The M4 macro processor")

if(M4_EXECUTABLE)
    set(SAFE_MATH_HEADERS
        ${CMAKE_BINARY_DIR}/safe_math_macros.h
        ${CMAKE_BINARY_DIR}/cl_safe_math_macros.h
    )

    add_custom_target(SAFE_MATH_HEADERS ALL
        DEPENDS ${SAFE_MATH_HEADERS}
    )

    add_custom_command(
        OUTPUT safe_math_macros.h
        COMMAND ${M4_EXECUTABLE} ${CMAKE_SOURCE_DIR}/runtime/safe_math_macros.m4 > safe_math_macros.h
        DEPENDS ${CMAKE_SOURCE_DIR}/runtime/safe_math_macros.m4
        VERBATIM
    )

    add_custom_command(
        OUTPUT cl_safe_math_macros.h
        COMMAND ${M4_EXECUTABLE} ${CMAKE_SOURCE_DIR}/runtime/cl_safe_math_macros.m4 > cl_safe_math_macros.h
        DEPENDS ${CMAKE_SOURCE_DIR}/runtime/cl_safe_math_macros.m4
        VERBATIM
    )

    install(TARGETS CUDASmith
        RUNTIME DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )

    install(FILES ${SAFE_MATH_HEADERS}
        DESTINATION include/CLSmith
    )
else()
    message(WARNING "Cannot build the safe math runtime header files because m4 was not found")
endif()

install(FILES ${CMAKE_SOURCE_DIR}/runtime/CLSmith.h
    DESTINATION include/CLSmith
)

find_package(OpenCL)

if(OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    add_executable(cl_launcher
            src/CUDASmith/cl_launcher.c
    )

    target_link_libraries(cl_launcher ${OpenCL_LIBRARIES})

    install(TARGETS cl_launcher
            RUNTIME DESTINATION bin
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
else()
    message(WARNING "Cannot build cl_launcher because OpenCL was not found")
endif()
