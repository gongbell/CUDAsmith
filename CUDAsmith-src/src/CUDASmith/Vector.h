// Vectors!

#ifndef _CUDASMITH_VECTOR_H_
#define _CUDASMITH_VECTOR_H_

#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <utility>

#include "ArrayVariable.h"
#include "CGContext.h"

class Block;
class CVQualifiers;
class Expression;
class Type;
class Variable;

namespace CUDASmith
{

// Handles OpenCL's vector type.
// Treated mostly the same as an array during generation. Only the output and
// size need special handling.
class Vector : public ArrayVariable
{
  public:
    // Basic constructor with parameters required by ArrayVariable.
    // Will only accept 1-dimensional sizes.
    Vector(Block *blk, const std::string &name, const Type *type,
           const Expression *init, const CVQualifiers *qfer, unsigned size,
           const Variable *isFieldVarOf)
        : ArrayVariable(blk, name, type, init, qfer, {size}, isFieldVarOf, true)
    {
    }

    // Copy constructor required, as ArrayVariable frequently uses it during
    // itemisation.
    Vector(const Vector &other) = default;
    Vector &operator=(const Vector &other) = default;
    // No move constructors in ArrayVariable, copy will be performed.
    Vector(Vector &&other) = default;
    Vector &operator=(Vector &&other) = default;
    virtual ~Vector() {}

    // Factory for producing a vector variable.
    // Parameters are mostly just forwarded on to the ArrayVariable constructor.
    static Vector *CreateVectorVariable(const CGContext &cg_context, Block *blk,
                                        const std::string &name, const Type *type, const Expression *init,
                                        const CVQualifiers *qfer, const Variable *isFieldVarOf);

    // Itemise methods for using a specific entry in the vector. All accesses must
    // be a constant index, so the last two methods will randomise the access.
    Vector *itemize(void) const;
    Vector *itemize(const std::vector<int> &const_indices) const;
    Vector *itemize(
        const std::vector<const Variable *> &indices, Block *blk) const
    {
        return itemize();
    }
    Vector *itemize(
        const std::vector<const Expression *> &indices, Block *blk) const
    {
        return itemize();
    }
    // Itemise for accessing multiple components at once.
    virtual Vector *itemize_simd(void) const;
    virtual Vector *itemize_simd(int access_count) const;
    virtual Vector *itemize_simd(const std::vector<int> &const_indices) const;

    // Methods that control how the array is printed.
    // In all cases, there is only one index, which is printed as a character
    // after a dot:
    //  'vec[2]' -> 'vec.z' or 'vec.s2'
    //  '{vec[3], vec[1]}' -> 'vec.wy' or 'vec.s31'
    void Output(std::ostream &out) const;
    void OutputDef(std::ostream &out, int indent) const;
    void OutputDecl(std::ostream &out) const;
    void hash(std::ostream &out) const;
    void Output_Self(std::ostream &out) const;

    // Creates a string used for initialising this vector. This has the form
    // (typen)(...) with the number of elements in the second set of brackets
    // matching the size of the vector.
    // The initialiser list can have elements grouped into smaller vectors:
    //   int8 vec = (int8)(4, (int4)(1, 2, 3, 4), 6, (int2)(9, 8))
    std::string build_initializer_str(
        const std::vector<std::string> &init_strings) const;

    // Get a random valid vector length no greater than the specified maximum.
    // If max is 0, no limit is imposed on the size of the vector.
    static int GetRandomVectorLength(int max);

    // Convert a simple type to a vector type. Giving a size of 0 will create a
    // random vector length. The type can already be a vector type, so this can be
    // used to change the length.
    static const Type *PromoteTypeToVectorType(const Type *type, int size);
    // Convert from a vector type to the underlying simple type.
    static const Type &DemoteVectorTypeToType(const Type *type);

    // Returns the character corresponding to the component that is accessed.
    // If the size of the vector is greater than 4, the access character is a hex
    // digit, otherwise, it is one of xyzw.
    static char GetComponentChar(int vector_size, int index);
    // Returns the string corresponding to the suffix by which the vector is being
    // accessed.  // TODO migrate suffixes here.
    static const char *GetSuffixString(/*enum*/ int suffix);

    // Outputs the vector type, without qualifiers, wrapped in a macro:
    //   VECTOR(int , 8) -> int8
    static void OutputVectorType(std::ostream &out, const Type *type,
                                 int vector_size);
    static void OutputVectorType_Make(std::ostream &out, const Type *type,
                                      int vector_size);

    // Must be called on startup, generates all the vector types.
    static void GenerateVectorTypes();

  private:
    // Keeps track of multiple accesses in a single itemisation.
    // e.g. 'vec.wy' will be {3, 1}
    std::vector<int> comp_access_;

    // All vector types. These will be pre-generated, as many of the functions
    // that check for type compatibilities between variables rely on the types
    // being identical.
    static std::map<std::pair<enum eSimpleType, unsigned>, const Type *>
        vector_types_;
};

} // namespace CUDASmith

#endif // CUDASMITH_VECTOR_H_
