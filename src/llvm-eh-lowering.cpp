// This file is a part of Julia. License is MIT: https://julialang.org/license

#define DEBUG_TYPE "eh_lowering"

// LLVM pass lowering the call to outlined exception handling function.

// The pass needs to be run after GC frame lowering since it can add addrspace casts
// that breaks the invariance needed by the GC frame lowering pass.

#include "llvm-version.h"
#include "support/dtypes.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>

#include "julia.h"
#include "julia_internal.h"
#include "codegen_shared.h"
#include "julia_assert.h"

using namespace llvm;

namespace {

struct EHLowering: public ModulePass {
    static char ID;
    EHLowering()
        : ModulePass(ID)
    {}

private:
    bool runOnModule(Module &M) override;
    void lowerEH(CallInst *call);

    LLVMContext *ctx;
    const DataLayout *DL;
    uint32_t ptr_bits;
    IntegerType *T_int32;
    IntegerType *T_size;
    PointerType *T_pint8;
    Function *catcher;
    Function *catch_1;
    Function *catch_generic;
};

void EHLowering::lowerEH(CallInst *call)
{
    struct ArgInfo {
        Value *val;
        enum ArgClass {
            // Combine into a big alloca
            Alloca,
            // Can be passed by value directly
            PtrVal,
            // Can be zext to a pointer size integer and passed directly
            IntZext,
            // Must be passed on the stack
            Spill
        };
        ArgClass arg_cls;
        AllocaInst *base; // Alloca only
        uint64_t offset; // Alloca only
        ArgInfo(Value *val)
            : val(val)
        {}
    };
    SetVector<AllocaInst*> allocas;
    SmallVector<ArgInfo,16> args;
    SmallVector<uint32_t,8> ptrarg_idx;
    Function *eh_func = cast<Function>(call->getArgOperand(0));
    bool has_spill = false;
    uint32_t num_val_args = 0;
    // Skip the first argument assuming there's at least one argument
    for (auto ai = call->arg_begin(), ae = call->arg_end(); ++ai != ae;) {
        args.emplace_back(ai->get());
        auto &info = args.back();
        auto ty = info.val->getType();
        if (auto ptr_ty = dyn_cast<PointerType>(ty)) {
            if (ptr_ty->getAddressSpace() == 0) {
                APInt offset(ptr_bits, 0);
                auto base = info.val->stripAndAccumulateInBoundsConstantOffsets(*DL, offset);
                if (auto alloca_base = dyn_cast<AllocaInst>(base)) {
                    if (alloca_base->isStaticAlloca()) {
                        allocas.insert(alloca_base);
                        info.arg_cls = ArgInfo::Alloca;
                        info.base = alloca_base;
                        info.offset = offset.getLimitedValue();
                        continue;
                    }
                }
            }
            info.arg_cls = ArgInfo::PtrVal;
            num_val_args++;
            continue;
        }
        if (auto int_ty = dyn_cast<IntegerType>(ty)) {
            if (int_ty->getBitWidth() <= ptr_bits) {
                info.arg_cls = ArgInfo::IntZext;
                num_val_args++;
                continue;
            }
        }
        info.arg_cls = ArgInfo::Spill;
        has_spill = true;
    }

    bool has_alloca_arg = has_spill || !allocas.empty();
    constexpr static uint32_t catch_narg = 5;
    uint32_t max_val_arg;
    if (has_alloca_arg || num_val_args > catch_narg) {
        max_val_arg = catch_narg - 1;
    }
    else {
        max_val_arg = catch_narg;
    }
    if (max_val_arg < num_val_args) {
    }
}

bool EHLowering::runOnModule(Module &M)
{
    catcher = M.getFunction("julia.catch_exception");
    if (!catcher)
        return false;
    ctx = &M.getContext();
    DL = &M.getDataLayout();
    ptr_bits = DL->getPointerSizeInBits(0);
    T_int32 = Type::getInt32Ty(*ctx);
    T_size = IntegerType::get(*ctx, ptr_bits);
    T_pint8 = Type::getInt8PtrTy(*ctx);
    Type *args[] = {T_pint8, T_pint8, T_pint8, T_pint8, T_pint8, T_pint8};
    catch_1 = Function::Create(FunctionType::get(T_int32, args, true),
                               Function::ExternalLinkage, "jl_catch_exception", &M);
    catch_generic = Function::Create(FunctionType::get(T_int32, args, true),
                                     Function::ExternalLinkage,
                                     "jl_catch_exception_generic", &M);

    bool changed = false;
    while (!catcher->use_empty())
        lowerEH(cast<CallInst>(*catcher->user_begin()));
    return changed;
}

char EHLowering::ID = 0;

static RegisterPass<EHLowering> X("EHLowering", "EHLowering Pass",
                                  false /* Only looks at CFG */,
                                  false /* Analysis Pass */);

} // anonymous namespace

Pass *createEHLoweringPass()
{
    return new EHLowering();
}
