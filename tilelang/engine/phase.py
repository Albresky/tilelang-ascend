# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
from tvm import tir, IRModule
from tvm.target import Target
import tilelang
from tilelang.transform import PassContext
from tilelang.contrib.nvcc import have_tma
from typing import Optional
from ..utils.mlogger import *

def allow_warp_specialized(pass_ctx: Optional[PassContext] = None,
                           target: Optional[Target] = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if not is_cuda_target(target):
        return False
    disable_warp_specialized = pass_ctx.config.get("tl.disable_warp_specialized", False)
    return not disable_warp_specialized


def allow_tma_and_warp_specialized(pass_ctx: Optional[PassContext] = None,
                                   target: Optional[Target] = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if not is_cuda_target(target) or not have_tma(target):
        return False
    disable_tma_lower = pass_ctx.config.get("tl.disable_tma_lower", False)
    return not disable_tma_lower and allow_warp_specialized(pass_ctx=pass_ctx, target=target)


def allow_fence_proxy(target: Optional[Target] = None) -> bool:
    # avoid circular import
    from tilelang.jit.adapter.utils import is_cuda_target

    return is_cuda_target(target) and have_tma(target)


def allow_vectorize(pass_ctx: Optional[PassContext] = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tir.disable_vectorize", False)
    return not disable_vectorize


def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    # Bind the target device information to the module
    mod = tir.transform.BindTarget(target)(mod)
    log2file("01-original_mod.log", mod.astext(show_meta_data=False))

    # Identify and filter host tiling data for npu
    mod = tilelang.transform.HostProcesser()(mod)
    log2file("02-after_host_processer.log", mod.astext(show_meta_data=False))

    mod = tilelang.transform.FrontendLegalize()(mod)
    log2file("03-after_frontend_legalize.log", mod.astext(show_meta_data=False))

    # Simplify the IR expressions
    mod = tir.transform.Simplify()(mod)
    log2file("04-after_simplify.log", mod.astext(show_meta_data=False))
    # Infer memory layouts for fragments and shared memory
    mod = tilelang.transform.LayoutInference()(mod)
    log2file("05-after_layout_inference.log", mod.astext(show_meta_data=False))
    # Lower high-level tile operations to low-level operations
    mod = tilelang.transform.LowerTileOp()(mod)
    log2file("06-after_lower_tile_op.log", mod.astext(show_meta_data=False))
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    log2file("07-after_legalize_vectorized_loop.log", mod.astext(show_meta_data=False))
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    log2file("08-after_legalize_safe_memory_access.log", mod.astext(show_meta_data=False))
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    mod = tir.transform.Simplify()(mod)
    log2file("09-after_simplify_2.log", mod.astext(show_meta_data=False))
    # Try to vectorize loop with dynamic shape
    mod = tilelang.transform.LoopVectorizeDynamic()(mod)
    log2file("10-after_loop_vectorize_dynamic.log", mod.astext(show_meta_data=False))

    return mod


def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()
    log2file("11-before_optimize_for_target.log", mod.astext(show_meta_data=False))
    mod = tir.transform.LowerOpaqueBlock()(mod)
    log2file("12-after_lower_opaque_block.log", mod.astext(show_meta_data=False))
    mod = tir.transform.NarrowDataType(32)(mod)
    log2file("13-after_narrow_data_type.log", mod.astext(show_meta_data=False))
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    log2file("14-after_config_index_bitwidth.log", mod.astext(show_meta_data=False))
    mod = tilelang.transform.FlattenBuffer()(mod)
    log2file("15-after_flatten_buffer.log", mod.astext(show_meta_data=False))
    mod = tir.transform.Simplify()(mod)
    log2file("16-after_simplify_3.log", mod.astext(show_meta_data=False))
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    log2file("17-after_vectorize_loop.log", mod.astext(show_meta_data=False))
    mod = tir.transform.StorageRewrite()(mod)
    log2file("18-after_storage_rewrite.log", mod.astext(show_meta_data=False))
    mod = tir.transform.UnrollLoop()(mod)
    log2file("19-after_unroll_loop.log", mod.astext(show_meta_data=False))
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    log2file("20-after_renormalize_split_pattern.log", mod.astext(show_meta_data=False))
    mod = tir.transform.Simplify()(mod)
    log2file("21-after_simplify_4.log", mod.astext(show_meta_data=False))
    mod = tir.transform.RemoveNoOp()(mod)
    log2file("22-after_remove_no_op.log", mod.astext(show_meta_data=False))
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    log2file("23-after_rewrite_unsafe_select.log", mod.astext(show_meta_data=False))
    mod = tir.transform.HoistIfThenElse()(mod)
    log2file("24-after_hoist_if_then_else.log", mod.astext(show_meta_data=False))
    return mod
