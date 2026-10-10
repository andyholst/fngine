#pragma once

// Programmatic dependent launch (PDL) for the decode/verify kernel chain.
//
// A kernel launched with pdl_launch may become resident while its stream predecessor still runs
// (once every CTA of the predecessor executed pdl_trigger(), or exited). Since every kernel of the
// chain triggers early, that includes any earlier kernel in the stream. Until it calls pdl_wait() a
// kernel may therefore only read immutable data: weights, codebooks, scales, constant tables (not
// recurrent state or KV: a back-to-back launch of the same kernel updates them; the one exception
// is described below). That is the point of it: each weight-streaming kernel issues its first
// weight loads, then waits, so the next weight stream overlaps the current kernel's tail instead
// of starting after a launch gap and a DRAM round trip. pdl_wait() returns once every prerequisite
// grid completed and its writes are visible, so every read of an activation, state, work list or
// residual, and every write a running kernel might still read, must come after it.
//
// Program order alone does not keep a read behind pdl_wait(): a const __restrict__ kernel pointer
// compiles to ld.global.nc, whose data must be read-only for the whole kernel, and ptxas
// schedules those loads above griddepcontrol.wait. Data written earlier in the step is therefore
// read through pdl_coherent(pointer), which compiles to ordinary loads. Conversely ptxas may sink
// .nc weight loads below the wait; loads that must be in flight before it use ld.global.cg
// (asm volatile).
//
// Kernels call pdl_trigger() first thing: dependents only become resident in slots the trigger
// grid no longer needs (they launch after the grid's last CTA started). A kernel whose dependent
// reads its predecessor's output before the dependent's own pdl_wait() triggers after its own
// pdl_wait() instead: hot_gate_up does, so the routing work list that hot_down reads early (to
// stage the right expert's weights) is complete before hot_down can launch; hot_down reads it with
// pdl_load_settled. Both device helpers are no-ops in ordinary launches, so the same kernel serves
// eager prefill and captured decode.
#include <cuda_runtime.h>

#include <utility>

namespace fngine::kernels {

__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;"); }

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// `pointer` without its kernel-parameter provenance (see above): loads through it are coherent.
template <class T>
__device__ __forceinline__ T* pdl_coherent(T* pointer) {
    asm("" : "+l"(pointer));
    return pointer;
}

// Load ahead of pdl_wait() of a value whose writer grid completed before this kernel could launch
// (see above). gpu-scope loads bypass L1, which may still hold the line from an earlier kernel.
__device__ __forceinline__ int pdl_load_settled(const int* p) {
    int v;
    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

// Launches `kernel` as a programmatic dependent of the preceding kernel in `stream`.
template <class... Params, class... Args>
cudaError_t pdl_launch(void (*kernel)(Params...), dim3 grid, dim3 block, std::size_t smem,
                       cudaStream_t stream, Args&&... args) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim          = grid;
    cfg.blockDim         = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream           = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs                                          = attr;
    cfg.numAttrs                                       = 1;
    return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

} // namespace fngine::kernels
