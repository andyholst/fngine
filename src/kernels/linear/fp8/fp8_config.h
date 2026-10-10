// Modified for fngine from cinference:src/ops/linear/fp8/fp8_config.h @ 383e5db. See NOTICE.
// Modified by satellitedown for Cinference: add K-split staging depth and L2 cache mode.
// See NOTICE and upstream-provenance.json for upstream attribution.
// fngine: fold fp8_geometry.h in; 4-value lanes (K = 384 / 640); intra-CTA warp K-split for the
// GEMV/SIMT schedules (narrow-N shapes such as the GR down projection).

#pragma once

#include <cstdint>

namespace fngine::kernels::detail {

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Fp8Geometry {
    static_assert(OutputRows > 0 && (OutputRows % 16) == 0);
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kOutputRows = OutputRows;
    static constexpr std::int32_t kInputRows  = InputRows;
};

enum class Fp8CodeCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Fp8SimtActivationAccess : std::uint8_t {
    TokenPacked,
    SharedPhase,
};

enum class Fp8SimtBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

// RowWarps warps own disjoint row sets; KSplit warps share each row set and split its K phases.
template <int RowWarps, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Fp8CodeCache CodeCache, int PhaseUnroll, int MinBlocksPerSm, int KSplit = 1>
struct Fp8GemvSchedule {
    static_assert(RowWarps > 0 && KSplit > 0 && RowWarps * KSplit <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 4 || ValuesPerLane == 8 || ValuesPerLane == 16 ||
                  ValuesPerLane == 32);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kRowWarps          = RowWarps;
    static constexpr int kKSplit            = KSplit;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kWarps             = RowWarps * KSplit;
    static constexpr int kThreads           = kWarps * 32;
    static constexpr int kRowsPerCta        = RowWarps * RowsPerWarp;
};

template <int RowWarps, int RowsPerWarp, int ValuesPerLane, int TokenTile, int AccumulatorChains,
          Fp8SimtActivationAccess ActivationAccess, Fp8CodeCache CodeCache, int PhaseUnroll,
          Fp8SimtBlockOrder BlockOrder, int MinBlocksPerSm, int KSplit = 1>
struct Fp8SimtSchedule {
    static_assert(RowWarps > 0 && KSplit > 0 && RowWarps * KSplit <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 4 || ValuesPerLane == 8 || ValuesPerLane == 16 ||
                  ValuesPerLane == 32);
    static_assert(TokenTile > 0);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);
    // Shared-phase staging serves one K phase to every warp; split warps walk different phases.
    static_assert(KSplit == 1 || ActivationAccess == Fp8SimtActivationAccess::TokenPacked);

    static constexpr int kRowWarps          = RowWarps;
    static constexpr int kKSplit            = KSplit;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kTokenTile         = TokenTile;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kWarps             = RowWarps * KSplit;
    static constexpr int kThreads           = kWarps * 32;
    static constexpr int kRowsPerCta        = RowWarps * RowsPerWarp;
};

enum class Fp8A16KSplitActivationStage : std::uint8_t {
    ActiveOnly,
    PaddedZero,
};

// Default fills L1 and L2; Streaming bypasses L1; EvictFirst also marks the L2 fill evict-first,
// for a weight stream that a single wave reads once per call while the activation and record
// working set of the surrounding layer should stay resident.
enum class Fp8A16KSplitCache : std::uint8_t {
    Default,
    Streaming,
    EvictFirst,
};

// Stages > 1 multi-buffers the per-group code and activation staging so the next group's copies
// are in flight while the current group multiplies. One-wave shapes need it to keep the weight
// stream busy; many-wave shapes hide latency across CTAs and keep the single-stage footprint.
template <int KWarps, int TileTokens, int MinBlocksPerSm,
          Fp8A16KSplitCache ActivationCache           = Fp8A16KSplitCache::Default,
          Fp8A16KSplitCache WeightCache               = Fp8A16KSplitCache::Streaming,
          Fp8A16KSplitActivationStage ActivationStage = Fp8A16KSplitActivationStage::ActiveOnly,
          int Stages                                  = 1>
struct Fp8A16KSplitSchedule {
    static_assert(KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens == 8 || TileTokens == 16 || TileTokens == 24 || TileTokens == 32 ||
                  TileTokens == 40 || TileTokens == 48);
    static_assert(MinBlocksPerSm > 0);
    static_assert(Stages >= 1);

    static constexpr int kKWarps            = KWarps;
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kStages            = Stages;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = KWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / KWarps;
};

} // namespace fngine::kernels::detail
