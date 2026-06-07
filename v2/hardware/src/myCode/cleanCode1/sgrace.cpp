/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <hls_math.h>
#include <string>
#include <fstream>
#include <sstream>

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"
#include "matrix_mult.h"
#include "hls_streamofblocks.h"


/* ────────────────────────────────────────────────────────────────────────────
 * Stream-of-blocks tile types.
 * Each C/A buffer tile holds one row-partition × one column-block of the
 * output or pre-activation matrix.
 * bufl uses the linear-branch quantization type (QLTYPE).
 * ──────────────────────────────────────────────────────────────────────────── */
typedef QTYPE  buf [F_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK];
typedef QLTYPE bufl[F_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK];

typedef unsigned long u32;
/* ────────────────────────────────────────────────────────────────────────────
 * Compile-time constants derived from the project parameters in matrix_mult.h.
 * These are pulled out here so the rest of the code can use readable names.
 * ──────────────────────────────────────────────────────────────────────────── */

/* Output column-block width (must be <= B_WIDTH_BLOCK) */
const int BLOCK     = B_WIDTH_BLOCK;

/* SpMM row-block size for multi-row tiling */
const int SBLOCK     = SPMM_BLOCK;

/* Linear path always processes one row at a time */
const int SBLOCK_LIN = 1;

/* Number of parallel processing elements in the row dimension */
const int PARALLEL_ROW = B_BLOCK_PARALLEL;

/* FIFO depth for the standard inter-task FIFOs */
const int FIFO_DEPTH = MAX_FIFO;

/* Total number of elements in one linear-projection weight tile */
const int LINEAR_DEPTH = B_WIDTH_BLOCK * B_HEIGHT;

/* Attention FIFO depths:
 * FIFO_DEPTH_ATTN  – per-row attention score FIFOs (edge values).
 * FIFO_DEPTH_ATTN2 – support (exp) FIFO, needs ATEN_BLOCK rows of storage.
 * OPT_ATTN controls expected adjacency sparsity (higher = sparser assumed,
 * smaller FIFOs).  OPT_ATTN = 1 is the worst-case (fully dense adjacency). */
const int FIFO_DEPTH_ATTN  = A_HEIGHT / OPT_ATTN;
const int FIFO_DEPTH_ATTN2 = A_HEIGHT * ATEN_BLOCK / OPT_ATTN;

/* Floating-point adder pipeline latencies — used by the latency-hiding
 * partial-accumulator pattern in the float/half DSP kernels. */
const int FADD_LATENCY_ADJ = FTYPE_LATENCY_ADJ;
const int FADD_LATENCY_FEA = FTYPE_LATENCY_FEA;


/* ────────────────────────────────────────────────────────────────────────────
 * FIFO telemetry counters.
 * Updated by the check_fifo_* relay tasks to profile pipeline back-pressure.
 * ──────────────────────────────────────────────────────────────────────────── */
static ap_int<64> fifo_full_0,  fifo_full_1,  fifo_full_2;
static ap_int<64> fifo_empty_0, fifo_empty_1, fifo_empty_2;
static ap_int<64> fifo_read_0,  fifo_read_1,  fifo_read_2;
static ap_int<64> fifo_write_0, fifo_write_1, fifo_write_2;
static ap_int<64> fifo_cycle_0, fifo_cycle_1, fifo_cycle_2;


/* ────────────────────────────────────────────────────────────────────────────
 * Simulation-only range monitors.
 * Enabled by defining SIMULATION at compile time; used to verify that
 * accumulator values stay within the expected fixed-point range.
 * ──────────────────────────────────────────────────────────────────────────── */
#ifdef simulation
extern float max_adj,   min_adj;
extern float max_fea,   min_fea;
extern float acc2_fea_min, acc2_fea_max;
extern float acc2_adj_min, acc2_adj_max;
#endif


/* ════════════════════════════════════════════════════════════════════════════
 * Quantization functions for feature, adjacency, weight, and linear values.
 *
 * All four functions follow the same pattern:
 *   1. Scale and shift:  vfloat = quantization_scale * B + zero_point
 *   2. Round to integer: vround = round(vfloat)   (or binary sign for f_align==7)
 *   3. Clip to [ialpha_q, ibeta_q]
 *   4. Right-shift by (qbits - f_align - 1) to produce the fixed-point value.
 *
 * Clip range depends on SIGNED_MODE (quanta/quantf) or is always symmetric
 * ±(beta_qu>>1) (quantl/quantw).
 * ════════════════════════════════════════════════════════════════════════════ */

// =============================================================================================
// =============================================================================================
/**
 * quanta - Quantize a floating-point adjacency value to ATYPE.
 *
 * Clip range:
 *   SIGNED_MODE == 0: [0,        beta_qu]
 *   SIGNED_MODE == 1: [-beta_q,  +beta_q]   where beta_q = beta_qu >> 1
 *
 * Binary device mode (qbits == 1): right-shift = 1.
 *
 * @param BW                  Output: quantized adjacency value.
 * @param B                   Input: raw floating-point adjacency value.
 * @param quantization_scale  Single scale factor (not per-layer).
 * @param f_align             Fractional alignment bits (7 = bipolar binary).
 * @param beta_qu             Quantization range (full width).
 */
void quanta(
    ATYPE &BW,
    float  B,
    float  quantization_scale,
    int    f_align,
    int    beta_qu
)
{
    float vfloat = quantization_scale * B + zero_point;
    float vround = hls::round(vfloat);

    ITYPE vquant = ITYPE(vround);

#if (SIGNED_MODE == 0)
    ITYPE ibeta_q  = (ITYPE)beta_qu;
    ITYPE ialpha_q = (ITYPE)(0.0);
#else
    ITYPE beta_q   = ITYPE(beta_qu >> 1);
    ITYPE ibeta_q  = (ITYPE)beta_q;
    ITYPE ialpha_q = -(ITYPE)beta_q;
#endif

    /* Clip to representable range */
    if      (vquant > ibeta_q)  vquant = ibeta_q;
    else if (vquant < ialpha_q) vquant = ialpha_q;

    if (f_align == 7)   // bipolar binary: use f_align = 6 for shift
        f_align = 6;

#if (qbits == 1)
    ITYPE vnorm = vquant >> 1;
#else
    ITYPE vnorm = vquant >> (qbits - f_align - 1);
#endif

    BW = ATYPE(vnorm);
}

// =============================================================================================
// =============================================================================================
/**
 * quantf - Quantize a floating-point feature value to FTYPE (GNN branch).
 *
 * Identical to quanta but uses per-layer scale (quantization_scale[B_index])
 * and outputs FTYPE.
 *
 * @param BW                  Output: quantized feature value.
 * @param B                   Input: raw floating-point feature value.
 * @param quantization_scale  Per-layer scale factor array.
 * @param f_align             Fractional alignment bits (7 = bipolar binary).
 * @param beta_qu             Quantization range.
 * @param B_index             Current layer index.
 */
void quantf(
    FTYPE &BW,
    float  B,
    float  quantization_scale[5],
    int    f_align,
    int    beta_qu,
    int    B_index
)
{
    float vfloat = quantization_scale[B_index] * B + zero_point;
    float vround = hls::round(vfloat);

    ITYPE vquant = ITYPE(vround);

#if (SIGNED_MODE == 0)
    ITYPE ibeta_q  = (ITYPE)beta_qu;
    ITYPE ialpha_q = (ITYPE)(0.0);
#else
    ITYPE beta_q   = ITYPE(beta_qu >> 1);
    ITYPE ibeta_q  = (ITYPE)beta_q;
    ITYPE ialpha_q = -(ITYPE)beta_q;
#endif

    if      (vquant > ibeta_q)  vquant = ibeta_q;
    else if (vquant < ialpha_q) vquant = ialpha_q;

    if (f_align == 7)
        f_align = 6;

#if (qbits == 1)
    ITYPE vnorm = vquant >> 1;
#else
    ITYPE vnorm = vquant >> (qbits - f_align - 1);
#endif

    BW = FTYPE(vnorm);
}

// =============================================================================================
// =============================================================================================
/**
 * quantl - Quantize a floating-point feature value to LTYPE (linear-projection branch).
 *
 * Uses the same bipolar-binary / general-fixed-point split as quantwl.
 * Clip range is always symmetric: [-beta_q, +beta_q].
 * Output uses qbitsl instead of qbits in the final shift.
 *
 * @param BW                  Output: quantized linear-branch feature value.
 * @param B                   Input: raw floating-point feature value.
 * @param quantization_scale  Per-layer scale factor array.
 * @param f_align             Fractional alignment bits (7 = bipolar binary).
 * @param beta_qu             Quantization range.
 * @param B_index             Current layer index.
 */
void quantl(
    LTYPE &BW,
    float  B,
    float  quantization_scale[5],
    int    f_align,
    int    beta_qu,
    int    B_index
)
{
    float vfloat = quantization_scale[B_index] * B + zero_point;
    float vround;

    ITYPE ibeta_q, ialpha_q, beta_q;

    if (f_align == 7)
    {
        /* Bipolar binary mode */
        ibeta_q  = 1;
        ialpha_q = -1;
        vround   = (vfloat < 0.0f) ? -1.0f : 1.0f;
    }
    else
    {
        /* General fixed-point */
        beta_q   = ITYPE(beta_qu >> 1);
        ibeta_q  = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
        vround   = hls::round(vfloat);
    }

    ITYPE vquant = ITYPE(vround);

    if      (vquant > ibeta_q)  vquant = ibeta_q;
    else if (vquant < ialpha_q) vquant = ialpha_q;

    if (f_align == 7)
        f_align = 6;

    ITYPE vnorm = vquant >> (qbitsl - f_align - 1);
    BW = LTYPE(vnorm);
}
// =============================================================================================
// =============================================================================================
/**
 * quantw - Quantize a floating-point weight to BTYPE (GNN branch).
 *
 * Maps:  vfloat = quantization_scale[B_index] * B + zero_point
 * then clips, rounds, and right-shifts to produce a BTYPE fixed-point value.
 *
 * Three quantization modes selected at compile time:
 *
 *   qbits == 1  (binary):
 *     vfloat < 0  → vround = 1.0  (encodes negative weight as bit 1)
 *     vfloat >= 0 → vround = 0.0
 *     Clip range: [0, 1].
 *
 *   f_align == 7  (bipolar binary, called separately):
 *     vfloat < 0  → vround = -1.0
 *     vfloat >= 0 → vround =  1.0
 *     Clip range: [-1, 1].
 *     f_align is then forced to 6 before the final shift.
 *
 *   General fixed-point:
 *     vround = round(vfloat)
 *     beta_q = beta_qu >> 1  (half-range)
 *     Clip range: [-beta_q, beta_q].
 *
 * Final normalization:  BW = BTYPE( vquant >> (qbits - f_align - 1) )
 *
 * @param BW                   Output: quantized weight value.
 * @param B                    Input: raw floating-point weight.
 * @param quantization_scale   Per-layer scale factors.
 * @param f_align              Fractional alignment bits (7 = bipolar binary mode).
 * @param beta_qu              Full quantization range [-(beta_qu>>1), +(beta_qu>>1)].
 * @param B_index              Current layer index.
 */
void quantw(
    BTYPE  &BW,
    float   B,
    float   quantization_scale[5],
    int     f_align,
    int     beta_qu,
    int     B_index
)
{
    float vfloat = quantization_scale[B_index] * B + zero_point;
    float vround;

    ITYPE ibeta_q, ialpha_q, beta_q;

#if (qbits == 1)
    /* Binary mode: encode sign as 1/0 */
    ibeta_q  = 1;
    ialpha_q = 0;
    vround   = (vfloat < 0.0f) ? 1.0f : 0.0f;

#else
    if (f_align == 7)
    {
        /* Bipolar binary mode */
        ibeta_q  = 1;
        ialpha_q = -1;
        vround   = (vfloat < 0.0f) ? -1.0f : 1.0f;
    }
    else
    {
        /* General fixed-point: symmetric clip around beta_q */
        beta_q   = ITYPE(beta_qu >> 1);
        ibeta_q  = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
        vround   = hls::round(vfloat);
    }
#endif

    ITYPE vquant = ITYPE(vround);

    /* Clip to representable range */
    if      (vquant > ibeta_q)  vquant = ibeta_q;
    else if (vquant < ialpha_q) vquant = ialpha_q;

    /* In bipolar binary mode the shift uses f_align = 6 */
    if (f_align == 7)
        f_align = 6;

    ITYPE vnorm = vquant >> (qbits - f_align - 1);
    BW = BTYPE(vnorm);
}

// =============================================================================================
// =============================================================================================
/**
 * quantwl - Quantize a floating-point weight to BLTYPE (linear-projection branch).
 *
 * Identical to quantw but:
 *   - No qbits == 1 binary mode (linear branch always uses multi-bit weights).
 *   - Output type is BLTYPE.
 *   - Uses qbitsl instead of qbits in the final shift.
 *
 * @param BW                   Output: quantized weight value (BLTYPE).
 * @param B                    Input: raw floating-point weight.
 * @param quantization_scale   Per-layer scale factors.
 * @param f_align              Fractional alignment bits (7 = bipolar binary mode).
 * @param beta_qu              Full quantization range.
 * @param B_index              Current layer index.
 */
void quantwl(
    BLTYPE &BW,
    float   B,
    float   quantization_scale[5],
    int     f_align,
    int     beta_qu,
    int     B_index
)
{
    float vfloat = quantization_scale[B_index] * B + zero_point;
    float vround;

    ITYPE ibeta_q, ialpha_q, beta_q;

    if (f_align == 7)
    {
        /* Bipolar binary mode */
        ibeta_q  = 1;
        ialpha_q = -1;
        vround   = (vfloat < 0.0f) ? -1.0f : 1.0f;
    }
    else
    {
        /* General fixed-point */
        beta_q   = ITYPE(beta_qu >> 1);
        ibeta_q  = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
        vround   = hls::round(vfloat);
    }

    ITYPE vquant = ITYPE(vround);

    /* Clip to representable range */
    if      (vquant > ibeta_q)  vquant = ibeta_q;
    else if (vquant < ialpha_q) vquant = ialpha_q;

    if (f_align == 7)
        f_align = 6;

    ITYPE vnorm = vquant >> (qbitsl - f_align - 1);
    BW = BLTYPE(vnorm);
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_float_adj_1 - Float/half inner multiply kernel, single C-tile partition (adjacency).
 *
 * Computes:  acc[j] = (ITYPE)a_value × (ITYPE)b_block[b_row][j]
 *
 * b_row indexes directly into b_block (no partition selection needed;
 * the caller ensures b_row < B_HEIGHT).
 *
 * Inlined by the HLS tool to allow the outer pipeline to absorb the multiplies.
 *
 * @param a_value        Adjacency scalar for this non-zero.
 * @param b_block        C-tile [B_HEIGHT][B_WIDTH_BLOCK].
 * @param b_row          Row index into b_block.
 * @param zero_point_lhs Quantization zero point (unused in float path).
 * @param zero_point_rhs Quantization zero point (unused in float path).
 * @param acc            Output: product results [B_WIDTH_BLOCK].
 */
void dsp_kernel_float_adj_1(
    ATYPE      a_value,
    BTYPE      b_block[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
)
{
    #pragma HLS INLINE

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        BTYPE b_val = b_block[b_row][j];
        ATYPE a_val = a_value;
        acc[j]      = (ITYPE)a_val * (ITYPE)b_val;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_float_adj_2 - Float/half inner multiply kernel, two C-tile partitions (adjacency).
 *
 * Selects partition based on b_row vs block_size, then computes acc[j] = a × b.
 *
 *   b_row in [0,          block_size)  → b_block1
 *   b_row in [block_size, 2*block_size)→ b_block2
 *
 * @param block_size     Number of rows per partition.
 * @param a_value        Adjacency scalar.
 * @param b_block1..2    C-tile partitions [B_HEIGHT][B_WIDTH_BLOCK].
 * @param b_row          Global row index.
 * @param zero_point_lhs Unused.
 * @param zero_point_rhs Unused.
 * @param acc            Output: product results [B_WIDTH_BLOCK].
 */
void dsp_kernel_float_adj_2(
    int        block_size,
    ATYPE      a_value,
    BTYPE      b_block1[B_HEIGHT][B_WIDTH_BLOCK],
    BTYPE      b_block2[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
)
{
    #pragma HLS INLINE

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        ATYPE a_val = a_value;

        int b_row_block, sel_block;

        if (b_row < block_size)
        {
            b_row_block = b_row;
            sel_block   = 0;
        }
        if (b_row > (block_size - 1))
        {
            b_row_block = b_row - block_size;
            sel_block   = 1;
        }

        BTYPE b_val1 = b_block1[b_row_block][j];
        BTYPE b_val2 = b_block2[b_row_block][j];

        BTYPE b_val;
        switch (sel_block)
        {
            case 0:  b_val = b_val1;  break;
            case 1:  b_val = b_val2;  break;
        }

        acc[j] = (ITYPE)a_val * (ITYPE)b_val;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_float_adj_4 - Float/half inner multiply kernel, four C-tile partitions (adjacency).
 *
 * Selects partition based on b_row vs block_size boundaries, then computes acc[j] = a × b.
 *
 *   [0,           block_size)  → b_block1
 *   [block_size,  2*block_size)→ b_block2  (note: overlaps with prev — preserves original behavior)
 *   [2*block_size,3*block_size)→ b_block3
 *   [3*block_size, ∞)          → b_block4
 *
 * @param block_size     Number of rows per partition.
 * @param a_value        Adjacency scalar.
 * @param b_block1..4    C-tile partitions [B_HEIGHT][B_WIDTH_BLOCK].
 * @param b_row          Global row index.
 * @param zero_point_lhs Unused.
 * @param zero_point_rhs Unused.
 * @param acc            Output: product results [B_WIDTH_BLOCK].
 */
void dsp_kernel_float_adj_4(
    int        block_size,
    ATYPE      a_value,
    BTYPE      b_block1[B_HEIGHT][B_WIDTH_BLOCK],
    BTYPE      b_block2[B_HEIGHT][B_WIDTH_BLOCK],
    BTYPE      b_block3[B_HEIGHT][B_WIDTH_BLOCK],
    BTYPE      b_block4[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
)
{
    #pragma HLS INLINE

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        ATYPE a_val = a_value;

        int b_row_block, sel_block;

        if (b_row < block_size)
        {
            b_row_block = b_row;
            sel_block   = 0;
        }
        if (b_row > (block_size - 1))
        {
            b_row_block = b_row - block_size;
            sel_block   = 1;
        }
        if (b_row > (2 * block_size - 1) && b_row < 3 * block_size)
        {
            b_row_block = b_row - 2 * block_size;
            sel_block   = 2;
        }
        if (b_row > (3 * block_size - 1))
        {
            b_row_block = b_row - 3 * block_size;
            sel_block   = 3;
        }

        BTYPE b_val1 = b_block1[b_row_block][j];
        BTYPE b_val2 = b_block2[b_row_block][j];
        BTYPE b_val3 = b_block3[b_row_block][j];
        BTYPE b_val4 = b_block4[b_row_block][j];

        BTYPE b_val;
        switch (sel_block)
        {
            case 0:  b_val = b_val1;  break;
            case 1:  b_val = b_val2;  break;
            case 2:  b_val = b_val3;  break;
            case 3:  b_val = b_val4;  break;
        }

        acc[j] = (ITYPE)a_val * (ITYPE)b_val;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_float_fea - Float/half inner multiply kernel for the feature SpMM (GNN branch).
 *
 * Computes:  acc[j] = (ITYPE)a_value × (ITYPE)b_block[b_row][j]
 *
 * b_row indexes directly into the single weight tile (no partition selection).
 * Inlined by the HLS tool.
 *
 * @param a_value        Feature non-zero value.
 * @param b_block        Weight tile [B_HEIGHT][B_WIDTH_BLOCK].
 * @param b_row          Row index into b_block.
 * @param zero_point_lhs Unused.
 * @param zero_point_rhs Unused.
 * @param acc            Output: product results [B_WIDTH_BLOCK].
 */
void dsp_kernel_float_fea(
    ATYPE      a_value,
    BTYPE      b_block[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
)
{
    #pragma HLS INLINE

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        BTYPE b_val = b_block[b_row][j];
        ATYPE a_val = a_value;
        acc[j]      = (ITYPE)a_val * (ITYPE)b_val;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_int_adj_1 - Single-multiply fixed-point inner kernel for adjacency SpMM,
 *                        one C-tile column partition.
 *
 * For a single (adjacency value, column index) pair, computes:
 *   acc[j] = a_value × b_block1[b_row][j]   for all j in [0, B_WIDTH_BLOCK)
 *
 * b_row selects a row from b_block1 only (partition 0); the column index is
 * already restricted to [0, block_size) by the caller.
 *
 * Binary weight mode (qbits == 1):
 *   b_val == 0 → b_val_i = +0.5
 *   b_val == 1 → b_val_i = -0.5
 *
 * Note: acc[] is written (not accumulated) here — accumulation across
 * non-zeros is performed by the calling wrapper.
 *
 * @param block_size     Number of rows per column partition (currently only partition 0 used).
 * @param a_value        Attention-weighted adjacency scalar for this non-zero.
 * @param b_block1       C-tile column partition [B_HEIGHT][B_WIDTH_BLOCK].
 * @param b_row          Row index into b_block1.
 * @param zero_point_lhs Quantization zero point (currently unused; zero-point subtraction removed).
 * @param zero_point_rhs Quantization zero point (currently unused).
 * @param acc            Output: product results [B_WIDTH_BLOCK].
 */
void dsp_kernel_int_adj_1(
    int       block_size,
    TTYPE     a_value,
    QTYPE     b_block1[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
)
{
    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        TTYPE a_val = a_value;

        /* Select the column partition and intra-partition row index */
        int b_row_block, sel_block;

        if (b_row < block_size)
        {
            b_row_block = b_row;
            sel_block   = 0;
        }

        QTYPE b_val = b_block1[b_row_block][j];   // only partition 0 active

        /* Cast with optional binary-weight encoding */
        ITYPE a_val_i = (ITYPE)a_val;
        ITYPE b_val_i;

#if (qbits == 1)
        b_val_i = (b_val == 0) ? ITYPE(0.5) : ITYPE(-0.5);
#else
        b_val_i = (ITYPE)b_val;
#endif

        acc[j] = a_val_i * b_val_i;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_int_adj_2 - Single-multiply fixed-point inner kernel for adjacency SpMM,
 *                        two C-tile column partitions.
 *
 * Extends dsp_kernel_int_adj_1 to two partitions:
 *   b_row in [0,          block_size)   → partition 0 (b_block1)
 *   b_row in [block_size, 2*block_size) → partition 1 (b_block2)
 *
 * acc[] is zeroed then written (single product, no accumulation here).
 *
 * @param block_size     Number of rows per column partition.
 * @param a_value        Adjacency scalar for this non-zero.
 * @param b_block1       C-tile partition 0 [B_HEIGHT/2][B_WIDTH_BLOCK].
 * @param b_block2       C-tile partition 1 [B_HEIGHT/2][B_WIDTH_BLOCK].
 * @param b_row          Global row index; partition selected automatically.
 * @param zero_point_lhs Quantization zero point (unused).
 * @param zero_point_rhs Quantization zero point (unused).
 * @param acc            Output: product results [B_WIDTH_BLOCK].
 */
void dsp_kernel_int_adj_2(
    int       block_size,
    ITYPE     a_value,
    QTYPE     b_block1[B_HEIGHT / 2][B_WIDTH_BLOCK],
    QTYPE     b_block2[B_HEIGHT / 2][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
)
{
    /* Zero output */
    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        acc[j] = 0;
    }

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        ATYPE a_val = a_value;

        /* Determine which partition and intra-partition row index */
        int b_row_block, sel_block;

        if (b_row < block_size)
        {
            b_row_block = b_row;
            sel_block   = 0;
        }
        if (b_row > (block_size - 1))
        {
            b_row_block = b_row - block_size;
            sel_block   = 1;
        }

        BTYPE b_val1 = b_block1[b_row_block][j];
        BTYPE b_val2 = b_block2[b_row_block][j];

        BTYPE b_val;
        switch (sel_block)
        {
            case 0:  b_val = b_val1;  break;
            case 1:  b_val = b_val2;  break;
        }

        ITYPE a_val_i = (ITYPE)a_val;
        ITYPE b_val_i = (ITYPE)b_val;

        acc[j] += a_val_i * b_val_i;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_int_adj_4 - Single-multiply fixed-point inner kernel for adjacency SpMM,
 *                        four C-tile column partitions.
 *
 * Extends dsp_kernel_int_adj_2 to four partitions:
 *   [0,           block_size)  → b_block1
 *   [block_size,  2*block_size)→ b_block2
 *   [2*block_size,3*block_size)→ b_block3
 *   [3*block_size, ∞)          → b_block4
 *
 * acc[] is zeroed then accumulated with a single product per column.
 *
 * @param block_size     Number of rows per column partition.
 * @param a_value        Attention-weighted adjacency scalar.
 * @param b_block1..4    C-tile column partitions [B_HEIGHT/4][B_WIDTH_BLOCK].
 * @param b_row          Global row index; partition selected automatically.
 * @param zero_point_lhs Quantization zero point (unused).
 * @param zero_point_rhs Quantization zero point (unused).
 * @param acc            Output: accumulated product results [B_WIDTH_BLOCK].
 */
void dsp_kernel_int_adj_4(
    int       block_size,
    TTYPE     a_value,
    QTYPE     b_block1[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE     b_block2[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE     b_block3[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE     b_block4[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
)
{
    /* Zero output */
    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        acc[j] = 0;
    }

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        TTYPE a_val = a_value;

        /* Determine partition and intra-partition row index */
        int b_row_block, sel_block;

        if (b_row < block_size)
        {
            b_row_block = b_row;
            sel_block   = 0;
        }
        if (b_row > (block_size - 1) && b_row < 2 * block_size)
        {
            b_row_block = b_row - block_size;
            sel_block   = 1;
        }
        if (b_row > (2 * block_size - 1) && b_row < 3 * block_size)
        {
            b_row_block = b_row - 2 * block_size;
            sel_block   = 2;
        }
        if (b_row > (3 * block_size - 1))
        {
            b_row_block = b_row - 3 * block_size;
            sel_block   = 3;
        }

        QTYPE b_val1 = b_block1[b_row_block][j];
        QTYPE b_val2 = b_block2[b_row_block][j];
        QTYPE b_val3 = b_block3[b_row_block][j];
        QTYPE b_val4 = b_block4[b_row_block][j];

        QTYPE b_val;
        switch (sel_block)
        {
            case 0:  b_val = b_val1;  break;
            case 1:  b_val = b_val2;  break;
            case 2:  b_val = b_val3;  break;
            case 3:  b_val = b_val4;  break;
        }

        ITYPE a_val_i = (ITYPE)a_val;
        ITYPE b_val_i = (ITYPE)b_val;

        acc[j] += a_val_i * b_val_i;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_int_fea - Single-multiply fixed-point inner kernel for the feature SpMM (GNN branch).
 *
 * For a single (feature value, row index) pair, computes:
 *   acc[j] = a_value × b_block[b_row][j]   for all j in [0, B_WIDTH_BLOCK)
 *
 * b_row indexes directly into the single weight tile; no partition selection
 * is needed because the caller selects the correct partition before calling.
 *
 * Binary weight mode (qbits == 1):
 *   b_val == 0 → b_val_i = +0.5
 *   b_val == 1 → b_val_i = -0.5
 *
 * acc[] is written (not accumulated) — accumulation is performed by the wrapper.
 *
 * @param a_value        Feature non-zero value.
 * @param b_block        Weight tile [B_HEIGHT/4][B_WIDTH_BLOCK].
 * @param b_row          Row index into b_block.
 * @param zero_point_lhs Quantization zero point (unused).
 * @param zero_point_rhs Quantization zero point (unused).
 * @param acc            Output: product results [B_WIDTH_BLOCK].
 */
void dsp_kernel_int_fea(
    FTYPE      a_value,
    BTYPE      b_block[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
)
{
    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        FTYPE a_val = a_value;
        BTYPE b_val = b_block[b_row][j];

        ITYPE a_val_i = (ITYPE)a_val;
        ITYPE b_val_i;

#if (qbits == 1)
        b_val_i = (b_val == 0) ? ITYPE(0.5) : ITYPE(-0.5);
#else
        b_val_i = (ITYPE)b_val;
#endif

        acc[j] = a_val_i * b_val_i;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_int_lin - Single-multiply fixed-point inner kernel for the linear-projection branch.
 *
 * Identical to dsp_kernel_int_fea but uses LTYPE/BLTYPE (linear-branch precision).
 * No binary-weight special case (linear branch always uses multi-bit weights).
 *
 * acc[] is written (not accumulated) — accumulation is performed by the wrapper.
 *
 * @param a_value        Linear-branch feature non-zero value (LTYPE).
 * @param b_block        Linear-projection weight tile [B_HEIGHT/4][B_WIDTH_BLOCK].
 * @param b_row          Row index into b_block.
 * @param zero_point_lhs Quantization zero point (unused).
 * @param zero_point_rhs Quantization zero point (unused).
 * @param acc            Output: product results [B_WIDTH_BLOCK].
 */
void dsp_kernel_int_lin(
    LTYPE      a_value,
    BLTYPE     b_block[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
)
{
    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        LTYPE  a_val = a_value;
        BLTYPE b_val = b_block[b_row][j];

        ITYPE a_val_i = (ITYPE)a_val;
        ITYPE b_val_i = (ITYPE)b_val;

        acc[j] = a_val_i * b_val_i;
    }
}
// =============================================================================================
// =============================================================================================
/**
 * writec - Dequantize and forward SpMM results to the writeout stage.
 *
 * For each output row i and active column j (j < P[B_index]):
 *   1. Reads the GNN accumulator value from write_fifo[j]    (when gcn_path == 1).
 *   2. Reads the linear residual from linear_pipo[i][j]      (when LINEAR_ENABLE and linear_mode).
 *   3. Dequantizes:
 *        output = C_out * deq_factor[B_index] + residual * deq_factor[B_index]
 *   4. Writes the result to the CS output stream for writeout.
 *
 * Columns j >= P[B_index] are skipped (sparse output: only the active
 * output width is forwarded downstream).
 *
 * @param deq_factor   Per-layer dequantization scale factors.
 * @param model        Per-layer mode flags [layer][bit].
 * @param first_row    First row index (unused in body; kept for symmetry).
 * @param row_count    Number of rows to process.
 * @param N_adj        Total adjacency rows (unused in body; kept for symmetry).
 * @param P            Per-layer output column widths.
 * @param write_fifo   Input FIFOs: GNN accumulator values [B_WIDTH_BLOCK].
 * @param linear_pipo  Input tile: linear-projection residuals [B_HEIGHT][B_WIDTH_BLOCK].
 * @param CS           Output stream: dequantized results → writeout.
 * @param B_index      Current layer index.
 * @param layer_loop   Total number of layer iterations (unused in body).
 */
void writec(
    float               deq_factor[5],
    ap_uint<1>          model[5][8],
    int                 first_row,
    int                 row_count,
    int                 N_adj,
    ap_uint<8>          P[5],
    hls::stream<ITYPE>  write_fifo[B_WIDTH_BLOCK],
    QLTYPE              linear_pipo[B_HEIGHT][B_WIDTH_BLOCK],
    hls::stream<OUTTYPE> &CS,
    int                 B_index,
    int                 layer_loop
)
{
    bool linear_mode = model[B_index][6];
    bool sage_mode   = model[B_index][7];
    bool gcn_path    = !(linear_mode ^ sage_mode);

    LOOP_WRITE42: for (int i = 0; i < row_count; i++)
    {
        LOOP_WRITE52: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS PIPELINE II=1

            DTYPE C_out = DTYPE(0.0);
            DTYPE residual;

            if (gcn_path == 1)
                C_out = DTYPE(write_fifo[j].read());

#if LINEAR_ENABLE == 1
            residual = (linear_mode == 1) ? DTYPE(linear_pipo[i][j]) : DTYPE(0.0);
#else
            residual = DTYPE(0.0);
#endif

#if (INT_DEQUANT == 1)
            OUTTYPE C_float = (OUTTYPE)C_out    * deq_factor[B_index]
                            + (OUTTYPE)residual * deq_factor[B_index];
#else
            OUTTYPE C_float = (OUTTYPE)C_out;
#endif

            /* Emit only active output columns */
            if (j < P[B_index])
                CS.write(C_float);
        }
    }
}


// =============================================================================================
// =============================================================================================
/**
 * writeout - Write dequantized adjacency outputs to DDR or AXI-stream.
 *
 * Reads dequantized values from write_fifo and routes them based on stream_mode:
 *
 *   stream_mode == 1 (AXI-stream output):
 *     Writes values to CS (value stream).
 *     When gemm_mode for the next layer == 0 (sparse), also writes row/column
 *     indices to CSR and CSC streams (for downstream COO reconstruction).
 *     TLAST is asserted on the last element.
 *
 *   stream_mode == 0 (DDR output):
 *     Writes to C[i * B_WIDTH_INT + j + first_row * B_WIDTH_BLOCK].
 *
 * Note: the TLAST condition  i*j == (WL-1)*(B_WIDTH_INT-1)  is taken directly
 * from the original code and preserved intentionally.
 *
 * @param model       Per-layer mode flags [layer][bit]:
 *                      bit 2 of B_index   = stream_mode (output to AXI-stream).
 *                      bit 1 of B_index+1 = gemm_mode of the next layer.
 * @param first_row   First row offset for DDR addressing.
 * @param row_count   Number of rows to write.
 * @param N_adj       Total adjacency rows (unused in body; kept for symmetry).
 * @param P           Per-layer active output column widths.
 * @param write_fifo  Input stream: dequantized values from writec.
 * @param C           DDR output array (stream_mode == 0).
 * @param CS          AXI-stream: output values (stream_mode == 1).
 * @param CSR         AXI-stream: row indices for sparse output (stream_mode == 1, sparse next layer).
 * @param CSC         AXI-stream: column indices for sparse output.
 * @param B_index     Current layer index.
 * @param layer_loop  Total number of layer iterations (unused in body).
 */
void writeout(
    ap_uint<1>           model[5][8],
    int                  first_row,
    int                  row_count,
    int                  N_adj,
    ap_uint<8>           P[5],
    hls::stream<OUTTYPE> &write_fifo,
    OUTTYPE             *C,
    hls::stream<ASTYPE>  &CS,
    hls::stream<ASTYPE>  &CSR,
    hls::stream<ASTYPE>  &CSC,
    int                  B_index,
    int                  layer_loop
)
{
    int        B_WIDTH_INT = P[B_index];
    int        WL          = row_count;
    ap_uint<1> stream_mode = model[B_index][2];
    ap_uint<1> next_gemm   = model[B_index + 1][1];   // next layer's mode

    if (stream_mode == 1)
    {
        /* ── AXI-stream output ── */
        bool last = 0;

        LOOP_WRITE42: for (int i = 0; i < WL; i++)
        {
            LOOP_WRITE52: for (int j = 0; j < B_WIDTH_INT; j++)
            {
                #pragma HLS PIPELINE II=1

                /* Assert TLAST on the final element */
                if (i * j == (WL - 1) * (B_WIDTH_INT - 1))
                    last = 1;

                OUTTYPE  C_float = OUTTYPE(write_fifo.read());
                fp_int   C_int;
                C_int.f  = C_float;

                ASTYPE temp;
                temp.data = C_int.i;
                temp.last = last;

                if (next_gemm == 1)
                {
                    /* Dense next layer: value stream only */
                    CS.write(temp);
                }
                else
                {
                    /* Sparse next layer: emit value + row + column index streams.
                     * Zero values are suppressed except: first element of each row (j==0)
                     * and the last element (last==1). */
                    if (j == 0 || C_float != 0 || last == 1)
                    {
                        CS.write(temp);

                        temp.data = i;
                        CSR.write(temp);

                        temp.data = j;
                        CSC.write(temp);
                    }
                }
            }
        }
    }
    else
    {
        /* ── DDR output ── */
        LOOP_WRITE45: for (int i = 0; i < WL; i++)
        {
            LOOP_WRITE55: for (int j = 0; j < B_WIDTH_INT; j++)
            {
                #pragma HLS PIPELINE II=1
                OUTTYPE C_float = OUTTYPE(write_fifo.read());
                C[i * B_WIDTH_INT + j + first_row * B_WIDTH_BLOCK] = C_float;
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * writec_transpose - Write transposed SpMM output to DDR or AXI-stream.
 *
 * Processes output in tiles of FIFO_DEPTH rows to amortize inner loop overhead.
 * For each (j, z) pair writes:
 *   C[i * FIFO_DEPTH + j * WL + z + first_row * B_WIDTH_BLOCK
 *     + B_index * N_adj * B_WIDTH_BLOCK] = dequantized value
 *
 * When STREAM_MODE_OUT == 1, writes to the AXI-stream CS instead of DDR.
 *
 * @param deq_factor  Scalar dequantization factor.
 * @param stream_mode Unused (routing controlled by STREAM_MODE_OUT macro).
 * @param first_row   Row offset for DDR addressing.
 * @param row_count   Number of rows.
 * @param N_adj       Total adjacency rows (used in DDR address formula).
 * @param P           Active output column width (unused in body; kept for symmetry).
 * @param write_fifo  Input FIFOs: accumulator values [B_WIDTH_BLOCK].
 * @param C           DDR output array.
 * @param CS          AXI-stream output (STREAM_MODE_OUT == 1).
 * @param B_index     Current layer index (used in DDR address formula).
 */
void writec_transpose(
    float               deq_factor,
    bool                stream_mode,
    int                 first_row,
    int                 row_count,
    int                 N_adj,
    int                 P,
    hls::stream<ITYPE>  write_fifo[B_WIDTH_BLOCK],
    OUTTYPE            *C,
    hls::stream<ASTYPE> &CS,
    int                 B_index
)
{
    int WL = row_count;

    LOOP_WRITE4: for (int i = 0; i < WL; i += FIFO_DEPTH)
    {
        LOOP_WRITE5: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            LOOP_WRITE6: for (int z = 0; z < FIFO_DEPTH; z++)
            {
                #pragma HLS PIPELINE II=1

                DTYPE C_out;
                if ((i + z) < WL)
                    C_out = DTYPE(write_fifo[j].read());
                else
                    C_out = 0.0;

#if (INT_DEQUANT == 1)
                OUTTYPE C_float = (OUTTYPE)C_out * deq_factor;
#else
                OUTTYPE C_float = (OUTTYPE)C_out;
#endif

#if (STREAM_MODE_OUT == 1)
                ASTYPE temp;
                temp.data = C_float;
                CS.write(temp);
#else
                C[i * FIFO_DEPTH + j * WL + z
                  + first_row * B_WIDTH_BLOCK
                  + B_index * N_adj * B_WIDTH_BLOCK] = C_float;
#endif
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * writes - Write edge-attention scores to DDR.
 *
 * Active only when gcn_path == 1 AND gat_mode == 1.
 * Reads the total non-zero count from rnnz_fifo, then drains write_fifo
 * and writes each dequantized value to C[i].
 *
 * @param deq_factor  Per-layer dequantization scale factors.
 * @param model       Per-layer mode flags [layer][bit].
 * @param first_row   First row offset (unused in body; kept for symmetry).
 * @param row_count   Number of rows (unused in body; kept for symmetry).
 * @param N_adj       Total adjacency rows (unused in body).
 * @param P           Per-layer output column widths (unused in body).
 * @param write_fifo  Input stream: attention scores (TTYPE).
 * @param rnnz_fifo   Input FIFO: total non-zeros to write.
 * @param C           DDR output array: edge scores.
 * @param B_index     Current layer index.
 */
void writes(
    float               deq_factor[5],
    ap_uint<1>          model[5][8],
    int                 first_row,
    int                 row_count,
    int                 N_adj,
    ap_uint<8>          P[5],
    hls::stream<TTYPE>  &write_fifo,
    hls::stream<int>    &rnnz_fifo,
    OUTTYPE             *C,
    int                  B_index
)
{
    bool linear_mode = model[B_index][6];
    bool gat_mode    = model[B_index][5];
    bool sage_mode   = model[B_index][7];
    bool gcn_path    = !(linear_mode ^ sage_mode);

    if (gcn_path == 1 && gat_mode == 1)
    {
        int rnnz = rnnz_fifo.read();

        LOOP_WRITE5: for (int i = 0; i < rnnz; i++)
        {
            #pragma HLS PIPELINE
            DTYPE   C_out = write_fifo.read();
#if (INT_DEQUANT == 1)
            OUTTYPE C_float = (OUTTYPE)C_out * deq_factor[B_index];
#else
            OUTTYPE C_float = (OUTTYPE)C_out;
#endif
            C[i] = C_float;
        }
    }
}
// =============================================================================================
// =============================================================================================
/**
 * writesx4 - Write attention scores or softmax values for four row partitions to DDR.
 *
 * Reads total non-zero counts from rnnz_fifo1..4, then for each partition
 * drains the corresponding write_fifo and writes the values contiguously
 * into the DDR array C at offset rnnz_total (which accumulates across partitions).
 *
 * Optional dequantization (INT_DEQUANT): output = (OUTTYPE)val * deq_factor.
 *
 * The function body executes only when gat_mode == 1; non-GAT layers skip it.
 *
 * Note: write_fifo3 uses the INT_QUANT guard instead of INT_DEQUANT — this
 * preserves the original behavior and is kept intentionally.
 *
 * @param deq_factor     Scalar dequantization factor applied to each output value.
 * @param gat_mode       True when GAT attention is active; no output otherwise.
 * @param row_count1..4  Number of rows in each partition (currently unused; retained for symmetry).
 * @param write_fifo1..4 Input FIFOs: attention/softmax values for each partition.
 * @param rnnz_fifo1..4  Input FIFOs: total non-zeros for each partition.
 * @param C              DDR output array (edge scores or softmax values).
 * @param B_index        Current layer index (currently unused; retained for symmetry).
 */
void writesx4(
    float                deq_factor,
    bool                 gat_mode,
    int                  row_count1,
    int                  row_count2,
    int                  row_count3,
    int                  row_count4,
    hls::stream<TTYPE>  &write_fifo1,
    hls::stream<TTYPE>  &write_fifo2,
    hls::stream<TTYPE>  &write_fifo3,
    hls::stream<TTYPE>  &write_fifo4,
    hls::stream<int>    &rnnz_fifo1,
    hls::stream<int>    &rnnz_fifo2,
    hls::stream<int>    &rnnz_fifo3,
    hls::stream<int>    &rnnz_fifo4,
    OUTTYPE             *C,
    int                  B_index
)
{
    if (gat_mode == 1)
    {
        /* Read total nnz for all four partitions up front */
        int rnnz1 = rnnz_fifo1.read();
        int rnnz2 = rnnz_fifo2.read();
        int rnnz3 = rnnz_fifo3.read();
        int rnnz4 = rnnz_fifo4.read();

        int rnnz_total = 0;   // running DDR write offset

        /* ── Partition 1 ── */
        LOOP_WRITE51: for (int j = 0; j < rnnz1; j++)
        {
            #pragma HLS PIPELINE
            DTYPE   val = write_fifo1.read();
#if (INT_DEQUANT == 1)
            OUTTYPE out = (OUTTYPE)val * deq_factor;
#else
            OUTTYPE out = (OUTTYPE)val;
#endif
            C[j + rnnz_total] = out;
        }
        rnnz_total += rnnz1;

        /* ── Partition 2 ── */
        LOOP_WRITE52: for (int j = 0; j < rnnz2; j++)
        {
            #pragma HLS PIPELINE
            DTYPE   val = write_fifo2.read();
#if (INT_DEQUANT == 1)
            OUTTYPE out = (OUTTYPE)val * deq_factor;
#else
            OUTTYPE out = (OUTTYPE)val;
#endif
            C[j + rnnz_total] = out;
        }
        rnnz_total += rnnz2;

        /* ── Partition 3 (uses INT_QUANT guard — preserves original behavior) ── */
        LOOP_WRITE53: for (int j = 0; j < rnnz3; j++)
        {
            #pragma HLS PIPELINE
            DTYPE   val = write_fifo3.read();
#if (INT_QUANT == 1)
            OUTTYPE out = (OUTTYPE)val * deq_factor;
#else
            OUTTYPE out = (OUTTYPE)val;
#endif
            C[j + rnnz_total] = out;
        }
        rnnz_total += rnnz3;

        /* ── Partition 4 ── */
        LOOP_WRITE54: for (int j = 0; j < rnnz4; j++)
        {
            #pragma HLS PIPELINE
            DTYPE   val = write_fifo4.read();
#if (INT_DEQUANT == 1)
            OUTTYPE out = (OUTTYPE)val * deq_factor;
#else
            OUTTYPE out = (OUTTYPE)val;
#endif
            C[j + rnnz_total] = out;
        }
        rnnz_total += rnnz4;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * readptr_csr_fea - Emit per-row non-zero counts for the feature CSR matrix.
 *
 * SpMM (gemm_mode == 0): computes nnz[i] = rowPtr[i+1] - rowPtr[i].
 * GEMM (gemm_mode == 1): all rows have exactly M non-zeros.
 *
 * @param gemm_mode  True = dense GEMM mode.
 * @param N          Number of rows.
 * @param M          Number of columns (nnz per row in GEMM mode).
 * @param rowPtr     DDR: CSR row pointer array.
 * @param rnnz_fifo  Output FIFO: per-row nnz counts.
 */
void readptr_csr_fea(
    bool             gemm_mode,
    int              N,
    int              M,
    int             *rowPtr,
    hls::stream<int> &rnnz_fifo
)
{
    #pragma HLS INLINE OFF

    int current_index = rowPtr[0];

    if (gemm_mode == 0)
    {
        LOOP_A_INDEX_SPMM1: for (int A_index = 0; A_index < N; A_index++)
        {
            #pragma HLS PIPELINE
            int next_index = rowPtr[A_index + 1];
            rnnz_fifo      << (next_index - current_index);
            current_index  = next_index;
        }
    }
    else
    {
        LOOP_A_INDEX_SPMM2: for (int A_index = 0; A_index < N; A_index++)
        {
            #pragma HLS PIPELINE
            rnnz_fifo << M;
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * read_ptr2 - Load COO row indices from DDR into a FIFO (adjacency path).
 *
 * Reads nnz_fea + 1 entries from rowPtr (COO row index array) and pushes
 * them into index_fifo for downstream processing by proc_ptr.
 * The extra entry (+1) is required by proc_ptr to detect the final row boundary.
 *
 * @param nnz_fea    Total number of non-zeros.
 * @param rowPtr     DDR: COO row index array (length ≥ nnz_fea + 1).
 * @param index_fifo Output FIFO: raw row indices.
 */
void read_ptr2(
    int              nnz_fea,
    int             *rowPtr,
    hls::stream<int> &index_fifo
)
{
    LOOP_A_INDEX0: for (int A_index = 0; A_index < nnz_fea + 1; A_index++)
    {
        #pragma HLS PIPELINE
        index_fifo << rowPtr[A_index];
    }
}

// =============================================================================================
// =============================================================================================
/**
 * read_ptr - Load COO row indices from DDR into a FIFO (feature path).
 *
 * Identical to read_ptr2 but gated by stream_mode:
 *   stream_mode == 0: reads from DDR and pushes into index_fifo.
 *   stream_mode == 1: no DDR read; row indices will arrive via AXI-stream
 *                     and are handled directly by proc_ptr2.
 *
 * @param stream_mode True when row indices come from AXI-stream (skip DDR read).
 * @param nnz_fea     Total number of non-zeros.
 * @param rowPtr      DDR: COO row index array (length ≥ nnz_fea + 1).
 * @param index_fifo  Output FIFO: raw row indices (stream_mode == 0 only).
 */
void read_ptr(
    bool             stream_mode,
    int              nnz_fea,
    int             *rowPtr,
    hls::stream<int> &index_fifo
)
{
    if (stream_mode == 0)
    {
        LOOP_A_INDEX0: for (int A_index = 0; A_index < nnz_fea + 1; A_index++)
        {
            #pragma HLS PIPELINE
            index_fifo << rowPtr[A_index];
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * proc_ptr - Convert a flat COO row-index FIFO into per-row non-zero counts.
 *
 * Reads nnz_fea row indices from index_fifo, groups consecutive identical
 * indices, and emits the count (nnz per row) to rnnz_fifo.
 *
 * Algorithm:
 *   1. Prime with the first index; rnnz = 1.
 *   2. For each subsequent index:
 *        - Same as current  → increment rnnz.
 *        - Different        → emit rnnz, reset to 1, advance current_index.
 *   3. After the loop emit the final rnnz.
 *   4. Read and discard the trailing padding token (nnz_fea + 1 th entry
 *      written by read_ptr2 / read_ptr to satisfy the loop bound).
 *
 * @param nnz_fea    Total number of non-zeros (loop bound = nnz_fea - 1).
 * @param index_fifo Input FIFO: flat COO row indices.
 * @param rnnz_fifo  Output FIFO: per-row non-zero counts.
 */
void proc_ptr(
    int              nnz_fea,
    hls::stream<int> &index_fifo,
    hls::stream<int> &rnnz_fifo
)
{
    int next_index;
    int current_index = index_fifo.read();
    int rnnz          = 1;
    int loop_idx      = 0;

    LOOP_A_INDEX1: while (loop_idx < nnz_fea - 1)
    {
        #pragma HLS PIPELINE

        next_index = index_fifo.read();
        loop_idx++;

        if (next_index == current_index)
        {
            rnnz++;
        }
        else
        {
            rnnz_fifo     << rnnz;
            current_index  = next_index;
            rnnz           = 1;
        }
    }

    /* Emit count for the final row */
    rnnz_fifo << rnnz;

    /* Drain the trailing padding token written by read_ptr / read_ptr2 */
    index_fifo.read();
}

// =============================================================================================
// =============================================================================================
/**
 * proc_ptr2 - Convert a flat COO row-index stream into per-row non-zero counts.
 *
 * Reads a sequence of row indices (from either a DDR FIFO or an AXI-stream),
 * groups consecutive identical indices, and emits the count (nnz per row) to
 * rnnz_fifo and/or rnnz_fifo_sage depending on the active path flags.
 *
 * Two source modes:
 *
 *   DDR    (stream_mode == 0):
 *     Row indices arrive via index_fifo (read from DDR by read_ptr).
 *     Loops for exactly nnz_fea - 1 comparisons, then emits the final count.
 *     A trailing read drains any padding token from the FIFO.
 *
 *   AXI-stream (stream_mode == 1):
 *     Row indices arrive via rowPtrs (AXI-stream with TLAST).
 *     Loop terminates when temp.last == 1.
 *
 * Output routing:
 *   gcn_path  == 1 → rnnz_fifo.
 *   linear_mode == 1 → rnnz_fifo_sage  (LINEAR_ENABLE guard).
 *
 * @param gcn_path       True when the GNN aggregation path is active.
 * @param linear_mode    True when the linear projection path is active.
 * @param stream_mode    True when row indices come from AXI-stream (not DDR FIFO).
 * @param nnz_fea        Total number of non-zeros (DDR mode: loop bound).
 * @param index_fifo     Input FIFO: row indices from DDR (stream_mode == 0).
 * @param rowPtrs        AXI-stream: row indices with TLAST (stream_mode == 1).
 * @param rnnz_fifo      Output FIFO: per-row nnz counts for the GNN branch.
 * @param rnnz_fifo_sage Output FIFO: per-row nnz counts for the linear branch.
 */
void proc_ptr2(
    bool                 gcn_path,
    bool                 linear_mode,
    bool                 stream_mode,
    int                  nnz_fea,
    hls::stream<int>    &index_fifo,
    hls::stream<ASTYPE> &rowPtrs,
    hls::stream<int>    &rnnz_fifo,
    hls::stream<int>    &rnnz_fifo_sage
)
{
    int    next_index;
    int    rnnz          = 0;
    int    current_index = 0;
    int    loop_idx      = 0;
    ASTYPE temp;

    if (stream_mode == 0)
    {
        /* ── DDR mode: row indices arrive via index_fifo ── */
        current_index = index_fifo.read();
        rnnz          = 1;

        LOOP_A_INDEX1: while (loop_idx < nnz_fea - 1)
        {
            #pragma HLS PIPELINE

            next_index = index_fifo.read();
            loop_idx++;

            if (next_index == current_index)
            {
                rnnz++;
            }
            else
            {
                /* Row boundary: emit count for completed row */
                if (gcn_path == 1)
                    rnnz_fifo << rnnz;

#if (LINEAR_ENABLE == 1)
                if (linear_mode == 1)
                    rnnz_fifo_sage << rnnz;
#endif

                current_index = next_index;
                rnnz          = 1;
            }
        }

        /* Emit count for the final row */
        if (gcn_path == 1)
            rnnz_fifo << rnnz;

#if (LINEAR_ENABLE == 1)
        if (linear_mode == 1)
            rnnz_fifo_sage << rnnz;
#endif

        index_fifo.read();   // drain trailing padding token
    }
    else
    {
        /* ── AXI-stream mode: row indices arrive with TLAST ── */
        temp          = rowPtrs.read();
        rnnz          = 1;
        current_index = temp.data;

        if (temp.last != 1)
        {
            LOOP_A_INDEX2: do
            {
                #pragma HLS PIPELINE

                temp       = rowPtrs.read();
                next_index = temp.data;

                if (next_index == current_index)
                {
                    rnnz++;
                }
                else
                {
                    if (gcn_path == 1)
                        rnnz_fifo << rnnz;

#if (LINEAR_ENABLE == 1)
                    if (linear_mode == 1)
                        rnnz_fifo_sage << rnnz;
#endif

                    current_index = next_index;
                    rnnz          = 1;
                }

            } while (temp.last != 1);
        }

        /* Emit count for the final row */
        if (gcn_path == 1)
            rnnz_fifo << rnnz;

#if (LINEAR_ENABLE == 1)
        if (linear_mode == 1)
            rnnz_fifo_sage << rnnz;
#endif
    }
}

// =============================================================================================
// =============================================================================================
/**
 * read_dataflow2 - Feature COO row-pointer dataflow pipeline.
 *
 * Orchestrates two pipelined tasks for COO feature row-pointer processing:
 *   1. read_ptr   – reads raw row indices from DDR into index_fifo.
 *   2. proc_ptr2  – converts the index stream into per-row nnz counts.
 *
 * @param gcn_path       True when the GNN aggregation path is active.
 * @param linear_mode    True when the linear projection path is active.
 * @param stream_mode    True when row indices come from AXI-stream.
 * @param nnz_fea        Total number of non-zeros.
 * @param rowPtr         DDR: COO row index array.
 * @param rowPtrs        AXI-stream: row indices (stream_mode == 1).
 * @param rnnz_fifo      Output FIFO: per-row nnz counts (GNN branch).
 * @param rnnz_fifo_sage Output FIFO: per-row nnz counts (linear branch).
 */
void read_dataflow2(
    bool                 gcn_path,
    bool                 linear_mode,
    bool                 stream_mode,
    int                  nnz_fea,
    int                 *rowPtr,
    hls::stream<ASTYPE> &rowPtrs,
    hls::stream<int>    &rnnz_fifo,
    hls::stream<int>    &rnnz_fifo_sage
)
{
    hls::stream<int> index_fifo("index fifo");
    #pragma HLS STREAM variable=index_fifo depth=FIFO_DEPTH

    #pragma HLS DATAFLOW
    read_ptr(stream_mode, nnz_fea, rowPtr, index_fifo);
    proc_ptr2(gcn_path, linear_mode, stream_mode, nnz_fea,
              index_fifo, rowPtrs, rnnz_fifo, rnnz_fifo_sage);
}

// =============================================================================================
// =============================================================================================
/**
 * read_dataflow - Adjacency COO row-pointer dataflow pipeline.
 *
 * Orchestrates two pipelined tasks for COO adjacency row-pointer processing:
 *   1. read_ptr2 – reads raw row indices from DDR into index_fifo.
 *   2. proc_ptr  – converts the index stream into per-row nnz counts.
 *
 * @param nnz_fea    Total number of non-zeros.
 * @param rowPtr     DDR: COO row index array.
 * @param rnnz_fifo  Output FIFO: per-row nnz counts.
 */
void read_dataflow(
    int              nnz_fea,
    int             *rowPtr,
    hls::stream<int> &rnnz_fifo
)
{
    hls::stream<int> index_fifo("index fifo");
    #pragma HLS STREAM variable=index_fifo depth=FIFO_DEPTH

    #pragma HLS DATAFLOW
    read_ptr2(nnz_fea, rowPtr, index_fifo);
    proc_ptr(nnz_fea, index_fifo, rnnz_fifo);
}

// =============================================================================================
// =============================================================================================
/**
 * readptr_coo_fea - Emit per-row non-zero counts for the feature COO matrix.
 *
 * Dispatches to either:
 *   SpMM (gemm_mode == 0): calls read_dataflow2 to decode COO row indices.
 *   GEMM (gemm_mode == 1): each row has exactly M non-zeros; emits M per row.
 *
 * Output is routed to rnnz_fifo (gcn_path) and/or rnnz_fifo_sage (linear_mode).
 *
 * @param nnz_fea        Total non-zeros (SpMM mode).
 * @param sage_mode      Layer is a SAGE aggregation layer.
 * @param linear_mode    Layer has a linear projection.
 * @param stream_mode    Row pointer source is AXI-stream.
 * @param gemm_mode      True = dense GEMM mode.
 * @param N              Number of rows.
 * @param M              Number of columns (nnz per row in GEMM mode).
 * @param rowPtr         DDR: COO row index array.
 * @param rowPtrs        AXI-stream: row indices (stream_mode == 1).
 * @param rnnz_fifo      Output FIFO: per-row nnz counts (GNN branch).
 * @param rnnz_fifo_sage Output FIFO: per-row nnz counts (linear branch).
 */
void readptr_coo_fea(
    int                  nnz_fea,
    bool                 sage_mode,
    bool                 linear_mode,
    bool                 stream_mode,
    bool                 gemm_mode,
    int                  N,
    int                  M,
    int                 *rowPtr,
    hls::stream<ASTYPE> &rowPtrs,
    hls::stream<int>    &rnnz_fifo,
    hls::stream<int>    &rnnz_fifo_sage
)
{
    #pragma HLS INLINE OFF

    bool gcn_path = !(linear_mode ^ sage_mode);

    if (gemm_mode == 0)
    {
        /* SpMM: decode per-row nnz from COO row indices */
        read_dataflow2(gcn_path, linear_mode, stream_mode, nnz_fea,
                       rowPtr, rowPtrs, rnnz_fifo, rnnz_fifo_sage);
    }
    else
    {
        /* GEMM: all rows have exactly M non-zeros */
        LOOP_A_INDEX2: for (int A_index = 0; A_index < N; A_index++)
        {
            #pragma HLS PIPELINE

            if (gcn_path == 1)
                rnnz_fifo << M;

#if (LINEAR_ENABLE == 1)
            if (linear_mode == 1)
                rnnz_fifo_sage << M;
#endif
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * readptr_csr_adj - Emit per-row non-zero counts for the adjacency CSR matrix.
 *
 * SpMM (gemm_mode == 0): computes nnz per row as rowPtr[i+1] - rowPtr[i].
 * GEMM (gemm_mode == 1): each row has exactly M non-zeros.
 *
 * @param gemm_mode  True = dense GEMM mode.
 * @param N          Number of rows.
 * @param M          Number of columns (nnz per row in GEMM mode).
 * @param rowPtr     DDR: CSR row pointer array.
 * @param rnnz_fifo  Output FIFO: per-row nnz counts.
 */
void readptr_csr_adj(
    bool             gemm_mode,
    int              N,
    int              M,
    int             *rowPtr,
    hls::stream<int> &rnnz_fifo
)
{
    #pragma HLS INLINE OFF

    int current_index = rowPtr[0];

    if (gemm_mode == 0)
    {
        /* SpMM: delta between consecutive row pointers = nnz for that row */
        LOOP_A_INDEX_SPMM1: for (int A_index = 0; A_index < N; A_index++)
        {
            int next_index = rowPtr[A_index + 1];
            int rnnz       = next_index - current_index;
            current_index  = next_index;
            rnnz_fifo      << rnnz;
        }
    }
    else
    {
        /* GEMM: fixed nnz = M per row */
        LOOP_A_INDEX_SPMM2: for (int A_index = 0; A_index < N; A_index++)
        {
            #pragma HLS PIPELINE
            rnnz_fifo << M;
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * readptr_coo_adj - Emit per-row non-zero counts for the adjacency COO matrix.
 *
 * Skipped entirely when gcn_path == 0 (linear-only layer).
 *
 * SpMM (gemm_mode == 0): calls read_dataflow to decode COO row indices.
 * GEMM (gemm_mode == 1): each row has exactly M non-zeros.
 *
 * @param nnz_adj    Total non-zeros in this partition.
 * @param sage_mode  Layer is a SAGE aggregation layer.
 * @param linear_mode Layer has a linear projection.
 * @param gemm_mode  True = dense GEMM mode.
 * @param N          Number of rows.
 * @param M          Number of columns (nnz per row in GEMM mode).
 * @param rowPtr     DDR: COO row index array.
 * @param rnnz_fifo  Output FIFO: per-row nnz counts.
 */
void readptr_coo_adj(
    int              nnz_adj,
    bool             sage_mode,
    bool             linear_mode,
    bool             gemm_mode,
    int              N,
    int              M,
    int             *rowPtr,
    hls::stream<int> &rnnz_fifo
)
{
    #pragma HLS INLINE OFF

    bool gcn_path = !(linear_mode ^ sage_mode);

    if (gcn_path == 1)
    {
        if (gemm_mode == 0)
        {
            read_dataflow(nnz_adj, rowPtr, rnnz_fifo);
        }
        else
        {
            LOOP_A_INDEX2: for (int A_index = 0; A_index < N; A_index++)
            {
                #pragma HLS PIPELINE
                rnnz_fifo << M;
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * readval_csr_adj - Stream CSR adjacency values and column indices (ATYPE output).
 *
 * Reads last_index non-zeros from DDR and emits them into A_fifo (ATYPE) and
 * col_indices_fifo, optionally applying fixed-point quantization (INT_QUANT_A).
 *
 * Two modes:
 *   SpMM (gemm_mode == 0): column index read from columnIndex[j].
 *   GEMM (gemm_mode == 1): column index is a running counter [0 .. ccount-1].
 *
 * @param beta_qu               Zero-point shift for quantization.
 * @param f_align               Fractional alignment bits.
 * @param quantization_scale_fea Adjacency value scale factor.
 * @param gemm_mode             True = dense GEMM mode.
 * @param ccount                Number of columns (counter wrap for GEMM mode).
 * @param last_index            Total non-zeros to read.
 * @param A_fifo                Output FIFO: quantized adjacency values (ATYPE).
 * @param col_indices_fifo      Output FIFO: column indices.
 * @param values                DDR: adjacency value array.
 * @param columnIndex           DDR: adjacency column index array (SpMM mode).
 */
void readval_csr_adj(
    int                  beta_qu,
    int                  f_align,
    float                quantization_scale_fea,
    bool                 gemm_mode,
    int                  ccount,
    int                  last_index,
    hls::stream<ATYPE>  &A_fifo,
    hls::stream<int>    &col_indices_fifo,
    INTYPE              *values,
    int                 *columnIndex
)
{
    #pragma HLS INLINE OFF

    if (gemm_mode == 0)
    {
        /* ── SpMM: explicit column indices from DDR ── */
        LOOP_J_SPMM: for (int j = 0; j < last_index; j++)
        {
            #pragma HLS PIPELINE

            INTYPE raw_val   = (INTYPE)values[j];
            ATYPE  quant_val;

#if (INT_QUANT_A == 1)
            quanta(quant_val, raw_val, quantization_scale_fea, f_align, beta_qu);
#else
            quant_val = raw_val;
#endif

            A_fifo           << quant_val;
            col_indices_fifo << columnIndex[j];
        }
    }
    else
    {
        /* ── GEMM: column index = running counter ── */
        int col = 0;

        LOOP_J_SPMM2: for (int j = 0; j < last_index; j++)
        {
            #pragma HLS PIPELINE

            INTYPE raw_val   = (INTYPE)values[j];
            ATYPE  quant_val;

#if (INT_QUANT_A == 1)
            quanta(quant_val, raw_val, quantization_scale_fea, f_align, beta_qu);
#else
            quant_val = raw_val;
#endif

            A_fifo           << quant_val;
            col_indices_fifo << col;

            col = (col == (ccount - 1)) ? 0 : col + 1;
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * readval_coo_adj - Stream COO adjacency values and column indices (ATYPE output).
 *
 * Identical to readval_csr_adj but gated by gcn_path:
 *   gcn_path = !(linear_mode ^ sage_mode)
 * No output is produced for linear-only layers.
 *
 * @param beta_qu               Zero-point shift for quantization.
 * @param f_align               Fractional alignment bits.
 * @param quantization_scale_fea Adjacency value scale factor.
 * @param sage_mode             Layer is a SAGE aggregation layer.
 * @param linear_mode           Layer has an active linear projection.
 * @param gemm_mode             True = dense GEMM mode.
 * @param ccount                Number of columns (counter wrap for GEMM mode).
 * @param last_index            Total non-zeros to read.
 * @param A_fifo                Output FIFO: quantized adjacency values (ATYPE).
 * @param col_indices_fifo      Output FIFO: column indices.
 * @param values                DDR: adjacency value array.
 * @param columnIndex           DDR: adjacency column index array (SpMM mode).
 */
void readval_coo_adj(
    int                  beta_qu,
    int                  f_align,
    float                quantization_scale_fea,
    bool                 sage_mode,
    bool                 linear_mode,
    bool                 gemm_mode,
    int                  ccount,
    int                  last_index,
    hls::stream<ATYPE>  &A_fifo,
    hls::stream<int>    &col_indices_fifo,
    INTYPE              *values,
    int                 *columnIndex
)
{
    #pragma HLS INLINE OFF

    bool gcn_path = !(linear_mode ^ sage_mode);

    if (gcn_path == 1)
    {
        if (gemm_mode == 0)
        {
            LOOP_J_SPMM: for (int j = 0; j < last_index; j++)
            {
                #pragma HLS PIPELINE

                INTYPE raw_val   = (INTYPE)values[j];
                ATYPE  quant_val;

#if (INT_QUANT_A == 1)
                quanta(quant_val, raw_val, quantization_scale_fea, f_align, beta_qu);
#else
                quant_val = raw_val;
#endif

                A_fifo           << quant_val;
                col_indices_fifo << columnIndex[j];
            }
        }
        else
        {
            int col = 0;

            LOOP_J_SPMM2: for (int j = 0; j < last_index; j++)
            {
                #pragma HLS PIPELINE

                INTYPE raw_val   = (INTYPE)values[j];
                ATYPE  quant_val;

#if (INT_QUANT_A == 1)
                quanta(quant_val, raw_val, quantization_scale_fea, f_align, beta_qu);
#else
                quant_val = raw_val;
#endif

                A_fifo           << quant_val;
                col_indices_fifo << col;

                col = (col == (ccount - 1)) ? 0 : col + 1;
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * readval_csr_adj2 - Stream CSR adjacency values and column indices (ITYPE output).
 *
 * Identical to readval_csr_adj but casts the quantized ATYPE value to ITYPE
 * before writing to A_fifo.  Used by the GCN pass-through path where the
 * downstream compute kernel expects ITYPE adjacency values.
 *
 * @param beta_qu               Zero-point shift for quantization.
 * @param f_align               Fractional alignment bits.
 * @param quantization_scale_fea Adjacency value scale factor.
 * @param gemm_mode             True = dense GEMM mode.
 * @param ccount                Number of columns (counter wrap for GEMM mode).
 * @param last_index            Total non-zeros to read.
 * @param A_fifo                Output FIFO: adjacency values cast to ITYPE.
 * @param col_indices_fifo      Output FIFO: column indices.
 * @param values                DDR: adjacency value array.
 * @param columnIndex           DDR: adjacency column index array (SpMM mode).
 */
void readval_csr_adj2(
    int                  beta_qu,
    int                  f_align,
    float                quantization_scale_fea,
    bool                 gemm_mode,
    int                  ccount,
    int                  last_index,
    hls::stream<ITYPE>  &A_fifo,
    hls::stream<int>    &col_indices_fifo,
    INTYPE              *values,
    int                 *columnIndex
)
{
    #pragma HLS INLINE OFF

    if (gemm_mode == 0)
    {
        LOOP_J_SPMM: for (int j = 0; j < last_index; j++)
        {
            #pragma HLS PIPELINE

            INTYPE raw_val   = (INTYPE)values[j];
            ATYPE  quant_val;

#if (INT_QUANT_A == 1)
            quanta(quant_val, raw_val, quantization_scale_fea, f_align, beta_qu);
#else
            quant_val = raw_val;
#endif

            A_fifo           << (ITYPE)quant_val;
            col_indices_fifo << columnIndex[j];
        }
    }
    else
    {
        int col = 0;

        LOOP_J_SPMM2: for (int j = 0; j < last_index; j++)
        {
            #pragma HLS PIPELINE

            INTYPE raw_val   = (INTYPE)values[j];
            ATYPE  quant_val;

#if (INT_QUANT_A == 1)
            quanta(quant_val, raw_val, quantization_scale_fea, f_align, beta_qu);
#else
            quant_val = raw_val;
#endif

            A_fifo           << (ITYPE)quant_val;
            col_indices_fifo << col;

            col = (col == (ccount - 1)) ? 0 : col + 1;
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * readval_coo_adj2 - Stream COO adjacency values and column indices (ITYPE output).
 *
 * Identical to readval_csr_adj2 but gated by gcn_path (same logic as
 * readval_coo_adj).  No output is produced for linear-only layers.
 *
 * Note: the SpMM path uses INT_QUANT (not INT_QUANT_A) — this matches the
 * original source and is preserved intentionally.
 *
 * @param beta_qu               Zero-point shift for quantization.
 * @param f_align               Fractional alignment bits.
 * @param quantization_scale_fea Adjacency value scale factor.
 * @param sage_mode             Layer is a SAGE aggregation layer.
 * @param linear_mode           Layer has an active linear projection.
 * @param gemm_mode             True = dense GEMM mode.
 * @param ccount                Number of columns (counter wrap for GEMM mode).
 * @param last_index            Total non-zeros to read.
 * @param A_fifo                Output FIFO: adjacency values cast to ITYPE.
 * @param col_indices_fifo      Output FIFO: column indices.
 * @param values                DDR: adjacency value array.
 * @param columnIndex           DDR: adjacency column index array (SpMM mode).
 */
void readval_coo_adj2(
    int                  beta_qu,
    int                  f_align,
    float                quantization_scale_fea,
    bool                 sage_mode,
    bool                 linear_mode,
    bool                 gemm_mode,
    int                  ccount,
    int                  last_index,
    hls::stream<ITYPE>  &A_fifo,
    hls::stream<int>    &col_indices_fifo,
    INTYPE              *values,
    int                 *columnIndex
)
{
    #pragma HLS INLINE OFF

    bool gcn_path = !(linear_mode ^ sage_mode);

    if (gcn_path == 1)
    {
        if (gemm_mode == 0)
        {
            LOOP_J_SPMM: for (int j = 0; j < last_index; j++)
            {
                #pragma HLS PIPELINE

                INTYPE raw_val   = (INTYPE)values[j];
                ATYPE  quant_val;

                /* Note: uses INT_QUANT guard (not INT_QUANT_A) — preserves original behavior */
#if (INT_QUANT == 1)
                quanta(quant_val, raw_val, quantization_scale_fea, f_align, beta_qu);
#else
                quant_val = raw_val;
#endif

                A_fifo           << (ITYPE)quant_val;
                col_indices_fifo << columnIndex[j];
            }
        }
        else
        {
            int col = 0;

            LOOP_J_SPMM2: for (int j = 0; j < last_index; j++)
            {
                #pragma HLS PIPELINE

                INTYPE raw_val   = (INTYPE)values[j];
                ATYPE  quant_val;

#if (INT_QUANT_A == 1)
                quanta(quant_val, raw_val, quantization_scale_fea, f_align, beta_qu);
#else
                quant_val = raw_val;
#endif

                A_fifo           << (ITYPE)quant_val;
                col_indices_fifo << col;

                col = (col == (ccount - 1)) ? 0 : col + 1;
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
// 
// =============================================================================================
// =============================================================================================

/**
 * readval_coo_fea - Stream feature matrix values and column indices into dataflow FIFOs.
 *
 * Handles all four combinations of {gemm_mode, stream_mode}:
 *
 *   SpMM + DDR    (gemm_mode=0, stream_mode=0): reads from DDR arrays values[]/columnIndex[].
 *   SpMM + Stream (gemm_mode=0, stream_mode=1): reads from AXI-stream valuess/columnIndex_feas.
 *   GEMM + DDR    (gemm_mode=1, stream_mode=0): reads dense values[]; column index = running counter.
 *   GEMM + Stream (gemm_mode=1, stream_mode=1): reads dense values from AXI-stream; column counter.
 *
 * For each non-zero, optionally applies fixed-point quantization (INT_QUANT_F):
 *   quantf() → FTYPE value for the GNN branch (A_fifo).
 *   quantl() → LTYPE value for the SAGE/linear branch (A_fifo_sage).
 *
 * Output routing:
 *   gcn_path  == 1: emits to A_fifo + col_indices_fifo.
 *   linear_mode== 1: emits to A_fifo_sage + col_indices_fifo_sage  (LINEAR_ENABLE guard).
 *
 * The AXI-stream path uses the TLAST field (temp.last) to detect end-of-frame
 * in SpMM+Stream mode (do-while loop).
 *
 * @param beta_qu                Zero-point shift for GNN quantization.
 * @param f_align                Fractional bits for GNN quantization.
 * @param beta_qul               Zero-point shift for linear quantization.
 * @param f_alignl               Fractional bits for linear quantization.
 * @param quantization_scale_fea Per-layer GNN scale factors.
 * @param quantization_scale_lin Per-layer linear scale factors.
 * @param sage_mode              Layer is a SAGE aggregation layer.
 * @param linear_mode            Layer has an active linear projection.
 * @param stream_mode            Read source is AXI-stream (not DDR).
 * @param gemm_mode              Dense (GEMM) mode; column index generated internally.
 * @param ccount                 Number of columns (used as the dense column counter wrap).
 * @param last_index             Total number of non-zeros to read.
 * @param A_fifo                 Output FIFO: GNN feature values (FTYPE).
 * @param col_indices_fifo       Output FIFO: column indices (GNN branch).
 * @param A_fifo_sage            Output FIFO: SAGE/linear feature values (LTYPE).
 * @param col_indices_fifo_sage  Output FIFO: column indices (SAGE/linear branch).
 * @param values                 DDR: value array (used when stream_mode == 0).
 * @param valuess                AXI-stream: value stream (used when stream_mode == 1).
 * @param columnIndex            DDR: column index array (SpMM+DDR mode).
 * @param columnIndex_feas       AXI-stream: column index stream (SpMM+Stream mode).
 * @param B_index                Current layer index.
 */
void readval_coo_fea(
    int                  beta_qu,
    int                  f_align,
    int                  beta_qul,
    int                  f_alignl,
    float                quantization_scale_fea[5],
    float                quantization_scale_lin[5],
    bool                 sage_mode,
    bool                 linear_mode,
    bool                 stream_mode,
    bool                 gemm_mode,
    int                  ccount,
    int                  last_index,
    hls::stream<FTYPE>  &A_fifo,
    hls::stream<int>    &col_indices_fifo,
    hls::stream<LTYPE>  &A_fifo_sage,
    hls::stream<int>    &col_indices_fifo_sage,
    INTYPE              *values,
    hls::stream<ASTYPE> &valuess,
    int                 *columnIndex,
    hls::stream<ASTYPE> &columnIndex_feas,
    int                  B_index
)
{
    #pragma HLS INLINE OFF

    bool gcn_path = !(linear_mode ^ sage_mode);

    /* ── Helper lambda-style macro: quantize one raw value into both branches ── */
    /* (Expanded inline inside each path to keep the pragma HLS PIPELINE intact.) */

    if (gemm_mode == 0)
    {
        /* ════════════════════════════════════════════
         * SpMM mode: each non-zero has an explicit column index.
         * ════════════════════════════════════════════ */

        fp_int C_float_int;

        if (stream_mode == 1)
        {
            /* ── SpMM + AXI-stream source ──
             * Reads until TLAST is asserted (do-while on last_index1). */
            bool last_index1;

            LOOP_J_SPMM11: do
            {
                #pragma HLS PIPELINE

                INTYPE raw_val;
                FTYPE  q_fea;
                LTYPE  q_lin;

                ASTYPE temp  = valuess.read();
                C_float_int.i = temp.data;
                raw_val       = (INTYPE)C_float_int.f;

                temp         = columnIndex_feas.read();
                last_index1  = temp.last;
                int col      = temp.data;

#if (INT_QUANT_F == 1)
                quantf(q_fea, raw_val, quantization_scale_fea, f_align,  beta_qu,  B_index);
                quantl(q_lin, raw_val, quantization_scale_lin, f_alignl, beta_qul, B_index);
#else
                q_fea = raw_val;
                q_lin = raw_val;
#endif

                if (gcn_path)
                {
                    A_fifo           << q_fea;
                    col_indices_fifo << col;
                }

#if (LINEAR_ENABLE == 1)
                if (linear_mode)
                {
                    A_fifo_sage           << q_lin;
                    col_indices_fifo_sage << col;
                }
#endif

            } while (last_index1 == 0);
        }
        else
        {
            /* ── SpMM + DDR source ── */
            LOOP_J_SPMM12: for (int j = 0; j < last_index; j++)
            {
                #pragma HLS PIPELINE

                INTYPE raw_val = (INTYPE)values[j];
                int    col     = columnIndex[j];
                FTYPE  q_fea;
                LTYPE  q_lin;

#if (INT_QUANT_F == 1)
                quantf(q_fea, raw_val, quantization_scale_fea, f_align,  beta_qu,  B_index);
                quantl(q_lin, raw_val, quantization_scale_lin, f_alignl, beta_qul, B_index);
#else
                q_fea = raw_val;
                q_lin = raw_val;
#endif

                if (gcn_path)
                {
                    A_fifo           << q_fea;
                    col_indices_fifo << col;
                }

#if (LINEAR_ENABLE == 1)
                if (linear_mode)
                {
                    A_fifo_sage           << q_lin;
                    col_indices_fifo_sage << col;
                }
#endif
            }
        }
    }
    else
    {
        /* ════════════════════════════════════════════
         * GEMM (dense) mode: column index is a running counter [0 .. ccount-1].
         * ════════════════════════════════════════════ */

        fp_int C_float_int;
        int    col = 0;

        if (stream_mode == 1)
        {
            /* ── GEMM + AXI-stream source ── */
            LOOP_J_SPMM21: for (int j = 0; j < last_index; j++)
            {
                #pragma HLS PIPELINE

                INTYPE raw_val;
                FTYPE  q_fea;
                LTYPE  q_lin;

                ASTYPE temp   = valuess.read();
                C_float_int.i  = temp.data;
                raw_val        = (INTYPE)C_float_int.f;

#if (INT_QUANT_F == 1)
                quantf(q_fea, raw_val, quantization_scale_fea, f_align,  beta_qu,  B_index);
                quantl(q_lin, raw_val, quantization_scale_lin, f_alignl, beta_qul, B_index);
#else
                q_fea = raw_val;
                q_lin = raw_val;
#endif

                if (gcn_path)
                {
                    A_fifo           << q_fea;
                    col_indices_fifo << col;
                }

#if (LINEAR_ENABLE == 1)
                if (linear_mode)
                {
                    A_fifo_sage           << q_lin;
                    col_indices_fifo_sage << col;
                }
#endif

                col = (col == (ccount - 1)) ? 0 : col + 1;
            }
        }
        else
        {
            /* ── GEMM + DDR source ── */
            LOOP_J_SPMM22: for (int j = 0; j < last_index; j++)
            {
                #pragma HLS PIPELINE

                INTYPE raw_val = (INTYPE)values[j];
                FTYPE  q_fea;
                LTYPE  q_lin;

#if (INT_QUANT_F == 1)
                quantf(q_fea, raw_val, quantization_scale_fea, f_align,  beta_qu,  B_index);
                quantl(q_lin, raw_val, quantization_scale_lin, f_alignl, beta_qul, B_index);
#else
                q_fea  = raw_val;
                q_lin  = raw_val;
#endif

                if (gcn_path)
                {
                    A_fifo           << q_fea;
                    col_indices_fifo << col;
                }

#if (LINEAR_ENABLE == 1)
                if (linear_mode)
                {
                    A_fifo_sage           << q_lin;
                    col_indices_fifo_sage << col;
                }
#endif

                col = (col == (ccount - 1)) ? 0 : col + 1;
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * check_fifo_0 - Non-blocking FIFO relay with elastic buffering (FIFO 0).
 *
 * Transfers exactly a_values elements from A_fifo to A_fifo_out using
 * non-blocking reads and writes to absorb back-pressure from a slow consumer
 * or a bursty producer.
 *
 * When the downstream FIFO (A_fifo_out) is full, the current element is held
 * in a 1-element local buffer (data_buffer) until the downstream makes space.
 *
 * Telemetry counters (fifo_cycle_0, fifo_read_0, fifo_write_0, fifo_full_0)
 * are updated each cycle for performance profiling.
 *
 * @param a_values  Total number of elements to relay.
 * @param A_fifo    Input FIFO.
 * @param A_fifo_out Output FIFO.
 */
void check_fifo_0(
    int                 a_values,
    hls::stream<ITYPE> &A_fifo,
    hls::stream<ITYPE> &A_fifo_out
)
{
    ITYPE data_buffer;
    int   data_count    = 0;
    bool  data_in_buffer = 0;   // true when data_buffer holds an unsent element

    while ((data_count < a_values) || (data_in_buffer == 1))
    {
        #pragma HLS PIPELINE

        fifo_cycle_0++;

        if (data_in_buffer == 0)
        {
            /* Buffer empty: try to read from input */
            if (A_fifo.read_nb(data_buffer) == 1)
            {
                fifo_read_0++;
                data_count++;

                if (A_fifo_out.write_nb(data_buffer) == 0)
                {
                    /* Downstream full: hold element in buffer */
                    fifo_full_0++;
                    data_in_buffer = 1;
                }
                else
                {
                    fifo_write_0++;
                }
            }
        }
        else
        {
            /* Buffer occupied: drain to output before reading more */
            if (A_fifo_out.write_nb(data_buffer) == 1)
            {
                fifo_write_0++;

                /* Immediately try to refill buffer from input */
                if (A_fifo.read_nb(data_buffer) == 0)
                    data_in_buffer = 0;   // buffer now empty
                else
                {
                    fifo_read_0++;
                    data_count++;
                    /* data_in_buffer stays 1: new element is in buffer */
                }
            }
            else
            {
                fifo_full_0++;
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
// 
// =============================================================================================
// =============================================================================================
/**
 * check_fifo_2 - Non-blocking FIFO relay with elastic buffering (FIFO 2).
 *
 * Transfers exactly N elements from C_fifo to C_fifo_out.
 * Semantics and telemetry counters are identical to check_fifo_0.
 *
 * @param N          Total number of elements to relay.
 * @param C_fifo     Input FIFO.
 * @param C_fifo_out Output FIFO.
 */
void check_fifo_2(
    int                 N,
    hls::stream<ITYPE> &C_fifo,
    hls::stream<ITYPE> &C_fifo_out
)
{
    ITYPE data_buffer;
    int   data_count    = 0;
    bool  data_in_buffer = 0;

    while (data_count < N)
    {
        #pragma HLS PIPELINE

        fifo_cycle_2++;

        if (data_in_buffer == 0)
        {
            if (C_fifo.read_nb(data_buffer) == 1)
            {
                fifo_read_2++;

                if (C_fifo_out.write_nb(data_buffer) == 0)
                {
                    fifo_full_2++;
                    data_in_buffer = 1;
                }
                else
                {
                    data_count++;
                    fifo_write_2++;
                }
            }
            else
            {
                fifo_empty_2++;
            }
        }
        else
        {
            if (C_fifo_out.write_nb(data_buffer) == 1)
            {
                fifo_write_2++;

                if (C_fifo.read_nb(data_buffer) == 0)
                {
                    fifo_empty_2++;
                    data_in_buffer = 0;
                }
                else
                {
                    fifo_read_2++;
                }

                data_count++;
            }
            else
            {
                fifo_full_2++;
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * check_fifo_1 - Non-blocking FIFO relay with elastic buffering (FIFO 1).
 *
 * Transfers exactly N elements from C_fifo to C_fifo_out.
 * Semantics and telemetry counters are identical to check_fifo_0 / check_fifo_2.
 *
 * The B_index / B_index_loop / tail parameters are accepted for interface
 * symmetry with the caller but are not used in the current implementation
 * (the B_WIDTH_INT calculation is commented out).
 *
 * @param N           Total number of elements to relay.
 * @param B_index     Current layer index (unused).
 * @param B_index_loop Total layer iterations (unused).
 * @param tail        Final-iteration column width (unused).
 * @param C_fifo      Input FIFO.
 * @param C_fifo_out  Output FIFO.
 */
void check_fifo_1(
    int                 N,
    int                 B_index,
    int                 B_index_loop,
    int                 tail,
    hls::stream<ITYPE> &C_fifo,
    hls::stream<ITYPE> &C_fifo_out
)
{
    ITYPE data_buffer;
    int   data_count    = 0;
    bool  data_in_buffer = 0;

    while (data_count < N)
    {
        #pragma HLS PIPELINE

        fifo_cycle_1++;

        if (data_in_buffer == 0)
        {
            if (C_fifo.read_nb(data_buffer) == 1)
            {
                fifo_read_1++;

                if (C_fifo_out.write_nb(data_buffer) == 0)
                {
                    fifo_full_1++;
                    data_in_buffer = 1;
                }
                else
                {
                    data_count++;
                    fifo_write_1++;
                }
            }
        }
        else
        {
            if (C_fifo_out.write_nb(data_buffer) == 1)
            {
                fifo_write_1++;

                if (C_fifo.read_nb(data_buffer) == 0)
                    data_in_buffer = 0;
                else
                    fifo_read_1++;

                data_count++;
            }
            else
            {
                fifo_full_1++;
            }
        }
    }
}
// =============================================================================================
// =============================================================================================
/**
 * reada1_coo - Read feature sparse matrix A in COO format into dataflow FIFOs.
 *
 * Decodes mode flags from the model array, adjusts the DDR base pointers for
 * the requested row partition, then delegates to two sub-tasks:
 *
 *   readptr_coo_fea  – emits per-row non-zero counts into rnnz FIFOs.
 *   readval_coo_fea  – emits (value, column-index) pairs into A/col FIFOs,
 *                      applying quantization for both the GNN and SAGE/linear
 *                      branches simultaneously.
 *
 * Supports two read modes (selected by gemm_mode):
 *   SpMM (gemm_mode == 0): adjusts column/value/rowPtr pointers to the
 *                           partition start; total nnz = nnz_fea.
 *   GEMM (gemm_mode == 1): treats A as dense; last_index = row_count × M_int.
 *
 * The feature width M_int depends on the layer:
 *   Layer 0     : M_int = M   (input feature dimension).
 *   Hidden layers: M_int = B_WIDTH_BLOCK (output of previous layer).
 *
 * @param nnz_fea              Total non-zeros in this partition.
 * @param beta_qu              Zero-point shift for GNN quantization.
 * @param f_align              Fractional alignment bits for GNN quantization.
 * @param beta_qul             Zero-point shift for linear-branch quantization.
 * @param f_alignl             Fractional alignment bits for linear quantization.
 * @param quantization_scale_fea Per-layer GNN feature scale factors.
 * @param quantization_scale_lin Per-layer linear-branch scale factors.
 * @param last_index           Output: total non-zeros streamed (set internally).
 * @param model                Per-layer mode flags [layer][bit].
 * @param M                    Input feature width at layer 0.
 * @param first_row            First row index of this partition.
 * @param row_count            Number of rows in this partition.
 * @param A_fifo_fea           Output FIFO: quantized GNN feature values.
 * @param col_indices_fifo_fea Output FIFO: column indices (GNN branch).
 * @param rnnz_fifo_fea        Output FIFO: per-row nnz counts (GNN branch).
 * @param A_fifo_fea_sage      Output FIFO: quantized SAGE/linear values.
 * @param col_indices_fifo_fea_sage Output FIFO: column indices (SAGE/linear).
 * @param rnnz_fifo_fea_sage   Output FIFO: per-row nnz counts (SAGE/linear).
 * @param rowPtr_fea           DDR: CSR row pointer array.
 * @param columnIndex_fea      DDR: CSR column index array.
 * @param values_fea           DDR: CSR value array.
 * @param rowPtr_feas          AXI-stream row pointers (streaming interface).
 * @param columnIndex_feas     AXI-stream column indices (streaming interface).
 * @param values_feas          AXI-stream values (streaming interface).
 * @param B_index              Current layer index.
 * @param layer_loop           Total number of layer iterations.
 */
void reada1_coo(
    int                  nnz_fea,
    int                  beta_qu,
    int                  f_align,
    int                  beta_qul,
    int                  f_alignl,
    float                quantization_scale_fea[5],
    float                quantization_scale_lin[5],
    int                 &last_index,
    ap_uint<1>           model[5][8],
    int                  M,
    int                  first_row,
    int                  row_count,
    hls::stream<FTYPE>  &A_fifo_fea,
    hls::stream<int>    &col_indices_fifo_fea,
    hls::stream<int>    &rnnz_fifo_fea,
    hls::stream<LTYPE>  &A_fifo_fea_sage,
    hls::stream<int>    &col_indices_fifo_fea_sage,
    hls::stream<int>    &rnnz_fifo_fea_sage,
    int                 *rowPtr_fea,
    int                 *columnIndex_fea,
    INTYPE              *values_fea,
    hls::stream<ASTYPE> &rowPtr_feas,
    hls::stream<ASTYPE> &columnIndex_feas,
    hls::stream<ASTYPE> &values_feas,
    int                  B_index,
    int                  layer_loop
)
{
    /* ── Decode layer mode flags ── */
    bool gemm_mode   = model[B_index][1];
    bool stream_mode = model[B_index][3];
    bool linear_mode = model[B_index][6];
    bool sage_mode   = model[B_index][7];

    /* Feature width: full input dim at layer 0, hidden dim for later layers */
    int M_int = (B_index == 0) ? M : B_WIDTH_BLOCK;

    /* ── Adjust DDR pointers and compute total non-zeros ── */
    int last_index_fea;

    if (gemm_mode == 0)
    {
        /* SpMM mode: advance pointers to the partition start row */
        columnIndex_fea += first_row;
        values_fea      += first_row;
        rowPtr_fea      += first_row;
        last_index_fea   = nnz_fea;
    }
    else
    {
        /* GEMM (dense) mode: treat A as dense, advance to partition start */
        values_fea    += first_row * M_int;
        last_index_fea = row_count * M_int;
    }

    /* ── Stage 1: emit per-row non-zero counts ── */
    readptr_coo_fea(nnz_fea, sage_mode, linear_mode, stream_mode, gemm_mode,
                    row_count, M_int,
                    rowPtr_fea, rowPtr_feas,
                    rnnz_fifo_fea, rnnz_fifo_fea_sage);

    /* ── Stage 2: emit (value, column-index) pairs for both branches ── */
    readval_coo_fea(beta_qu, f_align, beta_qul, f_alignl,
                    quantization_scale_fea, quantization_scale_lin,
                    sage_mode, linear_mode, stream_mode, gemm_mode,
                    M_int, last_index_fea,
                    A_fifo_fea,      col_indices_fifo_fea,
                    A_fifo_fea_sage, col_indices_fifo_fea_sage,
                    values_fea, values_feas,
                    columnIndex_fea, columnIndex_feas,
                    B_index);
}


// =============================================================================================
// =============================================================================================
/**
 * reada2_csr - Read adjacency sparse matrix A in CSR format into dataflow FIFOs.
 *
 * Adjusts DDR pointers to the row partition, pushes the total non-zero count
 * into rnnz_fifo_adj_total_e/s (used by the edge/softmax write tasks to size
 * their DDR output), then calls:
 *
 *   readptr_csr_adj – emits per-row nnz counts.
 *   readval_csr_adj – emits (value, column-index) pairs with quantization.
 *
 * @param beta_qu                 Zero-point shift for quantization.
 * @param f_align                 Fractional alignment bits.
 * @param quantization_scale_adj  Adjacency value scale factor.
 * @param gemm_mode               True = dense mode; False = sparse CSR mode.
 * @param M                       Number of adjacency columns.
 * @param first_row               First row index of this partition.
 * @param row_count               Number of rows in this partition.
 * @param A_fifo_adj              Output FIFO: quantized adjacency values.
 * @param col_indices_fifo_adj    Output FIFO: column indices.
 * @param rnnz_fifo_adj_total_e   Output FIFO: total nnz for edge-score DDR write.
 * @param rnnz_fifo_adj_total_s   Output FIFO: total nnz for softmax DDR write.
 * @param rnnz_fifo_adj           Output FIFO: per-row nnz counts.
 * @param rowPtr_adj              DDR: CSR row pointer array.
 * @param columnIndex_adj         DDR: CSR column index array.
 * @param values_adj              DDR: CSR value array.
 */
void reada2_csr(
    int                  beta_qu,
    int                  f_align,
    float                quantization_scale_adj,
    bool                 gemm_mode,
    int                  M,
    int                  first_row,
    int                  row_count,
    hls::stream<ATYPE>  &A_fifo_adj,
    hls::stream<int>    &col_indices_fifo_adj,
    hls::stream<int>    &rnnz_fifo_adj_total_e,
    hls::stream<int>    &rnnz_fifo_adj_total_s,
    hls::stream<int>    &rnnz_fifo_adj,
    int                 *rowPtr_adj,
    int                 *columnIndex_adj,
    INTYPE              *values_adj
)
{
    /* ── Adjust DDR pointers and compute total non-zeros ── */
    int last_index_adj;

    if (gemm_mode == 0)
    {
        last_index_adj  = rowPtr_adj[first_row + row_count] - rowPtr_adj[first_row];
        columnIndex_adj += rowPtr_adj[first_row];
        values_adj      += rowPtr_adj[first_row];
        rowPtr_adj      += first_row;
    }
    else
    {
        last_index_adj = row_count * M;
        values_adj    += first_row * M;
    }

    /* Forward total nnz to the edge-score and softmax DDR write tasks */
    rnnz_fifo_adj_total_e << last_index_adj;
    rnnz_fifo_adj_total_s << last_index_adj;

    /* ── Stage 1: emit per-row nnz counts ── */
    readptr_csr_adj(gemm_mode, row_count, M, rowPtr_adj, rnnz_fifo_adj);

    /* ── Stage 2: emit (value, column-index) pairs ── */
    readval_csr_adj(beta_qu, f_align, quantization_scale_adj,
                    gemm_mode, M, last_index_adj,
                    A_fifo_adj, col_indices_fifo_adj,
                    values_adj, columnIndex_adj);
}

// =============================================================================================
// =============================================================================================
/**
 * reada2_coo - Read adjacency sparse matrix A in COO format into dataflow FIFOs
 *              for the GAT attention path.
 *
 * When gat_mode == 1, also pushes the total nnz into rnnz_fifo_adj_total_e/s
 * so that the edge-score and softmax DDR write tasks know how much data to expect.
 *
 * @param nnz_adj                 Total non-zeros in this partition.
 * @param beta_qu                 Zero-point shift for quantization.
 * @param f_align                 Fractional alignment bits.
 * @param quantization_scale_adj  Adjacency value scale factor.
 * @param model                   Per-layer mode flags [layer][bit].
 * @param M                       Number of adjacency columns.
 * @param first_row               First row index of this partition.
 * @param row_count               Number of rows in this partition.
 * @param A_fifo_adj              Output FIFO: quantized adjacency values (ATYPE).
 * @param col_indices_fifo_adj    Output FIFO: column indices.
 * @param rnnz_fifo_adj_total_e   Output FIFO: total nnz for edge-score DDR write.
 * @param rnnz_fifo_adj_total_s   Output FIFO: total nnz for softmax DDR write.
 * @param rnnz_fifo_adj           Output FIFO: per-row nnz counts.
 * @param rowPtr_adj              DDR: COO row pointer array.
 * @param columnIndex_adj         DDR: COO column index array.
 * @param values_adj              DDR: COO value array.
 * @param B_index                 Current layer index.
 */
void reada2_coo(
    int                  nnz_adj,
    int                  beta_qu,
    int                  f_align,
    float                quantization_scale_adj,
    ap_uint<1>           model[5][8],
    int                  M,
    int                  first_row,
    int                  row_count,
    hls::stream<ATYPE>  &A_fifo_adj,
    hls::stream<int>    &col_indices_fifo_adj,
    hls::stream<int>    &rnnz_fifo_adj_total_e,
    hls::stream<int>    &rnnz_fifo_adj_total_s,
    hls::stream<int>    &rnnz_fifo_adj,
    int                 *rowPtr_adj,
    int                 *columnIndex_adj,
    INTYPE              *values_adj,
    int                  B_index
)
{
    /* ── Decode layer mode flags ── */
    bool gemm_mode   = model[B_index][0];
    bool linear_mode = model[B_index][6];
    bool gat_mode    = model[B_index][5];
    bool sage_mode   = model[B_index][7];

    /* ── Adjust DDR pointers and compute total non-zeros ── */
    int last_index_adj;

    if (gemm_mode == 0)
    {
        columnIndex_adj += rowPtr_adj[first_row];
        values_adj      += rowPtr_adj[first_row];
        rowPtr_adj      += first_row;
        last_index_adj   = nnz_adj;
    }
    else
    {
        values_adj    += first_row * M;
        last_index_adj = row_count * M;
    }

    /* Forward total nnz to attention write tasks (GAT only) */
    if (gat_mode == 1)
    {
        rnnz_fifo_adj_total_e << nnz_adj;
        rnnz_fifo_adj_total_s << nnz_adj;
    }

    /* ── Stage 1: emit per-row nnz counts ── */
    readptr_coo_adj(nnz_adj, sage_mode, linear_mode, gemm_mode,
                    row_count, M, rowPtr_adj, rnnz_fifo_adj);

    /* ── Stage 2: emit (value, column-index) pairs ── */
    readval_coo_adj(beta_qu, f_align, quantization_scale_adj,
                    sage_mode, linear_mode, gemm_mode,
                    M, last_index_adj,
                    A_fifo_adj, col_indices_fifo_adj,
                    values_adj, columnIndex_adj);
}


// =============================================================================================
// =============================================================================================
/**
 * reada22_coo - Read adjacency sparse matrix A in COO format for the GCN
 *               pass-through path (no attention scoring).
 *
 * Identical to reada2_coo but:
 *   - Emits values to A_fifo_adj typed as ITYPE (not ATYPE).
 *   - Does NOT emit to rnnz_fifo_adj_total_e/s (no edge-score DDR write).
 *   - Calls readval_coo_adj2 which targets the ITYPE output FIFO.
 *
 * @param nnz_adj              Total non-zeros in this partition.
 * @param beta_qu              Zero-point shift.
 * @param f_align              Fractional alignment bits.
 * @param quantization_scale_adj Adjacency value scale.
 * @param model                Per-layer mode flags.
 * @param M                    Number of adjacency columns.
 * @param first_row            First row index of this partition.
 * @param row_count            Number of rows in this partition.
 * @param A_fifo_adj           Output FIFO: adjacency values (ITYPE).
 * @param col_indices_fifo_adj Output FIFO: column indices.
 * @param rnnz_fifo_adj        Output FIFO: per-row nnz counts.
 * @param rowPtr_adj           DDR: COO row pointer array.
 * @param columnIndex_adj      DDR: COO column index array.
 * @param values_adj           DDR: COO value array.
 * @param B_index              Current layer index.
 */
void reada22_coo(
    int                  nnz_adj,
    int                  beta_qu,
    int                  f_align,
    float                quantization_scale_adj,
    ap_uint<1>           model[5][8],
    int                  M,
    int                  first_row,
    int                  row_count,
    hls::stream<ITYPE>  &A_fifo_adj,
    hls::stream<int>    &col_indices_fifo_adj,
    hls::stream<int>    &rnnz_fifo_adj,
    int                 *rowPtr_adj,
    int                 *columnIndex_adj,
    INTYPE              *values_adj,
    int                  B_index
)
{
    /* ── Decode layer mode flags ── */
    bool gemm_mode   = model[B_index][0];
    bool linear_mode = model[B_index][6];
    bool sage_mode   = model[B_index][7];

    /* ── Adjust DDR pointers and compute total non-zeros ── */
    int last_index_adj;

    if (gemm_mode == 0)
    {
        columnIndex_adj += rowPtr_adj[first_row];
        values_adj      += rowPtr_adj[first_row];
        rowPtr_adj      += first_row;
        last_index_adj   = nnz_adj;
    }
    else
    {
        values_adj    += first_row * M;
        last_index_adj = row_count * M;
    }

    /* ── Stage 1: emit per-row nnz counts ── */
    readptr_coo_adj(nnz_adj, sage_mode, linear_mode, gemm_mode,
                    row_count, M, rowPtr_adj, rnnz_fifo_adj);

    /* ── Stage 2: emit (value, column-index) pairs (ITYPE output) ── */
    readval_coo_adj2(beta_qu, f_align, quantization_scale_adj,
                     sage_mode, linear_mode, gemm_mode,
                     M, last_index_adj,
                     A_fifo_adj, col_indices_fifo_adj,
                     values_adj, columnIndex_adj);
}

// =============================================================================================
// =============================================================================================
/**
 * reada22_csr - Read adjacency sparse matrix A in CSR format for the GCN
 *               pass-through path (no attention scoring).
 *
 * Identical to reada2_csr but:
 *   - Emits values to A_fifo_adj typed as ITYPE (not ATYPE).
 *   - Does NOT emit to rnnz_fifo_adj_total_e/s.
 *   - Calls readval_csr_adj2 which targets the ITYPE output FIFO.
 *
 * @param beta_qu                Zero-point shift.
 * @param f_align                Fractional alignment bits.
 * @param quantization_scale_adj Adjacency value scale.
 * @param gemm_mode              True = dense; False = sparse CSR.
 * @param M                      Number of adjacency columns.
 * @param first_row              First row index of this partition.
 * @param row_count              Number of rows in this partition.
 * @param A_fifo_adj             Output FIFO: adjacency values (ITYPE).
 * @param col_indices_fifo_adj   Output FIFO: column indices.
 * @param rnnz_fifo_adj          Output FIFO: per-row nnz counts.
 * @param rowPtr_adj             DDR: CSR row pointer array.
 * @param columnIndex_adj        DDR: CSR column index array.
 * @param values_adj             DDR: CSR value array.
 */
void reada22_csr(
    int                  beta_qu,
    int                  f_align,
    float                quantization_scale_adj,
    bool                 gemm_mode,
    int                  M,
    int                  first_row,
    int                  row_count,
    hls::stream<ITYPE>  &A_fifo_adj,
    hls::stream<int>    &col_indices_fifo_adj,
    hls::stream<int>    &rnnz_fifo_adj,
    int                 *rowPtr_adj,
    int                 *columnIndex_adj,
    INTYPE              *values_adj
)
{
    /* ── Adjust DDR pointers and compute total non-zeros ── */
    int last_index_adj;

    if (gemm_mode == 0)
    {
        last_index_adj  = rowPtr_adj[first_row + row_count] - rowPtr_adj[first_row];
        columnIndex_adj += rowPtr_adj[first_row];
        values_adj      += rowPtr_adj[first_row];
        rowPtr_adj      += first_row;
    }
    else
    {
        last_index_adj = row_count * M;
        values_adj    += first_row * M;
    }

    /* ── Stage 1: emit per-row nnz counts ── */
    readptr_csr_adj(gemm_mode, row_count, M, rowPtr_adj, rnnz_fifo_adj);

    /* ── Stage 2: emit (value, column-index) pairs (ITYPE output) ── */
    readval_csr_adj2(beta_qu, f_align, quantization_scale_adj,
                     gemm_mode, M, last_index_adj,
                     A_fifo_adj, col_indices_fifo_adj,
                     values_adj, columnIndex_adj);
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_wrapper_adj_4 - Inner SpMM accumulation for the adjacency kernel,
 *                            four C-tile column partitions.
 *
 * Accumulates Adj[row] × C for a single row, where C is split across four
 * equal-sized column partitions (b_block1..4, each B_HEIGHT/4 rows).
 * The correct partition is selected inside dsp_kernel_*_adj_4 based on
 * the column index and block_size.
 *
 * Two implementations (FLOAT/HALF and FIXEDPOINT) share the same structure
 * as dsp_kernel_wrapper_adj_1; see that function for a detailed explanation
 * of the latency-hiding partial-accumulator pattern.
 *
 * @param block_size     Number of rows per column partition (B_HEIGHT/4).
 * @param M              Number of non-zeros for this single row.
 * @param A_fifo         Input: attention-weighted adjacency values (ITYPE).
 * @param col_indices_fifo Input: column indices.
 * @param b_block1..4    C-tile column partitions [B_HEIGHT/4][B_WIDTH_BLOCK].
 * @param zero_point_lhs Quantization zero point for adjacency values.
 * @param zero_point_rhs Quantization zero point for C tile values.
 * @param acc2           Output: accumulated results [B_WIDTH_BLOCK].
 */
void dsp_kernel_wrapper_adj_4(
    int                 block_size,
    int                 M,
    hls::stream<ITYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    QTYPE               b_block1[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE               b_block2[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE               b_block3[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE               b_block4[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    ITYPE               acc2[B_WIDTH_BLOCK]
)
{
#if defined FLOAT || defined HALF

    FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_part complete

    FTYPE acc_float[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_float complete

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        acc_float[j] = 0;
    }

    RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        for (int l = 0; l < FADD_LATENCY_ADJ; l++)
        {
            #pragma HLS UNROLL
            for (int z = 0; z < SPMM_BLOCK; z++)
            {
                #pragma HLS UNROLL
                acc_part[l][j][z] = 0;
            }
        }
    }

    int BM = M[SPMM_BLOCK - 1];
    int M_aux[SPMM_BLOCK + 1];
    M_aux[0] = 0;
    for (int j = 1; j < SPMM_BLOCK + 1; j++)
    {
        #pragma HLS UNROLL
        M_aux[j] = M[j - 1];
    }

    DSP_LOOP_SPMM: for (int k = 0; k < BM; k += FADD_LATENCY_ADJ)
    {
        #pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

        DSP_LOOP_SPMM2: for (int i = 0; i < FADD_LATENCY_ADJ; i++)
        {
            DTYPE v;
            int   ci;

            if ((k + i) < BM)
            {
                v  = A_fifo.read();
                ci = col_indices_fifo.read();
            }
            else
            {
                v  = 0;
                ci = 0;
            }

            dsp_kernel_float_adj_4(block_size, v,
                                   b_block1, b_block2, b_block3, b_block4,
                                   ci, zero_point_lhs, zero_point_rhs, acc_float);

            for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                for (int z = 0; z < SPMM_BLOCK; z++)
                {
                    #pragma HLS UNROLL
                    if ((k + i) >= M_aux[z] && (k + i) < M_aux[z + 1])
                        acc_part[i][j][z] += acc_float[j];
                }
            }
        }
    }

    /* ── Reduce partial accumulators ── */
    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        for (int l = 1; l < FADD_LATENCY_ADJ; l++)
        {
            for (int z = 0; z < SPMM_BLOCK; z++)
                acc_part[0][j][z] += acc_part[l][j][z];
        }
    }

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        for (int z = 0; z < SPMM_BLOCK; z++)
        {
            #pragma HLS UNROLL
            acc2[j][z] = acc_part[0][j][z];
        }
    }

#endif  /* FLOAT / HALF */

#ifdef FIXEDPOINT

    ITYPE acc[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    DSP_LOOP_SPMM: for (int i = 0; i < M; i++)
    {
        #pragma HLS PIPELINE

        DTYPE v  = A_fifo.read();
        int   ci = col_indices_fifo.read();

        dsp_kernel_int_adj_4(block_size, v,
                             b_block1, b_block2, b_block3, b_block4,
                             ci, zero_point_lhs, zero_point_rhs, acc);

        for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS UNROLL
            acc2[j] += acc[j];
        }
    }

#endif  /* FIXEDPOINT */
}


// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_wrapper_adj_2 - Inner SpMM accumulation for the adjacency kernel,
 *                            two C-tile column partitions.
 *
 * Extends dsp_kernel_wrapper_adj_4 to process SPMM_BLOCK rows simultaneously,
 * accumulating into a 2D array acc2[B_WIDTH_BLOCK][SPMM_BLOCK].
 * C is split across two column partitions (b_block1, b_block2).
 *
 * Row-slot assignment uses the M_aux[] cumulative-nnz boundary array:
 *   non-zero k belongs to row-slot z  iff  M_aux[z] <= k < M_aux[z+1]
 *
 * @param block_size     Number of rows per column partition (B_HEIGHT/4).
 * @param M              Cumulative nnz counts per SPMM_BLOCK row slot [SPMM_BLOCK].
 * @param A_fifo         Input: attention-weighted adjacency values (ITYPE).
 * @param col_indices_fifo Input: column indices.
 * @param b_block1..2    C-tile column partitions [B_HEIGHT/4][B_WIDTH_BLOCK].
 * @param zero_point_lhs Quantization zero point for adjacency values.
 * @param zero_point_rhs Quantization zero point for C tile values.
 * @param acc2           Output: accumulated results [B_WIDTH_BLOCK][SPMM_BLOCK].
 */
void dsp_kernel_wrapper_adj_2(
    int                 block_size,
    int                 M[SPMM_BLOCK],
    hls::stream<ITYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    QTYPE               b_block1[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE               b_block2[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    ITYPE               acc2[B_WIDTH_BLOCK][SPMM_BLOCK]
)
{
#if defined FLOAT || defined HALF

    FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_part complete

    FTYPE acc_float[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_float complete

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        acc_float[j] = 0;
    }

    RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        for (int l = 0; l < FADD_LATENCY_ADJ; l++)
        {
            #pragma HLS UNROLL
            for (int z = 0; z < SPMM_BLOCK; z++)
                acc_part[l][j][z] = 0;
        }
    }

    int BM = M[SPMM_BLOCK - 1];
    int M_aux[SPMM_BLOCK + 1];
    M_aux[0] = 0;
    for (int j = 1; j < SPMM_BLOCK + 1; j++)
    {
        #pragma HLS UNROLL
        M_aux[j] = M[j - 1];
    }

    DSP_LOOP_SPMM: for (int k = 0; k < BM; k += FADD_LATENCY_ADJ)
    {
        #pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

        DSP_LOOP_SPMM2: for (int i = 0; i < FADD_LATENCY_ADJ; i++)
        {
            DTYPE v;
            int   ci;

            if ((k + i) < BM)
            {
                v  = A_fifo.read();
                ci = col_indices_fifo.read();
            }
            else
            {
                v  = 0;
                ci = 0;
            }

            dsp_kernel_float_adj_2(block_size, v,
                                   b_block1, b_block2,
                                   ci, zero_point_lhs, zero_point_rhs, acc_float);

            for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                for (int z = 0; z < SPMM_BLOCK; z++)
                {
                    #pragma HLS UNROLL
                    if ((k + i) >= M_aux[z] && (k + i) < M_aux[z + 1])
                        acc_part[i][j][z] += acc_float[j];
                }
            }
        }
    }

    /* ── Reduce partial accumulators ── */
    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        for (int l = 1; l < FADD_LATENCY_ADJ; l++)
        {
            #pragma HLS UNROLL
            for (int z = 0; z < SPMM_BLOCK; z++)
                acc_part[0][j][z] += acc_part[l][j][z];
        }
    }

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        for (int z = 0; z < SPMM_BLOCK; z++)
            acc2[j][z] = acc_part[0][j][z];
    }

#endif  /* FLOAT / HALF */

#ifdef FIXEDPOINT

    ITYPE acc[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    int BM = M[SPMM_BLOCK - 1];
    int M_aux[SPMM_BLOCK + 1];
    M_aux[0] = 0;
    for (int j = 1; j < SPMM_BLOCK + 1; j++)
    {
        #pragma HLS UNROLL
        M_aux[j] = M[j - 1];
    }

    DSP_LOOP_SPMM: for (int i = 0; i < BM; i++)
    {
        #pragma HLS PIPELINE

        DTYPE v  = A_fifo.read();
        int   ci = col_indices_fifo.read();

        dsp_kernel_int_adj_2(block_size, v,
                             b_block1, b_block2,
                             ci, zero_point_lhs, zero_point_rhs, acc);

        for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS UNROLL
            for (int z = 0; z < SPMM_BLOCK; z++)
            {
                #pragma HLS UNROLL
                if (i >= M_aux[z] && i < M_aux[z + 1])
                    acc2[j][z] += acc[j];
            }
        }
    }

#endif  /* FIXEDPOINT */
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_wrapper_adj_1 - Inner SpMM accumulation for the adjacency kernel,
 *                            single C-tile column partition.
 *
 * Iterates over M non-zeros of one adjacency row and accumulates:
 *   acc2[j] += A_fifo[k] * b_block1[col_indices_fifo[k]][j]
 *
 * Two implementations are selected at compile time:
 *
 *   FLOAT / HALF path:
 *     Uses a latency-hiding pipeline with FADD_LATENCY_ADJ parallel partial
 *     accumulators (acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK]).
 *     Each partial accumulator absorbs one pipeline-stage worth of products,
 *     avoiding WAW hazards on the accumulator.  The partials are summed after
 *     the main loop.
 *
 *   FIXEDPOINT path:
 *     Simple pipelined loop; the inner dsp_kernel_int_adj_1() call handles
 *     the fixed-point multiply-accumulate.
 *
 * @param block_size     Number of nodes in the column partition (B_HEIGHT).
 * @param M              Number of non-zeros for this row.
 * @param A_fifo         Input: attention-weighted adjacency values (TTYPE).
 * @param col_indices_fifo Input: column indices.
 * @param b_block1       C-tile column partition [B_HEIGHT][B_WIDTH_BLOCK].
 * @param zero_point_lhs Quantization zero point for adjacency values.
 * @param zero_point_rhs Quantization zero point for C tile values.
 * @param acc2           Output: accumulated partial results [B_WIDTH_BLOCK].
 */
void dsp_kernel_wrapper_adj_1(
    int                 block_size,
    int                 M,
    hls::stream<TTYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    QTYPE               b_block1[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    ITYPE               acc2[B_WIDTH_BLOCK]
)
{
#if defined FLOAT || defined HALF

    /* ── Latency-hiding partial accumulators ── */
    FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0

    FTYPE acc_float[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_float complete

    /* Zero scalar and partial accumulators */
    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        acc_float[j] = 0;
    }

    RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        for (int l = 0; l < FADD_LATENCY_ADJ; l++)
        {
            #pragma HLS UNROLL
            for (int z = 0; z < SPMM_BLOCK; z++)
                acc_part[l][j][z] = 0;
        }
    }

    /* Build cumulative nnz boundary array for row-slot assignment */
    int BM = M[SPMM_BLOCK - 1];
    int M_aux[SPMM_BLOCK + 1];
    M_aux[0] = 0;
    for (int j = 1; j < SPMM_BLOCK + 1; j++)
    {
        #pragma HLS UNROLL
        M_aux[j] = M[j - 1];
    }

    /* ── Main accumulation loop (pipeline II = FADD_LATENCY_ADJ) ── */
    DSP_LOOP_SPMM: for (int k = 0; k < BM; k += FADD_LATENCY_ADJ)
    {
        #pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

        DSP_LOOP_SPMM2: for (int i = 0; i < FADD_LATENCY_ADJ; i++)
        {
            DTYPE v;
            int   ci;

            if ((k + i) < BM)
            {
                v  = A_fifo.read();
                ci = col_indices_fifo.read();
            }
            else
            {
                v  = 0;
                ci = 0;
            }

            dsp_kernel_float_adj_1(v, b_block1, ci,
                                   zero_point_lhs, zero_point_rhs, acc_float);

            /* Scatter acc_float into the correct row-slot partial accumulator */
            for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                for (int z = 0; z < SPMM_BLOCK; z++)
                {
                    #pragma HLS UNROLL
                    if ((k + i) >= M_aux[z] && (k + i) < M_aux[z + 1])
                        acc_part[i][j][z] += acc_float[j];
                }
            }
        }
    }

    /* ── Reduce partial accumulators ── */
    ACC_PART1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        ACC_PART2: for (int z = 0; z < SPMM_BLOCK; z++)
        {
            #pragma HLS UNROLL
            ACC_PART3: for (int l = 1; l < FADD_LATENCY_ADJ; l++)
            {
                #pragma HLS PIPELINE II=1
                acc_part[0][j][z] += acc_part[l][j][z];
            }
        }
    }

    FLOAT_PART1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        FLOAT_PART2: for (int z = 0; z < SPMM_BLOCK; z++)
        {
            #pragma HLS UNROLL
            acc2[j][z] = acc_part[0][j][z];
        }
    }

#endif  /* FLOAT / HALF */

#ifdef FIXEDPOINT

    ITYPE acc[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    /* ── Simple pipelined fixed-point accumulation ── */
    DSP_LOOP_SPMM: for (int i = 0; i < M; i++)
    {
        #pragma HLS PIPELINE

        TTYPE v  = A_fifo.read();
        int   ci = col_indices_fifo.read();

        dsp_kernel_int_adj_1(block_size, v, b_block1,
                             ci, zero_point_lhs, zero_point_rhs, acc);

        for (int j = 0; j < B_WIDTH_BLOCK; j++)
            acc2[j] += acc[j];
    }

#endif  /* FIXEDPOINT */
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_wrapper_fea - Inner SpMM accumulation for the feature kernel (GNN branch).
 *
 * Accumulates A × B for SPMM_BLOCK rows simultaneously:
 *   acc2[j][z] += A_fifo[k] * b_block[col][j]
 *                 for each row-slot z whose nnz range covers index k.
 *
 * Two implementations:
 *
 *   FLOAT / HALF:
 *     Same latency-hiding FADD_LATENCY_FEA partial-accumulator pattern as
 *     dsp_kernel_wrapper_adj_1.  Row-slot assignment is done via M_aux[].
 *
 *   FIXEDPOINT:
 *     Single pipelined loop.  The accumulator array `acc` is mapped to DSP
 *     adders via #pragma HLS bind_op.  Row-slot scattering is done inline
 *     using M_aux[].
 *
 * @param gemm_mode      True = GEMM (dense) mode; False = SpMM mode.
 *                       (Currently unused in body; column index always read.)
 * @param M              Cumulative nnz counts per SPMM_BLOCK row slot [SPMM_BLOCK].
 * @param A_fifo         Input: feature non-zero values (FTYPE).
 * @param col_indices_fifo Input: column indices.
 * @param b_block        Weight tile [B_HEIGHT/4][B_WIDTH_BLOCK].
 * @param zero_point_lhs Quantization zero point for features.
 * @param zero_point_rhs Quantization zero point for weights.
 * @param acc2           Output: accumulated results [B_WIDTH_BLOCK][SPMM_BLOCK].
 */
void dsp_kernel_wrapper_fea(
    bool                gemm_mode,
    int                 M[SPMM_BLOCK],
    hls::stream<FTYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    BTYPE               b_block[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    ITYPE               acc2[B_WIDTH_BLOCK][SPMM_BLOCK]
)
{
#if defined FLOAT || defined HALF

    ITYPE acc_part[FADD_LATENCY_FEA][B_WIDTH_BLOCK][SPMM_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0

    ITYPE acc_float[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_float complete

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        acc_float[j] = 0;
    }

    RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        for (int l = 0; l < FADD_LATENCY_FEA; l++)
        {
            #pragma HLS UNROLL
            for (int z = 0; z < SPMM_BLOCK; z++)
            {
                #pragma HLS UNROLL
                acc_part[l][j][z] = 0;
            }
        }
    }

    int BM = M[SPMM_BLOCK - 1];
    int M_aux[SPMM_BLOCK + 1];
    M_aux[0] = 0;
    for (int j = 1; j < SPMM_BLOCK + 1; j++)
    {
        #pragma HLS UNROLL
        M_aux[j] = M[j - 1];
    }

    DSP_LOOP_SPMM: for (int k = 0; k < BM; k += FADD_LATENCY_FEA)
    {
        #pragma HLS PIPELINE II=FADD_LATENCY_FEA

        DSP_LOOP_SPMM2: for (int i = 0; i < FADD_LATENCY_FEA; i++)
        {
            DTYPE v;
            int   ci;

            if ((k + i) < BM)
            {
                v  = A_fifo.read();
                ci = col_indices_fifo.read();
            }
            else
            {
                v  = 0;
                ci = 0;
            }

            dsp_kernel_float_fea(v, b_block, ci,
                                 zero_point_lhs, zero_point_rhs, acc_float);

            SPMM_BLOCK_LOOP1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                #pragma HLS UNROLL
                SPMM_BLOCK_LOOP2: for (int z = 0; z < SPMM_BLOCK; z++)
                {
                    #pragma HLS PIPELINE II=1
                    if ((k + i) >= M_aux[z] && (k + i) < M_aux[z + 1])
                        acc_part[i][j][z] += acc_float[j];
                }
            }
        }
    }

    ACC_PART1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        ACC_PART2: for (int z = 0; z < SPMM_BLOCK; z++)
        {
            #pragma HLS UNROLL
            ACC_PART3: for (int l = 1; l < FADD_LATENCY_FEA; l++)
            {
                #pragma HLS PIPELINE II=1
                acc_part[0][j][z] += acc_part[l][j][z];
            }
        }
    }

    ACC_PART_FLOAT1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        ACC_PART_FLOAT2: for (int z = 0; z < SPMM_BLOCK; z++)
        {
            #pragma HLS UNROLL
            acc2[j][z] = acc_part[0][j][z];
        }
    }

#endif  /* FLOAT / HALF */

#ifdef FIXEDPOINT

    ITYPE acc[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    #pragma HLS bind_op variable=acc op=add impl=dsp

    /* Build cumulative nnz boundary array */
    int BM = M[SPMM_BLOCK - 1];
    int M_aux[SPMM_BLOCK + 1];
    M_aux[0] = 0;
    for (int j = 1; j < SPMM_BLOCK + 1; j++)
    {
        #pragma HLS UNROLL
        M_aux[j] = M[j - 1];
    }

    DSP_LOOP_SPMM: for (int i = 0; i < BM; i++)
    {
        #pragma HLS PIPELINE

        FTYPE v  = A_fifo.read();
        int   ci = col_indices_fifo.read();

        dsp_kernel_int_fea(v, b_block, ci,
                           zero_point_lhs, zero_point_rhs, acc);

        for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS UNROLL
            for (int z = 0; z < SPMM_BLOCK; z++)
            {
                #pragma HLS UNROLL
                if (i >= M_aux[z] && i < M_aux[z + 1])
                    acc2[j][z] += acc[j];
            }
        }
    }

#endif  /* FIXEDPOINT */
}

// =============================================================================================
// =============================================================================================
/**
 * dsp_kernel_wrapper_lin - Inner SpMM accumulation for the linear-projection branch.
 *
 * Identical in structure to dsp_kernel_wrapper_fea but:
 *   - Processes a single row (scalar M, not M[SPMM_BLOCK]).
 *   - Uses BLTYPE / LTYPE types (linear-branch precision).
 *   - Calls dsp_kernel_int_lin() in the fixed-point path.
 *   - Output is a 1D array acc2[B_WIDTH_BLOCK] (no SPMM_BLOCK dimension).
 *
 * @param gemm_mode      Dense (GEMM) vs sparse mode flag (currently unused).
 * @param M              Number of non-zeros for this single row.
 * @param A_fifo         Input: linear-branch feature values (LTYPE).
 * @param col_indices_fifo Input: column indices.
 * @param b_block        Linear-projection weight tile [B_HEIGHT/4][B_WIDTH_BLOCK].
 * @param zero_point_lhs Quantization zero point for features.
 * @param zero_point_rhs Quantization zero point for weights.
 * @param acc2           Output: accumulated results [B_WIDTH_BLOCK].
 */
void dsp_kernel_wrapper_lin(
    bool                gemm_mode,
    int                 M,
    hls::stream<LTYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    BLTYPE              b_block[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    ITYPE               acc2[B_WIDTH_BLOCK]
)
{
#if defined FLOAT || defined HALF

    /* Float path shares the same latency-hiding pattern as dsp_kernel_wrapper_fea.
     * SPMM_BLOCK dimension is retained for structural symmetry even though
     * the linear kernel processes one row at a time (z is always 0). */
    ITYPE acc_part[FADD_LATENCY_FEA][B_WIDTH_BLOCK][SPMM_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0

    ITYPE acc_float[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_float complete

    for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        acc_float[j] = 0;
    }

    RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        for (int l = 0; l < FADD_LATENCY_FEA; l++)
        {
            #pragma HLS UNROLL
            for (int z = 0; z < SPMM_BLOCK; z++)
            {
                #pragma HLS UNROLL
                acc_part[l][j][z] = 0;
            }
        }
    }

    /* Single-row: BM = M, M_aux = {0, M} */
    int BM = M;
    int M_aux[SPMM_BLOCK + 1];
    M_aux[0] = 0;
    for (int j = 1; j < SPMM_BLOCK + 1; j++)
    {
        #pragma HLS UNROLL
        M_aux[j] = M;
    }

    DSP_LOOP_SPMM: for (int k = 0; k < BM; k += FADD_LATENCY_FEA)
    {
        #pragma HLS PIPELINE II=FADD_LATENCY_FEA

        DSP_LOOP_SPMM2: for (int i = 0; i < FADD_LATENCY_FEA; i++)
        {
            DTYPE v;
            int   ci;

            if ((k + i) < BM)
            {
                v  = A_fifo.read();
                ci = col_indices_fifo.read();
            }
            else
            {
                v  = 0;
                ci = 0;
            }

            dsp_kernel_float_fea(v, b_block, ci,
                                 zero_point_lhs, zero_point_rhs, acc_float);

            SPMM_BLOCK_LOOP1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                #pragma HLS UNROLL
                SPMM_BLOCK_LOOP2: for (int z = 0; z < SPMM_BLOCK; z++)
                {
                    #pragma HLS PIPELINE II=1
                    if ((k + i) >= M_aux[z] && (k + i) < M_aux[z + 1])
                        acc_part[i][j][z] += acc_float[j];
                }
            }
        }
    }

    ACC_PART1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        ACC_PART2: for (int z = 0; z < SPMM_BLOCK; z++)
        {
            #pragma HLS UNROLL
            ACC_PART3: for (int l = 1; l < FADD_LATENCY_FEA; l++)
            {
                #pragma HLS PIPELINE II=1
                acc_part[0][j][z] += acc_part[l][j][z];
            }
        }
    }

    ACC_PART_FLOAT1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
    {
        #pragma HLS UNROLL
        ACC_PART_FLOAT2: for (int z = 0; z < SPMM_BLOCK; z++)
        {
            #pragma HLS UNROLL
            acc2[j][z] = acc_part[0][j][z];
        }
    }

#endif  /* FLOAT / HALF */

#ifdef FIXEDPOINT

    ITYPE acc[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    #pragma HLS bind_op variable=acc op=add impl=dsp

    DSP_LOOP_SPMM: for (int i = 0; i < M; i++)
    {
        #pragma HLS PIPELINE

        LTYPE v  = A_fifo.read();
        int   ci = col_indices_fifo.read();

        dsp_kernel_int_lin(v, b_block, ci,
                           zero_point_lhs, zero_point_rhs, acc);

        for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS UNROLL
            acc2[j] += acc[j];
        }
    }

#endif  /* FIXEDPOINT */
}

// =============================================================================================
// =============================================================================================
/**
 * scale - Post-accumulation bias-add, requantization, and clamping.
 *
 * Reads raw accumulator values from C_fifo, applies:
 *   output = clamp( (C + bias) * quantized_multiplier >> shift + zero_point_dst,
 *                   clamp_min, clamp_max )
 * and writes the result to write_fifo.
 *
 * Two implementations:
 *
 *   FLOAT / HALF:
 *     Reinterpret-casts the integer FIFO values to float, adds float bias and
 *     zero_point_dst, clamps, then reinterpret-casts back to DTYPE.
 *     Scaling is gated by ENABLE_SCALING; without it, values pass through.
 *
 *   FIXEDPOINT:
 *     Processes 4 output channels per outer iteration to amortize the
 *     bias/shift/multiplier DDR read overhead.  Uses 64-bit intermediate
 *     arithmetic to avoid overflow.  Packs four 8-bit outputs into one
 *     32-bit word for efficient downstream writes.
 *
 * The output column width is clipped to `tail` on the last B_index iteration.
 *
 * @param quantized_multiplier Per-channel integer multiplier array.
 * @param shift                Per-channel right-shift amounts.
 * @param bias                 Per-channel bias values.
 * @param zero_point_dst       Output quantization zero point.
 * @param clamp_max            Upper clamp bound (e.g. 127 for INT8).
 * @param clamp_min            Lower clamp bound (e.g. -128 for INT8).
 * @param N                    Number of output rows.
 * @param M                    Number of input columns (unused; kept for symmetry).
 * @param P                    Number of output columns (unused; kept for symmetry).
 * @param C_fifo               Input: raw accumulator FIFOs [B_WIDTH_BLOCK].
 * @param B_index              Current layer index.
 * @param B_index_loop         Total layer iterations.
 * @param tail                 Active column count for the final iteration.
 * @param write_fifo           Output: requantized result FIFOs [B_WIDTH_BLOCK].
 */
void scale(
    ap_int<32>         *quantized_multiplier,
    ap_int<32>         *shift,
    ap_int<32>         *bias,
    ap_int<8>           zero_point_dst,
    ap_int<8>           clamp_max,
    ap_int<8>           clamp_min,
    int                 N,
    int                 M,
    int                 P,
    hls::stream<ITYPE>  C_fifo[B_WIDTH_BLOCK],
    int                 B_index,
    int                 B_index_loop,
    int                 tail,
    hls::stream<ITYPE>  write_fifo[B_WIDTH_BLOCK]
)
{
    int B_WIDTH_INT = (B_index < (B_index_loop - 1)) ? B_WIDTH_BLOCK : tail;

#if defined FLOAT || defined HALF

    LOOP_CH1f: for (int i = 0; i < N; i++)
    {
        LOOP_CW1f: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS PIPELINE II=1
            if (j < B_WIDTH_INT)
            {
#ifdef ENABLE_SCALING
                /* Reinterpret raw int as float, add float bias and zero point, clamp */
                int    bias_int         = bias[i];
                FTYPE  bias_float       = *(FTYPE *)&bias_int;
                DTYPE  C_raw            = C_fifo[j].read();
                FTYPE  C_float          = *(FTYPE *)&C_raw;
                FTYPE  zero_point_float = (FTYPE)zero_point_dst;
                FTYPE  clamp_min_float  = (FTYPE)clamp_min;
                FTYPE  clamp_max_float  = (FTYPE)clamp_max;

                FTYPE  C_out_float = C_float + bias_float + zero_point_float;
                if (C_out_float < clamp_min_float) C_out_float = clamp_min_float;
                if (C_out_float > clamp_max_float) C_out_float = clamp_max_float;

                DTYPE C_out = *(int *)&C_out_float;
                write_fifo[j] << C_out;
#else
                /* Pass through without scaling */
                DTYPE C_raw = C_fifo[j].read();
                write_fifo[j] << C_raw;
#endif
            }
        }
    }

#endif  /* FLOAT / HALF */

#ifdef FIXEDPOINT

    /* Process 4 output channels per outer iteration to amortize DDR reads */
    LOOP_CH1: for (int i = 0; i < N; i += 4)
    {
        /* Prefetch bias, shift, and multiplier for 4 channels */
        ap_int<32> bias_val[4], shift_val[4], mult_val[4];
        bias_val[0]  = bias[i];      bias_val[1]  = bias[i + 1];
        bias_val[2]  = bias[i + 2];  bias_val[3]  = bias[i + 3];
        shift_val[0] = shift[i];     shift_val[1] = shift[i + 1];
        shift_val[2] = shift[i + 2]; shift_val[3] = shift[i + 3];
        mult_val[0]  = quantized_multiplier[i];
        mult_val[1]  = quantized_multiplier[i + 1];
        mult_val[2]  = quantized_multiplier[i + 2];
        mult_val[3]  = quantized_multiplier[i + 3];

        LOOP_CW1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS PIPELINE II=4

            DTYPE C_out = 0;

            /* Pack 4 × 8-bit quantized outputs into one 32-bit word */
            LOOP_CH3: for (int z = 0; z < 4; z++)
            {
                #pragma HLS loop_tripcount min=1 max=1 avg=1
                if (j < B_WIDTH_INT)
                {
#ifdef ENABLE_SCALING
                    /* 64-bit intermediate to avoid overflow in multiply */
                    ap_int<64>  C_temp   = C_fifo[j].read() + bias_val[z];
                    ap_int<32>  tot_shift = 31 - shift_val[z];
                    ap_int<64>  round     = (ap_int<64>)1 << (tot_shift - 1);

                    C_temp = C_temp * mult_val[z] + round;
                    C_temp = (C_temp >> tot_shift) + zero_point_dst;

                    /* Clamp to [clamp_min, clamp_max] */
                    ap_int<8> C_temp8 = C_temp;
                    if (C_temp < clamp_min) C_temp8 = clamp_min;
                    if (C_temp > clamp_max) C_temp8 = clamp_max;

                    /* Pack into upper byte of C_out, shifting previous bytes down */
                    C_out = (C_out >> 8) | ((int)C_temp8 << 24);

                    if (z == 3)
                        write_fifo[j].write(C_out);
#else
                    C_out = C_fifo[j].read();
                    write_fifo[j].write(C_out);
#endif
                }
            }
        }
    }

#endif  /* FIXEDPOINT */
}

// =============================================================================================
// =============================================================================================
/**
 * compute2_4 - Adjacency SpMM kernel with 4 C-tile column partitions.
 *
 * Computes D = Adj × C for one row partition, where C is split across
 * four column-partition tiles (B_accel1..4).  One row is processed at a
 * time; the accumulator is reset per row.
 *
 * After accumulation, an optional shift-ReLU threshold is applied:
 *   output = (acc2[j] < relu_t) ? 0 : acc2[j]     when relu == true
 *   output = acc2[j]                                when relu == false
 *
 * @param relu           Enable shift-ReLU activation.
 * @param relu_t         ReLU threshold value.
 * @param block_size     Number of nodes per column partition (= B_HEIGHT/4).
 * @param zero_point_lhs Quantization zero point for the adjacency values.
 * @param zero_point_rhs Quantization zero point for the C tile values.
 * @param first_row      First row index of this partition (unused in body).
 * @param row_count      Number of rows to process.
 * @param A_fifo         Input: attention-weighted adjacency values.
 * @param col_indices_fifo Input: column indices.
 * @param rnnz_fifo      Input: per-row non-zero counts.
 * @param B_accel1..4    Input: C-tile column partitions [B_HEIGHT/4][B_WIDTH_BLOCK].
 * @param C_fifo         Output: per-column-block result FIFOs [B_WIDTH_BLOCK].
 */
void compute2_4(
    bool                relu,
    float               relu_t,
    int                 block_size,
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    int                 first_row,
    int                 row_count,
    hls::stream<ITYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    hls::stream<int>   &rnnz_fifo,
    QTYPE               B_accel1[B_HEIGHT / 2][B_WIDTH_BLOCK],
    QTYPE               B_accel2[B_HEIGHT / 2][B_WIDTH_BLOCK],
    QTYPE               B_accel3[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE               B_accel4[B_HEIGHT / 4][B_WIDTH_BLOCK],
    hls::stream<ITYPE>  C_fifo[B_WIDTH_BLOCK]
)
{
    ITYPE acc2[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc2 complete

    for (int A_index = 0; A_index < row_count; A_index++)
    {
        /* ── Zero accumulators ── */
        LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS UNROLL
            acc2[j] = 0;
        }

        int rnnz = rnnz_fifo.read();

        /* ── Accumulate Adj[row] × C (all 4 column partitions) ── */
        dsp_kernel_wrapper_adj_4(block_size, rnnz,
                                 A_fifo, col_indices_fifo,
                                 B_accel1, B_accel2, B_accel3, B_accel4,
                                 zero_point_lhs, zero_point_rhs, acc2);

        /* ── Apply optional shift-ReLU and emit results ── */
        LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS UNROLL

            ITYPE C_fifo_val;
            if (relu == 0)
                C_fifo_val = acc2[j];
            else
                C_fifo_val = (acc2[j] < (ITYPE)relu_t) ? ITYPE(0.0) : acc2[j];

            C_fifo[j].write(C_fifo_val);
        }
    }
}


// =============================================================================================
// =============================================================================================
// 
// =============================================================================================
// =============================================================================================
/**
 * compute2_2 - Adjacency SpMM kernel with 2 C-tile column partitions.
 *
 * Processes SPMM_BLOCK rows per iteration to amortize the loop overhead.
 * C is split across two column-partition tiles (B_accel1, B_accel2).
 *
 * Unlike compute2_4, this variant:
 *   - Uses a 2D accumulator acc2[B_WIDTH_BLOCK][SPMM_BLOCK] for parallelism.
 *   - Tracks the number of valid rows (crows) within the current tile to
 *     avoid writing padding entries to C_fifo.
 *   - The output column width is reduced to `tail` on the last B_index
 *     iteration (partial-width final block).
 *
 * @param block_size     Number of nodes per column partition.
 * @param zero_point_lhs Quantization zero point for adjacency values.
 * @param zero_point_rhs Quantization zero point for C tile values.
 * @param first_row      First row index of this partition (unused in body).
 * @param row_count      Number of rows to process.
 * @param A_fifo         Input: attention-weighted adjacency values.
 * @param col_indices_fifo Input: column indices.
 * @param rnnz_fifo      Input: per-row nnz FIFOs [SPMM_BLOCK].
 * @param B_accel1..2    Input: C-tile column partitions [B_HEIGHT/2][B_WIDTH_BLOCK].
 * @param C_fifo         Output: result FIFOs [B_WIDTH_BLOCK][SPMM_BLOCK].
 * @param B_index        Current layer index.
 * @param B_index_loop   Total number of layer iterations.
 * @param tail           Output column width for the final iteration.
 */
void compute2_2(
    int                 block_size,
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    int                 first_row,
    int                 row_count,
    hls::stream<ITYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    hls::stream<int>    rnnz_fifo[SPMM_BLOCK],
    QTYPE               B_accel1[B_HEIGHT / 2][B_WIDTH_BLOCK],
    QTYPE               B_accel2[B_HEIGHT / 2][B_WIDTH_BLOCK],
    hls::stream<ITYPE>  C_fifo[B_WIDTH_BLOCK][SPMM_BLOCK],
    int                 B_index,
    int                 B_index_loop,
    int                 tail
)
{
    ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

    /* Last iteration may have fewer active output columns */
    int B_WIDTH_INT = (B_index < (B_index_loop - 1)) ? B_WIDTH_BLOCK : tail;

    for (int A_index = 0; A_index < row_count; A_index += SPMM_BLOCK)
    {
        /* ── Zero accumulators ── */
        LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS UNROLL
            LOOP_ACC22: for (int i = 0; i < SPMM_BLOCK; i++)
            {
                #pragma HLS UNROLL
                acc2[j][i] = 0;
            }
        }

        /* ── Read per-row nnz counts and count valid rows in this tile ── */
        int rnnz[SPMM_BLOCK];
        int crows = 0;

        LOOP_RNNZ: for (int i = 0; i < SPMM_BLOCK; i++)
        {
            #pragma HLS UNROLL
            rnnz[i] = rnnz_fifo[i].read();
            if ((A_index + i) < row_count)
                crows++;
        }

        /* ── Accumulate Adj[tile] × C (2 column partitions) ── */
        dsp_kernel_wrapper_adj_2(block_size, rnnz,
                                 A_fifo, col_indices_fifo,
                                 B_accel1, B_accel2,
                                 zero_point_lhs, zero_point_rhs, acc2);

        /* ── Emit results for valid rows only ── */
        LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS UNROLL
            if (j < B_WIDTH_INT)
            {
                LOOP_C_BUF2: for (int i = 0; i < SPMM_BLOCK; i++)
                {
                    #pragma HLS UNROLL
                    if (i < crows)
                        C_fifo[j][i].write(acc2[j][i]);
                }
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * compute2_1 - Adjacency SpMM kernel with a single C-tile column partition.
 *
 * Processes one row at a time.  An optional shift-ReLU activation is applied
 * to the accumulator output before writing to C_fifo.
 *
 * Skipped entirely when gcn_path == 0 (linear-only layer).
 *
 * ReLU behavior (when relu == true):
 *   output = (acc2[j] < relu_t) ? 0 : acc2[j]
 *
 * @param model          Per-layer mode flags [layer][bit]:
 *                         bit 4 = relu enable, bit 6 = linear_mode, bit 7 = sage_mode.
 * @param srelu          Per-layer shift-ReLU thresholds.
 * @param block_size     Number of nodes in the single column partition.
 * @param zero_point_lhs Quantization zero point for adjacency values.
 * @param zero_point_rhs Quantization zero point for C tile values.
 * @param first_row      First row index of this partition (unused in body).
 * @param row_count      Number of rows to process.
 * @param A_fifo         Input: attention-weighted adjacency values (TTYPE).
 * @param col_indices_fifo Input: column indices.
 * @param rnnz_fifo      Input: per-row non-zero counts.
 * @param B_accel1       Input: C-tile [B_HEIGHT/2][B_WIDTH_BLOCK].
 * @param C_fifo         Output: result FIFOs [B_WIDTH_BLOCK].
 * @param B_index        Current layer index.
 */
void compute2_1(
    ap_uint<1>          model[5][8],
    float               srelu[5],
    int                 block_size,
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    int                 first_row,
    int                 row_count,
    hls::stream<TTYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    hls::stream<int>   &rnnz_fifo,
    QTYPE               B_accel1[B_HEIGHT / 2][B_WIDTH_BLOCK],
    hls::stream<ITYPE>  C_fifo[B_WIDTH_BLOCK],
    int                 B_index
)
{
    ITYPE acc2[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc2 complete

    /* ── Decode layer mode flags ── */
    bool  relu        = model[B_index][4];
    float relu_t      = srelu[B_index];
    bool  linear_mode = model[B_index][6];
    bool  sage_mode   = model[B_index][7];
    bool  gcn_path    = !(linear_mode ^ sage_mode);

    if (gcn_path == 1)
    {
        for (int A_index = 0; A_index < row_count; A_index++)
        {
            /* ── Zero accumulators ── */
            LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                #pragma HLS UNROLL
                acc2[j] = 0;
            }

            int rnnz = rnnz_fifo.read();

            /* ── Accumulate Adj[row] × C (single column partition) ── */
            dsp_kernel_wrapper_adj_1(block_size, rnnz,
                                     A_fifo, col_indices_fifo,
                                     B_accel1,
                                     zero_point_lhs, zero_point_rhs, acc2);

            /* ── Apply optional shift-ReLU and emit results ── */
            LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                #pragma HLS UNROLL

                ITYPE C_fifo_val;
                if (relu == 0)
                    C_fifo_val = acc2[j];
                else
                    C_fifo_val = (acc2[j] < (ITYPE)relu_t) ? ITYPE(0.0) : acc2[j];

                C_fifo[j].write(C_fifo_val);
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * float_to_fix - Convert a float to a fixed-point value with n_bits fractional bits.
 *
 * Note: the multiply/divide by (1<<n_bits) is intentional — it maps the float
 * through the fixed-point grid so the result is representable in QTYPE8.
 *
 * @param f_in   Input floating-point value.
 * @param n_bits Number of fractional bits in the target format.
 * @return       Quantized fixed-point value.
 */
QTYPE8 float_to_fix(float f_in, int n_bits)
{
    float  scale  = (1 << n_bits);
    QTYPE8 i_out  = (f_in * scale) * (1.0 / scale);
    return i_out;
}


// =============================================================================================
// =============================================================================================
/**
 * compute1_1 - GNN SpMM kernel: computes C = A × B with output quantization.
 *
 * For each tile of SPMM_BLOCK rows of the sparse feature matrix A:
 *   1. Reads per-row non-zero counts from rnnz_fifo.
 *   2. Calls dsp_kernel_wrapper_fea() to accumulate A[row] × B into acc2.
 *   3. Right-shifts the accumulator by scale_fea[B_index] to dequantize.
 *   4. Casts the result to the precision selected by quantized_multiplier
 *      (1/2/4/8/16 bits) and writes to C_buf1 (and A_buf1 for GAT).
 *
 * The output precision is selected by quantized_multiplier:
 *   1  → QTYPE1 / QTYPE2 (1-bit binary, written via range() to avoid rounding)
 *   2  → QTYPE2
 *   4  → QTYPE4
 *   8  → QTYPE8
 *   else → QTYPE (16-bit)
 *
 * Skipped entirely when gcn_path == 0 (linear-only layer).
 *
 * @param scale_fea          Per-layer right-shift amounts for dequantization.
 * @param max_fea            Output: maximum activation value (currently unused/zeroed).
 * @param quantized_multiplier Selects output bit-width (1/2/4/8/else=16).
 * @param model              Per-layer mode flags [layer][bit].
 * @param zero_point_lhs     Quantization zero point for the feature matrix.
 * @param zero_point_rhs     Quantization zero point for the weight matrix.
 * @param first_row          First row index of this partition (unused in body, kept for symmetry).
 * @param row_count          Number of rows to process.
 * @param A_fifo             Input: sparse feature values.
 * @param col_indices_fifo   Input: sparse column indices.
 * @param rnnz_fifo          Input: per-row non-zero counts.
 * @param B_accel            Local weight tile [B_HEIGHT][B_WIDTH_BLOCK].
 * @param C_buf1             Output: quantized result tile (GNN output).
 * @param A_buf1             Output: copy of C_buf1 written to A_buffer for GAT attention.
 * @param B_index            Current layer index.
 */
void compute1_1(
    STYPE               scale_fea[5],
    ITYPE              *max_fea,
    int                 quantized_multiplier,
    ap_uint<1>          model[5][8],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    int                 first_row,
    int                 row_count,
    hls::stream<FTYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    hls::stream<int>   &rnnz_fifo,
    BTYPE               B_accel[B_HEIGHT][B_WIDTH_BLOCK],
    QTYPE               C_buf1[B_HEIGHT][B_WIDTH_BLOCK],
    QTYPE               A_buf1[B_HEIGHT][B_WIDTH_BLOCK],
    int                 B_index
)
{
    ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

    /* ── Decode layer mode flags ── */
    bool gemm_mode   = model[B_index][1];
    bool linear_mode = model[B_index][6];
    bool sage_mode   = model[B_index][7];
    bool gcn_path    = !(linear_mode ^ sage_mode);  // skip for linear-only layers

    if (gcn_path == 1)
    {
        for (int A_index = 0; A_index < row_count; A_index += SPMM_BLOCK)
        {
            #pragma HLS DATAFLOW

            /* ── Zero accumulators ── */
            LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                #pragma HLS UNROLL
                LOOP_ACC22: for (int i = 0; i < SPMM_BLOCK; i++)
                {
                    #pragma HLS UNROLL
                    acc2[j][i] = 0;
                }
            }

            /* ── Read cumulative nnz for this SPMM_BLOCK tile ──
             * rnnz[0] holds the absolute cumulative count for the first row;
             * rnnz[i] = rnnz[i-1] + nnz of row (A_index+i).
             * Rows beyond row_count are padded with 0. */
            int rnnz[SPMM_BLOCK];
            rnnz[0] = rnnz_fifo.read();

            LOOP_RNNZ_SPMM: for (int i = 1; i < SPMM_BLOCK; i++)
            {
                #pragma HLS PIPELINE II=2
                int rnnz_temp = ((A_index + i) < row_count) ? rnnz_fifo.read() : 0;
                rnnz[i] = rnnz_temp + rnnz[i - 1];
            }

            /* ── Accumulate A[tile] × B ── */
            dsp_kernel_wrapper_fea(gemm_mode, rnnz, A_fifo, col_indices_fifo,
                                   B_accel, zero_point_lhs, zero_point_rhs, acc2);

            /* ── Quantize and write output tile ── */
            LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                #pragma HLS UNROLL

                LOOP_C_BUF2: for (int i = 0; i < SPMM_BLOCK; i++)
                {
                    #pragma HLS UNROLL

                    /* Promote to ap_fixed<32,16> before shifting to avoid truncation */
                    ap_fixed<32, 16> acc2_temp = acc2[j][i];

                    /* Compute all precisions; only the selected one is used */
                    QTYPE1 q1  = QTYPE1(acc2_temp >> scale_fea[B_index]);
                    QTYPE2 q2  = QTYPE2(acc2_temp >> scale_fea[B_index]);
                    QTYPE4 q4  = QTYPE4(acc2_temp >> scale_fea[B_index]);
                    QTYPE8 q8  = QTYPE8(acc2_temp >> scale_fea[B_index]);
                    QTYPE  q16 = QTYPE (acc2_temp >> scale_fea[B_index]);

                    *max_fea = 0;   // placeholder; max tracking currently disabled

#if GAT_ENABLE == 1
                    /* Write to both C_buf1 (SpMM output) and A_buf1 (attention input) */
                    if (quantized_multiplier == 1)
                    {
#if (qbits == 1)
                        /* Use range() to avoid rounding: only 0 and -1 representable */
                        C_buf1[A_index + i][j].range(0, 0) = q2[1];
                        A_buf1[A_index + i][j].range(0, 0) = q2[1];
#else
                        q2[0] = 1;   // force LSB for binary encoding
                        C_buf1[A_index + i][j] = q2;
                        A_buf1[A_index + i][j] = q2;
#endif
                    }
                    else if (quantized_multiplier == 2)  { C_buf1[A_index+i][j] = q2;  A_buf1[A_index+i][j] = q2;  }
                    else if (quantized_multiplier == 4)  { C_buf1[A_index+i][j] = q4;  A_buf1[A_index+i][j] = q4;  }
                    else if (quantized_multiplier == 8)  { C_buf1[A_index+i][j] = q8;  A_buf1[A_index+i][j] = q8;  }
                    else                                 { C_buf1[A_index+i][j] = q16; A_buf1[A_index+i][j] = q16; }

#else  /* GAT_ENABLE == 0: write to C_buf1 only */

                    if (quantized_multiplier == 1)
                    {
#if (qbits == 1)
                        C_buf1[A_index + i][j] = q1;
#else
                        q2[0] = 1;
                        C_buf1[A_index + i][j] = q2;
#endif
                    }
                    else if (quantized_multiplier == 2) C_buf1[A_index+i][j] = q2;
                    else if (quantized_multiplier == 4) C_buf1[A_index+i][j] = q4;
                    else if (quantized_multiplier == 8) C_buf1[A_index+i][j] = q8;
                    else                                C_buf1[A_index+i][j] = q16;

#endif  /* GAT_ENABLE */

                }  /* SPMM_BLOCK rows */
            }  /* B_WIDTH_BLOCK columns */

        }  /* A_index tile loop */
    }  /* gcn_path */
}

// =============================================================================================
// =============================================================================================
// 
// =============================================================================================
// =============================================================================================
/**
 * compute1_12 - Linear-projection SpMM kernel: computes linear_pipo = A × B_linear.
 *
 * Structurally identical to compute1_1 but:
 *   - Processes one row at a time (no SPMM_BLOCK tiling).
 *   - Uses BLTYPE / QLTYPE types (linear-branch precision).
 *   - Calls dsp_kernel_wrapper_lin() instead of the GNN variant.
 *   - Tracks the running maximum activation in *max_fea.
 *   - Activated only when linear_mode == 1.
 *
 * Output precision selection by quantized_multiplierl:
 *   2  → QTYPE2, 4 → QTYPE4, 8 → QTYPE8, else → QLTYPE (16-bit)
 *
 * @param scale_fea          Per-layer right-shift amounts for dequantization.
 * @param max_fea            Output: running maximum activation value.
 * @param quantized_multiplierl Selects output bit-width for linear branch.
 * @param model              Per-layer mode flags [layer][bit].
 * @param zero_point_lhs     Quantization zero point for the feature matrix.
 * @param zero_point_rhs     Quantization zero point for the weight matrix.
 * @param first_row          First row index (unused in body, kept for symmetry).
 * @param row_count          Number of rows to process.
 * @param A_fifo             Input: sparse feature values (linear type).
 * @param col_indices_fifo   Input: sparse column indices.
 * @param rnnz_fifo          Input: per-row non-zero counts.
 * @param B_accel            Local linear-projection weight tile [B_HEIGHT][B_WIDTH_BLOCK].
 * @param linear_pipo        Output: quantized linear-projection result tile.
 * @param B_index            Current layer index.
 */
void compute1_12(
    STYPE               scale_fea[5],
    ITYPE              *max_fea,
    int                 quantized_multiplierl,
    ap_uint<1>          model[5][8],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    int                 first_row,
    int                 row_count,
    hls::stream<LTYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    hls::stream<int>   &rnnz_fifo,
    BLTYPE              B_accel[B_HEIGHT][B_WIDTH_BLOCK],
    QLTYPE              linear_pipo[B_HEIGHT][B_WIDTH_BLOCK],
    int                 B_index
)
{
    ITYPE acc2[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc2 complete

    bool gemm_mode   = model[B_index][1];
    bool linear_mode = model[B_index][6];

    if (linear_mode == 1)
    {
        for (int A_index = 0; A_index < row_count; A_index++)
        {
            /* ── Zero accumulators ── */
            LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                #pragma HLS UNROLL
                acc2[j] = 0;
            }

            int rnnz = rnnz_fifo.read();

            /* ── Accumulate A[row] × B_linear ── */
            dsp_kernel_wrapper_lin(gemm_mode, rnnz, A_fifo, col_indices_fifo,
                                   B_accel, zero_point_lhs, zero_point_rhs, acc2);

            /* ── Quantize and write output row ── */
            LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
            {
                #pragma HLS UNROLL

                ITYPE cur_val = ITYPE(acc2[j]);

                /* Track running maximum */
                if (cur_val > *max_fea)
                    *max_fea = cur_val;

                ap_fixed<32, 16> acc2_temp = acc2[j];

                QTYPE2  q2  = QTYPE2 (acc2_temp >> scale_fea[B_index]);
                QTYPE4  q4  = QTYPE4 (acc2_temp >> scale_fea[B_index]);
                QTYPE8  q8  = QTYPE8 (acc2_temp >> scale_fea[B_index]);
                QLTYPE  q16 = QLTYPE (acc2_temp >> scale_fea[B_index]);

                if      (quantized_multiplierl == 2) linear_pipo[A_index][j] = q2;
                else if (quantized_multiplierl == 4) linear_pipo[A_index][j] = q4;
                else if (quantized_multiplierl == 8) linear_pipo[A_index][j] = q8;
                else                                 linear_pipo[A_index][j] = q16;
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * mxv - Matrix-vector products for GAT attention: computes WH1 and WH2.
 *
 * Computes the two attention dot products for the single-partition case:
 *   WH1[i] = Σ_j  C_mxv[i][j] * ate_m[j]              (source projection)
 *   WH2[i] = Σ_j  C_mxv[i][j] * ate_m[j + B_WIDTH_BLOCK] (destination projection)
 *
 * The inner loop is unrolled by 2 (j+=2) to expose two parallel MAC chains
 * (acc11/acc12 for WH1, acc21/acc22 for WH2), achieving II=1 with two DSPs.
 *
 * When GATV2 == 1, LeakyReLU is applied to WH1 and WH2 here (pre-activation).
 * When GATV2 == 0, values are stored linearly and LeakyReLU is applied later
 * in generate_attention_candidates on the sum WH1[i] + WH2[col].
 *
 * @param M        Number of nodes in this partition.
 * @param P_w      Number of attention projection dimensions (currently unused; loop bound = B_WIDTH_BLOCK).
 * @param C_mxv    Pre-activation feature tile [B_HEIGHT][B_WIDTH_BLOCK] (read-locked).
 * @param A        Attention parameter vector [a_src || a_dst], length 2×B_WIDTH_BLOCK.
 * @param WH1      Output: source projections, length M.
 * @param WH2      Output: destination projections, length M.
 */
void mxv(
    int    M,
    int    P_w,
    QTYPE  C_mxv[B_HEIGHT][B_WIDTH_BLOCK],
    BTYPE *A,
    TTYPE *WH1,
    TTYPE *WH2
)
{
    /* Preload attention parameter vector into a fully-partitioned local array
     * so all elements are accessible in parallel within LOOP_MXV4. */
    BTYPE ate_m_int[B_WIDTH_BLOCK * 2];
    #pragma HLS ARRAY_PARTITION variable=ate_m_int type=complete

    PRELOAD: for (int i = 0; i < B_WIDTH_BLOCK * 2; i++)
    {
        #pragma HLS PIPELINE
        ate_m_int[i] = A[i];
    }

    LOOP_MXV3: for (int i = 0; i < M; i++)
    {
        /* Two parallel MAC chains per projection (unrolled by 2) */
        ITYPE acc11 = 0, acc12 = 0;   // WH1 partial sums
        ITYPE acc21 = 0, acc22 = 0;   // WH2 partial sums

        LOOP_MXV4: for (int j = 0; j < B_WIDTH_BLOCK; j += 2)
        {
            #pragma HLS PIPELINE II=1
            acc11 += C_mxv[i][j]     * ate_m_int[j];
            acc12 += C_mxv[i][j + 1] * ate_m_int[j + 1];
            acc21 += C_mxv[i][j]     * ate_m_int[j + B_WIDTH_BLOCK];
            acc22 += C_mxv[i][j + 1] * ate_m_int[j + 1 + B_WIDTH_BLOCK];
        }

        ITYPE acc1 = acc11 + acc12;
        ITYPE acc2 = acc21 + acc22;

#if GATV2 == 1
        /* GATv2: apply LeakyReLU to each projection individually */
        WH1[i] = (acc1 > 0) ? acc1 : acc1 * ITYPE(0.2);
        WH2[i] = (acc2 > 0) ? acc2 : acc2 * ITYPE(0.2);
#else
        /* Standard GAT: store linear projections; LeakyReLU applied on the sum later */
        WH1[i] = acc1;
        WH2[i] = acc2;
#endif
    }
}

// =============================================================================================
// =============================================================================================
/**
 * mxv1 - Per-partition matrix-vector products for the four-partition GAT case.
 *
 * Computes WH1 (source projection) and WH21..WH24 (destination projections,
 * one per column partition) for a single row partition of size M.
 *
 * WH2 is replicated into all four destination arrays because each column
 * partition of generate_attention_candidates reads from a separate array to
 * avoid port conflicts.  The replication is identical in value:
 *   WH21[i] = WH22[i] = WH23[i] = WH24[i] = a_dst · Wh_i
 *
 * The attention parameter vector ate_m_int is preloaded by the caller (mxv4)
 * and passed in to avoid redundant DDR accesses across the four partitions.
 *
 * @param M            Number of nodes in this row partition.
 * @param P_w          Attention projection dimensions (loop bound = B_WIDTH_BLOCK).
 * @param C_mxv        Pre-activation tile [B_HEIGHT/FEA_THREADS][B_WIDTH_BLOCK].
 * @param ate_m_int    Preloaded attention vector [a_src || a_dst], length 2×B_WIDTH_BLOCK.
 * @param WH1          Output: source projections.
 * @param WH21..WH24   Output: destination projections (replicated for 4 column partitions).
 */
void mxv1(
    int    M,
    int    P_w,
    QTYPE  C_mxv[B_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK],
    BTYPE  ate_m_int[B_WIDTH_BLOCK * 2],
    TTYPE *WH1,
    TTYPE *WH21,
    TTYPE *WH22,
    TTYPE *WH23,
    TTYPE *WH24
)
{
    LOOP_MXV1: for (int i = 0; i < M; i++)
    {
        ITYPE acc11 = 0, acc12 = 0;
        ITYPE acc21 = 0, acc22 = 0;

        LOOP_MXV5: for (int j = 0; j < B_WIDTH_BLOCK; j += 2)
        {
            #pragma HLS PIPELINE
            acc11 += C_mxv[i][j]     * ate_m_int[j];
            acc12 += C_mxv[i][j + 1] * ate_m_int[j + 1];
            acc21 += C_mxv[i][j]     * ate_m_int[j + B_WIDTH_BLOCK];
            acc22 += C_mxv[i][j + 1] * ate_m_int[j + 1 + B_WIDTH_BLOCK];
        }

        ITYPE acc1 = acc11 + acc12;
        ITYPE acc2 = acc21 + acc22;

#if GATV2 == 1
        /* GATv2: apply LeakyReLU per projection */
        WH1[i]  = (acc1 > 0) ? acc1 : acc1 * ITYPE(0.2);

        ITYPE wh2 = (acc2 > 0) ? acc2 : acc2 * ITYPE(0.2);
        WH21[i] = wh2;
        WH22[i] = wh2;
        WH23[i] = wh2;
        WH24[i] = wh2;
#else
        /* Standard GAT: linear projections, replicated to all column partitions */
        WH1[i]  = acc1;
        WH21[i] = acc2;
        WH22[i] = acc2;
        WH23[i] = acc2;
        WH24[i] = acc2;
#endif
    }
}

// =============================================================================================
// =============================================================================================
/**
 * mxv4 - Matrix-vector products for all four row partitions.
 *
 * Wrapper that:
 *   1. Preloads the attention parameter vector from DDR into a local
 *      fully-partitioned array (ate_m_int) — done once for all partitions.
 *   2. Calls mxv1() for each of the four row partitions, passing the same
 *      ate_m_int so each partition reads from BRAM rather than DDR.
 *
 * @param M1..M4       Number of nodes in each row partition.
 * @param P_w          Attention projection dimensions.
 * @param C_mxv1..4    Pre-activation tiles, one per row partition.
 * @param A            Attention parameter vector in DDR [a_src || a_dst].
 * @param WH11..WH244  Output projection arrays (see mxv1 for naming convention).
 */
void mxv4(
    int    M1,
    int    M2,
    int    M3,
    int    M4,
    int    P_w,
    QTYPE  C_mxv1[B_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK],
    QTYPE  C_mxv2[B_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK],
    QTYPE  C_mxv3[B_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK],
    QTYPE  C_mxv4[B_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK],
    BTYPE *A,
    TTYPE *WH11,  TTYPE *WH211, TTYPE *WH212, TTYPE *WH213, TTYPE *WH214,
    TTYPE *WH12,  TTYPE *WH221, TTYPE *WH222, TTYPE *WH223, TTYPE *WH224,
    TTYPE *WH13,  TTYPE *WH231, TTYPE *WH232, TTYPE *WH233, TTYPE *WH234,
    TTYPE *WH14,  TTYPE *WH241, TTYPE *WH242, TTYPE *WH243, TTYPE *WH244
)
{
    /* Preload attention vector once into BRAM; shared across all 4 mxv1 calls */
    BTYPE ate_m_int[B_WIDTH_BLOCK * 2];
    #pragma HLS ARRAY_PARTITION variable=ate_m_int type=complete

    PRELOAD: for (int i = 0; i < B_WIDTH_BLOCK * 2; i++)
    {
        #pragma HLS PIPELINE
        ate_m_int[i] = A[i];
    }

    mxv1(M1, P_w, C_mxv1, ate_m_int, WH11, WH211, WH212, WH213, WH214);
    mxv1(M2, P_w, C_mxv2, ate_m_int, WH12, WH221, WH222, WH223, WH224);
    mxv1(M3, P_w, C_mxv3, ate_m_int, WH13, WH231, WH232, WH233, WH234);
    mxv1(M4, P_w, C_mxv4, ate_m_int, WH14, WH241, WH242, WH243, WH244);
}
// =============================================================================================
// =============================================================================================
/**
 * prepare_attentional_mechanism_input2 - Single-partition GAT attention preparation.
 *
 * Implements the first half of the GAT attention equation for one row partition:
 *
 *   e_ij = LeakyReLU( aᵀ [Wh_i || Wh_j] )
 *        = LeakyReLU( WH1[i] + WH2[j] )
 *
 * Steps:
 *   1. Acquire a read lock on the pre-activation buffer A_buffer11 (written
 *      by loop_fea) and compute the dot products:
 *        WH1[i] = a_src · Wh_i    for all i in this partition
 *        WH2[j] = a_dst · Wh_j    for all j (neighbors)
 *      via mxv().
 *   2. For each row i, read its non-zero neighbors from the adjacency FIFOs
 *      and compute e_ij = LeakyReLU(WH1[i] + WH2[col]).
 *   3. Emit:
 *        E_fifo         – raw scores for softmax (compute_attention2).
 *        EO_fifo        – raw scores for DDR write (writes).
 *        A_fifo         – adjacency values passed through for normalization.
 *        E_col_indices_fifo / E_rnnz_fifo – structure for downstream stages.
 *        max_fifo       – per-row maximum for numerically-stable softmax.
 *
 * When gat_mode == 0 or the layer is linear-only (gcn_path == 0),
 * the mxv() is skipped and only the adjacency structure is forwarded.
 *
 * @param model              Per-layer mode flags [layer][bit].
 * @param N_adj              Number of adjacency rows in this partition.
 * @param P_w                Per-layer output column widths.
 * @param A_buffer11         Pre-activation buffer from loop_fea (read side).
 * @param A_fifo_adj         Input: adjacency non-zero values.
 * @param col_indices_fifo_adj Input: adjacency column indices.
 * @param rnnz_fifo_adj      Input: per-row non-zero counts.
 * @param E_col_indices_fifo Output: column indices for softmax stage.
 * @param E_rnnz_fifo        Output: per-row nnz counts for softmax stage.
 * @param A_fifo             Output: adjacency values (structure copy).
 * @param E_fifo             Output: raw attention scores e_ij.
 * @param max_fifo           Output: per-row maximum e_ij.
 * @param ate_m              Attention parameter vector [a_src || a_dst] (2×C_WIDTH).
 * @param EO_fifo            Output: raw scores for DDR recording.
 * @param EO_rnnz_fifo       Output: nnz counts for DDR recording.
 * @param B_index            Current layer index.
 */
void prepare_attentional_mechanism_input2(
    ap_uint<1>  model[5][8],
    int         N_adj,
    ap_uint<8>  P_w[5],
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf> &A_buffer11,
#else
    buf                         A_buffer11,
#endif
    hls::stream<ATYPE> &A_fifo_adj,
    hls::stream<int>   &col_indices_fifo_adj,
    hls::stream<int>   &rnnz_fifo_adj,
    hls::stream<int>   &E_col_indices_fifo,
    hls::stream<int>   &E_rnnz_fifo,
    hls::stream<ATYPE> &A_fifo,
    hls::stream<TTYPE> &E_fifo,
    hls::stream<TTYPE> &max_fifo,
    BTYPE               ate_m[2 * C_WIDTH],
    hls::stream<TTYPE> &EO_fifo,
    hls::stream<int>   &EO_rnnz_fifo,
    int                 B_index
)
{
    /* ── Per-node attention score vectors ── */
    TTYPE WH1[A_HEIGHT] = {0};   // a_src · Wh_i  (source projections)
    TTYPE WH2[A_HEIGHT] = {0};   // a_dst · Wh_j  (destination projections)

    TTYPE relu_out, relu_in;
    TTYPE max_val, A_val;
    int   rnnz, col;

    /* ── Decode layer mode flags ── */
    ap_uint<8> P_w_attention = P_w[B_index];
    bool gat_mode  = model[B_index][5];
    bool linear_mode = model[B_index][6];
    bool sage_mode   = model[B_index][7];
    bool gcn_path    = !(sage_mode ^ linear_mode);   // active GNN path

    /* ── Step 1: Project node features onto attention vectors ── */
#if (PIPO_BLOCKS >= 2)
    hls::read_lock<buf> C_mxv(A_buffer11);
    if (gat_mode == 1 && gcn_path == 1)
        mxv(N_adj, P_w_attention, C_mxv, ate_m, WH1, WH2);
#else
    if (gat_mode == 1 && gcn_path == 1)
        mxv(N_adj, P_w_attention, A_buffer11, ate_m, WH1, WH2);
#endif

    /* ── Step 2: Compute per-edge scores and emit to downstream stages ── */
    if (gcn_path == 1)
    {
        LOOP_WH1: for (int i = 0; i < N_adj; i++)
        {
            rnnz = rnnz_fifo_adj.read();
            E_rnnz_fifo << rnnz;

            max_val = 0.0;

            LOOP_WH2: for (int j = 0; j < rnnz; j++)
            {
                #pragma HLS PIPELINE

                col   = col_indices_fifo_adj.read();
                A_val = A_fifo_adj.read();

                /* Forward adjacency structure to softmax stage */
                E_col_indices_fifo << col;
                A_fifo             << A_val;

                if (gat_mode == 1)
                {
                    relu_in = WH1[i] + WH2[col];

#if GATV2 == 1
                    /* GATv2: linear pre-activation (no LeakyReLU) */
                    relu_out = relu_in;
#else
                    /* Standard GAT: LeakyReLU with negative slope 0.2 */
                    relu_out = (relu_in >= 0) ? relu_in : relu_in * ITYPE(0.2);
#endif
                    if (relu_out > max_val)
                        max_val = relu_out;

                    E_fifo  << relu_out;   // → compute_attention2
                    EO_fifo << relu_out;   // → DDR via writes
                }
            }

            if (gat_mode == 1)
                max_fifo << max_val;
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * generate_attention_candidates - Per-partition inner loop of GAT score generation.
 *
 * Called once per row partition by prepare_attentional_mechanism_inputx4.
 * For each row i in this partition, reads its non-zero neighbors and computes:
 *
 *   e_ij = LeakyReLU( WH1[i] + WH2_col )
 *
 * where WH2_col is read from the appropriate quarter of the partitioned WH2
 * array (WH21..WH24), selected by the column index relative to N_block.
 *
 * @param gat_mode            True when attention scoring is active.
 * @param N_adj               Number of rows in this partition.
 * @param N_block             Number of rows per FEA_THREADS partition (= N_adj/4).
 * @param A_fifo_adj          Input: adjacency non-zero values.
 * @param col_indices_fifo_adj Input: adjacency column indices.
 * @param rnnz_fifo_adj       Input: per-row non-zero counts.
 * @param E_col_indices_fifo  Output: column indices for softmax stage.
 * @param E_rnnz_fifo         Output: per-row nnz counts for softmax stage.
 * @param A_fifo              Output: adjacency values (structure copy).
 * @param E_fifo              Output: raw attention scores e_ij.
 * @param max_fifo            Output: per-row maximum e_ij.
 * @param EO_fifo             Output: raw scores for DDR recording.
 * @param EO_rnnz_fifo        Output: nnz counts for DDR recording.
 * @param WH1                 Source projection vector for this partition.
 * @param WH21..WH24          Destination projection vectors for each of the 4 column partitions.
 */
void generate_attention_candidates(
    bool                gat_mode,
    int                 N_adj,
    int                 N_block,
    hls::stream<ATYPE> &A_fifo_adj,
    hls::stream<int>   &col_indices_fifo_adj,
    hls::stream<int>    rnnz_fifo_adj[SPMM_BLOCK],
    hls::stream<int>   &E_col_indices_fifo,
    hls::stream<int>    E_rnnz_fifo[SPMM_BLOCK],
    hls::stream<ATYPE> &A_fifo,
    hls::stream<TTYPE> &E_fifo,
    hls::stream<TTYPE> &max_fifo,
    hls::stream<TTYPE> &EO_fifo,
    hls::stream<int>   &EO_rnnz_fifo,
    TTYPE              *WH1,
    TTYPE              *WH21,
    TTYPE              *WH22,
    TTYPE              *WH23,
    TTYPE              *WH24
)
{
    LOOP_WH11: for (int i = 0; i < N_adj; i++)
    {
        int   rnnz    = rnnz_fifo_adj[0].read();
        TTYPE max_val = 0.0;
        TTYPE relu_out, relu_in;

        E_rnnz_fifo[0] << rnnz;

        LOOP_WH21: for (int j = 0; j < rnnz; j++)
        {
            #pragma HLS PIPELINE

            int   col   = col_indices_fifo_adj.read();
            TTYPE A_val = A_fifo_adj.read();

            /* Select WH2 from the correct column partition */
            TTYPE WH2_col;
            if      (col < N_block)          WH2_col = WH21[col];
            else if (col < N_block * 2)      WH2_col = WH22[col - N_block];
            else if (col < N_block * 3)      WH2_col = WH23[col - 2 * N_block];
            else                             WH2_col = WH24[col - 3 * N_block];

            E_col_indices_fifo << col;
            A_fifo             << A_val;

            if (gat_mode == 1)
            {
                relu_in = WH1[i] + WH2_col;

#if GATV2 == 1
                relu_out = relu_in;
#else
                relu_out = (relu_in >= 0) ? relu_in : relu_in * ITYPE(0.2);
#endif
                if (relu_out > max_val)
                    max_val = relu_out;

                E_fifo  << relu_out;
                EO_fifo << relu_out;
            }
        }

        if (gat_mode == 1)
            max_fifo << max_val;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * prepare_attentional_mechanism_inputx4 - Four-partition GAT attention preparation.
 *
 * Extends prepare_attentional_mechanism_input2 to four parallel row partitions.
 *
 * Steps:
 *   1. Acquire read locks on all four A_buffer tiles (written by loop_fea).
 *   2. Project all node features onto attention vectors via mxv4(), which
 *      computes WH1 (source) and WH2x1..WH2x4 (destination, one per column
 *      partition) for each of the four row partitions.
 *   3. Call generate_attention_candidates() independently for each partition;
 *      these run as DATAFLOW processes in the caller.
 *
 * The WH arrays are dimensioned A_HEIGHT/FEA_THREADS since each partition
 * handles only 1/FEA_THREADS of the full node set.
 *
 * @param gat_mode        True when attention scoring is active.
 * @param N_adj1..4       Number of rows in each partition (partition 4 may have remainder).
 * @param P_w             Number of attention projection dimensions.
 * @param A_buffer11..41  Pre-activation buffers from loop_fea (one per partition).
 * @param A_fifo_adj1..4  Input: adjacency values per partition.
 * @param col_indices_fifo_adj1..4 Input: column indices per partition.
 * @param rnnz_fifo_adj1..4 Input: per-row nnz counts per partition.
 * @param E_col_indices_fifo1..4 Output: column indices for softmax stages.
 * @param E_rnnz_fifo1..4 Output: nnz counts for softmax stages.
 * @param A_fifo1..4      Output: adjacency structure copies.
 * @param E_fifo1..4      Output: raw attention scores.
 * @param max_fifo1..4    Output: per-row maxima.
 * @param ate_m           Attention parameter vector [a_src || a_dst] (shared across partitions).
 * @param EO_fifo1..4     Output: raw scores for DDR recording.
 * @param EO_rnnz_fifo1..4 Output: nnz counts for DDR recording.
 */
void prepare_attentional_mechanism_inputx4(
    bool        gat_mode,
    int         N_adj1,
    int         N_adj2,
    int         N_adj3,
    int         N_adj4,
    int         P_w,
    hls::stream_of_blocks<buf> &A_buffer11,
    hls::stream_of_blocks<buf> &A_buffer21,
    hls::stream_of_blocks<buf> &A_buffer31,
    hls::stream_of_blocks<buf> &A_buffer41,
    hls::stream<ATYPE> &A_fifo_adj1,
    hls::stream<ATYPE> &A_fifo_adj2,
    hls::stream<ATYPE> &A_fifo_adj3,
    hls::stream<ATYPE> &A_fifo_adj4,
    hls::stream<int>   &col_indices_fifo_adj1,
    hls::stream<int>   &col_indices_fifo_adj2,
    hls::stream<int>   &col_indices_fifo_adj3,
    hls::stream<int>   &col_indices_fifo_adj4,
    hls::stream<int>    rnnz_fifo_adj1[SPMM_BLOCK],
    hls::stream<int>    rnnz_fifo_adj2[SPMM_BLOCK],
    hls::stream<int>    rnnz_fifo_adj3[SPMM_BLOCK],
    hls::stream<int>    rnnz_fifo_adj4[SPMM_BLOCK],
    hls::stream<int>   &E_col_indices_fifo1,
    hls::stream<int>   &E_col_indices_fifo2,
    hls::stream<int>   &E_col_indices_fifo3,
    hls::stream<int>   &E_col_indices_fifo4,
    hls::stream<int>    E_rnnz_fifo1[SPMM_BLOCK],
    hls::stream<int>    E_rnnz_fifo2[SPMM_BLOCK],
    hls::stream<int>    E_rnnz_fifo3[SPMM_BLOCK],
    hls::stream<int>    E_rnnz_fifo4[SPMM_BLOCK],
    hls::stream<ATYPE> &A_fifo1,
    hls::stream<ATYPE> &A_fifo2,
    hls::stream<ATYPE> &A_fifo3,
    hls::stream<ATYPE> &A_fifo4,
    hls::stream<TTYPE> &E_fifo1,
    hls::stream<TTYPE> &E_fifo2,
    hls::stream<TTYPE> &E_fifo3,
    hls::stream<TTYPE> &E_fifo4,
    hls::stream<TTYPE> &max_fifo1,
    hls::stream<TTYPE> &max_fifo2,
    hls::stream<TTYPE> &max_fifo3,
    hls::stream<TTYPE> &max_fifo4,
    BTYPE               ate_m[2 * C_WIDTH],
    hls::stream<TTYPE> &EO_fifo1,
    hls::stream<TTYPE> &EO_fifo2,
    hls::stream<TTYPE> &EO_fifo3,
    hls::stream<TTYPE> &EO_fifo4,
    hls::stream<int>   &EO_rnnz_fifo1,
    hls::stream<int>   &EO_rnnz_fifo2,
    hls::stream<int>   &EO_rnnz_fifo3,
    hls::stream<int>   &EO_rnnz_fifo4
)
{
    /* ── Per-partition attention projection arrays ──────────────────────
     * WH1x  : source projection for partition x  (a_src · Wh_i)
     * WH2yx : destination projection for column-partition y, row-partition x
     *         (a_dst · Wh_j), where j falls in column partition y.
     * Dimensioned A_HEIGHT/FEA_THREADS since each partition covers 1/4 of nodes.
     * ─────────────────────────────────────────────────────────────────── */
    TTYPE WH11[A_HEIGHT / FEA_THREADS];
    TTYPE WH211[A_HEIGHT / FEA_THREADS];
    TTYPE WH221[A_HEIGHT / FEA_THREADS];
    TTYPE WH231[A_HEIGHT / FEA_THREADS];
    TTYPE WH241[A_HEIGHT / FEA_THREADS];

    TTYPE WH12[A_HEIGHT / FEA_THREADS];
    TTYPE WH212[A_HEIGHT / FEA_THREADS];
    TTYPE WH222[A_HEIGHT / FEA_THREADS];
    TTYPE WH232[A_HEIGHT / FEA_THREADS];
    TTYPE WH242[A_HEIGHT / FEA_THREADS];

    TTYPE WH13[A_HEIGHT / FEA_THREADS];
    TTYPE WH213[A_HEIGHT / FEA_THREADS];
    TTYPE WH223[A_HEIGHT / FEA_THREADS];
    TTYPE WH233[A_HEIGHT / FEA_THREADS];
    TTYPE WH243[A_HEIGHT / FEA_THREADS];

    TTYPE WH14[A_HEIGHT / FEA_THREADS];
    TTYPE WH214[A_HEIGHT / FEA_THREADS];
    TTYPE WH224[A_HEIGHT / FEA_THREADS];
    TTYPE WH234[A_HEIGHT / FEA_THREADS];
    TTYPE WH244[A_HEIGHT / FEA_THREADS];

    /* ── Step 1: Acquire read locks and project all features ── */
    hls::read_lock<buf> C_mxv1(A_buffer11);
    hls::read_lock<buf> C_mxv2(A_buffer21);
    hls::read_lock<buf> C_mxv3(A_buffer31);
    hls::read_lock<buf> C_mxv4(A_buffer41);

    if (gat_mode == 1)
        mxv4(N_adj1, N_adj2, N_adj3, N_adj4, P_w,
             C_mxv1, C_mxv2, C_mxv3, C_mxv4,
             ate_m,
             WH11,  WH211, WH212, WH213, WH214,
             WH12,  WH221, WH222, WH223, WH224,
             WH13,  WH231, WH232, WH233, WH234,
             WH14,  WH241, WH242, WH243, WH244);

    /* ── Step 2: Generate per-edge attention scores for each partition ── */
    generate_attention_candidates(
        gat_mode, N_adj1, N_adj1,
        A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
        E_col_indices_fifo1, E_rnnz_fifo1,
        A_fifo1, E_fifo1, max_fifo1, EO_fifo1, EO_rnnz_fifo1,
        WH11, WH211, WH221, WH231, WH241);

    generate_attention_candidates(
        gat_mode, N_adj2, N_adj1,
        A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,
        E_col_indices_fifo2, E_rnnz_fifo2,
        A_fifo2, E_fifo2, max_fifo2, EO_fifo2, EO_rnnz_fifo2,
        WH12, WH212, WH222, WH232, WH242);

    generate_attention_candidates(
        gat_mode, N_adj3, N_adj1,
        A_fifo_adj3, col_indices_fifo_adj3, rnnz_fifo_adj3,
        E_col_indices_fifo3, E_rnnz_fifo3,
        A_fifo3, E_fifo3, max_fifo3, EO_fifo3, EO_rnnz_fifo3,
        WH13, WH213, WH223, WH233, WH243);

    generate_attention_candidates(
        gat_mode, N_adj4, N_adj1,
        A_fifo_adj4, col_indices_fifo_adj4, rnnz_fifo_adj4,
        E_col_indices_fifo4, E_rnnz_fifo4,
        A_fifo4, E_fifo4, max_fifo4, EO_fifo4, EO_rnnz_fifo4,
        WH14, WH214, WH224, WH234, WH244);
}
// =============================================================================================
// =============================================================================================
/**
 * func_rnnz - Stage 1 of the tiled softmax pipeline.
 *
 * Reads ATEN_BLOCK cumulative non-zero counts and row-max values from
 * the input FIFOs and fans them out into per-slot staging FIFOs for the
 * next stage (func_exp).  Tail slots beyond N_adj are padded with the
 * last valid values so downstream stages always see a full ATEN_BLOCK.
 *
 * @param i          Current tile base row index.
 * @param N_adj      Total number of adjacency rows in this partition.
 * @param max_fifo   Input: per-row maximum e_ij (for numerical stability).
 * @param rnnz_fifo  Input: cumulative per-row non-zero counts.
 * @param rnnz_f     Output: per-slot cumulative nnz staging FIFOs.
 * @param val_f      Output: per-slot row-max staging FIFOs.
 */
void func_rnnz(
    int                 i,
    int                 N_adj,
    hls::stream<ATYPE> &max_fifo,
    hls::stream<int>    rnnz_fifo[SPMM_BLOCK],
    hls::stream<int>    rnnz_f[ATEN_BLOCK],
    hls::stream<ATYPE>  val_f[ATEN_BLOCK]
)
{
    int   rnnz_val = 0;   // last valid cumulative nnz (used for tail padding)
    ATYPE max_val  = 0;   // last valid row max (used for tail padding)

    LOOP_RNNZ: for (int z = 0; z < ATEN_BLOCK; z++)
    {
        #pragma HLS PIPELINE II=1

        if ((i + z) < N_adj)
        {
            /* Active row: read from input FIFOs */
            rnnz_val = rnnz_fifo[0].read();
            max_val  = max_fifo.read();
        }
        /* else: tail slot – reuse last valid values for padding */

        rnnz_f[z] << rnnz_val;
        val_f[z]  << max_val;
    }
}

// =============================================================================================
// =============================================================================================
/**
 * func_exp - Stage 2 of the tiled softmax pipeline.
 *
 * Reads the ATEN_BLOCK row metadata (nnz, max) from staging FIFOs, then
 * for each non-zero in the tile:
 *   1. Reads the raw attention score e_ij from E_fifo.
 *   2. Computes  support = exp(e_ij - max_i)  (numerically stable).
 *   3. Accumulates the per-row sum.
 *   4. Forwards support to support_f for the division stage (func_div).
 *
 * After all non-zeros are processed, forwards row sums, row maxima, and
 * cumulative nnz arrays to func_fixed via staging FIFOs.
 *
 * @param rnnz_f    Input:  per-slot cumulative nnz from func_rnnz.
 * @param val_f     Input:  per-slot row max from func_rnnz.
 * @param E_fifo    Input:  raw attention scores e_ij.
 * @param sum_f     Output: per-slot partial row sums → func_fixed.
 * @param val_f2    Output: per-slot row maxima passthrough → func_fixed.
 * @param rnnz_f2   Output: per-slot cumulative nnz passthrough → func_fixed.
 * @param support_f Output: exp(e_ij - max_i) values → func_div.
 */
void func_exp(
    hls::stream<int>    rnnz_f[ATEN_BLOCK],
    hls::stream<ATYPE>  val_f[ATEN_BLOCK],
    hls::stream<FTYPE> &E_fifo,
    hls::stream<ATYPE>  sum_f[ATEN_BLOCK],
    hls::stream<ATYPE>  val_f2[ATEN_BLOCK],
    hls::stream<int>    rnnz_f2[ATEN_BLOCK],
    hls::stream<ATYPE> &support_f
)
{
    int   val_rnnz[ATEN_BLOCK + 1];
    ATYPE val_max[ATEN_BLOCK];
    ATYPE sum[ATEN_BLOCK];
    ATYPE support;
    ATYPE attention_candidate;

    /* ── Drain staging FIFOs into local arrays ── */
    val_rnnz[0] = 0;
    LOOP_1: for (int z = 0; z < ATEN_BLOCK; z++)
    {
        #pragma HLS UNROLL
        val_rnnz[z + 1] = rnnz_f[z].read();
        val_max[z]      = val_f[z].read();
        sum[z]          = 0;
    }

    /* ── Compute exp(e_ij - max_i) and accumulate per-row sums ── */
    LOOP_SOFTMAX2: for (int j = 0; j < val_rnnz[ATEN_BLOCK]; j++)
    {
        #pragma HLS PIPELINE II=1

        attention_candidate = E_fifo.read();

        /* Determine which row slot non-zero j belongs to */
        int row_index1 = 0;
        for (int k = 0; k < ATEN_BLOCK; k++)
        {
            if ((j >= val_rnnz[k]) && (j < val_rnnz[k + 1]))
                row_index1 = k;
        }

#ifdef FIXEDPOINT
        support = hls::exp(attention_candidate - val_max[row_index1]);
#else
        support = hls::half_exp(attention_candidate - val_max[row_index1]);
#endif
        sum[row_index1] += support;
        support_f       << support;
    }

    /* ── Forward per-row metadata to func_fixed ── */
    LOOP_2: for (int z = 0; z < ATEN_BLOCK; z++)
    {
        #pragma HLS UNROLL
        sum_f[z]   << sum[z];
        val_f2[z]  << val_max[z];
        rnnz_f2[z] << val_rnnz[z + 1];
    }
}

// =============================================================================================
// =============================================================================================
/**
 * func_fixed - Stage 3 of the tiled softmax pipeline.
 *
 * Corrects the row-wise softmax denominator to account for zero-masked edges.
 * PyTorch GAT computes:
 *   attention = softmax(torch.where(adj > 0, e, -9e15))
 * On sparse inputs, the zero entries are never stored.  This stage adds their
 * contribution back:
 *   corrected_sum_i = Σ_{j∈nnz} exp(e_ij - max_i)
 *                   + (N_adj - nnz_i) * exp(-9e3 - max_i)
 *
 * @param N_adj    Total number of nodes (= number of possible edges per row).
 * @param sum_f    Input:  partial row sums from func_exp.
 * @param val_f2   Input:  row maxima from func_exp.
 * @param rnnz_f2  Input:  cumulative nnz from func_exp.
 * @param sum_f2   Output: corrected row sums → func_div.
 * @param rnnz_f3  Output: cumulative nnz passthrough → func_div.
 */
void func_fixed(
    int                N_adj,
    hls::stream<ATYPE> sum_f[ATEN_BLOCK],
    hls::stream<ATYPE> val_f2[ATEN_BLOCK],
    hls::stream<int>   rnnz_f2[ATEN_BLOCK],
    hls::stream<ATYPE> sum_f2[ATEN_BLOCK],
    hls::stream<int>   rnnz_f3[ATEN_BLOCK]
)
{
    int   val_rnnz2[ATEN_BLOCK + 1];
    ATYPE val_max2[ATEN_BLOCK];
    ATYPE val_sum[ATEN_BLOCK];
    ATYPE fixed_val     = ATYPE(-9e3);  // large-negative proxy for masked edges
    ATYPE fixed_support;

    /* ── Drain staging FIFOs into local arrays ── */
    val_rnnz2[0] = 0;
    LOOP_3: for (int z = 0; z < ATEN_BLOCK; z++)
    {
        #pragma HLS UNROLL
        val_rnnz2[z + 1] = rnnz_f2[z].read();
        val_max2[z]      = val_f2[z].read();
        val_sum[z]       = sum_f[z].read();
    }

    /* ── Add masked-edge contribution to each row sum ── */
    LOOP_FIXED: for (int z = 0; z < ATEN_BLOCK; z++)
    {
        #pragma HLS PIPELINE II=1

        int masked_count = N_adj - val_rnnz2[z + 1] + val_rnnz2[z];

#ifdef FIXEDPOINT
        fixed_support = hls::exp(fixed_val - val_max2[z]);
#else
        fixed_support = hls::half_exp(fixed_val - val_max2[z]);
#endif
        fixed_support *= masked_count;
        sum_f2[z] << (val_sum[z] + fixed_support);
    }

    /* ── Forward cumulative nnz to func_div ── */
    LOOP_4: for (int z = 0; z < ATEN_BLOCK; z++)
    {
        #pragma HLS UNROLL
        rnnz_f3[z] << val_rnnz2[z + 1];
    }
}
// =============================================================================================
// =============================================================================================
/**
 * func_div - Final softmax division stage: normalizes exp(e_ij) by the row sum.
 *
 * Consumes:
 *   - support_f        : exp(e_ij - max_i) values produced by func_exp.
 *   - A_fifo           : original adjacency values (passed through to output).
 *   - col_indices_fifo : column indices (passed through to output).
 *   - sum_f2           : per-row softmax denominators (sum of exp values).
 *   - rnnz_f3          : cumulative non-zero counts per ATEN_BLOCK row slot.
 *
 * Produces:
 *   - val_att_fifo     : softmax-normalized attention weights  α_ij = exp(e_ij) / Σ exp(e_ik).
 *   - col_att_fifo     : column indices matching val_att_fifo.
 *   - rnnz_att_fifo[0] : per-row non-zero counts for the downstream SpMM.
 */
void func_div(
    hls::stream<int>   rnnz_att_fifo[SPMM_BLOCK],  // output: per-row nnz counts
    hls::stream<ATYPE> &A_fifo,                     // input:  adjacency values (pass-through)
    hls::stream<ATYPE> &support_f,                  // input:  exp(e_ij - max_i) values
    hls::stream<int>   &col_indices_fifo,            // input:  column indices
    hls::stream<ATYPE> sum_f2[ATEN_BLOCK],          // input:  per-row softmax denominators
    hls::stream<int>   rnnz_f3[ATEN_BLOCK],         // input:  cumulative nnz per row slot
    hls::stream<ATYPE> &val_att_fifo,               // output: normalized attention weights
    hls::stream<int>   &col_att_fifo                // output: column indices
)
{
    /* ── Drain cumulative nnz and row sums; emit per-row nnz to downstream ── */
    int   val_rnnz3[ATEN_BLOCK + 1];
    ATYPE val_sum2[ATEN_BLOCK];

    val_rnnz3[0] = 0;
    int rnnz_old = 0;

    LOOP_5: for (int z = 0; z < ATEN_BLOCK; z++)
    {
        #pragma HLS UNROLL
        int rnnz_val    = rnnz_f3[z].read();
        val_rnnz3[z + 1] = rnnz_val;
        val_sum2[z]      = sum_f2[z].read();

        /* Emit per-row count (delta from previous cumulative value) */
        rnnz_att_fifo[0] << (rnnz_val - rnnz_old);
        rnnz_old = rnnz_val;
    }

    /* ── Normalize: α_ij = exp(e_ij - max_i) / Σ_k exp(e_ik - max_i) ── */
    LOOP_SOFTMAX4: for (int j = 0; j < val_rnnz3[ATEN_BLOCK]; j++)
    {
        #pragma HLS PIPELINE II=1

        /* Determine which row slot this non-zero belongs to */
        int row_index2 = 0;
        for (int k = 0; k < ATEN_BLOCK; k++)
        {
            if ((j >= val_rnnz3[k]) && (j < val_rnnz3[k + 1]))
                row_index2 = k;
        }

        ATYPE val = A_fifo.read();
        int   col = col_indices_fifo.read();

        /* Normalize by the row softmax denominator */
        ATYPE out_val = support_f.read() / val_sum2[row_index2];

        col_att_fifo << col;
        val_att_fifo << out_val;
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 * The func_loops variant of compute_attention2 (loop-decomposed into
 * func_rnnz / func_exp / func_fixed / func_div sub-tasks) is kept here for
 * reference but is disabled at compile time.  The active implementation is
 * the #ifndef func_loops version below, which inlines all stages into a
 * single function for better II.
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifdef func_loops

void compute_attention2(
    bool                gat_mode,
    int                 N_adj,
    hls::stream<ATYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    hls::stream<int>    rnnz_fifo[SPMM_BLOCK],
    hls::stream<ATYPE> &E_fifo,
    hls::stream<ATYPE> &max_fifo,
    hls::stream<ATYPE> &val_att_fifo,
    hls::stream<int>   &col_att_fifo,
    hls::stream<int>    rnnz_att_fifo[SPMM_BLOCK]
)
{
    hls::stream<int>   rnnz_f[ATEN_BLOCK];
    #pragma HLS STREAM variable=rnnz_f depth=FIFO_DEPTH

    hls::stream<int>   rnnz_f2[ATEN_BLOCK];
    #pragma HLS STREAM variable=rnnz_f2 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_f3[ATEN_BLOCK];
    #pragma HLS STREAM variable=rnnz_f3 depth=FIFO_DEPTH

    hls::stream<ATYPE> val_f[ATEN_BLOCK];
    #pragma HLS STREAM variable=val_f depth=FIFO_DEPTH

    hls::stream<ATYPE> val_f2[ATEN_BLOCK];
    #pragma HLS STREAM variable=val_f2 depth=FIFO_DEPTH

    hls::stream<ATYPE> sum_f[ATEN_BLOCK];
    #pragma HLS STREAM variable=sum_f depth=FIFO_DEPTH

    hls::stream<ATYPE> sum_f2[ATEN_BLOCK];
    #pragma HLS STREAM variable=sum_f2 depth=FIFO_DEPTH

    hls::stream<ATYPE> support_f;
    #pragma HLS STREAM variable=support_f depth=FIFO_DEPTH_ATTN2

    if (gat_mode == 1)
    {
        ATEN_LOOP: for (int i = 0; i < N_adj; i += ATEN_BLOCK)
        {
            #pragma HLS DATAFLOW
            func_rnnz(i, N_adj, max_fifo, rnnz_fifo, rnnz_f, val_f);
            func_exp(rnnz_f, val_f, E_fifo, sum_f, val_f2, rnnz_f2, support_f);
            func_fixed(N_adj, sum_f, val_f2, rnnz_f2, sum_f2, rnnz_f3);
            func_div(rnnz_att_fifo, A_fifo, support_f, col_indices_fifo,
                     sum_f2, rnnz_f3, val_att_fifo, col_att_fifo);
        }
    }
    else
    {
        /* GCN pass-through: forward adjacency values unchanged */
        LOOP_GCN: for (int i = 0; i < N_adj; i += ATEN_BLOCK)
        {
            #pragma HLS DATAFLOW
            int rnnz_old = 0;
            int rnnz_val;

            LOOP_RNNZ2: for (int z = 0; z < ATEN_BLOCK; z++)
            {
                #pragma HLS PIPELINE II=1
                if ((i + z) < N_adj)
                {
                    rnnz_val = rnnz_fifo[0].read();
                    max_fifo.read();                    // consume and discard max
                    rnnz_att_fifo[0] << (rnnz_val - rnnz_old);
                    rnnz_old = rnnz_val;
                }
            }

            LOOP_SOFTMAX5: for (int j = 0; j < rnnz_old; j++)
            {
                #pragma HLS PIPELINE II=1
                ATYPE val = A_fifo.read();
                int   col = col_indices_fifo.read();
                col_att_fifo << col;
                val_att_fifo << val;
            }
        }
    }
}

#endif  /* func_loops */

// =============================================================================================
// =============================================================================================
/*
 * compute_attention2 - Row-wise softmax attention over a sparse adjacency.
 *
 * Active implementation (~218K cycles with ATEN_BLOCK = 1).
 *
 * Each ATEN_BLOCK rows are processed together in a tiled DATAFLOW pipeline:
 *
 *   Stage 1 – LOOP_RNNZ:
 *     Reads per-row non-zero counts and row max values from the input FIFOs,
 *     emits delta-encoded counts to rnnz_att_fifo (downstream SpMM needs
 *     per-row counts, not cumulative indices).
 *
 *   Stage 2 – LOOP_SOFTMAX2:
 *     For each non-zero: reads e_ij from E_fifo, computes
 *       support = exp(e_ij - max_i)
 *     and accumulates the per-row sum.  Emits support values to support_f.
 *
 *   Stage 3 – LOOP_FIXED:
 *     Adds the contribution of zero-masked edges
 *       (N_adj - nnz_i) * exp(-9e3 - max_i)
 *     to the per-row sum to reproduce PyTorch's
 *       torch.where(adj > 0, e, -9e15)  →  softmax
 *     behavior on sparse inputs.
 *
 *   Stage 4 – LOOP_SOFTMAX4:
 *     Normalizes each support value by the corrected row sum:
 *       α_ij = support / corrected_sum_i
 *     Emits normalized weights to val_att_fifo and column indices to
 *     col_att_fifo.  Also mirrors values to SO_fifo for DDR recording.
 *
 * When gat_mode == 0 (plain GCN): adjacency values are forwarded unchanged.
 * When gcn_path  == 0 (linear-only layer): entire function is skipped.
 *
 * @param model          Per-layer mode flags: [5]=gat, [6]=linear, [7]=sage.
 * @param N_adj          Number of rows in the adjacency partition.
 * @param A_fifo         Adjacency values (structure copy from prepare stage).
 * @param col_indices_fifo Column indices matching A_fifo.
 * @param rnnz_fifo      Cumulative per-row nnz from prepare stage.
 * @param E_fifo         Raw attention scores e_ij from prepare stage.
 * @param max_fifo       Per-row maximum e_ij for numerical stability.
 * @param val_att_fifo   Output: softmax-normalized attention weights.
 * @param col_att_fifo   Output: column indices matching val_att_fifo.
 * @param rnnz_att_fifo  Output: per-row nnz counts for downstream SpMM.
 * @param SO_fifo        Output: mirror of val_att_fifo written to DDR.
 * @param SO_rnnz_fifo   Output: nnz counts for SO_fifo (DDR write sizing).
 * @param B_index        Current layer index (selects model flags).
 */
#ifndef func_loops

void compute_attention2(
    ap_uint<1>          model[5][8],
    int                 N_adj,
    hls::stream<ATYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    hls::stream<int>   &rnnz_fifo,
    hls::stream<TTYPE> &E_fifo,
    hls::stream<TTYPE> &max_fifo,
    hls::stream<TTYPE> &val_att_fifo,
    hls::stream<int>   &col_att_fifo,
    hls::stream<int>   &rnnz_att_fifo,
    hls::stream<TTYPE> &SO_fifo,
    hls::stream<int>   &SO_rnnz_fifo,
    int                 B_index
)
{
    /* Scalar temporaries */
    int   col;
    TTYPE val, attention_candidate;
    TTYPE fixed_val     = TTYPE(-9e3);  // large-negative proxy for masked edges
    TTYPE sum[ATEN_BLOCK];
    TTYPE support;
    TTYPE fixed_support;

    /* ── Internal inter-stage FIFOs ── */

    /* support values between Stage 2 and Stage 4 (large: nnz-sized) */
    hls::stream<TTYPE> support_f;
    #pragma HLS STREAM variable=support_f depth=FIFO_DEPTH_ATTN2
    #pragma HLS bind_storage variable=support_f type=FIFO impl=URAM

    /* Cumulative nnz staging between Stage 1 and Stage 2/3 */
    hls::stream<int>   rnnz_f[ATEN_BLOCK];
    #pragma HLS STREAM variable=rnnz_f depth=FIFO_DEPTH

    hls::stream<int>   rnnz_f2[ATEN_BLOCK];
    #pragma HLS STREAM variable=rnnz_f2 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_f3[ATEN_BLOCK];
    #pragma HLS STREAM variable=rnnz_f3 depth=FIFO_DEPTH

    /* Row max staging between Stage 1 and Stage 2/3 */
    hls::stream<TTYPE> val_f[ATEN_BLOCK];
    #pragma HLS STREAM variable=val_f depth=FIFO_DEPTH

    hls::stream<TTYPE> val_f2[ATEN_BLOCK];
    #pragma HLS STREAM variable=val_f2 depth=FIFO_DEPTH

    /* Row sum staging between Stage 2 and Stage 3/4 */
    hls::stream<TTYPE> sum_f[ATEN_BLOCK];
    #pragma HLS STREAM variable=sum_f depth=FIFO_DEPTH

    hls::stream<TTYPE> sum_f2[ATEN_BLOCK];
    #pragma HLS STREAM variable=sum_f2 depth=FIFO_DEPTH

    /* ── Decode layer mode flags ── */
    bool gat_mode  = model[B_index][5];
    bool linear_mode = model[B_index][6];
    bool sage_mode   = model[B_index][7];
    bool gcn_path    = !(linear_mode ^ sage_mode); // active GNN path (not linear-only)

    if (gcn_path == 1)
    {
        /* ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
         * GAT path: compute full softmax attention.
         * ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── */
        if (gat_mode == 1)
        {
            ATEN_LOOP: for (int i = 0; i < N_adj; i += ATEN_BLOCK)
            {
                #pragma HLS DATAFLOW

                /* ── Stage 1: Read nnz counts and row maxima ── */
                int   rnnz_old = 0;
                ITYPE max_old  = 0;

                LOOP_RNNZ: for (int z = 0; z < ATEN_BLOCK; z++)
                {
                    #pragma HLS PIPELINE II=1
                    int   rnnz_val;
                    ITYPE max_val;

                    if ((i + z) < N_adj)
                    {
                        rnnz_val = rnnz_fifo.read();
                        max_val  = max_fifo.read();
                        rnnz_att_fifo << (rnnz_val - rnnz_old);   // delta nnz → SpMM
                        rnnz_old = rnnz_val;
                        max_old  = max_val;
                    }
                    else
                    {
                        /* Pad tail slots with last valid values */
                        rnnz_val = rnnz_old;
                        max_val  = max_old;
                    }

                    rnnz_f[z] << rnnz_val;
                    val_f[z]  << max_val;
                }

                /* ── Stage 2: Drain staging FIFOs into local arrays ── */
                int   val_rnnz[4];
                ATYPE val_max[4];

                LOOP_1: for (int z = 0; z < ATEN_BLOCK; z++)
                {
                    #pragma HLS UNROLL
                    val_rnnz[z] = rnnz_f[z].read();
                    val_max[z]  = val_f[z].read();
                    sum[z]      = 0;
                }

                /* ── Stage 2 (cont.): Compute exp(e_ij - max_i) and accumulate row sums ── */
                LOOP_SOFTMAX2: for (int j = 0; j < val_rnnz[ATEN_BLOCK - 1]; j++)
                {
                    #pragma HLS PIPELINE II=1

                    attention_candidate = E_fifo.read();

                    /* Map non-zero index j to its row slot */
                    int row_index1;
                    if      (j < val_rnnz[0]) row_index1 = 0;
                    else if (j < val_rnnz[1]) row_index1 = 1;
                    else if (j < val_rnnz[2]) row_index1 = 2;
                    else                       row_index1 = 3;

                    /* Numerically-stable exponential */
#ifdef FIXEDPOINT
                    support = hls::exp(attention_candidate - val_max[row_index1]);
#else
                    support = hls::half_exp(attention_candidate - val_max[row_index1]);
#endif
                    sum[row_index1] += support;
                    support_f       << support;
                }

                /* Forward row sums and index arrays to Stage 3 */
                LOOP_2: for (int z = 0; z < ATEN_BLOCK; z++)
                {
                    #pragma HLS UNROLL
                    sum_f[z]   << sum[z];
                    val_f2[z]  << val_max[z];
                    rnnz_f2[z] << val_rnnz[z];
                }

                /* ── Stage 3: Correct row sum for zero-masked edges ── */
                /* Adds  (N_adj - nnz_i) * exp(-9e3 - max_i) to each row sum,
                 * matching torch.where(adj > 0, e, -9e15) → softmax semantics. */
                int   val_rnnz2[ATEN_BLOCK + 1];
                TTYPE val_max2[ATEN_BLOCK];
                TTYPE val_sum[ATEN_BLOCK];

                val_rnnz2[0] = 0;

                LOOP_3: for (int z = 0; z < ATEN_BLOCK; z++)
                {
                    #pragma HLS UNROLL
                    val_rnnz2[z + 1] = rnnz_f2[z].read();
                    val_max2[z]      = val_f2[z].read();
                    val_sum[z]       = sum_f[z].read();
                }

                LOOP_FIXED: for (int z = 0; z < ATEN_BLOCK; z++)
                {
                    #pragma HLS PIPELINE II=1
                    int   masked_count = N_adj - val_rnnz2[z + 1] + val_rnnz2[z];
#ifdef FIXEDPOINT
                    fixed_support = hls::exp(fixed_val - val_max2[z]);
#else
                    fixed_support = hls::half_exp(fixed_val - val_max2[z]);
#endif
                    fixed_support *= masked_count;
                    sum_f2[z]  << (val_sum[z] + fixed_support);
                    rnnz_f3[z] << val_rnnz2[z + 1];
                }

                /* ── Stage 4: Normalize by corrected row sum ── */
                int   val_rnnz3[4];
                TTYPE val_sum2[4];

                LOOP_5: for (int z = 0; z < ATEN_BLOCK; z++)
                {
                    #pragma HLS UNROLL
                    val_rnnz3[z] = rnnz_f3[z].read();
                    val_sum2[z]  = sum_f2[z].read();
                }

                LOOP_SOFTMAX4: for (int j = 0; j < val_rnnz3[ATEN_BLOCK - 1]; j++)
                {
                    #pragma HLS PIPELINE II=1

                    /* Map non-zero index j to its row slot */
                    int row_index2;
                    if      (j < val_rnnz3[0]) row_index2 = 0;
                    else if (j < val_rnnz3[1]) row_index2 = 1;
                    else if (j < val_rnnz3[2]) row_index2 = 2;
                    else                        row_index2 = 3;

                    val = (ATYPE)(A_fifo.read());
                    col = col_indices_fifo.read();

                    /* α_ij = exp(e_ij - max_i) / corrected_sum_i */
                    TTYPE out_val = support_f.read() / val_sum2[row_index2];

                    col_att_fifo << col;
                    val_att_fifo << out_val;   // to downstream SpMM (no cast: preserve precision)
                    SO_fifo      << out_val;   // mirror to DDR recording
                }
            }
        }

        /* ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
         * GCN pass-through: forward adjacency values unchanged.
         * ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── */
        else
        {
            LOOP_GCN: for (int i = 0; i < N_adj; i += ATEN_BLOCK)
            {
                #pragma HLS DATAFLOW
                int rnnz[ATEN_BLOCK + 1];
                rnnz[0] = 0;

                /* Emit per-row nnz delta counts */
                LOOP_RNNZ2: for (int z = 0; z < ATEN_BLOCK; z++)
                {
                    #pragma HLS PIPELINE II=1
                    rnnz[z + 1] = rnnz_fifo.read();
                    rnnz_att_fifo << (rnnz[z + 1] - rnnz[z]);
                }

                /* Forward adjacency values without modification */
                LOOP_SOFTMAX5: for (int j = 0; j < rnnz[ATEN_BLOCK]; j++)
                {
                    #pragma HLS PIPELINE II=1
                    val = A_fifo.read();
                    col = col_indices_fifo.read();
                    col_att_fifo << col;
                    val_att_fifo << (TTYPE)(val);
                }
            }
        }

    }  /* end gcn_path */
}

#endif  /* !func_loops */

// =============================================================================================
// =============================================================================================
/**
 * loop_attention - Per-edge attention scoring and softmax normalization for GAT layers.
 *
 * For each layer iteration (B_index), this function implements the GAT attention pipeline:
 *
 *   When GAT_ENABLE == 1  (attention active):
 *     1. Load attention parameter vector (ate_m) from DDR, optionally quantized.
 *     2. Read the sparse adjacency matrix (CSR or COO) into internal FIFOs.
 *     3. prepare_attentional_mechanism_input*:
 *          - Reads A_buffer (pre-activations from loop_fea).
 *          - Computes per-edge scores  e_ij = LeakyReLU(aᵀ [h_i || h_j]).
 *          - Emits raw scores (EO_fifo) and passes adj structure onward.
 *     4. writes / writesx4: write raw edge scores (E1) to DDR.
 *     5. compute_attention2:
 *          - Applies numerically-stable row-wise softmax over EO scores.
 *          - Emits attention-weighted sparse values (val_att_fifo) for loop_adj.
 *          - Emits softmax outputs (SO_fifo).
 *     6. writes / writesx4: write softmax values (S1) to DDR.
 *
 *   When GAT_ENABLE == 0  (plain GCN / no attention):
 *     Reads the adjacency matrix directly into the output attention FIFOs
 *     (val_att_fifo, col_att_fifo, rnnz_att_fifo) without modification.
 *
 * Parallelism is controlled at compile time by:
 *   ADJ_THREADS   – number of row partitions processed in parallel (1 or 4).
 *   GAT_ENABLE    – enables attention scoring; 0 = pass-through.
 *   FAST_ATTENTION– selects the optimised fused attention kernel.
 *   COO_MODE      – selects sparse format: 0 = CSR, 1 = COO.
 *   PIPO_BLOCKS   – enables ping-pong layer loop when >= 2.
 */
void loop_attention(
    /* ── Per-layer dequantization factors ── */
    float       deq_factor[5],

    /* ── Quantization parameters ── */
    int         beta_qu,                    // zero-point shift
    int         f_align,                    // fractional alignment bits
    float       quantization_scale_adj,     // adjacency value scale
    float       quantization_scale_w[5],    // per-layer weight scales

    /* ── Per-layer mode flags ── */
    ap_uint<1>  model[5][8],

    /* ── Non-zero counts for each adjacency partition ── */
    int         nnz_adj1,
    int         nnz_adj2,
    int         nnz_adj3,
    int         nnz_adj4,

    /* ── CSR row pointers for each adjacency partition ── */
    int        *rowPtr_adj1,
    int        *rowPtr_adj2,
    int        *rowPtr_adj3,
    int        *rowPtr_adj4,

    /* ── CSR column indices for each adjacency partition ── */
    int        *columnIndex_adj1,
    int        *columnIndex_adj2,
    int        *columnIndex_adj3,
    int        *columnIndex_adj4,

    /* ── CSR values for each adjacency partition ── */
    INTYPE     *values_adj1,
    INTYPE     *values_adj2,
    INTYPE     *values_adj3,
    INTYPE     *values_adj4,

    /* ── Adjacency dimensions ── */
    int         N_adj,                      // number of adjacency rows
    int         M_adj,                      // number of adjacency columns
    ap_uint<8>  P_w[5],                     // per-layer output column widths

    /* ── Attention parameter vector (DDR): concatenated [a_src || a_dst] ── */
    INTYPE     *A,

    /* ── Pre-activation buffers from loop_fea (one per row partition) ── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf> &A_buffer11,
    hls::stream_of_blocks<buf> &A_buffer21,
#else
    buf                         A_buffer11,
    hls::stream_of_blocks<buf> &A_buffer21,
#endif
    hls::stream_of_blocks<buf> &A_buffer31,
    hls::stream_of_blocks<buf> &A_buffer41,

    /* ── DDR outputs: raw edge scores (E1) and softmax values (S1) ── */
    OUTTYPE    *E1,
    OUTTYPE    *S1,

    /* ── Output attention-weighted sparse streams → loop_adj (one set per partition) ── */
    hls::stream<int>   &rnnz_att_fifo1,
    hls::stream<int>   &col_att_fifo1,
    hls::stream<TTYPE> &val_att_fifo1,

    hls::stream<int>   &rnnz_att_fifo2,
    hls::stream<int>   &col_att_fifo2,
    hls::stream<TTYPE> &val_att_fifo2,

    hls::stream<int>   &rnnz_att_fifo3,
    hls::stream<int>   &col_att_fifo3,
    hls::stream<TTYPE> &val_att_fifo3,

    hls::stream<int>   &rnnz_att_fifo4,
    hls::stream<int>   &col_att_fifo4,
    hls::stream<TTYPE> &val_att_fifo4,

    /* ── Number of GNN layers to iterate over ── */
    int         layer_loop
)
{
    /* ────────────────────────────────────────────────────────────────────
     * Internal FIFOs – all mapped to URAM for the large attention buffers.
     *
     * Pipeline stages inside each partition:
     *
     *   reada2_*   →  [A_fifo_adj, col_indices_fifo_adj, rnnz_fifo_adj*]
     *                        ↓
     *   prepare_*  →  [E_fifo, E_col_indices_fifo, E_rnnz_fifo,
     *                   A_fifo (adj structure copy), max_fifo,
     *                   EO_fifo (raw scores), EO_rnnz_fifo]
     *                        ↓
     *   writes*    →  DDR (E1)
     *   compute_*  →  [val_att_fifo, col_att_fifo, rnnz_att_fifo  → loop_adj]
     *                   [SO_fifo, SO_rnnz_fifo]
     *                        ↓
     *   writes*    →  DDR (S1)
     * ──────────────────────────────────────────────────────────────────── */

    /* ── Raw edge score FIFOs (prepare → writes + compute_attention) ── */
    /* Stored in URAM: size proportional to nnz (large for real graphs).  */
    hls::stream<TTYPE> EO_fifo1("EO fifo1");
    #pragma HLS STREAM variable=EO_fifo1 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=EO_fifo1 type=FIFO impl=URAM

    hls::stream<int>   EO_rnnz_fifo1("EO rnnz fifo1");
    #pragma HLS STREAM variable=EO_rnnz_fifo1 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=EO_rnnz_fifo1 type=FIFO impl=URAM

    hls::stream<TTYPE> EO_fifo2("EO fifo2");
    #pragma HLS STREAM variable=EO_fifo2 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=EO_fifo2 type=FIFO impl=URAM

    hls::stream<int>   EO_rnnz_fifo2("EO rnnz fifo2");
    #pragma HLS STREAM variable=EO_rnnz_fifo2 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=EO_rnnz_fifo2 type=FIFO impl=URAM

    hls::stream<TTYPE> EO_fifo3("EO fifo3");
    #pragma HLS STREAM variable=EO_fifo3 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=EO_fifo3 type=FIFO impl=URAM

    hls::stream<int>   EO_rnnz_fifo3("EO rnnz fifo3");
    #pragma HLS STREAM variable=EO_rnnz_fifo3 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=EO_rnnz_fifo3 type=FIFO impl=URAM

    hls::stream<TTYPE> EO_fifo4("EO fifo4");
    #pragma HLS STREAM variable=EO_fifo4 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=EO_fifo4 type=FIFO impl=URAM

    hls::stream<int>   EO_rnnz_fifo4("EO rnnz fifo4");
    #pragma HLS STREAM variable=EO_rnnz_fifo4 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=EO_rnnz_fifo4 type=FIFO impl=URAM

    /* ── Softmax output FIFOs (compute_attention → writes) ── */
    hls::stream<TTYPE> SO_fifo1("SO fifo1");
    #pragma HLS STREAM variable=SO_fifo1 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=SO_fifo1 type=FIFO impl=URAM

    hls::stream<int>   SO_rnnz_fifo1("SO rnnz fifo1");
    #pragma HLS STREAM variable=SO_rnnz_fifo1 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=SO_rnnz_fifo1 type=FIFO impl=URAM

    hls::stream<TTYPE> SO_fifo2("SO fifo2");
    #pragma HLS STREAM variable=SO_fifo2 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=SO_fifo2 type=FIFO impl=URAM

    hls::stream<int>   SO_rnnz_fifo2("SO rnnz fifo2");
    #pragma HLS STREAM variable=SO_rnnz_fifo2 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=SO_rnnz_fifo2 type=FIFO impl=URAM

    hls::stream<TTYPE> SO_fifo3("SO fifo3");
    #pragma HLS STREAM variable=SO_fifo3 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=SO_fifo3 type=FIFO impl=URAM

    hls::stream<int>   SO_rnnz_fifo3("SO rnnz fifo3");
    #pragma HLS STREAM variable=SO_rnnz_fifo3 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=SO_rnnz_fifo3 type=FIFO impl=URAM

    hls::stream<TTYPE> SO_fifo4("SO fifo4");
    #pragma HLS STREAM variable=SO_fifo4 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=SO_fifo4 type=FIFO impl=URAM

    hls::stream<int>   SO_rnnz_fifo4("SO rnnz fifo4");
    #pragma HLS STREAM variable=SO_rnnz_fifo4 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=SO_rnnz_fifo4 type=FIFO impl=URAM

    /* ── Pre-softmax score staging FIFOs (prepare → compute_attention) ── */
    /* max_fifo: row-wise maximum for numerically-stable softmax.           */
    hls::stream<TTYPE> max_fifo1("max_fifo1");
    #pragma HLS STREAM variable=max_fifo1 depth=FIFO_DEPTH

    hls::stream<TTYPE> E_fifo1("E fifo1");
    #pragma HLS STREAM variable=E_fifo1 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=E_fifo1 type=FIFO impl=URAM

    /* A_fifo: copy of the adjacency structure for softmax denominator pass */
    hls::stream<ATYPE> A_fifo1("A fifo1");
    #pragma HLS STREAM variable=A_fifo1 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=A_fifo1 type=FIFO impl=URAM

    hls::stream<int>   E_col_indices_fifo1("E col fifo1");
    #pragma HLS STREAM variable=E_col_indices_fifo1 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=E_col_indices_fifo1 type=FIFO impl=URAM

    hls::stream<int>   E_rnnz_fifo1;
    #pragma HLS STREAM variable=E_rnnz_fifo1 depth=FIFO_DEPTH

    hls::stream<TTYPE> max_fifo2("max_fifo2");
    #pragma HLS STREAM variable=max_fifo2 depth=FIFO_DEPTH

    hls::stream<TTYPE> E_fifo2("E fifo2");
    #pragma HLS STREAM variable=E_fifo2 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=E_fifo2 type=FIFO impl=URAM

    hls::stream<ATYPE> A_fifo2("A fifo2");
    #pragma HLS STREAM variable=A_fifo2 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=A_fifo2 type=FIFO impl=URAM

    hls::stream<int>   E_col_indices_fifo2("E col fifo2");
    #pragma HLS STREAM variable=E_col_indices_fifo2 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=E_col_indices_fifo2 type=FIFO impl=URAM

    hls::stream<int>   E_rnnz_fifo2;
    #pragma HLS STREAM variable=E_rnnz_fifo2 depth=FIFO_DEPTH

    hls::stream<TTYPE> max_fifo3("max_fifo3");
    #pragma HLS STREAM variable=max_fifo3 depth=FIFO_DEPTH

    hls::stream<TTYPE> E_fifo3("E fifo3");
    #pragma HLS STREAM variable=E_fifo3 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=E_fifo3 type=FIFO impl=URAM

    hls::stream<ATYPE> A_fifo3("A fifo3");
    #pragma HLS STREAM variable=A_fifo3 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=A_fifo3 type=FIFO impl=URAM

    hls::stream<int>   E_col_indices_fifo3("E col fifo3");
    #pragma HLS STREAM variable=E_col_indices_fifo3 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=E_col_indices_fifo3 type=FIFO impl=URAM

    hls::stream<int>   E_rnnz_fifo3;
    #pragma HLS STREAM variable=E_rnnz_fifo3 depth=FIFO_DEPTH

    hls::stream<TTYPE> max_fifo4("max_fifo4");
    #pragma HLS STREAM variable=max_fifo4 depth=FIFO_DEPTH

    hls::stream<TTYPE> E_fifo4("E fifo4");
    #pragma HLS STREAM variable=E_fifo4 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=E_fifo4 type=FIFO impl=URAM

    hls::stream<ATYPE> A_fifo4("A fifo4");
    #pragma HLS STREAM variable=A_fifo4 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=A_fifo4 type=FIFO impl=URAM

    hls::stream<int>   E_col_indices_fifo4("E col fifo4");
    #pragma HLS STREAM variable=E_col_indices_fifo4 depth=FIFO_DEPTH_ATTN
    #pragma HLS bind_storage variable=E_col_indices_fifo4 type=FIFO impl=URAM

    hls::stream<int>   E_rnnz_fifo4;
    #pragma HLS STREAM variable=E_rnnz_fifo4 depth=FIFO_DEPTH

    /* ── Adjacency read FIFOs (reada2 → prepare_*) ── */
    /* rnnz_total_e/s: total non-zero counts forwarded to writes* for DDR sizing */
    hls::stream<int>   rnnz_fifo_adj1;
    #pragma HLS STREAM variable=rnnz_fifo_adj1 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj1_total_e;
    #pragma HLS STREAM variable=rnnz_fifo_adj1_total_e depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj1_total_s;
    #pragma HLS STREAM variable=rnnz_fifo_adj1_total_s depth=FIFO_DEPTH

    hls::stream<ATYPE> A_fifo_adj1("A fifo adj1");
    #pragma HLS STREAM variable=A_fifo_adj1 depth=FIFO_DEPTH

    hls::stream<int>   col_indices_fifo_adj1("col fifo1");
    #pragma HLS STREAM variable=col_indices_fifo_adj1 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj2;
    #pragma HLS STREAM variable=rnnz_fifo_adj2 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj2_total_e;
    #pragma HLS STREAM variable=rnnz_fifo_adj2_total_e depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj2_total_s;
    #pragma HLS STREAM variable=rnnz_fifo_adj2_total_s depth=FIFO_DEPTH

    hls::stream<ATYPE> A_fifo_adj2("A fifo adj2");
    #pragma HLS STREAM variable=A_fifo_adj2 depth=FIFO_DEPTH

    hls::stream<int>   col_indices_fifo_adj2("col fifo2");
    #pragma HLS STREAM variable=col_indices_fifo_adj2 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj3;
    #pragma HLS STREAM variable=rnnz_fifo_adj3 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj3_total_e;
    #pragma HLS STREAM variable=rnnz_fifo_adj3_total_e depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj3_total_s;
    #pragma HLS STREAM variable=rnnz_fifo_adj3_total_s depth=FIFO_DEPTH

    hls::stream<ATYPE> A_fifo_adj3("A fifo adj3");
    #pragma HLS STREAM variable=A_fifo_adj3 depth=FIFO_DEPTH

    hls::stream<int>   col_indices_fifo_adj3("col fifo3");
    #pragma HLS STREAM variable=col_indices_fifo_adj3 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj4;
    #pragma HLS STREAM variable=rnnz_fifo_adj4 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj4_total_e;
    #pragma HLS STREAM variable=rnnz_fifo_adj4_total_e depth=FIFO_DEPTH

    hls::stream<int>   rnnz_fifo_adj4_total_s;
    #pragma HLS STREAM variable=rnnz_fifo_adj4_total_s depth=FIFO_DEPTH

    hls::stream<ATYPE> A_fifo_adj4("A fifo adj4");
    #pragma HLS STREAM variable=A_fifo_adj4 depth=FIFO_DEPTH

    hls::stream<int>   col_indices_fifo_adj4("col fifo4");
    #pragma HLS STREAM variable=col_indices_fifo_adj4 depth=FIFO_DEPTH

    /* ── Attention parameter vector: [a_src || a_dst], size = 2 × C_WIDTH ── */
    BTYPE ate_m1[2 * C_WIDTH];

    /* ────────────────────────────────────────────────────────────────────
     * Layer loop (mirrors loop_fea / loop_adj ping-pong structure).
     * ──────────────────────────────────────────────────────────────────── */
#if (PIPO_BLOCKS >= 2)
    LOOP_ATTN:
    for (int B_index = 0; B_index < layer_loop; B_index++)
    {
#else
    {
        int B_index = 0;
#endif
        #pragma HLS DATAFLOW

        std::cout << "attention layer " << B_index << std::endl;

        /* ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
         * ADJ_THREADS == 1  :  single-partition attention path.
         * ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── */
#if ADJ_THREADS == 1

        /* ── Step 1: Load attention parameters from DDR, optionally quantized ── */
        for (int j = 0; j < 2 * B_WIDTH_BLOCK; j++)
        {
            #pragma HLS PIPELINE
            INTYPE raw_val = A[j];
            BTYPE  quant_val;
#if (INT_QUANT_W == 1)
            quantw(quant_val, raw_val, quantization_scale_w, f_align, beta_qu, B_index);
#else
            quant_val = raw_val;
#endif
            ate_m1[j] = quant_val;
        }

        /* ── Step 2: Assign full row range to partition 1 ── */
        int first_row1 = 0;
        int row_count1 = N_adj / ADJ_THREADS;

#if GAT_ENABLE == 1

        /* ── Step 3: Read sparse adjacency matrix ── */
#if (COO_MODE == 0)
        reada2_csr(beta_qu, f_align, quantization_scale_adj,
                   gemm_mode, M_adj, first_row1, row_count1,
                   A_fifo_adj1, col_indices_fifo_adj1,
                   rnnz_fifo_adj1_total_e, rnnz_fifo_adj1_total_s, rnnz_fifo_adj1,
                   rowPtr_adj1, columnIndex_adj1, values_adj1);
#else
        reada2_coo(nnz_adj1, beta_qu, f_align, quantization_scale_adj,
                   model, M_adj, first_row1, row_count1,
                   A_fifo_adj1, col_indices_fifo_adj1,
                   rnnz_fifo_adj1_total_e, rnnz_fifo_adj1_total_s, rnnz_fifo_adj1,
                   rowPtr_adj1, columnIndex_adj1, values_adj1, B_index);
#endif

#if (FAST_ATTENTION == 1)
        /* ── Step 4a: Compute per-edge scores e_ij = LeakyReLU(aᵀ[h_i||h_j]) ── */
        prepare_attentional_mechanism_input2(
            model, row_count1, P_w,
            A_buffer11,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            E_col_indices_fifo1, E_rnnz_fifo1,
            A_fifo1, E_fifo1, max_fifo1,
            ate_m1,
            EO_fifo1, EO_rnnz_fifo1,
            B_index);

        /* ── Step 5a: Write raw edge scores to DDR ── */
        writes(deq_factor, model,
               first_row1, row_count1, N_adj, P_w,
               EO_fifo1, rnnz_fifo_adj1_total_e,
               E1, B_index);

        /* ── Step 6a: Apply row-wise softmax, emit weighted sparse values ── */
        compute_attention2(
            model, row_count1,
            A_fifo1, E_col_indices_fifo1, E_rnnz_fifo1,
            E_fifo1, max_fifo1,
            val_att_fifo1, col_att_fifo1, rnnz_att_fifo1,
            SO_fifo1, SO_rnnz_fifo1,
            B_index);

        /* ── Step 7a: Write softmax values to DDR (scale = 1.0, no dequant) ── */
        float deq_dummy[5] = {1.0};
        writes(deq_dummy, model,
               first_row1, row_count1, N_adj, P_w,
               SO_fifo1, rnnz_fifo_adj1_total_s,
               S1, B_index);

#else  /* FAST_ATTENTION == 0: legacy non-fused path */
        prepare_attentional_mechanism_input(N_adj, P_w, C_buffer, E_fifo, ate_m);
        compute_attention(N_adj,
                          A_fifo_adj, col_indices_fifo_adj, rnnz_fifo_adj,
                          E_fifo,
                          val_att_fifo, col_att_fifo, rnnz_att_fifo);
#endif  /* FAST_ATTENTION */

#else  /* GAT_ENABLE == 0: pass adjacency directly as attention weights */
#if (COO_MODE == 0)
        reada22_csr(beta_qu, f_align, quantization_scale_adj,
                    gemm_mode, M_adj, first_row1, row_count1,
                    val_att_fifo1, col_att_fifo1, rnnz_att_fifo1,
                    rowPtr_adj1, columnIndex_adj1, values_adj1);
#else
        reada22_coo(nnz_adj1, beta_qu, f_align, quantization_scale_adj,
                    model, M_adj, first_row1, row_count1,
                    val_att_fifo1, col_att_fifo1, rnnz_att_fifo1,
                    rowPtr_adj1, columnIndex_adj1, values_adj1, B_index);
#endif
#endif  /* GAT_ENABLE */

#endif  /* ADJ_THREADS == 1 */


        /* ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
         * ADJ_THREADS == 4  :  four-partition attention path.
         * All 4 partitions execute in parallel as DATAFLOW processes.
         * ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── */
#if ADJ_THREADS == 4

        /* ── Step 1: Load attention parameters (same vector for all partitions) ── */
        for (int j = 0; j < 2 * B_WIDTH_BLOCK; j++)
        {
            #pragma HLS PIPELINE
            ate_m1[j] = A[j];
        }

        /* ── Step 2: Split N_adj rows across 4 partitions; remainder to partition 4 ── */
        int N_adj_block = N_adj / ADJ_THREADS;
        int N_adj_rest  = N_adj % 4;

        int first_row1 = 0;
        int row_count1 = N_adj_block;

        int first_row2 = N_adj_block;
        int row_count2 = N_adj_block;

        int first_row3 = 2 * N_adj_block;
        int row_count3 = N_adj_block;

        int first_row4 = 3 * N_adj_block;
        int row_count4 = N_adj_block + N_adj_rest;

#if GAT_ENABLE == 1

        /* ── Step 3: Read sparse adjacency for all 4 partitions ── */
#if (COO_MODE == 0)
        reada2_csr(beta_qu, f_align, quantization_scale_adj,
                   gemm_mode, M_adj, first_row1, row_count1,
                   A_fifo_adj1, col_indices_fifo_adj1,
                   rnnz_fifo_adj1_total_e, rnnz_fifo_adj1_total_s, rnnz_fifo_adj1,
                   rowPtr_adj1, columnIndex_adj1, values_adj1);

        reada2_csr(beta_qu, f_align, quantization_scale_adj,
                   gemm_mode, M_adj, first_row2, row_count2,
                   A_fifo_adj2, col_indices_fifo_adj2,
                   rnnz_fifo_adj2_total_e, rnnz_fifo_adj2_total_s, rnnz_fifo_adj2,
                   rowPtr_adj2, columnIndex_adj2, values_adj2);

        reada2_csr(beta_qu, f_align, quantization_scale_adj,
                   gemm_mode, M_adj, first_row3, row_count3,
                   A_fifo_adj3, col_indices_fifo_adj3,
                   rnnz_fifo_adj3_total_e, rnnz_fifo_adj3_total_s, rnnz_fifo_adj3,
                   rowPtr_adj3, columnIndex_adj3, values_adj3);

        reada2_csr(beta_qu, f_align, quantization_scale_adj,
                   gemm_mode, M_adj, first_row4, row_count4,
                   A_fifo_adj4, col_indices_fifo_adj4,
                   rnnz_fifo_adj4_total_e, rnnz_fifo_adj4_total_s, rnnz_fifo_adj4,
                   rowPtr_adj4, columnIndex_adj4, values_adj4);
#else
        reada2_coo(nnz_adj1, beta_qu, f_align, quantization_scale_adj,
                   gemm_mode, M_adj, first_row1, row_count1,
                   A_fifo_adj1, col_indices_fifo_adj1,
                   rnnz_fifo_adj1_total_e, rnnz_fifo_adj1_total_s, rnnz_fifo_adj1,
                   rowPtr_adj1, columnIndex_adj1, values_adj1);

        reada2_coo(nnz_adj2, beta_qu, f_align, quantization_scale_adj,
                   gemm_mode, M_adj, first_row2, row_count2,
                   A_fifo_adj2, col_indices_fifo_adj2,
                   rnnz_fifo_adj2_total_e, rnnz_fifo_adj2_total_s, rnnz_fifo_adj2,
                   rowPtr_adj2, columnIndex_adj2, values_adj2);

        reada2_coo(nnz_adj3, beta_qu, f_align, quantization_scale_adj,
                   gemm_mode, M_adj, first_row3, row_count3,
                   A_fifo_adj3, col_indices_fifo_adj3,
                   rnnz_fifo_adj3_total_e, rnnz_fifo_adj3_total_s, rnnz_fifo_adj3,
                   rowPtr_adj3, columnIndex_adj3, values_adj3);

        reada2_coo(nnz_adj4, beta_qu, f_align, quantization_scale_adj,
                   gemm_mode, M_adj, first_row4, row_count4,
                   A_fifo_adj4, col_indices_fifo_adj4,
                   rnnz_fifo_adj4_total_e, rnnz_fifo_adj4_total_s, rnnz_fifo_adj4,
                   rowPtr_adj4, columnIndex_adj4, values_adj4);
#endif  /* COO_MODE */

#if (FAST_ATTENTION == 1)
        /* ── Step 4b: Compute per-edge scores for all 4 partitions (fused kernel) ── */
        prepare_attentional_mechanism_inputx4(
            gat_mode,
            row_count1, row_count2, row_count3, row_count4,
            P_w,
            A_buffer11, A_buffer21, A_buffer31, A_buffer41,
            A_fifo_adj1, A_fifo_adj2, A_fifo_adj3, A_fifo_adj4,
            col_indices_fifo_adj1, col_indices_fifo_adj2,
            col_indices_fifo_adj3, col_indices_fifo_adj4,
            rnnz_fifo_adj1, rnnz_fifo_adj2, rnnz_fifo_adj3, rnnz_fifo_adj4,
            E_col_indices_fifo1, E_col_indices_fifo2,
            E_col_indices_fifo3, E_col_indices_fifo4,
            E_rnnz_fifo1, E_rnnz_fifo2, E_rnnz_fifo3, E_rnnz_fifo4,
            A_fifo1, A_fifo2, A_fifo3, A_fifo4,
            E_fifo1, E_fifo2, E_fifo3, E_fifo4,
            max_fifo1, max_fifo2, max_fifo3, max_fifo4,
            ate_m1,
            EO_fifo1, EO_fifo2, EO_fifo3, EO_fifo4,
            EO_rnnz_fifo1, EO_rnnz_fifo2, EO_rnnz_fifo3, EO_rnnz_fifo4);

        /* ── Step 5b: Write raw edge scores for all partitions to DDR ── */
        writesx4(deq_factor, gat_mode,
                 row_count1, row_count2, row_count3, row_count4,
                 EO_fifo1, EO_fifo2, EO_fifo3, EO_fifo4,
                 rnnz_fifo_adj1_total_e, rnnz_fifo_adj2_total_e,
                 rnnz_fifo_adj3_total_e, rnnz_fifo_adj4_total_e,
                 E1, B_index);

        /* ── Step 6b: Apply softmax for all 4 partitions ── */
        compute_attention2(gat_mode, row_count1,
                           A_fifo1, E_col_indices_fifo1, E_rnnz_fifo1,
                           E_fifo1, max_fifo1,
                           val_att_fifo1, col_att_fifo1, rnnz_att_fifo1,
                           SO_fifo1, SO_rnnz_fifo1);

        compute_attention2(gat_mode, row_count2,
                           A_fifo2, E_col_indices_fifo2, E_rnnz_fifo2,
                           E_fifo2, max_fifo2,
                           val_att_fifo2, col_att_fifo2, rnnz_att_fifo2,
                           SO_fifo2, SO_rnnz_fifo2);

        compute_attention2(gat_mode, row_count3,
                           A_fifo3, E_col_indices_fifo3, E_rnnz_fifo3,
                           E_fifo3, max_fifo3,
                           val_att_fifo3, col_att_fifo3, rnnz_att_fifo3,
                           SO_fifo3, SO_rnnz_fifo3);

        compute_attention2(gat_mode, row_count4,
                           A_fifo4, E_col_indices_fifo4, E_rnnz_fifo4,
                           E_fifo4, max_fifo4,
                           val_att_fifo4, col_att_fifo4, rnnz_att_fifo4,
                           SO_fifo4, SO_rnnz_fifo4);

        /* ── Step 7b: Write softmax values for all partitions (scale = 1.0) ── */
        float deq_dummy = 1.0;
        writesx4(deq_dummy, gat_mode,
                 row_count1, row_count2, row_count3, row_count4,
                 SO_fifo1, SO_fifo2, SO_fifo3, SO_fifo4,
                 rnnz_fifo_adj1_total_s, rnnz_fifo_adj2_total_s,
                 rnnz_fifo_adj3_total_s, rnnz_fifo_adj4_total_s,
                 S1, B_index);

#else  /* FAST_ATTENTION == 0: legacy non-fused path */
        prepare_attentional_mechanism_input(N_adj, P_w, C_buffer, E_fifo, ate_m);
        compute_attention(N_adj,
                          A_fifo_adj, col_indices_fifo_adj, rnnz_fifo_adj,
                          E_fifo,
                          val_att_fifo, col_att_fifo, rnnz_att_fifo);
#endif  /* FAST_ATTENTION */

#else  /* GAT_ENABLE == 0: pass adjacency directly as attention weights */
#if (COO_MODE == 0)
        reada22_csr(beta_qu, f_align, quantization_scale_adj,
                    gemm_mode, M_adj, first_row1, row_count1,
                    val_att_fifo1, col_att_fifo1, rnnz_att_fifo1,
                    rowPtr_adj1, columnIndex_adj1, values_adj1);

        reada22_csr(beta_qu, f_align, quantization_scale_adj,
                    gemm_mode, M_adj, first_row2, row_count2,
                    val_att_fifo2, col_att_fifo2, rnnz_att_fifo2,
                    rowPtr_adj2, columnIndex_adj2, values_adj2);

        reada22_csr(beta_qu, f_align, quantization_scale_adj,
                    gemm_mode, M_adj, first_row3, row_count3,
                    val_att_fifo3, col_att_fifo3, rnnz_att_fifo3,
                    rowPtr_adj3, columnIndex_adj3, values_adj3);

        reada22_csr(beta_qu, f_align, quantization_scale_adj,
                    gemm_mode, M_adj, first_row4, row_count4,
                    val_att_fifo4, col_att_fifo4, rnnz_att_fifo4,
                    rowPtr_adj4, columnIndex_adj4, values_adj4);
#else
        reada22_coo(nnz_adj1, beta_qu, f_align, quantization_scale_adj,
                    gemm_mode, M_adj, first_row1, row_count1,
                    val_att_fifo1, col_att_fifo1, rnnz_att_fifo1,
                    rowPtr_adj1, columnIndex_adj1, values_adj1);

        reada22_coo(nnz_adj2, beta_qu, f_align, quantization_scale_adj,
                    gemm_mode, M_adj, first_row2, row_count2,
                    val_att_fifo2, col_att_fifo2, rnnz_att_fifo2,
                    rowPtr_adj2, columnIndex_adj2, values_adj2);

        reada22_coo(nnz_adj3, beta_qu, f_align, quantization_scale_adj,
                    gemm_mode, M_adj, first_row3, row_count3,
                    val_att_fifo3, col_att_fifo3, rnnz_att_fifo3,
                    rowPtr_adj3, columnIndex_adj3, values_adj3);

        reada22_coo(nnz_adj4, beta_qu, f_align, quantization_scale_adj,
                    gemm_mode, M_adj, first_row4, row_count4,
                    val_att_fifo4, col_att_fifo4, rnnz_att_fifo4,
                    rowPtr_adj4, columnIndex_adj4, values_adj4);
#endif
#endif  /* GAT_ENABLE */

#endif  /* ADJ_THREADS == 4 */

    }  /* end LOOP_ATTN / single-pass block */
}
// =============================================================================================
// =============================================================================================
/**
/**
 * readb - Load GNN weight tile from DDR into local BRAM (B_accel).
 *
 * Reads a column block of the weight matrix B into the local tile B_accel,
 * applying optional integer quantization (INT_QUANT_W) on the fly.
 *
 * The memory layout of B in DDR is:
 *   - Layer 0  : B[0 .. M_fea * B_WIDTH_BLOCK - 1]          (input dim = M_fea)
 *   - Layer k>0: B[B_shift .. ]  where B_shift skips layer-0 and prior hidden layers
 *                (input dim = B_WIDTH_BLOCK for all hidden layers)
 *
 * The load is gated by two conditions:
 *   load_weights_gcn = load_weights AND (layer is NOT pure-linear AND NOT pure-SAGE).
 * In other words, skip loading when the layer is a standalone linear or SAGE path
 * that does not share the GNN weight tile.
 *
 * @param load_weights          Global enable: only load when true.
 * @param model                 Per-layer mode flags [layer][bit]:
 *                                bit 6 = linear_mode, bit 7 = sage_mode.
 * @param beta_qu               Zero-point shift for weight quantization.
 * @param f_align               Fractional alignment bits for quantization.
 * @param quantization_scale_w  Per-layer floating-point scale factors.
 * @param M_fea                 Input feature dimension (columns of B at layer 0).
 * @param P_w                   Per-layer number of output columns to load.
 * @param B_index               Current layer index.
 * @param B_accel               Output: local BRAM tile [B_HEIGHT][B_WIDTH_BLOCK].
 * @param B                     Input: DDR weight matrix (flat, row-major).
 */
void readb(
    bool        load_weights,
    ap_uint<1>  model[5][8],
    int         beta_qu,
    int         f_align,
    float       quantization_scale_w[5],
    int         M_fea,
    ap_uint<8>  P_w[5],
    int         B_index,
    BTYPE       B_accel[B_HEIGHT][B_WIDTH_BLOCK],
    INTYPES    *B
)
{
    /* ── Compute DDR base address and row count for this layer ──────────
     * Layer 0 starts at offset 0 with M_fea rows (input features).
     * Subsequent layers start after the layer-0 block and all prior
     * hidden-to-hidden blocks, each of size B_WIDTH_BLOCK × B_WIDTH_BLOCK.
     * ─────────────────────────────────────────────────────────────────── */
    int B_shift;
    int M_fea_current;

    if (B_index == 0)
    {
        B_shift       = 0;
        M_fea_current = M_fea;
    }
    else
    {
        B_shift       = B_WIDTH_BLOCK * M_fea + (B_index - 1) * B_WIDTH_BLOCK * B_WIDTH_BLOCK;
        M_fea_current = B_WIDTH_BLOCK;
    }

    /* ── Gate: only load for GNN (non-linear, non-SAGE-only) layers ──── */
    bool linear_mode      = model[B_index][6];
    bool sage_mode        = model[B_index][7];
    bool gcn_path         = !(linear_mode ^ sage_mode); // true when both flags agree (pure GNN)
    bool load_weights_gcn = load_weights & gcn_path;

    if (load_weights_gcn)
    {
        /* Iterate over output columns (up to P_w[B_index]) and input rows */
        LOOP_BLOCKB1: for (int j = 0; j < P_w[B_index]; j++)
        {
            LOOP_BLOCKB2: for (int i = 0; i < M_fea_current; i++)
            {
                #pragma HLS PIPELINE

                INTYPE raw_val   = (INTYPE)B[i + j * M_fea_current + B_shift];
                BTYPE  quant_val;

#if (INT_QUANT_W == 1)
                quantw(quant_val, raw_val, quantization_scale_w, f_align, beta_qu, B_index);
#else
                quant_val = raw_val;
#endif

                B_accel[i][j] = quant_val;
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * readbl - Load linear-projection weight tile from DDR into local BRAM (B_accel).
 *
 * Identical memory layout and addressing as readb, but:
 *   - Stores into a BLTYPE tile (linear-branch type, may differ in width/precision).
 *   - Uses quantwl instead of quantw for quantization.
 *   - Gated by linear_mode only (bit 6), not the GNN path check.
 *
 * @param load_weights          Global enable: only load when true.
 * @param model                 Per-layer mode flags [layer][bit]: bit 6 = linear_mode.
 * @param beta_qu               Zero-point shift for weight quantization.
 * @param f_align               Fractional alignment bits for quantization.
 * @param quantization_scale_w  Per-layer floating-point scale factors.
 * @param M_fea                 Input feature dimension (columns of B at layer 0).
 * @param P_w                   Per-layer number of output columns to load.
 * @param B_index               Current layer index.
 * @param B_accel               Output: local BRAM tile [B_HEIGHT][B_WIDTH_BLOCK].
 * @param B                     Input: DDR weight matrix (flat, row-major).
 */
void readbl(
    bool        load_weights,
    ap_uint<1>  model[5][8],
    int         beta_qu,
    int         f_align,
    float       quantization_scale_w[5],
    int         M_fea,
    ap_uint<8>  P_w[5],
    int         B_index,
    BLTYPE      B_accel[B_HEIGHT][B_WIDTH_BLOCK],
    INTYPES    *B
)
{
    /* ── Same DDR addressing as readb ─────────────────────────────────── */
    int B_shift;
    int M_fea_current;

    if (B_index == 0)
    {
        B_shift       = 0;
        M_fea_current = M_fea;
    }
    else
    {
        B_shift       = B_WIDTH_BLOCK * M_fea + (B_index - 1) * B_WIDTH_BLOCK * B_WIDTH_BLOCK;
        M_fea_current = B_WIDTH_BLOCK;
    }

    /* ── Gate: only load for layers with an active linear projection ─── */
    bool linear_mode         = model[B_index][6];
    bool load_weights_linear = load_weights & linear_mode;

    if (load_weights_linear)
    {
        LOOP_BLOCKB1: for (int j = 0; j < P_w[B_index]; j++)
        {
            LOOP_BLOCKB2: for (int i = 0; i < M_fea_current; i++)
            {
                #pragma HLS PIPELINE

                INTYPE  raw_val   = (INTYPE)B[i + j * M_fea_current + B_shift];
                BLTYPE  quant_val;

#if (INT_QUANT_W == 1)
                quantwl(quant_val, raw_val, quantization_scale_w, f_align, beta_qu, B_index);
#else
                quant_val = raw_val;
#endif

                B_accel[i][j] = quant_val;
            }
        }
    }
}

// =============================================================================================
// =============================================================================================
/**
 * loop_fea - Feature matrix SpMM (Sparse × Dense) engine for GNN layers.
 *
 * For each layer iteration (B_index), this function:
 *   1. Loads GNN weight matrix B into local BRAM tiles (B_accel).
 *   2. Reads the sparse feature matrix A from memory (CSR or COO format)
 *      and streams it through FIFOs.
 *   3. Multiplies A × B (SpMM) to produce output tiles C, written into
 *      double-buffered stream-of-blocks (ping-pong or single-buffer mode).
 *   4. Optionally computes a linear projection (A × B_linear) in parallel.
 *   5. Optionally writes attention pre-activations (A_buffer) for GAT layers.
 *
 * Parallelism is controlled at compile time by:
 *   FEA_THREADS  – number of row partitions of A processed in parallel (1/2/4).
 *   ADJ_THREADS  – number of output column-block streams per row partition (1/2/4).
 *   PIPO_BLOCKS  – enables ping-pong double-buffering when >= 2.
 *   COO_MODE     – selects sparse format: 0 = CSR, 1 = COO.
 *   LINEAR_ENABLE– enables parallel linear-projection branch.
 *   GAT_ENABLE   – enables writing attention pre-activations to A_buffer.
 */
void loop_fea(
    /* ── Control ── */
    bool        load_weights,

    /* ── Quantization parameters (GNN branch) ── */
    int         beta_qu,            // zero-point shift for feature quantization
    int         f_align,            // fractional alignment bits

    /* ── Quantization parameters (linear branch) ── */
    int         beta_qul,
    int         f_alignl,

    /* ── Per-layer quantization scales ── */
    float       quantization_scale_fea[5],
    float       quantization_scale_w[5],
    float       quantization_scale_lin[5],

    /* ── Layer-selection bitmask (which layers are active) ── */
    ap_uint<1>  model[5][8],

    /* ── Output scaling and max accumulator ── */
    STYPE       scale_fea[5],
    ITYPE      *max_fea,

    /* ── Quantization multipliers ── */
    int         quantized_multiplier,   // GNN branch
    int         quantized_multiplierl,  // linear branch

    /* ── Non-zero counts for each sparse partition ── */
    int         nnz_fea1,
    int         nnz_fea2,
    int         nnz_fea3,
    int         nnz_fea4,

    /* ── CSR row pointers for each sparse partition ── */
    int        *rowPtr_fea1,
    int        *rowPtr_fea2,
    int        *rowPtr_fea3,
    int        *rowPtr_fea4,

    /* ── CSR column indices for each sparse partition ── */
    int        *columnIndex_fea1,
    int        *columnIndex_fea2,
    int        *columnIndex_fea3,
    int        *columnIndex_fea4,

    /* ── CSR values for each sparse partition ── */
    INTYPE     *values_fea1,
    INTYPE     *values_fea2,
    INTYPE     *values_fea3,
    INTYPE     *values_fea4,

    /* ── AXI-stream row pointers (streaming interface, one per partition) ── */
    hls::stream<ASTYPE> &rowPtr_feas1,
    hls::stream<ASTYPE> &rowPtr_feas2,
    hls::stream<ASTYPE> &rowPtr_feas3,
    hls::stream<ASTYPE> &rowPtr_feas4,

    /* ── AXI-stream column indices (one per partition) ── */
    hls::stream<ASTYPE> &columnIndex_feas1,
    hls::stream<ASTYPE> &columnIndex_feas2,
    hls::stream<ASTYPE> &columnIndex_feas3,
    hls::stream<ASTYPE> &columnIndex_feas4,

    /* ── AXI-stream values (one per partition) ── */
    hls::stream<ASTYPE> &values_feas1,
    hls::stream<ASTYPE> &values_feas2,
    hls::stream<ASTYPE> &values_feas3,
    hls::stream<ASTYPE> &values_feas4,

    /* ── Dense weight matrices: GNN (B) and linear projection (B2) ── */
    INTYPES    *B,
    INTYPES    *B2,

    /* ── Feature matrix dimensions ── */
    int         N_fea,              // number of rows in the feature matrix
    int         M_fea,              // number of columns (features)

    /* ── Per-layer weight-column widths ── */
    ap_uint<8>  P_w[5],

    /* ── Quantization zero points ── */
    ap_int<8>   zero_point_lhs,
    ap_int<8>   zero_point_rhs,

    /* ── Output C ping-pong stream-of-blocks (row-partition 1, up to 4 adj-thread slots) ── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf> &C_buffer11,
#else
    buf                         C_buffer11,
#endif
    hls::stream_of_blocks<buf> &C_buffer12,
    hls::stream_of_blocks<buf> &C_buffer13,
    hls::stream_of_blocks<buf> &C_buffer14,

    /* ── Output C stream-of-blocks (row-partition 2) ── */
    hls::stream_of_blocks<buf> &C_buffer21,
    hls::stream_of_blocks<buf> &C_buffer22,
    hls::stream_of_blocks<buf> &C_buffer23,
    hls::stream_of_blocks<buf> &C_buffer24,

    /* ── Output C stream-of-blocks (row-partition 3) ── */
    hls::stream_of_blocks<buf> &C_buffer31,
    hls::stream_of_blocks<buf> &C_buffer32,
    hls::stream_of_blocks<buf> &C_buffer33,
    hls::stream_of_blocks<buf> &C_buffer34,

    /* ── Output C stream-of-blocks (row-partition 4) ── */
    hls::stream_of_blocks<buf> &C_buffer41,
    hls::stream_of_blocks<buf> &C_buffer42,
    hls::stream_of_blocks<buf> &C_buffer43,
    hls::stream_of_blocks<buf> &C_buffer44,

    /* ── Attention pre-activation buffers (GAT only, one per row-partition) ── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf> &A_buffer11,
    hls::stream_of_blocks<buf> &A_buffer21,
#else
    buf                         A_buffer11,
    hls::stream_of_blocks<buf> &A_buffer21,
#endif
    hls::stream_of_blocks<buf> &A_buffer31,
    hls::stream_of_blocks<buf> &A_buffer41,

    /* ── Linear-projection output ping-pong buffer ── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<bufl> &linear_pipo,
#else
    bufl                         linear_pipo,
#endif

    /* ── Number of GNN layers to iterate over ── */
    int         layer_loop
)
{
    /* ────────────────────────────────────────────────────────────────────
     * Local weight tiles (BRAM).
     * Each partition gets its own copy so all FEA_THREADS can read
     * simultaneously without port conflicts.
     * Partitioned along the column dimension (factor = BLOCK/2) to
     * expose enough read ports for the inner compute loop.
     * ──────────────────────────────────────────────────────────────────── */

    /* GNN weight tiles */
    BTYPE B_accel1[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel1 block factor=BLOCK/2 dim=2

    BTYPE B_accel2[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel2 block factor=BLOCK/2 dim=2

    BTYPE B_accel3[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel3 block factor=BLOCK/2 dim=2

    BTYPE B_accel4[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel4 block factor=BLOCK/2 dim=2

    /* Linear-projection weight tiles */
    BLTYPE B_accel12[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel12 block factor=BLOCK/2 dim=2

    BLTYPE B_accel22[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel22 block factor=BLOCK/2 dim=2

    BLTYPE B_accel32[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel32 block factor=BLOCK/2 dim=2

    BLTYPE B_accel42[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel42 block factor=BLOCK/2 dim=2

    /* ────────────────────────────────────────────────────────────────────
     * Inter-task FIFOs.
     * reada tasks produce into these FIFOs; compute tasks consume from them.
     * Separate FIFO sets for:
     *   - GNN branch  (FTYPE values,  suffix 1..4)
     *   - Linear branch (LTYPE values, suffix 12..42)
     * Each FIFO carries: non-zero values, column indices, row-nnz counts.
     * ──────────────────────────────────────────────────────────────────── */

    /* Row non-zero counts – GNN branch */
    hls::stream<int> rnnz_fifo_fea1;
    #pragma HLS STREAM variable=rnnz_fifo_fea1 depth=FIFO_DEPTH
    hls::stream<int> rnnz_fifo_fea2;
    #pragma HLS STREAM variable=rnnz_fifo_fea2 depth=FIFO_DEPTH
    hls::stream<int> rnnz_fifo_fea3;
    #pragma HLS STREAM variable=rnnz_fifo_fea3 depth=FIFO_DEPTH
    hls::stream<int> rnnz_fifo_fea4;
    #pragma HLS STREAM variable=rnnz_fifo_fea4 depth=FIFO_DEPTH

    /* Row non-zero counts – linear branch */
    hls::stream<int> rnnz_fifo_fea12;
    #pragma HLS STREAM variable=rnnz_fifo_fea12 depth=FIFO_DEPTH
    hls::stream<int> rnnz_fifo_fea22;
    #pragma HLS STREAM variable=rnnz_fifo_fea22 depth=FIFO_DEPTH
    hls::stream<int> rnnz_fifo_fea32;
    #pragma HLS STREAM variable=rnnz_fifo_fea32 depth=FIFO_DEPTH
    hls::stream<int> rnnz_fifo_fea42;
    #pragma HLS STREAM variable=rnnz_fifo_fea42 depth=FIFO_DEPTH

    /* Non-zero values – GNN branch */
    hls::stream<FTYPE> A_fifo_fea1;
    #pragma HLS STREAM variable=A_fifo_fea1 depth=FIFO_DEPTH
    hls::stream<FTYPE> A_fifo_fea2;
    #pragma HLS STREAM variable=A_fifo_fea2 depth=FIFO_DEPTH
    hls::stream<FTYPE> A_fifo_fea3;
    #pragma HLS STREAM variable=A_fifo_fea3 depth=FIFO_DEPTH
    hls::stream<FTYPE> A_fifo_fea4;
    #pragma HLS STREAM variable=A_fifo_fea4 depth=FIFO_DEPTH

    /* Non-zero values – linear branch */
    hls::stream<LTYPE> A_fifo_fea12;
    #pragma HLS STREAM variable=A_fifo_fea12 depth=FIFO_DEPTH
    hls::stream<LTYPE> A_fifo_fea22;
    #pragma HLS STREAM variable=A_fifo_fea22 depth=FIFO_DEPTH
    hls::stream<LTYPE> A_fifo_fea32;
    #pragma HLS STREAM variable=A_fifo_fea32 depth=FIFO_DEPTH
    hls::stream<LTYPE> A_fifo_fea42;
    #pragma HLS STREAM variable=A_fifo_fea42 depth=FIFO_DEPTH

    /* Column indices – GNN branch */
    hls::stream<int> col_indices_fifo_fea1;
    #pragma HLS STREAM variable=col_indices_fifo_fea1 depth=FIFO_DEPTH
    hls::stream<int> col_indices_fifo_fea2;
    #pragma HLS STREAM variable=col_indices_fifo_fea2 depth=FIFO_DEPTH
    hls::stream<int> col_indices_fifo_fea3;
    #pragma HLS STREAM variable=col_indices_fifo_fea3 depth=FIFO_DEPTH
    hls::stream<int> col_indices_fifo_fea4;
    #pragma HLS STREAM variable=col_indices_fifo_fea4 depth=FIFO_DEPTH

    /* Column indices – linear branch */
    hls::stream<int> col_indices_fifo_fea12;
    #pragma HLS STREAM variable=col_indices_fifo_fea12 depth=FIFO_DEPTH
    hls::stream<int> col_indices_fifo_fea22;
    #pragma HLS STREAM variable=col_indices_fifo_fea22 depth=FIFO_DEPTH
    hls::stream<int> col_indices_fifo_fea32;
    #pragma HLS STREAM variable=col_indices_fifo_fea32 depth=FIFO_DEPTH
    hls::stream<int> col_indices_fifo_fea42;
    #pragma HLS STREAM variable=col_indices_fifo_fea42 depth=FIFO_DEPTH

    /* ────────────────────────────────────────────────────────────────────
     * Layer loop.
     * When PIPO_BLOCKS >= 2, the hardware iterates over all GNN layers
     * and ping-pong buffers allow the next layer to load while the current
     * layer is being processed downstream (dataflow pipeline).
     * When PIPO_BLOCKS == 1, only a single layer pass is compiled; the
     * loop body runs exactly once (B_index = 0).
     * ──────────────────────────────────────────────────────────────────── */
#if (PIPO_BLOCKS >= 2)
    LOOP_FEA:
    for (int B_index = 0; B_index < layer_loop; B_index++)
    {
#else
    {
        int B_index = 0;
#endif
        #pragma HLS DATAFLOW

        /* Width of the current weight-column block (constant = B_WIDTH_BLOCK) */
        int B_WIDTH_INT = B_WIDTH_BLOCK;

        std::cout << "fea layer " << B_index << std::endl;

        /* ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
         * FEA_THREADS == 1  :  single-partition path.
         * All N_fea rows processed by one read + one compute task.
         * ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── */
#if FEA_THREADS == 1

        /* ── Step 1: Load GNN weight tile from DDR into local BRAM ── */
        readb(load_weights, model, beta_qu, f_align,
              quantization_scale_w, M_fea, P_w, B_index,
              B_accel1, B);

        /* ── Step 2 (optional): Load linear-projection weight tile ── */
#if LINEAR_ENABLE == 1
        readbl(load_weights, model, beta_qul, f_alignl,
               quantization_scale_w, M_fea, P_w, B_index,
               B_accel12, B2);
#endif

        /* ── Step 3: Acquire write locks on output ping-pong buffers ── */
#if (PIPO_BLOCKS >= 2)
        hls::write_lock<buf>  C_fea11(C_buffer11);  // GNN output for adj-loop

#if LINEAR_ENABLE == 1
        hls::write_lock<bufl> linear_fea(linear_pipo);
#else
        QLTYPE linear_fea[B_HEIGHT][B_WIDTH_BLOCK];  // dummy (not forwarded)
#endif

#if GAT_ENABLE == 1
        hls::write_lock<buf>  A_fea11(A_buffer11);  // attention pre-activation
#else
        QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];      // dummy (not forwarded)
#endif

#else  /* PIPO_BLOCKS == 1: no stream-of-blocks, use plain arrays */
#if GAT_ENABLE == 1
        /* A_buffer11 is already the plain-array parameter */
#else
        QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];      // dummy
#endif
#endif  /* PIPO_BLOCKS */

        /* ── Step 4: Read sparse feature matrix A into FIFOs ── */
        /* Full matrix assigned to partition 1 (no row splitting) */
        int first_row1  = 0;
        int row_count1  = N_fea;
        int last_index1;

#if (COO_MODE == 0)
        /* CSR format: row pointers + column indices + values */
        reada1_csr(beta_qu, f_align, quantization_scale_fea,
                   last_index1, stream_mode_int, gemm_mode_int, M_fea,
                   first_row1, row_count1,
                   A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                   rowPtr_fea1, columnIndex_fea1, values_fea1, values_feas1);
#else
        /* COO format: explicit (row, col, val) triplets; also feeds linear branch */
        reada1_coo(nnz_fea1,
                   beta_qu, f_align, beta_qul, f_alignl,
                   quantization_scale_fea, quantization_scale_lin,
                   last_index1, model, M_fea,
                   first_row1, row_count1,
                   A_fifo_fea1,  col_indices_fifo_fea1,  rnnz_fifo_fea1,
                   A_fifo_fea12, col_indices_fifo_fea12, rnnz_fifo_fea12,
                   rowPtr_fea1, columnIndex_fea1, values_fea1,
                   rowPtr_feas1, columnIndex_feas1, values_feas1,
                   B_index, layer_loop);
#endif

        /* ── Step 5: Compute SpMM  A × B → C (and optionally A × B_linear) ── */
        ITYPE max_fea1, max_fea2;

#if (PIPO_BLOCKS >= 2)
        compute1_1(scale_fea, &max_fea1, quantized_multiplier,
                   model, zero_point_lhs, zero_point_rhs,
                   first_row1, row_count1,
                   A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                   B_accel1, C_fea11, A_fea11, B_index);

#if LINEAR_ENABLE == 1
        compute1_12(scale_fea, &max_fea2, quantized_multiplierl,
                    model, zero_point_lhs, zero_point_rhs,
                    first_row1, row_count1,
                    A_fifo_fea12, col_indices_fifo_fea12, rnnz_fifo_fea12,
                    B_accel12, linear_fea, B_index);
#endif

#else  /* PIPO_BLOCKS == 1 */
        compute1_1(scale_fea, &max_fea1, quantized_multiplier,
                   gemm_mode_int, zero_point_lhs, zero_point_rhs,
                   first_row1, row_count1,
                   A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                   B_accel1, C_buffer11, A_buffer11);

#if LINEAR_ENABLE == 1
        compute1_12(scale_fea, &max_fea2, quantized_multiplier,
                    gemm_mode_int, zero_point_lhs, zero_point_rhs,
                    first_row1, row_count1,
                    A_fifo_fea12, col_indices_fifo_fea12, rnnz_fifo_fea12,
                    B_accel12, linear_pipo);
#endif
#endif  /* PIPO_BLOCKS */

        /* Expose the maximum activation value to the caller */
        *max_fea = max_fea1;

#endif  /* FEA_THREADS == 1 */


        /* ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
         * FEA_THREADS == 2  :  two-partition path.
         * The N_fea rows are split in half; two read + two compute tasks
         * run as parallel dataflow processes.
         * ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── */
#if FEA_THREADS == 2

        /* Acquire output write locks */
        hls::write_lock<buf> C_fea11(C_buffer11);
        hls::write_lock<buf> C_fea21(C_buffer21);
#if ADJ_THREADS == 2
        hls::write_lock<buf> C_fea12(C_buffer12);
        hls::write_lock<buf> C_fea22(C_buffer22);
#endif

        /* Load weight tile into all active partitions (same weights, broadcast) */
        for (int j = 0; j < B_WIDTH_INT; j++) {
            LOOP_BLOCKB:
            for (int i = 0; i < M_fea; i++) {
                #pragma HLS PIPELINE
                BTYPE val = B[i + j * M_fea + B_index * B_WIDTH_BLOCK * M_fea];
                B_accel1[i][j] = val;
                B_accel2[i][j] = val;
            }
        }

        /* Split rows evenly between the two partitions */
        int N_fea_block = N_fea / 2;
        int N_fea_rest  = N_fea % 2;   // remainder goes to partition 2

        int first_row1 = 0;
        int row_count1 = N_fea_block;

        int first_row2 = N_fea_block;
        int row_count2 = N_fea_block + N_fea_rest;

        /* Read sparse A – partition 1 and partition 2 */
        reada1(gemm_mode, M_fea, first_row1, row_count1,
               A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
               B_index_loop, tail,
               rowPtr_fea1, columnIndex_fea1, values_fea1);

        reada1(gemm_mode, M_fea, first_row2, row_count2,
               A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
               B_index_loop, tail,
               rowPtr_fea2, columnIndex_fea2, values_fea2);

        /* Compute SpMM for each partition */
#if ADJ_THREADS == 2
        compute1_2(gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row1, row_count1,
                   A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                   B_accel1, C_fea11, C_fea12,
                   B_index, B_index_loop, tail);

        compute1_2(gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row2, row_count2,
                   A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
                   B_accel2, C_fea21, C_fea22,
                   B_index, B_index_loop, tail);
#endif

#if ADJ_THREADS == 1
        compute1_1(gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row1, row_count1,
                   A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                   B_accel1, C_fea11);

        compute1_1(gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row2, row_count2,
                   A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
                   B_accel2, C_fea21);
#endif

#endif  /* FEA_THREADS == 2 */


        /* ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
         * FEA_THREADS == 4  :  four-partition path.
         * The N_fea rows are split into four blocks; four independent
         * read + compute task pairs execute as dataflow processes.
         * ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── */
#if FEA_THREADS == 4

        /* ── Acquire output write locks (set depends on ADJ_THREADS) ── */
#if ADJ_THREADS == 4
        hls::write_lock<buf> C_fea11(C_buffer11);
        hls::write_lock<buf> C_fea12(C_buffer12);
        hls::write_lock<buf> C_fea13(C_buffer13);
        hls::write_lock<buf> C_fea14(C_buffer14);
        hls::write_lock<buf> C_fea21(C_buffer21);
        hls::write_lock<buf> C_fea22(C_buffer22);
        hls::write_lock<buf> C_fea23(C_buffer23);
        hls::write_lock<buf> C_fea24(C_buffer24);
        hls::write_lock<buf> C_fea31(C_buffer31);
        hls::write_lock<buf> C_fea32(C_buffer32);
        hls::write_lock<buf> C_fea33(C_buffer33);
        hls::write_lock<buf> C_fea34(C_buffer34);
        hls::write_lock<buf> C_fea41(C_buffer41);
        hls::write_lock<buf> C_fea42(C_buffer42);
        hls::write_lock<buf> C_fea43(C_buffer43);
        hls::write_lock<buf> C_fea44(C_buffer44);

#if GAT_ENABLE == 1
        hls::write_lock<buf> A_fea11(A_buffer11);
        hls::write_lock<buf> A_fea21(A_buffer21);
        hls::write_lock<buf> A_fea31(A_buffer31);
        hls::write_lock<buf> A_fea41(A_buffer41);
#else
        /* Dummy local arrays when GAT is disabled */
        QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];
        QTYPE A_fea21[B_HEIGHT][B_WIDTH_BLOCK];
        QTYPE A_fea31[B_HEIGHT][B_WIDTH_BLOCK];
        QTYPE A_fea41[B_HEIGHT][B_WIDTH_BLOCK];
#endif
#endif  /* ADJ_THREADS == 4 */

#if ADJ_THREADS == 2
        hls::write_lock<buf> C_fea11(C_buffer11);
        hls::write_lock<buf> C_fea12(C_buffer12);
        hls::write_lock<buf> C_fea21(C_buffer21);
        hls::write_lock<buf> C_fea22(C_buffer22);
        hls::write_lock<buf> C_fea31(C_buffer31);
        hls::write_lock<buf> C_fea32(C_buffer32);
        hls::write_lock<buf> C_fea41(C_buffer41);
        hls::write_lock<buf> C_fea42(C_buffer42);
#endif  /* ADJ_THREADS == 2 */

        /* ── Load quantized weight tile; same data broadcast to all 4 partitions ── */
        for (int j = 0; j < B_WIDTH_INT; j++) {
            LOOP_BLOCKB:
            for (int i = 0; i < M_fea; i++) {
                #pragma HLS PIPELINE
                INTYPE   raw_val = (INTYPE)B[i + j * M_fea + B_index * B_WIDTH_BLOCK * M_fea];
                BTYPE    quant_val;
#if (INT_QUANT_W == 1)
                quantw(quant_val, raw_val, quantization_scale_w, f_align, beta_qu);
#else
                quant_val = raw_val;
#endif
                B_accel1[i][j] = quant_val;
                B_accel2[i][j] = quant_val;
                B_accel3[i][j] = quant_val;
                B_accel4[i][j] = quant_val;
            }
        }

        /* ── Split N_fea rows across 4 partitions; remainder to partition 4 ── */
        int N_fea_block = N_fea / 4;
        int N_fea_rest  = N_fea % 4;

        int first_row1 = 0;
        int row_count1 = N_fea_block;

        int first_row2 = N_fea_block;
        int row_count2 = N_fea_block;

        int first_row3 = 2 * N_fea_block;
        int row_count3 = N_fea_block;

        int first_row4 = 3 * N_fea_block;
        int row_count4 = N_fea_block + N_fea_rest;

        ITYPE max_fea1, max_fea2, max_fea3, max_fea4;
        int   last_index1, last_index2, last_index3, last_index4;

        /* ── Read sparse A – all 4 partitions ── */
#if (COO_MODE == 0)
        reada1_csr(beta_qu, f_align, quantization_scale_fea,
                   last_index1, stream_mode, gemm_mode, M_fea,
                   first_row1, row_count1,
                   A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                   rowPtr_fea1, columnIndex_fea1, values_fea1, values_feas1);

        reada1_csr(beta_qu, f_align, quantization_scale_fea,
                   last_index2, stream_mode, gemm_mode, M_fea,
                   first_row2, row_count2,
                   A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
                   rowPtr_fea2, columnIndex_fea2, values_fea2, values_feas2);

        reada1_csr(beta_qu, f_align, quantization_scale_fea,
                   last_index3, stream_mode, gemm_mode, M_fea,
                   first_row3, row_count3,
                   A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3,
                   rowPtr_fea3, columnIndex_fea3, values_fea3, values_feas3);

        reada1_csr(beta_qu, f_align, quantization_scale_fea,
                   last_index4, stream_mode, gemm_mode, M_fea,
                   first_row4, row_count4,
                   A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4,
                   rowPtr_fea4, columnIndex_fea4, values_fea4, values_feas4);
#else
        reada1_coo(nnz_fea1, beta_qu, f_align, quantization_scale_fea,
                   last_index1, stream_mode, gemm_mode, M_fea,
                   first_row1, row_count1,
                   A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                   rowPtr_fea1, columnIndex_fea1, values_fea1, values_feas1);

        reada1_coo(nnz_fea2, beta_qu, f_align, quantization_scale_fea,
                   last_index2, stream_mode, gemm_mode, M_fea,
                   first_row2, row_count2,
                   A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
                   rowPtr_fea2, columnIndex_fea2, values_fea2, values_feas2);

        reada1_coo(nnz_fea3, beta_qu, f_align, quantization_scale_fea,
                   last_index3, stream_mode, gemm_mode, M_fea,
                   first_row3, row_count3,
                   A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3,
                   rowPtr_fea3, columnIndex_fea3, values_fea3, values_feas3);

        reada1_coo(nnz_fea4, beta_qu, f_align, quantization_scale_fea,
                   last_index4, stream_mode, gemm_mode, M_fea,
                   first_row4, row_count4,
                   A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4,
                   rowPtr_fea4, columnIndex_fea4, values_fea4, values_feas4);
#endif  /* COO_MODE */

        /* ── Compute SpMM for each partition ── */
#if ADJ_THREADS == 4
        compute1_4(scale_fea, &max_fea1, quantized_multiplier,
                   gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row1, row_count1,
                   A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                   B_accel1, C_fea11, C_fea12, C_fea13, C_fea14, A_fea11);

        compute1_4(scale_fea, &max_fea2, quantized_multiplier,
                   gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row2, row_count2,
                   A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
                   B_accel2, C_fea21, C_fea22, C_fea23, C_fea24, A_fea21);

        compute1_4(scale_fea, &max_fea3, quantized_multiplier,
                   gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row3, row_count3,
                   A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3,
                   B_accel3, C_fea31, C_fea32, C_fea33, C_fea34, A_fea31);

        compute1_4(scale_fea, &max_fea4, quantized_multiplier,
                   gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row4, row_count4,
                   A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4,
                   B_accel4, C_fea41, C_fea42, C_fea43, C_fea44, A_fea41);

        *max_fea = max_fea1;
#endif  /* ADJ_THREADS == 4 */

#if ADJ_THREADS == 2
        compute1_2(gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row1, row_count1,
                   A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                   B_accel1, C_fea11, C_fea12,
                   B_index, B_index_loop, tail);

        compute1_2(gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row2, row_count2,
                   A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
                   B_accel2, C_fea21, C_fea22,
                   B_index, B_index_loop, tail);

        compute1_2(gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row3, row_count3,
                   A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3,
                   B_accel3, C_fea31, C_fea32,
                   B_index, B_index_loop, tail);

        compute1_2(gemm_mode, zero_point_lhs, zero_point_rhs,
                   first_row4, row_count4,
                   A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4,
                   B_accel4, C_fea41, C_fea42,
                   B_index, B_index_loop, tail);
#endif  /* ADJ_THREADS == 2 */

#endif  /* FEA_THREADS == 4 */

    }  /* end LOOP_FEA / single-pass block */

}


// =============================================================================================
// =============================================================================================
/**
 * loop_adj - Adjacency SpMM engine: computes D = Adj × C for each GNN layer.
 *
 * For each layer iteration (B_index), this function:
 *   1. Acquires read locks on the C tiles produced by loop_fea (stream-of-blocks).
 *   2. Multiplies the sparse adjacency matrix Adj × C using compute2_* tasks,
 *      writing partial results into D_fifo arrays.
 *   3. Dequantizes and writes the final output via writec / writeout.
 *
 * Row-partition parallelism is controlled at compile time by:
 *   ADJ_THREADS  – number of adjacency row partitions processed in parallel (1/2/4).
 *   FEA_THREADS  – number of C-tile column partitions consumed per adj partition (1/2/4).
 *   PIPO_BLOCKS  – enables ping-pong double-buffering when >= 2.
 *   LINEAR_ENABLE– enables reading the linear-projection buffer alongside C.
 */
void loop_adj(
    /* ── Per-layer dequantization factors and activation config ── */
    float       deq_factor[5],
    ap_uint<1>  model[5][8],        // per-layer mode flags
    float       srelu[5],           // per-layer shift-ReLU thresholds

    /* ── Sparse adjacency matrix streams (one set per adj-thread partition) ── */
    hls::stream<ITYPE> &A_fifo_adj1,
    hls::stream<int>   &col_indices_fifo_adj1,
    hls::stream<int>   &rnnz_fifo_adj1,

    hls::stream<TTYPE> &A_fifo_adj2,
    hls::stream<int>   &col_indices_fifo_adj2,
    hls::stream<int>   &rnnz_fifo_adj2,

    hls::stream<TTYPE> &A_fifo_adj3,
    hls::stream<int>   &col_indices_fifo_adj3,
    hls::stream<int>   &rnnz_fifo_adj3,

    hls::stream<TTYPE> &A_fifo_adj4,
    hls::stream<int>   &col_indices_fifo_adj4,
    hls::stream<int>   &rnnz_fifo_adj4,

    /* ── Adjacency / output dimensions ── */
    int         N_adj,              // number of rows in the adjacency matrix
    int         M_adj,              // number of columns
    ap_uint<8>  P_w[5],            // per-layer output column widths

    /* ── Quantization zero points ── */
    ap_int<8>   zero_point_lhs,
    ap_int<8>   zero_point_rhs,

    /* ── Input C ping-pong stream-of-blocks from loop_fea (row-partition 1) ── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf> &C_buffer11,
    hls::stream_of_blocks<buf> &C_buffer12,
#else
    buf                         C_buffer11,
    hls::stream_of_blocks<buf> &C_buffer12,
#endif
    hls::stream_of_blocks<buf> &C_buffer13,
    hls::stream_of_blocks<buf> &C_buffer14,

    /* ── Input C stream-of-blocks (row-partition 2) ── */
    hls::stream_of_blocks<buf> &C_buffer21,
    hls::stream_of_blocks<buf> &C_buffer22,
    hls::stream_of_blocks<buf> &C_buffer23,
    hls::stream_of_blocks<buf> &C_buffer24,

    /* ── Input C stream-of-blocks (row-partition 3) ── */
    hls::stream_of_blocks<buf> &C_buffer31,
    hls::stream_of_blocks<buf> &C_buffer32,
    hls::stream_of_blocks<buf> &C_buffer33,
    hls::stream_of_blocks<buf> &C_buffer34,

    /* ── Input C stream-of-blocks (row-partition 4) ── */
    hls::stream_of_blocks<buf> &C_buffer41,
    hls::stream_of_blocks<buf> &C_buffer42,
    hls::stream_of_blocks<buf> &C_buffer43,
    hls::stream_of_blocks<buf> &C_buffer44,

    /* ── Linear-projection ping-pong buffer (read side) ── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<bufl> &linear_pipo,
#else
    bufl                         linear_pipo,
#endif

    /* ── Number of GNN layers to iterate over ── */
    int         layer_loop,

    /* ── DDR output arrays (one per adj-thread partition) ── */
    OUTTYPE    *D1,
    OUTTYPE    *D2,
    OUTTYPE    *D3,
    OUTTYPE    *D4,

    /* ── AXI-stream output interfaces ── */
    hls::stream<ASTYPE> &DS1,
    hls::stream<ASTYPE> &DS1R,
    hls::stream<ASTYPE> &DS1C,
    hls::stream<ASTYPE> &DS2,
    hls::stream<ASTYPE> &DS3,
    hls::stream<ASTYPE> &DS4
)
{
    /* ────────────────────────────────────────────────────────────────────
     * Inter-task FIFOs.
     * compute2_* tasks produce partial D results into D_fifo arrays;
     * writec / writeout consume them.
     * write_fifo arrays are reserved for future use (currently unused paths).
     * Each array has B_WIDTH_BLOCK independent FIFOs to allow column-parallel
     * access without port conflicts.
     * ──────────────────────────────────────────────────────────────────── */

    /* Partial D results from compute2 tasks → writec */
    hls::stream<ITYPE> D_fifo1[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=D_fifo1 depth=FIFO_DEPTH

    hls::stream<ITYPE> D_fifo2[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=D_fifo2 depth=FIFO_DEPTH

    hls::stream<ITYPE> D_fifo3[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=D_fifo3 depth=FIFO_DEPTH

    hls::stream<ITYPE> D_fifo4[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=D_fifo4 depth=FIFO_DEPTH

    /* Dequantized output stream from writec → writeout */
    hls::stream<OUTTYPE> out_fifo1;
    #pragma HLS STREAM variable=out_fifo1 depth=FIFO_DEPTH

    /* Reserved write-back FIFOs (unused paths kept for future expansion) */
    hls::stream<ITYPE> write_fifo1[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=write_fifo1 depth=FIFO_DEPTH

    hls::stream<ITYPE> write_fifo2[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=write_fifo2 depth=FIFO_DEPTH

    hls::stream<ITYPE> write_fifo3[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=write_fifo3 depth=FIFO_DEPTH

    hls::stream<ITYPE> write_fifo4[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=write_fifo4 depth=FIFO_DEPTH

    /* ────────────────────────────────────────────────────────────────────
     * Layer loop (mirrors loop_fea ping-pong structure).
     * ──────────────────────────────────────────────────────────────────── */
#if (PIPO_BLOCKS >= 2)
    LOOP_ADJ:
    for (int B_index = 0; B_index < layer_loop; B_index++)
    {
#else
    {
        int B_index = 0;
#endif
        #pragma HLS DATAFLOW

        std::cout << "adj layer " << B_index << std::endl;

        /* ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
         * ADJ_THREADS == 1  :  single-partition path.
         * All N_adj rows processed by one compute + one write task.
         * ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── */
#if ADJ_THREADS == 1

        /* Acquire read locks on the C tiles written by loop_fea */
#if (PIPO_BLOCKS >= 2)
        hls::read_lock<buf>  C_adj11(C_buffer11);

#if (LINEAR_ENABLE == 1)
        hls::read_lock<bufl> linear_adj(linear_pipo);
#else
        QLTYPE linear_adj[B_HEIGHT][B_WIDTH_BLOCK];  // dummy when linear disabled
#endif
#endif

#if FEA_THREADS == 2
        hls::read_lock<buf>  C_adj21(C_buffer21);
#endif

        /* Full adjacency row range assigned to partition 1 */
        int first_row1 = 0;
        int row_count1 = N_adj / ADJ_THREADS;

        /* ── Multiply Adj × C → D_fifo ── */
#if FEA_THREADS == 1

#if (PIPO_BLOCKS >= 2)
        compute2_1(
            model,
            srelu,
            N_adj / ADJ_THREADS,
            zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            C_adj11,
            D_fifo1,
            B_index);
#else
        compute2_1(
            relu,
            N_adj / ADJ_THREADS,
            zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            C_buffer11,
            D_fifo1);
#endif

#endif  /* FEA_THREADS == 1 */

#if FEA_THREADS == 2
        /* Two C-tile columns contribute to this adj partition */
        compute2_2(
            N_adj / FEA_THREADS,
            zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            C_adj11, C_adj21,
            D_fifo1);
#endif

        /* ── Dequantize and forward to writeout (ping-pong path) ── */
#if (PIPO_BLOCKS >= 2)
        writec(
            deq_factor, model,
            first_row1, row_count1,
            N_adj, P_w,
            D_fifo1, linear_adj,
            out_fifo1,
            B_index, layer_loop);
#else
        writec(
            deq_factor, model,
            first_row1, row_count1,
            N_adj, P_w,
            D_fifo1, linear_pipo,
            D1, DS1,
            B_index, layer_loop);
#endif

        /* ── Write final output to DDR / AXI-stream ── */
        writeout(
            model,
            first_row1, row_count1,
            N_adj, P_w,
            out_fifo1,
            D1, DS1, DS1R, DS1C,
            B_index, layer_loop);

#endif  /* ADJ_THREADS == 1 */


        /* ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
         * ADJ_THREADS == 2  :  two-partition path.
         * N_adj rows split in half; two compute + two write tasks in parallel.
         * ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── */
#if ADJ_THREADS == 2

        /* Acquire read locks for both adj partitions */
        hls::read_lock<buf> C_adj11(C_buffer11);
        hls::read_lock<buf> C_adj12(C_buffer12);
        hls::read_lock<buf> C_adj21(C_buffer21);
        hls::read_lock<buf> C_adj22(C_buffer22);

#if FEA_THREADS == 4
        hls::read_lock<buf> C_adj31(C_buffer31);
        hls::read_lock<buf> C_adj32(C_buffer32);
        hls::read_lock<buf> C_adj41(C_buffer41);
        hls::read_lock<buf> C_adj42(C_buffer42);
#endif

        /* Split N_adj rows; remainder goes to partition 2 */
        int N_adj_block         = N_adj / ADJ_THREADS;
        int N_adj_rest          = N_adj % ADJ_THREADS;
        int N_adj_block_compute = N_adj / FEA_THREADS;

        int first_row1 = 0;
        int row_count1 = N_adj_block;

        int first_row2 = N_adj_block;
        int row_count2 = N_adj_block + N_adj_rest;

        /* Read sparse adjacency rows for each partition */
        reada2(
            first_row1, row_count1,
            B_index_loop, tail,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            rowPtr_adj1, columnIndex_adj1, values_adj1);

        reada2(
            first_row2, row_count2,
            B_index_loop, tail,
            A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,
            rowPtr_adj2, columnIndex_adj2, values_adj2);

        /* ── Multiply Adj × C → D_fifo for each partition ── */
#if FEA_THREADS == 2
        compute2_2(
            N_adj_block_compute,
            zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            C_adj11, C_adj21,
            D_fifo1,
            B_index, B_index_loop, tail);

        compute2_2(
            N_adj_block_compute,
            zero_point_lhs, zero_point_rhs,
            first_row2, row_count2,
            A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,
            C_adj12, C_adj22,
            D_fifo2,
            B_index, B_index_loop, tail);
#endif

#if FEA_THREADS == 4
        compute2_4(
            N_adj_block_compute,
            zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            C_adj11, C_adj21, C_adj31, C_adj41,
            D_fifo1,
            B_index, B_index_loop, tail);

        compute2_4(
            N_adj_block_compute,
            zero_point_lhs, zero_point_rhs,
            first_row2, row_count2,
            A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,
            C_adj12, C_adj22, C_adj32, C_adj42,
            D_fifo2,
            B_index, B_index_loop, tail);
#endif

        /* ── Write results to DDR ── */
        writec(first_row1, row_count1, P_w, D_fifo1, D1, B_index, B_index_loop, tail);
        writec(first_row2, row_count2, P_w, D_fifo2, D2, B_index, B_index_loop, tail);

#endif  /* ADJ_THREADS == 2 */


        /* ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
         * ADJ_THREADS == 4  :  four-partition path.
         * N_adj rows split into 4 blocks; four independent compute + write
         * task sets execute as parallel dataflow processes.
         * ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── */
#if ADJ_THREADS == 4

        /* Acquire read locks for all 16 C-tile slots (4 partitions × 4 adj columns) */
        hls::read_lock<buf> C_adj11(C_buffer11);
        hls::read_lock<buf> C_adj12(C_buffer12);
        hls::read_lock<buf> C_adj13(C_buffer13);
        hls::read_lock<buf> C_adj14(C_buffer14);

        hls::read_lock<buf> C_adj21(C_buffer21);
        hls::read_lock<buf> C_adj22(C_buffer22);
        hls::read_lock<buf> C_adj23(C_buffer23);
        hls::read_lock<buf> C_adj24(C_buffer24);

        hls::read_lock<buf> C_adj31(C_buffer31);
        hls::read_lock<buf> C_adj32(C_buffer32);
        hls::read_lock<buf> C_adj33(C_buffer33);
        hls::read_lock<buf> C_adj34(C_buffer34);

        hls::read_lock<buf> C_adj41(C_buffer41);
        hls::read_lock<buf> C_adj42(C_buffer42);
        hls::read_lock<buf> C_adj43(C_buffer43);
        hls::read_lock<buf> C_adj44(C_buffer44);

        /* Split N_adj rows across 4 partitions; remainder goes to partition 4 */
        int N_adj_block = N_adj / 4;
        int N_adj_rest  = N_adj % 4;

        int first_row1 = 0;
        int row_count1 = N_adj_block;

        int first_row2 = N_adj_block;
        int row_count2 = N_adj_block;

        int first_row3 = 2 * N_adj_block;
        int row_count3 = N_adj_block;

        int first_row4 = 3 * N_adj_block;
        int row_count4 = N_adj_block + N_adj_rest;

        /* ── Multiply Adj × C → D_fifo for each of the 4 partitions ── */
        compute2_4(
            relu, N_adj_block,
            zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            C_adj11, C_adj21, C_adj31, C_adj41,
            D_fifo1);

        compute2_4(
            relu, N_adj_block,
            zero_point_lhs, zero_point_rhs,
            first_row2, row_count2,
            A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,
            C_adj12, C_adj22, C_adj32, C_adj42,
            D_fifo2);

        compute2_4(
            relu, N_adj_block,
            zero_point_lhs, zero_point_rhs,
            first_row3, row_count3,
            A_fifo_adj3, col_indices_fifo_adj3, rnnz_fifo_adj3,
            C_adj13, C_adj23, C_adj33, C_adj43,
            D_fifo3);

        compute2_4(
            relu, N_adj_block,
            zero_point_lhs, zero_point_rhs,
            first_row4, row_count4,
            A_fifo_adj4, col_indices_fifo_adj4, rnnz_fifo_adj4,
            C_adj14, C_adj24, C_adj34, C_adj44,
            D_fifo4);

        /* ── Dequantize and write all 4 partitions to DDR / AXI-stream ── */
        writec(deq_factor, stream_mode, first_row1, row_count1, N_adj, P_w, D_fifo1, D1, DS1, B_index);
        writec(deq_factor, stream_mode, first_row2, row_count2, N_adj, P_w, D_fifo2, D2, DS2, B_index);
        writec(deq_factor, stream_mode, first_row3, row_count3, N_adj, P_w, D_fifo3, D3, DS3, B_index);
        writec(deq_factor, stream_mode, first_row4, row_count4, N_adj, P_w, D_fifo4, D4, DS4, B_index);

#endif  /* ADJ_THREADS == 4 */

    }  /* end LOOP_ADJ / single-pass block */
}

// =============================================================================================
// =============================================================================================
/**
 * loop_adj2 - GAT (Graph Attention) pipeline: attention scoring followed by SpMM.
 *
 * This function wraps two sequential dataflow stages:
 *
 *   Stage 1 – loop_attention:
 *     Reads the raw adjacency sparse matrix (CSR format), computes per-edge
 *     attention scores using the pre-activation buffers (A_buffer) produced
 *     by loop_fea, applies softmax normalization, and emits attention-weighted
 *     sparse streams (rnnz_att, columnIndex_att, values_att) for each partition.
 *     Also writes edge scores (E1) and softmax outputs (S1) to DDR.
 *
 *   Stage 2 – loop_adj:
 *     Consumes the attention-weighted sparse streams from Stage 1 and the
 *     feature output tiles C from loop_fea, computes the final aggregation
 *     SpMM (Att × C), dequantizes, and writes results to DDR / AXI-streams.
 *
 * The two stages are connected through internal FIFOs and execute as a
 * top-level DATAFLOW region, enabling overlap between attention computation
 * and the subsequent SpMM aggregation.
 */
void loop_adj2(
    /* ── Non-zero counts for each adjacency partition ── */
    int         nnz_adj1,
    int         nnz_adj2,
    int         nnz_adj3,
    int         nnz_adj4,

    /* ── Quantization parameters ── */
    int         beta_qu,                    // zero-point shift
    int         f_align,                    // fractional alignment bits
    float       quantization_scale_adj,     // adjacency value scale
    float       quantization_scale_w[5],    // per-layer weight scales

    /* ── Per-layer dequantization factors and activation config ── */
    float       deq_factor[5],
    ap_uint<1>  model[5][8],                // per-layer mode flags
    float       srelu[5],                   // per-layer shift-ReLU thresholds

    /* ── CSR row pointers for each adjacency partition ── */
    int        *rowPtr_adj1,
    int        *rowPtr_adj2,
    int        *rowPtr_adj3,
    int        *rowPtr_adj4,

    /* ── CSR column indices for each adjacency partition ── */
    int        *columnIndex_adj1,
    int        *columnIndex_adj2,
    int        *columnIndex_adj3,
    int        *columnIndex_adj4,

    /* ── CSR values for each adjacency partition ── */
    INTYPE     *values_adj1,
    INTYPE     *values_adj2,
    INTYPE     *values_adj3,
    INTYPE     *values_adj4,

    /* ── Adjacency / output dimensions ── */
    int         N_adj,                      // number of adjacency rows
    int         M_adj,                      // number of adjacency columns
    ap_uint<8>  P_w[5],                     // per-layer output column widths

    /* ── Quantization zero points ── */
    ap_int<8>   zero_point_lhs,
    ap_int<8>   zero_point_rhs,

    /* ── Input node features for attention scoring (DDR) ── */
    INTYPE     *A,

    /* ── Attention pre-activation buffers from loop_fea (one per partition) ── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf> &A_buffer11,
    hls::stream_of_blocks<buf> &A_buffer21,
#else
    buf                         A_buffer11,
    hls::stream_of_blocks<buf> &A_buffer21,
#endif
    hls::stream_of_blocks<buf> &A_buffer31,
    hls::stream_of_blocks<buf> &A_buffer41,

    /* ── DDR outputs: edge attention scores (E1) and softmax values (S1) ── */
    OUTTYPE    *E1,
    OUTTYPE    *S1,

    /* ── Input C ping-pong stream-of-blocks from loop_fea (row-partition 1) ── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf> &C_buffer11,
    hls::stream_of_blocks<buf> &C_buffer12,
#else
    buf                         C_buffer11,
    hls::stream_of_blocks<buf> &C_buffer12,
#endif
    hls::stream_of_blocks<buf> &C_buffer13,
    hls::stream_of_blocks<buf> &C_buffer14,

    /* ── Input C stream-of-blocks (row-partition 2) ── */
    hls::stream_of_blocks<buf> &C_buffer21,
    hls::stream_of_blocks<buf> &C_buffer22,
    hls::stream_of_blocks<buf> &C_buffer23,
    hls::stream_of_blocks<buf> &C_buffer24,

    /* ── Input C stream-of-blocks (row-partition 3) ── */
    hls::stream_of_blocks<buf> &C_buffer31,
    hls::stream_of_blocks<buf> &C_buffer32,
    hls::stream_of_blocks<buf> &C_buffer33,
    hls::stream_of_blocks<buf> &C_buffer34,

    /* ── Input C stream-of-blocks (row-partition 4) ── */
    hls::stream_of_blocks<buf> &C_buffer41,
    hls::stream_of_blocks<buf> &C_buffer42,
    hls::stream_of_blocks<buf> &C_buffer43,
    hls::stream_of_blocks<buf> &C_buffer44,

    /* ── Linear-projection ping-pong buffer (read side) ── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<bufl> &linear_pipo,
#else
    bufl                         linear_pipo,
#endif

    /* ── Number of GNN layers to iterate over ── */
    int         layer_loop,

    /* ── DDR output arrays for SpMM results (one per adj-thread partition) ── */
    OUTTYPE    *D1,
    OUTTYPE    *D2,
    OUTTYPE    *D3,
    OUTTYPE    *D4,

    /* ── AXI-stream output interfaces ── */
    hls::stream<ASTYPE> &DS1,
    hls::stream<ASTYPE> &DS1R,
    hls::stream<ASTYPE> &DS1C,
    hls::stream<ASTYPE> &DS2,
    hls::stream<ASTYPE> &DS3,
    hls::stream<ASTYPE> &DS4
)
{
    /* ────────────────────────────────────────────────────────────────────
     * Inter-stage FIFOs: attention-weighted sparse streams.
     * loop_attention produces into these; loop_adj consumes from them.
     * One set of (rnnz, columnIndex, values) FIFOs per row partition.
     * Partition 1 FIFOs are named explicitly for debug visibility.
     * ──────────────────────────────────────────────────────────────────── */

    hls::stream<int>   rnnz_att1("rnnz_att1 stream");
    #pragma HLS STREAM variable=rnnz_att1 depth=FIFO_DEPTH

    hls::stream<ITYPE> values_att1("values_att1 stream");
    #pragma HLS STREAM variable=values_att1 depth=FIFO_DEPTH

    hls::stream<int>   columnIndex_att1("columnIndex_att1 stream");
    #pragma HLS STREAM variable=columnIndex_att1 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_att2;
    #pragma HLS STREAM variable=rnnz_att2 depth=FIFO_DEPTH

    hls::stream<ITYPE> values_att2;
    #pragma HLS STREAM variable=values_att2 depth=FIFO_DEPTH

    hls::stream<int>   columnIndex_att2;
    #pragma HLS STREAM variable=columnIndex_att2 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_att3;
    #pragma HLS STREAM variable=rnnz_att3 depth=FIFO_DEPTH

    hls::stream<ITYPE> values_att3;
    #pragma HLS STREAM variable=values_att3 depth=FIFO_DEPTH

    hls::stream<int>   columnIndex_att3;
    #pragma HLS STREAM variable=columnIndex_att3 depth=FIFO_DEPTH

    hls::stream<int>   rnnz_att4;
    #pragma HLS STREAM variable=rnnz_att4 depth=FIFO_DEPTH

    hls::stream<ITYPE> values_att4;
    #pragma HLS STREAM variable=values_att4 depth=FIFO_DEPTH

    hls::stream<int>   columnIndex_att4;
    #pragma HLS STREAM variable=columnIndex_att4 depth=FIFO_DEPTH

    /* ────────────────────────────────────────────────────────────────────
     * Top-level DATAFLOW region: Stage 1 and Stage 2 overlap in hardware.
     * ──────────────────────────────────────────────────────────────────── */
    #pragma HLS DATAFLOW

    /* ── Stage 1: Compute per-edge attention scores and emit weighted sparse streams ── */
    loop_attention(
        deq_factor,
        beta_qu, f_align,
        quantization_scale_adj,
        quantization_scale_w,
        model,
        nnz_adj1, nnz_adj2, nnz_adj3, nnz_adj4,
        rowPtr_adj1,      rowPtr_adj2,      rowPtr_adj3,      rowPtr_adj4,
        columnIndex_adj1, columnIndex_adj2, columnIndex_adj3, columnIndex_adj4,
        values_adj1,      values_adj2,      values_adj3,      values_adj4,
        N_adj, M_adj, P_w,
        A,
        A_buffer11, A_buffer21, A_buffer31, A_buffer41,
        E1, S1,
        rnnz_att1,        columnIndex_att1, values_att1,
        rnnz_att2,        columnIndex_att2, values_att2,
        rnnz_att3,        columnIndex_att3, values_att3,
        rnnz_att4,        columnIndex_att4, values_att4,
        layer_loop);

    /* ── Stage 2: SpMM aggregation using attention-weighted adjacency ── */
    loop_adj(
        deq_factor,
        model,
        srelu,
        values_att1,      columnIndex_att1, rnnz_att1,
        values_att2,      columnIndex_att2, rnnz_att2,
        values_att3,      columnIndex_att3, rnnz_att3,
        values_att4,      columnIndex_att4, rnnz_att4,
        N_adj, M_adj, P_w,
        zero_point_lhs, zero_point_rhs,
        C_buffer11, C_buffer12, C_buffer13, C_buffer14,
        C_buffer21, C_buffer22, C_buffer23, C_buffer24,
        C_buffer31, C_buffer32, C_buffer33, C_buffer34,
        C_buffer41, C_buffer42, C_buffer43, C_buffer44,
        linear_pipo,
        layer_loop,
        D1, D2, D3, D4,
        DS1, DS1R, DS1C, DS2, DS3, DS4);
}

// =============================================================================================
// =============================================================================================

/**
 * mmult_wrapper - Top-level GNN accelerator dataflow wrapper.
 *
 * Instantiates all ping-pong buffers and orchestrates the two-stage pipeline:
 *
 *   Stage 1 – loop_fea:
 *     Loads weight tiles (B, B2), reads the sparse feature matrix in
 *     COO/CSR format, computes SpMM  C = Features × Weights, and writes
 *     quantized results into C_buffer and A_buffer tiles.
 *
 *   Stage 2 – loop_adj2:
 *     Reads the sparse adjacency matrix, computes attention scores via
 *     loop_attention (GAT), then aggregates  D = Adjacency × C via loop_adj,
 *     and writes the final node embeddings to DDR or AXI-stream.
 *
 * Buffer naming conventions:
 *   C_bufferXY  – feature SpMM output tile; X = row partition, Y = adj-thread slot.
 *   A_bufferX1  – attention pre-activation tile for row partition X (GAT only).
 *   linear_pipo – linear-projection output tile (LINEAR_ENABLE path).
 *
 * When PIPO_BLOCKS >= 2, all C/A/linear buffers are declared as
 * stream_of_blocks with depth PIPO_BLOCKS, enabling ping-pong overlap
 * between loop_fea and loop_adj2 across consecutive layer iterations.
 * When PIPO_BLOCKS == 1, plain arrays are used (single-buffered).
 *
 * The #pragma HLS DATAFLOW directive at the function level (PIPO_BLOCKS >= 2)
 * causes loop_fea and loop_adj2 to execute as overlapping tasks connected
 * by the stream-of-blocks channels.
 */
void mmult_wrapper(
    /* ── Control ── */
    bool            load_weights,

    /* ── Quantization parameters (GNN branch) ── */
    int             beta_qu,
    int             f_align,

    /* ── Quantization parameters (linear branch) ── */
    int             beta_qul,
    int             f_alignl,

    /* ── Per-layer quantization scales ── */
    float           quantization_scale_adj,
    float           quantization_scale_fea[5],
    float           quantization_scale_w[5],
    float           quantization_scale_lin[5],
    float           deq_factor[5],

    /* ── Per-layer mode flags and activations ── */
    ap_uint<1>      model[5][8],
    float           srelu[5],

    /* ── Output scaling ── */
    STYPE           scale_fea[5],
    ITYPE          *max_fea,
    int             quantized_multiplier,
    int             quantized_multiplierl,
    ap_int<32>     *shift,
    ap_int<32>     *bias,
    ap_int<32>      bias_count,

    /* ── Quantization zero points ── */
    ap_int<8>       zero_point_lhs,
    ap_int<8>       zero_point_rhs,
    ap_int<8>       zero_point_dst,
    ap_int<8>       clamp_max,
    ap_int<8>       clamp_min,

    /* ── Matrix dimensions ── */
    int             N_adj,
    int             M_adj,
    int             M_fea,
    ap_uint<8>      P_w[5],

    /* ── Dense weight matrices ── */
    INTYPES        *B,
    INTYPES        *B2,

    /* ── DDR output arrays (one per adj-thread partition) ── */
    OUTTYPE        *D1,
    OUTTYPE        *D2,
    OUTTYPE        *D3,
    OUTTYPE        *D4,

    /* ── AXI-stream output interfaces ── */
    hls::stream<ASTYPE> &DS1,
    hls::stream<ASTYPE> &DS1R,
    hls::stream<ASTYPE> &DS1C,
    hls::stream<ASTYPE> &DS2,
    hls::stream<ASTYPE> &DS3,
    hls::stream<ASTYPE> &DS4,

    /* ── DDR outputs for attention scores and softmax ── */
    OUTTYPE        *E1,
    OUTTYPE        *S1,
    INTYPE         *ate_m,

    /* ── Misc ── */
    int             array_c_adjust,
    ap_int<32>      layer_loop,

    /* ── Non-zero counts for feature partitions ── */
    int             nnz_fea1,
    int             nnz_fea2,
    int             nnz_fea3,
    int             nnz_fea4,

    /* ── CSR row pointers for feature partitions ── */
    int            *rowPtr_fea1,
    int            *rowPtr_fea2,
    int            *rowPtr_fea3,
    int            *rowPtr_fea4,

    /* ── CSR column indices for feature partitions ── */
    int            *columnIndex_fea1,
    int            *columnIndex_fea2,
    int            *columnIndex_fea3,
    int            *columnIndex_fea4,

    /* ── CSR values for feature partitions ── */
    INTYPE         *values_fea1,
    INTYPE         *values_fea2,
    INTYPE         *values_fea3,
    INTYPE         *values_fea4,

    /* ── AXI-stream feature row pointers ── */
    hls::stream<ASTYPE> &rowPtr_feas1,
    hls::stream<ASTYPE> &rowPtr_feas2,
    hls::stream<ASTYPE> &rowPtr_feas3,
    hls::stream<ASTYPE> &rowPtr_feas4,

    /* ── AXI-stream feature column indices ── */
    hls::stream<ASTYPE> &columnIndex_feas1,
    hls::stream<ASTYPE> &columnIndex_feas2,
    hls::stream<ASTYPE> &columnIndex_feas3,
    hls::stream<ASTYPE> &columnIndex_feas4,

    /* ── AXI-stream feature values ── */
    hls::stream<ASTYPE> &values_feas1,
    hls::stream<ASTYPE> &values_feas2,
    hls::stream<ASTYPE> &values_feas3,
    hls::stream<ASTYPE> &values_feas4,

    /* ── Non-zero counts for adjacency partitions ── */
    int             nnz_adj1,
    int             nnz_adj2,
    int             nnz_adj3,
    int             nnz_adj4,

    /* ── CSR row pointers for adjacency partitions ── */
    int            *rowPtr_adj1,
    int            *rowPtr_adj2,
    int            *rowPtr_adj3,
    int            *rowPtr_adj4,

    /* ── CSR column indices for adjacency partitions ── */
    int            *columnIndex_adj1,
    int            *columnIndex_adj2,
    int            *columnIndex_adj3,
    int            *columnIndex_adj4,

    /* ── CSR values for adjacency partitions ── */
    INTYPE         *values_adj1,
    INTYPE         *values_adj2,
    INTYPE         *values_adj3,
    INTYPE         *values_adj4
)
{
    /* ────────────────────────────────────────────────────────────────────
     * Linear-projection output ping-pong buffer.
     * Written by loop_fea (compute1_12), read by loop_adj (writec).
     * Partitioned to expose BLOCK/2 parallel column ports and SBLOCK_LIN
     * parallel row ports for the linear inner loop.
     * ──────────────────────────────────────────────────────────────────── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<bufl, PIPO_BLOCKS> linear_pipo;
#else
    bufl linear_pipo;
#endif
    #pragma HLS array_partition variable=linear_pipo block  factor=BLOCK/2    dim=2
    #pragma HLS array_partition variable=linear_pipo cyclic factor=SBLOCK_LIN dim=1

    /* ────────────────────────────────────────────────────────────────────
     * Feature SpMM output tiles (C_bufferXY).
     * X ∈ {1..4} = FEA_THREADS row partition.
     * Y ∈ {1..4} = ADJ_THREADS column slot.
     * Written by loop_fea (compute1_1), read by loop_adj2 (compute2_*).
     * The first buffer (C_buffer11) is single-buffered when PIPO_BLOCKS == 1.
     * ──────────────────────────────────────────────────────────────────── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer11;
#else
    buf C_buffer11;
#endif
    #pragma HLS array_partition variable=C_buffer11 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer11 cyclic factor=SBLOCK  dim=1

    /* Row-partition 1, adj slots 2-4 */
    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer12;
    #pragma HLS array_partition variable=C_buffer12 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer12 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer13;
    #pragma HLS array_partition variable=C_buffer13 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer13 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer14;
    #pragma HLS array_partition variable=C_buffer14 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer14 cyclic factor=SBLOCK  dim=1

    /* Row-partition 2 */
    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer21;
    #pragma HLS array_partition variable=C_buffer21 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer21 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer22;
    #pragma HLS array_partition variable=C_buffer22 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer22 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer23;
    #pragma HLS array_partition variable=C_buffer23 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer23 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer24;
    #pragma HLS array_partition variable=C_buffer24 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer24 cyclic factor=SBLOCK  dim=1

    /* Row-partition 3 */
    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer31;
    #pragma HLS array_partition variable=C_buffer31 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer31 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer32;
    #pragma HLS array_partition variable=C_buffer32 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer32 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer33;
    #pragma HLS array_partition variable=C_buffer33 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer33 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer34;
    #pragma HLS array_partition variable=C_buffer34 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer34 cyclic factor=SBLOCK  dim=1

    /* Row-partition 4 */
    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer41;
    #pragma HLS array_partition variable=C_buffer41 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer41 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer42;
    #pragma HLS array_partition variable=C_buffer42 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer42 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer43;
    #pragma HLS array_partition variable=C_buffer43 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer43 cyclic factor=SBLOCK  dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer44;
    #pragma HLS array_partition variable=C_buffer44 block  factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer44 cyclic factor=SBLOCK  dim=1

    /* ────────────────────────────────────────────────────────────────────
     * Attention pre-activation tiles (A_bufferX1).
     * Written by loop_fea (compute1_1 with GAT_ENABLE), read by loop_adj2
     * (prepare_attentional_mechanism_input*).
     * A_buffer11 is single-buffered when PIPO_BLOCKS == 1.
     * ──────────────────────────────────────────────────────────────────── */
#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer11;
#else
    buf A_buffer11;
#endif
    #pragma HLS array_partition variable=A_buffer11 block factor=BLOCK/2 dim=2

    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer21;
    #pragma HLS array_partition variable=A_buffer21 block factor=BLOCK/2 dim=2

    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer31;
    #pragma HLS array_partition variable=A_buffer31 block factor=BLOCK/2 dim=2

    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer41;
    #pragma HLS array_partition variable=A_buffer41 block factor=BLOCK/2 dim=2

    /* ────────────────────────────────────────────────────────────────────
     * Top-level DATAFLOW region.
     * loop_fea and loop_adj2 execute as overlapping hardware tasks
     * connected by the stream-of-blocks channels above.
     * ──────────────────────────────────────────────────────────────────── */
#if (PIPO_BLOCKS >= 2)
    #pragma HLS DATAFLOW
#endif

    /* ── Stage 1: Feature SpMM  C = Features × Weights ── */
    loop_fea(
        load_weights,
        beta_qu,        f_align,
        beta_qul,       f_alignl,
        quantization_scale_fea,
        quantization_scale_w,
        quantization_scale_lin,
        model,
        scale_fea,      max_fea,
        quantized_multiplier,
        quantized_multiplierl,
        nnz_fea1,       nnz_fea2,       nnz_fea3,       nnz_fea4,
        rowPtr_fea1,    rowPtr_fea2,    rowPtr_fea3,    rowPtr_fea4,
        columnIndex_fea1, columnIndex_fea2, columnIndex_fea3, columnIndex_fea4,
        values_fea1,    values_fea2,    values_fea3,    values_fea4,
        rowPtr_feas1,   rowPtr_feas2,   rowPtr_feas3,   rowPtr_feas4,
        columnIndex_feas1, columnIndex_feas2, columnIndex_feas3, columnIndex_feas4,
        values_feas1,   values_feas2,   values_feas3,   values_feas4,
        B, B2,
        M_adj, M_fea, P_w,
        zero_point_lhs, zero_point_rhs,
        C_buffer11, C_buffer12, C_buffer13, C_buffer14,
        C_buffer21, C_buffer22, C_buffer23, C_buffer24,
        C_buffer31, C_buffer32, C_buffer33, C_buffer34,
        C_buffer41, C_buffer42, C_buffer43, C_buffer44,
        A_buffer11, A_buffer21, A_buffer31, A_buffer41,
        linear_pipo,
        layer_loop
    );

    /* ── Stage 2: Attention + Adjacency SpMM  D = Adj × C ── */
    loop_adj2(
        nnz_adj1,       nnz_adj2,       nnz_adj3,       nnz_adj4,
        beta_qu,        f_align,
        quantization_scale_adj,
        quantization_scale_w,
        deq_factor,
        model,
        srelu,
        rowPtr_adj1,    rowPtr_adj2,    rowPtr_adj3,    rowPtr_adj4,
        columnIndex_adj1, columnIndex_adj2, columnIndex_adj3, columnIndex_adj4,
        values_adj1,    values_adj2,    values_adj3,    values_adj4,
        N_adj, M_adj, P_w,
        zero_point_lhs, zero_point_rhs,
        ate_m,
        A_buffer11, A_buffer21, A_buffer31, A_buffer41,
        E1, S1,
        C_buffer11, C_buffer12, C_buffer13, C_buffer14,
        C_buffer21, C_buffer22, C_buffer23, C_buffer24,
        C_buffer31, C_buffer32, C_buffer33, C_buffer34,
        C_buffer41, C_buffer42, C_buffer43, C_buffer44,
        linear_pipo,
        layer_loop,
        D1, D2, D3, D4,
        DS1, DS1R, DS1C, DS2, DS3, DS4
    );
}

// =============================================================================================
// =============================================================================================
/*
 * mmult_top - SGRACE GNN Accelerator top-level kernel.
 *
 * This is the AXI4-accessible entry point synthesised into the FPGA.
 * It performs one full GNN inference pass (loop_fea + loop_adj2) for
 * up to layer_count stacked GNN layers.
 *
 * Supported computation modes (selected per-layer via model[i] bits):
 *   Bit 0 (gemm_adj) + Bit 1 (gemm_fea):
 *     0,0 – dense feature,  dense adjacency  (unused in graph layers)
 *     0,1 – sparse feature, dense adjacency  (layer 2 normal mode)
 *     1,0 – dense feature,  sparse adjacency (training mode)
 *     1,1 – sparse feature, sparse adjacency (layer 1 normal mode)
 *
 * Steps performed here:
 *   1. Reset all FIFO telemetry counters.
 *   2. Load per-layer configuration arrays from DDR into on-chip registers
 *      (model_int, srelu_int, scale arrays, P_w_int) for fast access
 *      during the compute loop.
 *   3. Call mmult_wrapper to execute the two-stage dataflow pipeline.
 *   4. Forward the maximum activation value to the host via max_fea.
 *
 * Note: bias/shift/quantized_multiplier preloading is disabled (commented out)
 * because on-demand loading gives equivalent performance without the overhead.
 *
 * Memory interface summary (all DDR ports use m_axi + s_axilite address):
 *   Partition 1 of each array uses depth = 64000 (large graph support).
 *   Partitions 2-4 use depth = 4096 (smaller parallel partitions).
 * 
 * 
 * FPGA BRAM usage note:
 *
 * The amount of data stored locally in the FPGA is approximately:
 *
 *     B_HEIGHT * B_WIDTH_BLOCK + A_WIDTH + B_WIDTH_BLOCK
 *
 * This should be smaller than the available FPGA BRAM capacity.
 *
 * gemm_mode encoding:
 *
 *   gemm_mode | fea | adj | Meaning
 *   ----------|-----|-----|----------------------------------------
 *      0      |  0  |  0  | Dense feature, dense adjacency
 *      1      |  0  |  1  | Dense feature, sparse adjacency
 *      2      |  1  |  0  | Sparse feature, dense adjacency
 *      3      |  1  |  1  | Sparse feature, sparse adjacency
 */
 
void mmult_top(
    /* ── Control ── */
    bool            load_weights,

    /* ── Quantization parameters (GNN branch) ── */
    int             beta_qu,
    int             f_align,

    /* ── Quantization parameters (linear branch) ── */
    int             beta_qul,
    int             f_alignl,

    /* ── Per-layer quantization scales ── */
    float           quantization_scale_adj,
    float           quantization_scale_fea[5],
    float           quantization_scale_w[5],
    float           quantization_scale_lin[5],
    float           deq_factor[5],

    /* ── Per-layer mode flags and activations ── */
    ap_uint<8>      model[5],
    float           srelu[5],

    /* ── Output scaling ── */
    STYPE           scale_fea[5],
    ITYPE          *max_fea,
    ap_int<32>      layer_count,
    int             quantized_multiplier,
    int             quantized_multiplierl,
    ap_int<32>     *shift,
    ap_int<32>     *bias,
    ap_int<32>      bias_count,

    /* ── Profiling output ── */
    ap_int<64>     *profiling,

    /* ── Quantization zero points ── */
    ap_int<8>       zero_point_lhs,
    ap_int<8>       zero_point_rhs,
    ap_int<8>       zero_point_dst,
    ap_int<8>       clamp_max,
    ap_int<8>       clamp_min,

    /* ── Matrix dimensions ── */
    int             N_adj,
    int             M_adj,
    int             M_fea,
    ap_uint<8>      P_w[5],

    /* ── Dense weight matrices ── */
    INTYPES        *B,
    INTYPES        *B2,

    /* ── DDR output arrays (one per adj-thread partition) ── */
    OUTTYPE        *D1,
    OUTTYPE        *D2,
    OUTTYPE        *D3,
    OUTTYPE        *D4,

    /* ── AXI-stream output interfaces ── */
    hls::stream<ASTYPE> &DS1,
    hls::stream<ASTYPE> &DS1R,
    hls::stream<ASTYPE> &DS1C,
    hls::stream<ASTYPE> &DS2,
    hls::stream<ASTYPE> &DS3,
    hls::stream<ASTYPE> &DS4,

    /* ── Attention score and softmax outputs ── */
    OUTTYPE        *E1,
    OUTTYPE        *S1,
    INTYPE         *ate_m,

    int             array_c_adjust,

    /* ── Non-zero counts for feature partitions ── */
    int             nnz_fea1,
    int             nnz_fea2,
    int             nnz_fea3,
    int             nnz_fea4,

    /* ── CSR row pointers for feature partitions ── */
    int            *rowPtr_fea1,
    int            *rowPtr_fea2,
    int            *rowPtr_fea3,
    int            *rowPtr_fea4,

    /* ── CSR column indices for feature partitions ── */
    int            *columnIndex_fea1,
    int            *columnIndex_fea2,
    int            *columnIndex_fea3,
    int            *columnIndex_fea4,

    /* ── CSR values for feature partitions ── */
    INTYPE         *values_fea1,
    INTYPE         *values_fea2,
    INTYPE         *values_fea3,
    INTYPE         *values_fea4,

    /* ── AXI-stream feature matrix interfaces ── */
    hls::stream<ASTYPE> &rowPtr_feas1,
    hls::stream<ASTYPE> &rowPtr_feas2,
    hls::stream<ASTYPE> &rowPtr_feas3,
    hls::stream<ASTYPE> &rowPtr_feas4,
    hls::stream<ASTYPE> &columnIndex_feas1,
    hls::stream<ASTYPE> &columnIndex_feas2,
    hls::stream<ASTYPE> &columnIndex_feas3,
    hls::stream<ASTYPE> &columnIndex_feas4,
    hls::stream<ASTYPE> &values_feas1,
    hls::stream<ASTYPE> &values_feas2,
    hls::stream<ASTYPE> &values_feas3,
    hls::stream<ASTYPE> &values_feas4,

    /* ── Non-zero counts for adjacency partitions ── */
    int             nnz_adj1,
    int             nnz_adj2,
    int             nnz_adj3,
    int             nnz_adj4,

    /* ── CSR row pointers for adjacency partitions ── */
    int            *rowPtr_adj1,
    int            *rowPtr_adj2,
    int            *rowPtr_adj3,
    int            *rowPtr_adj4,

    /* ── CSR column indices for adjacency partitions ── */
    int            *columnIndex_adj1,
    int            *columnIndex_adj2,
    int            *columnIndex_adj3,
    int            *columnIndex_adj4,

    /* ── CSR values for adjacency partitions ── */
    INTYPE         *values_adj1,
    INTYPE         *values_adj2,
    INTYPE         *values_adj3,
    INTYPE         *values_adj4
)
{
    /* ── AXI4-Lite control register interface ── */
    #pragma HLS INTERFACE s_axilite port=return               bundle=control
    #pragma HLS INTERFACE s_axilite port=load_weights         bundle=control
    #pragma HLS INTERFACE s_axilite port=beta_qu              bundle=control
    #pragma HLS INTERFACE s_axilite port=f_align              bundle=control
    #pragma HLS INTERFACE s_axilite port=beta_qul             bundle=control
    #pragma HLS INTERFACE s_axilite port=f_alignl             bundle=control
    #pragma HLS INTERFACE s_axilite port=deq_factor           bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_fea1             bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_fea2             bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_fea3             bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_fea4             bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_adj1             bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_adj2             bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_adj3             bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_adj4             bundle=control
    #pragma HLS INTERFACE s_axilite port=quantization_scale_adj bundle=control
    #pragma HLS INTERFACE s_axilite port=quantization_scale_fea bundle=control
    #pragma HLS INTERFACE s_axilite port=quantization_scale_w   bundle=control
    #pragma HLS INTERFACE s_axilite port=quantization_scale_lin bundle=control
    #pragma HLS INTERFACE s_axilite port=bias_count           bundle=control
    #pragma HLS INTERFACE s_axilite port=zero_point_lhs       bundle=control
    #pragma HLS INTERFACE s_axilite port=zero_point_rhs       bundle=control
    #pragma HLS INTERFACE s_axilite port=zero_point_dst       bundle=control
    #pragma HLS INTERFACE s_axilite port=clamp_max            bundle=control
    #pragma HLS INTERFACE s_axilite port=clamp_min            bundle=control
    #pragma HLS INTERFACE s_axilite port=N_adj                bundle=control
    #pragma HLS INTERFACE s_axilite port=M_adj                bundle=control
    #pragma HLS INTERFACE s_axilite port=M_fea                bundle=control
    #pragma HLS INTERFACE s_axilite port=P_w                  bundle=control
    #pragma HLS INTERFACE s_axilite port=array_c_adjust       bundle=control
    #pragma HLS INTERFACE s_axilite port=model                bundle=control
    #pragma HLS INTERFACE s_axilite port=layer_count          bundle=control
    #pragma HLS INTERFACE s_axilite port=quantized_multiplier bundle=control
    #pragma HLS INTERFACE s_axilite port=quantized_multiplierl bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_fea1     bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_fea2     bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_fea3     bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_fea4     bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea1          bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea2          bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea3          bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea4          bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj1     bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj2     bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj3     bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj4     bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj1          bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj2          bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj3          bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj4          bundle=control
    #pragma HLS INTERFACE s_axilite port=values_adj1          bundle=control
    #pragma HLS INTERFACE s_axilite port=values_adj2          bundle=control
    #pragma HLS INTERFACE s_axilite port=values_adj3          bundle=control
    #pragma HLS INTERFACE s_axilite port=values_adj4          bundle=control
    #pragma HLS INTERFACE s_axilite port=values_fea1          bundle=control
    #pragma HLS INTERFACE s_axilite port=values_fea2          bundle=control
    #pragma HLS INTERFACE s_axilite port=values_fea3          bundle=control
    #pragma HLS INTERFACE s_axilite port=values_fea4          bundle=control
    #pragma HLS INTERFACE s_axilite port=B                    bundle=control
    #pragma HLS INTERFACE s_axilite port=B2                   bundle=control
    #pragma HLS INTERFACE s_axilite port=D1                   bundle=control
    #pragma HLS INTERFACE s_axilite port=D2                   bundle=control
    #pragma HLS INTERFACE s_axilite port=D3                   bundle=control
    #pragma HLS INTERFACE s_axilite port=D4                   bundle=control
    #pragma HLS INTERFACE s_axilite port=E1                   bundle=control
    #pragma HLS INTERFACE s_axilite port=S1                   bundle=control
    #pragma HLS INTERFACE s_axilite port=profiling            bundle=control
    #pragma HLS INTERFACE s_axilite port=quantized_multiplier bundle=control
    #pragma HLS INTERFACE s_axilite port=shift                bundle=control
    #pragma HLS INTERFACE s_axilite port=bias                 bundle=control
    #pragma HLS INTERFACE s_axilite port=ate_m                bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_fea            bundle=control
    #pragma HLS INTERFACE s_axilite port=max_fea              bundle=control
    #pragma HLS INTERFACE s_axilite port=srelu                bundle=control

    /* ── AXI4-Stream output interfaces ── */
    #pragma HLS INTERFACE axis port=DS1  depth=64000
    #pragma HLS INTERFACE axis port=DS1R depth=64000
    #pragma HLS INTERFACE axis port=DS1C depth=64000
    #pragma HLS INTERFACE axis port=DS2  depth=4096
    #pragma HLS INTERFACE axis port=DS3  depth=4096
    #pragma HLS INTERFACE axis port=DS4  depth=4096

    /* ── AXI4-Stream feature matrix input interfaces ── */
    #pragma HLS INTERFACE axis port=columnIndex_feas1 depth=4096
    #pragma HLS INTERFACE axis port=columnIndex_feas2 depth=4096
    #pragma HLS INTERFACE axis port=columnIndex_feas3 depth=4096
    #pragma HLS INTERFACE axis port=columnIndex_feas4 depth=4096
    #pragma HLS INTERFACE axis port=rowPtr_feas1      depth=4096
    #pragma HLS INTERFACE axis port=rowPtr_feas2      depth=4096
    #pragma HLS INTERFACE axis port=rowPtr_feas3      depth=4096
    #pragma HLS INTERFACE axis port=rowPtr_feas4      depth=4096
    #pragma HLS INTERFACE axis port=values_feas1      depth=64000
    #pragma HLS INTERFACE axis port=values_feas2      depth=4096
    #pragma HLS INTERFACE axis port=values_feas3      depth=4096
    #pragma HLS INTERFACE axis port=values_feas4      depth=4096

    /* ── AXI4 memory-mapped DDR interfaces ── */
    #pragma HLS INTERFACE m_axi port=profiling          depth=16    offset=slave bundle=profiling
    #pragma HLS INTERFACE m_axi port=rowPtr_fea1        depth=64000 offset=slave bundle=rowPtr_fea1
    #pragma HLS INTERFACE m_axi port=rowPtr_fea2        depth=4096  offset=slave bundle=rowPtr_fea2
    #pragma HLS INTERFACE m_axi port=rowPtr_fea3        depth=4096  offset=slave bundle=rowPtr_fea3
    #pragma HLS INTERFACE m_axi port=rowPtr_fea4        depth=4096  offset=slave bundle=rowPtr_fea4
    #pragma HLS INTERFACE m_axi port=columnIndex_fea1   depth=64000 offset=slave bundle=columnIndex_fea1
    #pragma HLS INTERFACE m_axi port=columnIndex_fea2   depth=4096  offset=slave bundle=columnIndex_fea2
    #pragma HLS INTERFACE m_axi port=columnIndex_fea3   depth=4096  offset=slave bundle=columnIndex_fea3
    #pragma HLS INTERFACE m_axi port=columnIndex_fea4   depth=4096  offset=slave bundle=columnIndex_fea4
    #pragma HLS INTERFACE m_axi port=values_fea1        depth=64000 offset=slave bundle=values_fea1
    #pragma HLS INTERFACE m_axi port=values_fea2        depth=4096  offset=slave bundle=values_fea2
    #pragma HLS INTERFACE m_axi port=values_fea3        depth=4096  offset=slave bundle=values_fea3
    #pragma HLS INTERFACE m_axi port=values_fea4        depth=4096  offset=slave bundle=values_fea4
    #pragma HLS INTERFACE m_axi port=rowPtr_adj1        depth=64000 offset=slave bundle=rowPtr_adj1
    #pragma HLS INTERFACE m_axi port=rowPtr_adj2        depth=4096  offset=slave bundle=rowPtr_adj2
    #pragma HLS INTERFACE m_axi port=rowPtr_adj3        depth=4096  offset=slave bundle=rowPtr_adj3
    #pragma HLS INTERFACE m_axi port=rowPtr_adj4        depth=4096  offset=slave bundle=rowPtr_adj4
    #pragma HLS INTERFACE m_axi port=columnIndex_adj1   depth=64000 offset=slave bundle=columnIndex_adj1
    #pragma HLS INTERFACE m_axi port=columnIndex_adj2   depth=4096  offset=slave bundle=columnIndex_adj2
    #pragma HLS INTERFACE m_axi port=columnIndex_adj3   depth=4096  offset=slave bundle=columnIndex_adj3
    #pragma HLS INTERFACE m_axi port=columnIndex_adj4   depth=4096  offset=slave bundle=columnIndex_adj4
    #pragma HLS INTERFACE m_axi port=values_adj1        depth=64000 offset=slave bundle=values_adj1
    #pragma HLS INTERFACE m_axi port=values_adj2        depth=4096  offset=slave bundle=values_adj2
    #pragma HLS INTERFACE m_axi port=values_adj3        depth=4096  offset=slave bundle=values_adj3
    #pragma HLS INTERFACE m_axi port=values_adj4        depth=4096  offset=slave bundle=values_adj4
    #pragma HLS INTERFACE m_axi port=B                  depth=32000 offset=slave bundle=B
    #pragma HLS INTERFACE m_axi port=B2                 depth=32000 offset=slave bundle=B2
    #pragma HLS INTERFACE m_axi port=D1                 depth=64000 offset=slave bundle=D1
    #pragma HLS INTERFACE m_axi port=D2                 depth=1000  offset=slave bundle=D2
    #pragma HLS INTERFACE m_axi port=D3                 depth=1000  offset=slave bundle=D3
    #pragma HLS INTERFACE m_axi port=D4                 depth=1000  offset=slave bundle=D4
    #pragma HLS INTERFACE m_axi port=E1                 depth=64000 offset=slave bundle=E1
    #pragma HLS INTERFACE m_axi port=S1                 depth=64000 offset=slave bundle=S1
    #pragma HLS INTERFACE m_axi port=ate_m              depth=1000  offset=slave bundle=ate_m
    #pragma HLS INTERFACE m_axi port=shift              depth=1024  offset=slave bundle=shift
    #pragma HLS INTERFACE m_axi port=bias               depth=1024  offset=slave bundle=bias
    #pragma HLS INTERFACE m_axi port=model              depth=1024  offset=slave bundle=model
    #pragma HLS INTERFACE m_axi port=quantization_scale_fea offset=slave bundle=quantization_scale_fea
    #pragma HLS INTERFACE m_axi port=quantization_scale_w   offset=slave bundle=quantization_scale_w
    #pragma HLS INTERFACE m_axi port=quantization_scale_lin offset=slave bundle=quantization_scale_lin
    #pragma HLS INTERFACE m_axi port=deq_factor         offset=slave bundle=deq_factor
    #pragma HLS INTERFACE m_axi port=scale_fea          offset=slave bundle=scale_fea
    #pragma HLS INTERFACE m_axi port=P_w                offset=slave bundle=P_w
    #pragma HLS INTERFACE m_axi port=srelu              offset=slave bundle=srelu

    /* ────────────────────────────────────────────────────────────────────
     * On-chip register copies of per-layer configuration arrays.
     * Copying DDR arrays into fully-partitioned local registers before the
     * compute loop ensures all per-layer parameters are accessible in a
     * single cycle without DDR arbitration latency.
     * ──────────────────────────────────────────────────────────────────── */
    ap_int<32> bias_data[1024];
    ap_int<32> shift_data[1024];
    float      srelu_int[5];
    ap_uint<8> P_w_int[5];
    #pragma HLS ARRAY_PARTITION variable=P_w_int complete

    ap_uint<1> model_int[5][8];
    #pragma HLS ARRAY_PARTITION variable=model_int complete

    float quantization_scale_lin_int[5];
    #pragma HLS ARRAY_PARTITION variable=quantization_scale_lin_int complete

    float quantization_scale_w_int[5];
    #pragma HLS ARRAY_PARTITION variable=quantization_scale_w_int complete

    float quantization_scale_fea_int[5];
    #pragma HLS ARRAY_PARTITION variable=quantization_scale_fea_int complete

    float deq_factor_int[5];
    #pragma HLS ARRAY_PARTITION variable=deq_factor_int complete

    STYPE scale_fea_int[5];
    #pragma HLS ARRAY_PARTITION variable=scale_fea_int complete

    /* ── Reset FIFO telemetry counters ── */
    fifo_empty_0 = fifo_empty_1 = fifo_empty_2 = 0;
    fifo_full_0  = fifo_full_1  = fifo_full_2  = 0;
    fifo_read_0  = fifo_read_1  = fifo_read_2  = 0;
    fifo_write_0 = fifo_write_1 = fifo_write_2 = 0;
    fifo_cycle_0 = fifo_cycle_1 = fifo_cycle_2 = 0;

    ap_int<32> layer_loop = layer_count;

    /* ── Load per-layer configuration from DDR into on-chip registers ── */
    for (int i = 0; i < layer_loop; i++)
    {
        /* Unpack the 8-bit model word into individual bit flags */
        model_int[i][0] = model[i][0];   // gemm_adj
        model_int[i][1] = model[i][1];   // gemm_fea
        model_int[i][2] = model[i][2];   // stream_out
        model_int[i][3] = model[i][3];   // stream_in
        model_int[i][4] = model[i][4];   // relu
        model_int[i][5] = model[i][5];   // gat
        model_int[i][6] = model[i][6];   // linear
        model_int[i][7] = model[i][7];   // sage

        srelu_int[i]                  = srelu[i];
        quantization_scale_lin_int[i] = quantization_scale_lin[i];
        quantization_scale_w_int[i]   = quantization_scale_w[i];
        quantization_scale_fea_int[i] = quantization_scale_fea[i];
        deq_factor_int[i]             = deq_factor[i];
        scale_fea_int[i]              = scale_fea[i];
        P_w_int[i]                    = P_w[i];

        std::cout << "Layer " << i << " instruction: "
                  << model_int[i][7] << model_int[i][6] << model_int[i][5]
                  << model_int[i][4] << model_int[i][3] << model_int[i][2]
                  << model_int[i][1] << model_int[i][0] << std::endl;
    }

    /* ── Execute the GNN pipeline ── */
    ITYPE max_fea_val = 0;

    mmult_wrapper(
        load_weights,
        beta_qu,           f_align,
        beta_qul,          f_alignl,
        quantization_scale_adj,
        quantization_scale_fea_int,
        quantization_scale_w_int,
        quantization_scale_lin_int,
        deq_factor_int,
        model_int,         srelu_int,
        scale_fea_int,     &max_fea_val,
        quantized_multiplier,
        quantized_multiplierl,
        shift_data,        bias_data,  bias_count,
        zero_point_lhs,    zero_point_rhs,
        zero_point_dst,    clamp_max,  clamp_min,
        N_adj, M_adj, M_fea, P_w_int,
        B, B2,
        D1, D2, D3, D4,
        DS1, DS1R, DS1C, DS2, DS3, DS4,
        E1, S1, ate_m,
        array_c_adjust, layer_loop,
        nnz_fea1, nnz_fea2, nnz_fea3, nnz_fea4,
        rowPtr_fea1,      rowPtr_fea2,      rowPtr_fea3,      rowPtr_fea4,
        columnIndex_fea1, columnIndex_fea2, columnIndex_fea3, columnIndex_fea4,
        values_fea1,      values_fea2,      values_fea3,      values_fea4,
        rowPtr_feas1,     rowPtr_feas2,     rowPtr_feas3,     rowPtr_feas4,
        columnIndex_feas1,columnIndex_feas2,columnIndex_feas3,columnIndex_feas4,
        values_feas1,     values_feas2,     values_feas3,     values_feas4,
        nnz_adj1, nnz_adj2, nnz_adj3, nnz_adj4,
        rowPtr_adj1,      rowPtr_adj2,      rowPtr_adj3,      rowPtr_adj4,
        columnIndex_adj1, columnIndex_adj2, columnIndex_adj3, columnIndex_adj4,
        values_adj1,      values_adj2,      values_adj3,      values_adj4
    );

    /* Forward maximum activation value to the host */
    *max_fea = max_fea_val;
}

// =============================================================================================
// =============================================================================================

/**
 * kernelmult1 - Single-layer GNN inference entry point (simplified API).
 *
 * A convenience wrapper around mmult_top that exposes a flatter parameter
 * set suitable for single-layer invocations or external test-bench calls.
 * It fills in defaults for parameters not exposed at this level:
 *
 *   quantized_multiplierl = 8   (linear-branch output bit-width selector)
 *   beta_qul              = 255 (linear-branch quantization range)
 *   f_alignl              = 0   (linear-branch fractional alignment)
 *   srelu[0]              = 0.0 (shift-ReLU disabled for layer 0)
 *   array_c_adjust        = N_adj
 *   P_w_int[0]            = P_w (scalar → per-layer array)
 *
 * After the kernel completes, spot-checks several output values to the
 * console for quick verification during simulation or hardware bring-up.
 *
 * @param load_weights              Load weight tiles into BRAM on this call.
 * @param beta_qu                   GNN quantization range.
 * @param f_align                   GNN fractional alignment bits.
 * @param quantization_scale_adj    Adjacency value scale factor.
 * @param quantization_scale_fea    Per-layer feature scale factors [5].
 * @param quantization_scale_w      Per-layer weight scale factors [5].
 * @param quantization_scale_lin    Per-layer linear-branch scale factors [5].
 * @param deq_factor                Per-layer dequantization factors [5].
 * @param layer_count               Number of GNN layers to process.
 * @param model                     Per-layer 8-bit mode flag word [5].
 * @param scale_fea                 Per-layer output right-shift amounts [5].
 * @param max_fea                   Output: maximum activation value.
 * @param quantized_multiplier      GNN output bit-width selector (1/2/4/8/16).
 * @param shift                     Per-channel requantization shifts.
 * @param bias                      Per-channel bias values.
 * @param bias_count                Number of valid bias/shift entries.
 * @param profiling                 Output: FIFO telemetry counters.
 * @param zero_point_lhs            Quantization zero point for features.
 * @param zero_point_rhs            Quantization zero point for weights.
 * @param zero_point_dst            Quantization zero point for output.
 * @param clamp_max                 Output clamp upper bound.
 * @param clamp_min                 Output clamp lower bound.
 * @param array_b                   DDR: GNN weight matrix.
 * @param array_b2                  DDR: linear-projection weight matrix.
 * @param array_d1..4               DDR output arrays (one per adj partition).
 * @param stream_d1/d1r/d1c/d2/d3/d4 AXI-stream outputs.
 * @param array_e1                  DDR: edge attention scores.
 * @param array_s1                  DDR: softmax attention values.
 * @param ate_m                     DDR: attention parameter vector.
 * @param values_fea1..4            DDR/stream: feature non-zero values.
 * @param values_feas1..4           AXI-stream: feature values.
 * @param colIndices_fea1..4        DDR: feature column indices.
 * @param columnIndex_feas1..4      AXI-stream: feature column indices.
 * @param nnz_fea1..4               Non-zero counts for feature partitions.
 * @param rowPtr_fea1..4            DDR: feature row pointer arrays.
 * @param rowPtr_feas1..4           AXI-stream: feature row pointers.
 * @param values_adj1..4            DDR: adjacency non-zero values.
 * @param colIndices_adj1..4        DDR: adjacency column indices.
 * @param nnz_adj1..4               Non-zero counts for adjacency partitions.
 * @param rowPtr_adj1..4            DDR: adjacency row pointer arrays.
 * @param N_adj                     Number of adjacency rows.
 * @param M_adj                     Number of adjacency columns.
 * @param M_fea                     Input feature dimension.
 * @param P_w                       Output column width for layer 0.
 */
void kernelmult1(
    bool            load_weights,
    int             beta_qu,
    int             f_align,
    float           quantization_scale_adj,
    float           quantization_scale_fea[5],
    float           quantization_scale_w[5],
    float           quantization_scale_lin[5],
    float           deq_factor[5],
    int             layer_count,
    ap_uint<8>      model[5],
    STYPE           scale_fea[5],
    ITYPE          *max_fea,
    int             quantized_multiplier,
    ap_int<32>     *shift,
    ap_int<32>     *bias,
    ap_int<32>      bias_count,
    ap_int<64>     *profiling,
    ap_int<8>       zero_point_lhs,
    ap_int<8>       zero_point_rhs,
    ap_int<8>       zero_point_dst,
    ap_int<8>       clamp_max,
    ap_int<8>       clamp_min,
    INTYPES        *array_b,
    INTYPES        *array_b2,
    OUTTYPE        *array_d1,
    OUTTYPE        *array_d2,
    OUTTYPE        *array_d3,
    OUTTYPE        *array_d4,
    hls::stream<ASTYPE> &stream_d1,
    hls::stream<ASTYPE> &stream_d1r,
    hls::stream<ASTYPE> &stream_d1c,
    hls::stream<ASTYPE> &stream_d2,
    hls::stream<ASTYPE> &stream_d3,
    hls::stream<ASTYPE> &stream_d4,
    OUTTYPE        *array_e1,
    OUTTYPE        *array_s1,
    INTYPE         *ate_m,
    INTYPE         *values_fea1,
    INTYPE         *values_fea2,
    INTYPE         *values_fea3,
    INTYPE         *values_fea4,
    hls::stream<ASTYPE> &values_feas1,
    hls::stream<ASTYPE> &values_feas2,
    hls::stream<ASTYPE> &values_feas3,
    hls::stream<ASTYPE> &values_feas4,
    int            *colIndices_fea1,
    int            *colIndices_fea2,
    int            *colIndices_fea3,
    int            *colIndices_fea4,
    hls::stream<ASTYPE> &columnIndex_feas1,
    hls::stream<ASTYPE> &columnIndex_feas2,
    hls::stream<ASTYPE> &columnIndex_feas3,
    hls::stream<ASTYPE> &columnIndex_feas4,
    int             nnz_fea1,
    int             nnz_fea2,
    int             nnz_fea3,
    int             nnz_fea4,
    int            *rowPtr_fea1,
    int            *rowPtr_fea2,
    int            *rowPtr_fea3,
    int            *rowPtr_fea4,
    hls::stream<ASTYPE> &rowPtr_feas1,
    hls::stream<ASTYPE> &rowPtr_feas2,
    hls::stream<ASTYPE> &rowPtr_feas3,
    hls::stream<ASTYPE> &rowPtr_feas4,
    INTYPE         *values_adj1,
    INTYPE         *values_adj2,
    INTYPE         *values_adj3,
    INTYPE         *values_adj4,
    int            *colIndices_adj1,
    int            *colIndices_adj2,
    int            *colIndices_adj3,
    int            *colIndices_adj4,
    int             nnz_adj1,
    int             nnz_adj2,
    int             nnz_adj3,
    int             nnz_adj4,
    int            *rowPtr_adj1,
    int            *rowPtr_adj2,
    int            *rowPtr_adj3,
    int            *rowPtr_adj4,
    int             N_adj,
    int             M_adj,
    int             M_fea,
    int             P_w
)
{
    /* ── Fill in defaults not exposed at this API level ── */
    int        array_c_adjust      = N_adj;
    int        quantized_multiplierl = 8;
    int        beta_qul             = 255;
    int        f_alignl             = 0;

    ap_uint<8> P_w_int[5];
    P_w_int[0] = P_w;   // only layer 0 is active at this level

    float srelu[5];
    srelu[0] = 0.0f;    // shift-ReLU disabled for layer 0

    /* ── Invoke the full top-level kernel ── */
    mmult_top(
        load_weights,
        beta_qu,           f_align,
        beta_qul,          f_alignl,
        quantization_scale_adj,
        quantization_scale_fea,
        quantization_scale_w,
        quantization_scale_lin,
        deq_factor,
        model,             srelu,
        scale_fea,         max_fea,
        layer_count,
        quantized_multiplier,
        quantized_multiplierl,
        shift,             bias,  bias_count,
        profiling,
        zero_point_lhs,    zero_point_rhs,
        zero_point_dst,    clamp_max,  clamp_min,
        N_adj, M_adj, M_fea, P_w_int,
        array_b, array_b2,
        array_d1, array_d2, array_d3, array_d4,
        stream_d1, stream_d1r, stream_d1c,
        stream_d2, stream_d3, stream_d4,
        array_e1,
        array_s1,
        ate_m,
        array_c_adjust,
        nnz_fea1, nnz_fea2, nnz_fea3, nnz_fea4,
        rowPtr_fea1,       rowPtr_fea2,       rowPtr_fea3,       rowPtr_fea4,
        colIndices_fea1,   colIndices_fea2,   colIndices_fea3,   colIndices_fea4,
        values_fea1,       values_fea2,       values_fea3,       values_fea4,
        rowPtr_feas1,      rowPtr_feas2,      rowPtr_feas3,      rowPtr_feas4,
        columnIndex_feas1, columnIndex_feas2, columnIndex_feas3, columnIndex_feas4,
        values_feas1,      values_feas2,      values_feas3,      values_feas4,
        nnz_adj1, nnz_adj2, nnz_adj3, nnz_adj4,
        rowPtr_adj1,       rowPtr_adj2,       rowPtr_adj3,       rowPtr_adj4,
        colIndices_adj1,   colIndices_adj2,   colIndices_adj3,   colIndices_adj4,
        values_adj1,       values_adj2,       values_adj3,       values_adj4
    );

    /* ── Spot-check output values for simulation verification ── */
    std::cout << "Output spot-check:"           << std::endl;
    std::cout << "  array_d1[ 0] = " << array_d1[ 0] << std::endl;
    std::cout << "  array_d1[ 3] = " << array_d1[ 3] << std::endl;
    std::cout << "  array_d1[ 7] = " << array_d1[ 7] << std::endl;
    std::cout << "  array_d1[ 9] = " << array_d1[ 9] << std::endl;
    std::cout << "  array_d1[13] = " << array_d1[13] << std::endl;
    std::cout << "  array_d1[20] = " << array_d1[20] << std::endl;
    std::cout << "  array_d1[33] = " << array_d1[33] << std::endl;
}