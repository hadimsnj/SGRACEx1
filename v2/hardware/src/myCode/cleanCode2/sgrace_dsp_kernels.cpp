/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include "sgrace_dsp_kernels.h"


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


