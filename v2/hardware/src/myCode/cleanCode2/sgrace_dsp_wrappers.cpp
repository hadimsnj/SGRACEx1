/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include "sgrace_dsp_wrappers.h"


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


