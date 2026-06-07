/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include "sgrace_compute.h"


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


