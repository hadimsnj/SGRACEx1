/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include "sgrace_loops_top.h"


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


