/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include "sgrace_attention.h"

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


