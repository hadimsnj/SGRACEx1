/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include "sgrace_fifo_reada.h"


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


