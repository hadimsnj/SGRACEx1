/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include "sgrace_readers.h"


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


