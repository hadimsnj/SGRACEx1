/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#ifndef __SGRACE_LOOPS_TOP_H__
#define __SGRACE_LOOPS_TOP_H__

#include "sgrace_common.h"
#include "sgrace_quant.h"
#include "sgrace_dsp_kernels.h"
#include "sgrace_dsp_wrappers.h"
#include "sgrace_readers.h"
#include "sgrace_fifo_reada.h"
#include "sgrace_writers.h"
#include "sgrace_compute.h"
#include "sgrace_attention.h"

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
);

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
);

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
);

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
);

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
);

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
);

#endif  /* __SGRACE_LOOPS_TOP_H__ */
