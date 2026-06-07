/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#ifndef __SGRACE_ATTENTION_H__
#define __SGRACE_ATTENTION_H__

#include "sgrace_common.h"
#include "sgrace_quant.h"
#include "sgrace_dsp_wrappers.h"
#include "sgrace_compute.h"

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
);

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
);

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
);

void func_rnnz(
    int                 i,
    int                 N_adj,
    hls::stream<ATYPE> &max_fifo,
    hls::stream<int>    rnnz_fifo[SPMM_BLOCK],
    hls::stream<int>    rnnz_f[ATEN_BLOCK],
    hls::stream<ATYPE>  val_f[ATEN_BLOCK]
);

void func_exp(
    hls::stream<int>    rnnz_f[ATEN_BLOCK],
    hls::stream<ATYPE>  val_f[ATEN_BLOCK],
    hls::stream<FTYPE> &E_fifo,
    hls::stream<ATYPE>  sum_f[ATEN_BLOCK],
    hls::stream<ATYPE>  val_f2[ATEN_BLOCK],
    hls::stream<int>    rnnz_f2[ATEN_BLOCK],
    hls::stream<ATYPE> &support_f
);

void func_fixed(
    int                N_adj,
    hls::stream<ATYPE> sum_f[ATEN_BLOCK],
    hls::stream<ATYPE> val_f2[ATEN_BLOCK],
    hls::stream<int>   rnnz_f2[ATEN_BLOCK],
    hls::stream<ATYPE> sum_f2[ATEN_BLOCK],
    hls::stream<int>   rnnz_f3[ATEN_BLOCK]
);

void func_div(
    hls::stream<int>   rnnz_att_fifo[SPMM_BLOCK],  // output: per-row nnz counts
    hls::stream<ATYPE> &A_fifo,                     // input:  adjacency values (pass-through)
    hls::stream<ATYPE> &support_f,                  // input:  exp(e_ij - max_i) values
    hls::stream<int>   &col_indices_fifo,            // input:  column indices
    hls::stream<ATYPE> sum_f2[ATEN_BLOCK],          // input:  per-row softmax denominators
    hls::stream<int>   rnnz_f3[ATEN_BLOCK],         // input:  cumulative nnz per row slot
    hls::stream<ATYPE> &val_att_fifo,               // output: normalized attention weights
    hls::stream<int>   &col_att_fifo                // output: column indices
);

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
);

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
);

#endif  /* __SGRACE_ATTENTION_H__ */
