/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#ifndef __SGRACE_READERS_H__
#define __SGRACE_READERS_H__

#include "sgrace_common.h"
#include "sgrace_quant.h"

void readptr_csr_fea(
    bool             gemm_mode,
    int              N,
    int              M,
    int             *rowPtr,
    hls::stream<int> &rnnz_fifo
);

void read_ptr2(
    int              nnz_fea,
    int             *rowPtr,
    hls::stream<int> &index_fifo
);

void read_ptr(
    bool             stream_mode,
    int              nnz_fea,
    int             *rowPtr,
    hls::stream<int> &index_fifo
);

void proc_ptr(
    int              nnz_fea,
    hls::stream<int> &index_fifo,
    hls::stream<int> &rnnz_fifo
);

void proc_ptr2(
    bool                 gcn_path,
    bool                 linear_mode,
    bool                 stream_mode,
    int                  nnz_fea,
    hls::stream<int>    &index_fifo,
    hls::stream<ASTYPE> &rowPtrs,
    hls::stream<int>    &rnnz_fifo,
    hls::stream<int>    &rnnz_fifo_sage
);

void read_dataflow2(
    bool                 gcn_path,
    bool                 linear_mode,
    bool                 stream_mode,
    int                  nnz_fea,
    int                 *rowPtr,
    hls::stream<ASTYPE> &rowPtrs,
    hls::stream<int>    &rnnz_fifo,
    hls::stream<int>    &rnnz_fifo_sage
);

void read_dataflow(
    int              nnz_fea,
    int             *rowPtr,
    hls::stream<int> &rnnz_fifo
);

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
);

void readptr_csr_adj(
    bool             gemm_mode,
    int              N,
    int              M,
    int             *rowPtr,
    hls::stream<int> &rnnz_fifo
);

void readptr_coo_adj(
    int              nnz_adj,
    bool             sage_mode,
    bool             linear_mode,
    bool             gemm_mode,
    int              N,
    int              M,
    int             *rowPtr,
    hls::stream<int> &rnnz_fifo
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

#endif  /* __SGRACE_READERS_H__ */
