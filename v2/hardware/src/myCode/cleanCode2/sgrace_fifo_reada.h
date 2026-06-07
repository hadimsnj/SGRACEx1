/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#ifndef __SGRACE_FIFO_READA_H__
#define __SGRACE_FIFO_READA_H__

#include "sgrace_common.h"

void check_fifo_0(
    int                 a_values,
    hls::stream<ITYPE> &A_fifo,
    hls::stream<ITYPE> &A_fifo_out
);

void check_fifo_2(
    int                 N,
    hls::stream<ITYPE> &C_fifo,
    hls::stream<ITYPE> &C_fifo_out
);

void check_fifo_1(
    int                 N,
    int                 B_index,
    int                 B_index_loop,
    int                 tail,
    hls::stream<ITYPE> &C_fifo,
    hls::stream<ITYPE> &C_fifo_out
);

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
);

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
);

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
);

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
);

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
);

#endif  /* __SGRACE_FIFO_READA_H__ */
