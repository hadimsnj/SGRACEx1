/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#ifndef __SGRACE_COMPUTE_H__
#define __SGRACE_COMPUTE_H__

#include "sgrace_common.h"
#include "sgrace_dsp_kernels.h"
#include "sgrace_dsp_wrappers.h"

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
);

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
);

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
);

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
);

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
);

#endif  /* __SGRACE_COMPUTE_H__ */
