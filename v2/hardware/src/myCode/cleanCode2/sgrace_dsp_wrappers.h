/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#ifndef __SGRACE_DSP_WRAPPERS_H__
#define __SGRACE_DSP_WRAPPERS_H__

#include "sgrace_common.h"
#include "sgrace_dsp_kernels.h"

void dsp_kernel_wrapper_adj_4(
    int                 block_size,
    int                 M,
    hls::stream<ITYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    QTYPE               b_block1[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE               b_block2[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE               b_block3[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE               b_block4[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    ITYPE               acc2[B_WIDTH_BLOCK]
);

void dsp_kernel_wrapper_adj_2(
    int                 block_size,
    int                 M[SPMM_BLOCK],
    hls::stream<ITYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    QTYPE               b_block1[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE               b_block2[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    ITYPE               acc2[B_WIDTH_BLOCK][SPMM_BLOCK]
);

void dsp_kernel_wrapper_adj_1(
    int                 block_size,
    int                 M,
    hls::stream<TTYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    QTYPE               b_block1[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    ITYPE               acc2[B_WIDTH_BLOCK]
);

void dsp_kernel_wrapper_fea(
    bool                gemm_mode,
    int                 M[SPMM_BLOCK],
    hls::stream<FTYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    BTYPE               b_block[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    ITYPE               acc2[B_WIDTH_BLOCK][SPMM_BLOCK]
);

void dsp_kernel_wrapper_lin(
    bool                gemm_mode,
    int                 M,
    hls::stream<LTYPE> &A_fifo,
    hls::stream<int>   &col_indices_fifo,
    BLTYPE              b_block[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<8>           zero_point_lhs,
    ap_int<8>           zero_point_rhs,
    ITYPE               acc2[B_WIDTH_BLOCK]
);

void scale(
    ap_int<32>         *quantized_multiplier,
    ap_int<32>         *shift,
    ap_int<32>         *bias,
    ap_int<8>           zero_point_dst,
    ap_int<8>           clamp_max,
    ap_int<8>           clamp_min,
    int                 N,
    int                 M,
    int                 P,
    hls::stream<ITYPE>  C_fifo[B_WIDTH_BLOCK],
    int                 B_index,
    int                 B_index_loop,
    int                 tail,
    hls::stream<ITYPE>  write_fifo[B_WIDTH_BLOCK]
);

void mxv(
    int    M,
    int    P_w,
    QTYPE  C_mxv[B_HEIGHT][B_WIDTH_BLOCK],
    BTYPE *A,
    TTYPE *WH1,
    TTYPE *WH2
);

void mxv1(
    int    M,
    int    P_w,
    QTYPE  C_mxv[B_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK],
    BTYPE  ate_m_int[B_WIDTH_BLOCK * 2],
    TTYPE *WH1,
    TTYPE *WH21,
    TTYPE *WH22,
    TTYPE *WH23,
    TTYPE *WH24
);

void mxv4(
    int    M1,
    int    M2,
    int    M3,
    int    M4,
    int    P_w,
    QTYPE  C_mxv1[B_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK],
    QTYPE  C_mxv2[B_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK],
    QTYPE  C_mxv3[B_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK],
    QTYPE  C_mxv4[B_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK],
    BTYPE *A,
    TTYPE *WH11,  TTYPE *WH211, TTYPE *WH212, TTYPE *WH213, TTYPE *WH214,
    TTYPE *WH12,  TTYPE *WH221, TTYPE *WH222, TTYPE *WH223, TTYPE *WH224,
    TTYPE *WH13,  TTYPE *WH231, TTYPE *WH232, TTYPE *WH233, TTYPE *WH234,
    TTYPE *WH14,  TTYPE *WH241, TTYPE *WH242, TTYPE *WH243, TTYPE *WH244
);

#endif  /* __SGRACE_DSP_WRAPPERS_H__ */
