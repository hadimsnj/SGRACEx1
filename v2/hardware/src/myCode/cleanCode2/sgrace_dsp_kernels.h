/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#ifndef __SGRACE_DSP_KERNELS_H__
#define __SGRACE_DSP_KERNELS_H__

#include "sgrace_common.h"

void dsp_kernel_float_adj_1(
    ATYPE      a_value,
    BTYPE      b_block[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
);

void dsp_kernel_float_adj_2(
    int        block_size,
    ATYPE      a_value,
    BTYPE      b_block1[B_HEIGHT][B_WIDTH_BLOCK],
    BTYPE      b_block2[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
);

void dsp_kernel_float_adj_4(
    int        block_size,
    ATYPE      a_value,
    BTYPE      b_block1[B_HEIGHT][B_WIDTH_BLOCK],
    BTYPE      b_block2[B_HEIGHT][B_WIDTH_BLOCK],
    BTYPE      b_block3[B_HEIGHT][B_WIDTH_BLOCK],
    BTYPE      b_block4[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
);

void dsp_kernel_float_fea(
    ATYPE      a_value,
    BTYPE      b_block[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
);

void dsp_kernel_int_adj_1(
    int       block_size,
    TTYPE     a_value,
    QTYPE     b_block1[B_HEIGHT][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
);

void dsp_kernel_int_adj_2(
    int       block_size,
    ITYPE     a_value,
    QTYPE     b_block1[B_HEIGHT / 2][B_WIDTH_BLOCK],
    QTYPE     b_block2[B_HEIGHT / 2][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
);

void dsp_kernel_int_adj_4(
    int       block_size,
    TTYPE     a_value,
    QTYPE     b_block1[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE     b_block2[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE     b_block3[B_HEIGHT / 4][B_WIDTH_BLOCK],
    QTYPE     b_block4[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
);

void dsp_kernel_int_fea(
    FTYPE      a_value,
    BTYPE      b_block[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
);

void dsp_kernel_int_lin(
    LTYPE      a_value,
    BLTYPE     b_block[B_HEIGHT / 4][B_WIDTH_BLOCK],
    ap_int<32> b_row,
    ap_int<8>  zero_point_lhs,
    ap_int<8>  zero_point_rhs,
    ITYPE      acc[B_WIDTH_BLOCK]
);

#endif  /* __SGRACE_DSP_KERNELS_H__ */
