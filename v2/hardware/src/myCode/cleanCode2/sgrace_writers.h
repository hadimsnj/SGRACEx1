/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#ifndef __SGRACE_WRITERS_H__
#define __SGRACE_WRITERS_H__

#include "sgrace_common.h"

void writec(
    float               deq_factor[5],
    ap_uint<1>          model[5][8],
    int                 first_row,
    int                 row_count,
    int                 N_adj,
    ap_uint<8>          P[5],
    hls::stream<ITYPE>  write_fifo[B_WIDTH_BLOCK],
    QLTYPE              linear_pipo[B_HEIGHT][B_WIDTH_BLOCK],
    hls::stream<OUTTYPE> &CS,
    int                 B_index,
    int                 layer_loop
);

void writeout(
    ap_uint<1>           model[5][8],
    int                  first_row,
    int                  row_count,
    int                  N_adj,
    ap_uint<8>           P[5],
    hls::stream<OUTTYPE> &write_fifo,
    OUTTYPE             *C,
    hls::stream<ASTYPE>  &CS,
    hls::stream<ASTYPE>  &CSR,
    hls::stream<ASTYPE>  &CSC,
    int                  B_index,
    int                  layer_loop
);

void writec_transpose(
    float               deq_factor,
    bool                stream_mode,
    int                 first_row,
    int                 row_count,
    int                 N_adj,
    int                 P,
    hls::stream<ITYPE>  write_fifo[B_WIDTH_BLOCK],
    OUTTYPE            *C,
    hls::stream<ASTYPE> &CS,
    int                 B_index
);

void writes(
    float               deq_factor[5],
    ap_uint<1>          model[5][8],
    int                 first_row,
    int                 row_count,
    int                 N_adj,
    ap_uint<8>          P[5],
    hls::stream<TTYPE>  &write_fifo,
    hls::stream<int>    &rnnz_fifo,
    OUTTYPE             *C,
    int                  B_index
);

void writesx4(
    float                deq_factor,
    bool                 gat_mode,
    int                  row_count1,
    int                  row_count2,
    int                  row_count3,
    int                  row_count4,
    hls::stream<TTYPE>  &write_fifo1,
    hls::stream<TTYPE>  &write_fifo2,
    hls::stream<TTYPE>  &write_fifo3,
    hls::stream<TTYPE>  &write_fifo4,
    hls::stream<int>    &rnnz_fifo1,
    hls::stream<int>    &rnnz_fifo2,
    hls::stream<int>    &rnnz_fifo3,
    hls::stream<int>    &rnnz_fifo4,
    OUTTYPE             *C,
    int                  B_index
);

#endif  /* __SGRACE_WRITERS_H__ */
