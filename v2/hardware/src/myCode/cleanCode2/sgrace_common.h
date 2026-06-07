/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#ifndef __SGRACE_COMMON_H__
#define __SGRACE_COMMON_H__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <hls_math.h>
#include <string>
#include <fstream>
#include <sstream>

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"
#include "matrix_mult.h"
#include "hls_streamofblocks.h"


/* ────────────────────────────────────────────────────────────────────────────
 * Stream-of-blocks tile types.
 * ──────────────────────────────────────────────────────────────────────────── */
typedef QTYPE  buf [F_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK];
typedef QLTYPE bufl[F_HEIGHT / FEA_THREADS][B_WIDTH_BLOCK];

typedef unsigned long u32;

/* ────────────────────────────────────────────────────────────────────────────
 * Compile-time constants derived from the project parameters in matrix_mult.h.
 * ──────────────────────────────────────────────────────────────────────────── */
const int BLOCK        = B_WIDTH_BLOCK;
const int SBLOCK       = SPMM_BLOCK;
const int SBLOCK_LIN   = 1;
const int PARALLEL_ROW = B_BLOCK_PARALLEL;
const int FIFO_DEPTH   = MAX_FIFO;
const int LINEAR_DEPTH = B_WIDTH_BLOCK * B_HEIGHT;
const int FIFO_DEPTH_ATTN  = A_HEIGHT / OPT_ATTN;
const int FIFO_DEPTH_ATTN2 = A_HEIGHT * ATEN_BLOCK / OPT_ATTN;
const int FADD_LATENCY_ADJ = FTYPE_LATENCY_ADJ;
const int FADD_LATENCY_FEA = FTYPE_LATENCY_FEA;

/* ────────────────────────────────────────────────────────────────────────────
 * FIFO telemetry counters (shared across translation units).
 * Defined once in sgrace_common.cpp; declared extern here.
 * ──────────────────────────────────────────────────────────────────────────── */
extern ap_int<64> fifo_full_0,  fifo_full_1,  fifo_full_2;
extern ap_int<64> fifo_empty_0, fifo_empty_1, fifo_empty_2;
extern ap_int<64> fifo_read_0,  fifo_read_1,  fifo_read_2;
extern ap_int<64> fifo_write_0, fifo_write_1, fifo_write_2;
extern ap_int<64> fifo_cycle_0, fifo_cycle_1, fifo_cycle_2;

#ifdef simulation
extern float max_adj,   min_adj;
extern float max_fea,   min_fea;
extern float acc2_fea_min, acc2_fea_max;
extern float acc2_adj_min, acc2_adj_max;
#endif

#endif  /* __SGRACE_COMMON_H__ */
