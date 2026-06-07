/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================
 *
 * Design notes:
 *
 *   - The writec loop ignores possible column tails.  The weight matrix width
 *     (P_w) must therefore be an exact multiple of the number of cores.
 *     Example: P_w = 64, 16 cores → 4 exact tiles.
 *
 *   - USE_SBLOCKS controls whether the write unit reads from multiple FIFO
 *     channels (USE_SBLOCKS = 1) or a single channel per core (0).
 *     Single-channel mode is generally better because it produces a tighter
 *     memory-write loop.
 *
 *   - SPMM_BLOCK controls multi-row fusion in the inner compute loop.
 *     SPMM_BLOCK = 1 means one row at a time (no fusion).
 *     SPMM_BLOCK = 4 fuses 4 rows per iteration, reducing loop-restart
 *     overhead when rows have few non-zeros.
 *===============================================================================*/

#ifndef __MATRIX_MULT_H__
#define __MATRIX_MULT_H__

#include <vector>


/* ═══════════════════════════════════════════════════════════════════════════
 * Dimension limits.
 * MAX_N / MAX_M / MAX_P bound the maximum supported matrix sizes.
 * These determine BRAM allocation for tiles and FIFOs.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MAX_N    8192    // maximum number of graph nodes (rows of A / adj)
#define MAX_M    2048    // maximum input feature width (cols of A / rows of B)
#define MAX_P    64      // maximum output feature width (cols of B)


/* ═══════════════════════════════════════════════════════════════════════════
 * Arithmetic precision mode.
 * Exactly one of FIXEDPOINT, HALF, or FLOAT must be defined.
 * qbits / qbitsl select the bit-width for GNN / linear-branch activations.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define FIXEDPOINT        // use HLS ap_fixed arithmetic (default)
#define qbits  8          // GNN activation bit-width  (1 = binary, 8 = INT8)
#define qbitsl 8          // linear-branch activation bit-width


/* ═══════════════════════════════════════════════════════════════════════════
 * Pipeline and buffering parameters.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MAX_FIFO     16   // standard inter-task FIFO depth
#define PIPO_BLOCKS  2    // stream-of-blocks depth; 1 = single-buffer, 2 = ping-pong


/* ═══════════════════════════════════════════════════════════════════════════
 * Matrix tile dimensions derived from the limits above.
 * A_HEIGHT / F_HEIGHT size the feature and attention buffers.
 * B_HEIGHT / B_WIDTH size the weight tile.
 * C_WIDTH / B_WIDTH_BLOCK set the output column-block width.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define A_HEIGHT      MAX_N    // attention buffer height (= number of nodes)
#define F_HEIGHT      MAX_N    // feature buffer height
#define B_HEIGHT      MAX_M    // weight tile height (= input feature width)
#define B_WIDTH       MAX_P    // weight tile width
#define C_WIDTH       MAX_P    // output tile width
#define B_WIDTH_BLOCK MAX_P    // output column-block width (must equal C_WIDTH for attention)


/* ═══════════════════════════════════════════════════════════════════════════
 * Quantization and sparsity configuration.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define SIGNED_MODE    0   // 0 = unsigned activations; 1 = signed (training config)
#define COO_MODE       1   // sparse format: 1 = COO row-index arrays, 0 = CSR

/* Internal hardware quantization enables.
 * Setting INT_QUANT_* to 1 moves the quantization step into hardware,
 * allowing the DDR interface to use float while the compute path uses
 * fixed-point arithmetic.  Disabling (0) assumes software pre-quantization. */
#define INT_QUANT    1     // global enable (used by readval_coo_adj2)
#define INT_QUANT_A  1     // adjacency value quantization
#define INT_QUANT_F  1     // feature value quantization
#define INT_QUANT_W  1     // weight quantization

/* Output dequantization: removes internal scale factor from the output. */
#define INT_DEQUANT  1


/* ═══════════════════════════════════════════════════════════════════════════
 * Parallelism configuration.
 * FEA_THREADS  – number of row partitions of the feature matrix.
 * ADJ_THREADS  – number of row partitions of the adjacency matrix.
 * SPMM_BLOCK   – number of rows fused in one compute-loop iteration.
 * ATEN_BLOCK   – attention softmax tile size (keep at 1 to avoid DATAFLOW issues).
 * OPT_ATTN     – adjacency sparsity assumption for attention FIFO sizing:
 *                1 = worst-case (fully dense), 8 = ~12.5% non-zeros.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define FEA_THREADS  1
#define ADJ_THREADS  1
#define SPMM_BLOCK   1     // 1 = no row fusion; 4 = 4-row fusion
#define ATEN_BLOCK   1
#define OPT_ATTN     1


/* ═══════════════════════════════════════════════════════════════════════════
 * Optional hardware features.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define GAT_ENABLE     0   // 1 = enable Graph Attention Network (GAT) support
#define LINEAR_ENABLE  1   // 1 = enable parallel linear-projection branch
#define GATV2          0   // 1 = use GATv2 formulation; 0 = standard GAT
#define FAST_ATTENTION 1   // 1 = use the fused attention kernel


/* ═══════════════════════════════════════════════════════════════════════════
 * Miscellaneous constants.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define USE_SBLOCKS      0   // 0 = single write-FIFO channel per core (recommended)
#define FAST_ATTENTION   1
#define A_HEIGHT_BLOCK   1   // software partition block size for sds_alloc alignment
#define C_HEIGHT_BLOCK   A_HEIGHT_BLOCK
#define B_BLOCK_PARALLEL 1
#define ENABLE_GEMM
#define ENABLE_SPMM


/* ═══════════════════════════════════════════════════════════════════════════
 * Arithmetic type definitions.
 *
 * All three precision modes (FIXEDPOINT, HALF, FLOAT) define the same set
 * of named types so the rest of the source is mode-agnostic.
 *
 * Key types:
 *   INTYPE   – DDR interface for sparse matrix values (float when INT_QUANT).
 *   INTYPES  – DDR interface for weight matrix values (float when INT_QUANT).
 *   ATYPE    – adjacency activation type (after quantization).
 *   BTYPE    – GNN weight type.
 *   BLTYPE   – linear-branch weight type.
 *   FTYPE    – feature activation type (GNN branch).
 *   LTYPE    – feature activation type (linear branch).
 *   DTYPE    – intermediate accumulation type passed between stages.
 *   ITYPE    – main accumulator type (wide enough to avoid overflow).
 *   TTYPE    – attention score computation type.
 *   QTYPE    – quantized output type (full precision, AP_SAT).
 *   QTYPE1/2/4/8 – quantized output types at 1/2/4/8-bit precisions.
 *   QLTYPE   – quantized output type for the linear branch.
 *   OUTTYPE  – final DDR output type (float when INT_QUANT).
 *   STYPE    – per-layer output right-shift type.
 *   ASTYPE   – AXI-stream data type (32-bit word + sideband signals).
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef FIXEDPOINT

    /* float↔int reinterpret union used by the AXI-stream read/write paths */
    union fp_int { int i; float f; };

    /* AXI4-Stream type: 32-bit data word with last/keep/strb sidebands */
    typedef ap_axis<32, 0, 0, 0> ASTYPE;

    /* DDR interface types */
#if (INT_QUANT == 1)
    typedef float   INTYPE;    // sparse matrix values arrive as float from DDR
    typedef float   INTYPES;   // weight matrix values arrive as float from DDR
    typedef float   OUTTYPE;   // output written as float to DDR
#else
    typedef ap_ufixed<8, 1> INTYPE;
    typedef ap_fixed<8,  1> INTYPES;
    typedef ap_fixed<32, 16> OUTTYPE;
#endif

#if (qbits == 8)
    /* 8-bit mode */
    typedef ap_ufixed<8, 1>  ATYPE;     // adjacency activation (unsigned)
    typedef ap_fixed<8,  1>  BTYPE;     // GNN weight
    typedef ap_fixed<8,  1>  BLTYPE;    // linear-branch weight
    typedef ap_ufixed<8, 1>  FTYPE;     // GNN feature activation (unsigned)
    typedef ap_fixed<8,  1>  LTYPE;     // linear-branch feature activation
    typedef ap_fixed<32, 16> DTYPE;     // inter-stage data word
    typedef ap_fixed<32, 16> ITYPE;     // accumulator
    typedef ap_fixed<32, 16> TTYPE;     // attention score computation
    typedef ap_fixed<8, 1, AP_TRN_ZERO, AP_SAT>  QTYPE;
    typedef ap_fixed<1, 1, AP_TRN_ZERO, AP_SAT>  QTYPE1;
    typedef ap_fixed<2, 1, AP_TRN_ZERO, AP_SAT>  QTYPE2;
    typedef ap_fixed<4, 1, AP_TRN_ZERO, AP_SAT>  QTYPE4;
    typedef ap_fixed<8, 1, AP_TRN_ZERO, AP_SAT>  QTYPE8;
    typedef ap_fixed<8, 1, AP_TRN_ZERO, AP_SAT>  QLTYPE;
    typedef ap_int<8> STYPE;            // per-layer output right-shift
    #define zero_point 0.0
    #define FTYPE_LATENCY_ADJ 1
    #define FTYPE_LATENCY_FEA 1
#endif

#if (qbits == 1)
    /* 1-bit (binary) mode */
    typedef ap_ufixed<2, 1>  ATYPE;
    typedef ap_fixed<1,  1>  BTYPE;
    typedef ap_fixed<qbitsl, 1> BLTYPE;
    typedef ap_ufixed<2, 1>  FTYPE;
    typedef ap_fixed<qbitsl, 1> LTYPE;
    typedef ap_fixed<32, 16> DTYPE;
    typedef ap_fixed<18, 16> ITYPE;     // narrower accumulator sufficient for binary
    typedef ap_fixed<18, 16> TTYPE;
    typedef ap_fixed<1, 1, AP_TRN_ZERO, AP_SAT>  QTYPE;
    typedef ap_fixed<1, 1, AP_TRN_ZERO, AP_SAT>  QTYPE1;
    typedef ap_fixed<2, 1, AP_TRN_ZERO, AP_SAT>  QTYPE2;
    typedef ap_fixed<4, 1, AP_TRN_ZERO, AP_SAT>  QTYPE4;
    typedef ap_fixed<8, 1, AP_TRN_ZERO, AP_SAT>  QTYPE8;
    typedef ap_fixed<qbitsl, 1, AP_TRN_ZERO, AP_SAT> QLTYPE;
    typedef ap_int<8> STYPE;
    #define zero_point 0.0
    #define FTYPE_LATENCY_ADJ 1
    #define FTYPE_LATENCY_FEA 1
#endif

#endif  /* FIXEDPOINT */


#ifdef HALF
    /* 16-bit IEEE half-precision floating-point mode */
    typedef half    ATYPE;
    typedef half    BTYPE;
    typedef half    DTYPE;
    typedef half    FTYPE;
    typedef half    ITYPE;
    typedef half    QTYPE;
    typedef half    STYPE;
    #define FTYPE_LATENCY_ADJ 4
    #define FTYPE_LATENCY_FEA 4
#endif


#ifdef FLOAT
    /* 32-bit IEEE single-precision floating-point mode */
    typedef float   ATYPE;
    typedef float   BTYPE;
    typedef float   DTYPE;
    typedef float   FTYPE;
    typedef float   ITYPE;
    #define FTYPE_LATENCY_ADJ 6
    #define FTYPE_LATENCY_FEA 6
#endif


/* Convenience alias */
typedef std::vector<int> vi;

#endif  /* __MATRIX_MULT_H__ */