/*===============================================================================
 * This file is part of the SGRACE GNN accelerator.
 * Written at Linkoping/UPM University.
 *
 * Author: Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 *===============================================================================
 */

#ifndef __MATRIX_MULT_H__
#define __MATRIX_MULT_H__

#include <vector>

/*-------------------------------------------------------------------------------
 * Design notes
 *-------------------------------------------------------------------------------
 * - The accelerator write-C loop ignores tail elements.
 * - Therefore, the weight matrix width must be an exact multiple of the number of
 *   processing cores.
 *   Example: P_w = 64 with 16 cores gives 4 exact tiles.
 *
 * - USE_SBLOCKS controls whether the write unit reads from multiple FIFO channels
 *   using sparse blocks, or from one FIFO channel per core.
 * - Disabling sparse blocks in the write unit can improve memory write efficiency
 *   when the write loop is the bottleneck.
 *
 * CSR generation note:
 * - Set DTYPE_LENGTH to 8 when SPMM packs the A matrix into 8-bit words.
 * - Set DTYPE_LENGTH to 32 when A is stored as 32-bit words.
 *-------------------------------------------------------------------------------
 */

/*-------------------------------------------------------------------------------
 * Matrix limits
 *-------------------------------------------------------------------------------
 */
#define MAX_N 8192
#define MAX_M 2048
#define MAX_P 64

#define A_HEIGHT MAX_N
#define F_HEIGHT MAX_N
#define B_HEIGHT MAX_M
#define B_WIDTH  MAX_P
#define C_WIDTH  MAX_P

/*-------------------------------------------------------------------------------
 * Numeric configuration
 *-------------------------------------------------------------------------------
 */
#define FIXEDPOINT
#define qbits  8
#define qbitsl 8

/*-------------------------------------------------------------------------------
 * Buffering and FIFO configuration
 *-------------------------------------------------------------------------------
 */
#define MAX_FIFO    16
#define PIPO_BLOCKS 2  // Use 1 block to save memory; 2 is the standard ping-pong setup.

/*-------------------------------------------------------------------------------
 * Execution mode configuration
 *-------------------------------------------------------------------------------
 */
#define SIGNED_MODE 0  // Use signed inputs, mainly for training configurations.
#define COO_MODE    1

#define INT_QUANT   1  // DDR interface uses float; quantization is done internally.
#define INT_QUANT_A 1  // Hardware quantization for adjacency data.
#define INT_QUANT_F 1  // Hardware quantization for feature data.
#define INT_QUANT_W 1  // Hardware quantization for weight data.
#define INT_DEQUANT 1  // Hardware dequantization/scaling removal at output.

#define FEA_THREADS 1
#define ADJ_THREADS 1

/*-------------------------------------------------------------------------------
 * GNN operator configuration
 *-------------------------------------------------------------------------------
 */
#define GAT_ENABLE    0  // Enable GAT support.
#define LINEAR_ENABLE 1  // Enable LINEAR support.
#define GATV2         0  // 0: standard GAT, 1: GATv2 modification.

/*-------------------------------------------------------------------------------
 * Sparse and attention configuration
 *-------------------------------------------------------------------------------
 * SPMM_BLOCK controls sparse-row fusion:
 * - 1: no row fusion
 * - 4: process four sparse rows together
 *
 * OPT_ATTN estimates adjacency sparsity per row:
 * - 1: worst case / dense adjacency
 * - 8: approximately 12.5% non-zeros per row
 *-------------------------------------------------------------------------------
 */
#define SPMM_BLOCK   1
#define ATEN_BLOCK   1  // Keep at 1; larger values can create dataflow issues.
#define OPT_ATTN     1
#define B_WIDTH_BLOCK MAX_P  // Attention does not support B-width blocking.

/*-------------------------------------------------------------------------------
 * Active type configuration
 *-------------------------------------------------------------------------------
 */
#if (SIGNED_MODE == 1)

    typedef ap_axis<32, 0, 0, 0> ASTYPE;  // AXI stream type.

    #if (INT_QUANT == 1)
        typedef float INTYPE;             // DDR interface type for activations/features.
        typedef float INTYPES;            // DDR interface type for weights.
        typedef float OUTTYPE;            // DDR output interface type.
    #else
        typedef ap_fixed<8, 1> INTYPE;
        typedef ap_fixed<8, 1> INTYPES;
        typedef ap_fixed<32, 16> OUTTYPE;
    #endif

    typedef ap_fixed<8, 1>  ATYPE;        // Adjacency/internal activation type.
    typedef ap_fixed<8, 1>  BTYPE;        // Weight type.
    typedef ap_fixed<8, 1>  FTYPE;        // Feature type.
    typedef ap_fixed<32, 16> DTYPE;       // Accumulation/output compute type.
    typedef ap_fixed<32, 16> ITYPE;       // Intermediate accumulation type.
    typedef ap_fixed<32, 16> TTYPE;       // Attention compute type.

    typedef ap_fixed<8, 1, AP_TRN_ZERO, AP_SAT> QTYPE;
    typedef ap_fixed<1, 1, AP_TRN_ZERO, AP_SAT> QTYPE1;
    typedef ap_fixed<2, 1, AP_TRN_ZERO, AP_SAT> QTYPE2;
    typedef ap_fixed<4, 1, AP_TRN_ZERO, AP_SAT> QTYPE4;
    typedef ap_fixed<8, 1, AP_TRN_ZERO, AP_SAT> QTYPE8;

    typedef ap_int<8> STYPE;              // Scaling-factor type.

    #define zero_point 0.0

#else

    #if (qbits == 8)

        union fp_int {
            int i;
            float f;
        };

        typedef ap_axis<32, 0, 0, 0> ASTYPE;  // AXI stream type.

        #if (INT_QUANT == 1)
            typedef float INTYPE;             // DDR interface type for activations/features.
            typedef float INTYPES;            // DDR interface type for weights.
            typedef float OUTTYPE;            // DDR output interface type.
        #else
            typedef ap_ufixed<8, 1> INTYPE;
            typedef ap_fixed<8, 1>  INTYPES;
            typedef ap_fixed<32, 16> OUTTYPE;
        #endif

        typedef ap_ufixed<8, 1> ATYPE;        // Unsigned adjacency/internal activation type.
        typedef ap_fixed<8, 1>  BTYPE;        // Weight type.
        typedef ap_fixed<8, 1>  BLTYPE;       // Linear weight type.
        typedef ap_ufixed<8, 1> FTYPE;        // Feature type.
        typedef ap_fixed<8, 1>  LTYPE;        // Linear operator type.
        typedef ap_fixed<32, 16> DTYPE;       // Accumulation/output compute type.
        typedef ap_fixed<32, 16> ITYPE;       // Intermediate accumulation type.
        typedef ap_fixed<32, 16> TTYPE;       // Attention compute type.

        typedef ap_fixed<8, 1, AP_TRN_ZERO, AP_SAT> QTYPE;
        typedef ap_fixed<1, 1, AP_TRN_ZERO, AP_SAT> QTYPE1;
        typedef ap_fixed<2, 1, AP_TRN_ZERO, AP_SAT> QTYPE2;
        typedef ap_fixed<4, 1, AP_TRN_ZERO, AP_SAT> QTYPE4;
        typedef ap_fixed<8, 1, AP_TRN_ZERO, AP_SAT> QTYPE8;
        typedef ap_fixed<8, 1, AP_TRN_ZERO, AP_SAT> QLTYPE;

        typedef ap_int<8> STYPE;              // Scaling-factor type.

        #define zero_point 0.0

    #endif

    #if (qbits == 1)

        union fp_int {
            int i;
            float f;
        };

        typedef ap_axis<32, 0, 0, 0> ASTYPE;  // AXI stream type.

        #if (INT_QUANT == 1)
            typedef float INTYPE;             // DDR interface type for activations/features.
            typedef float INTYPES;            // DDR interface type for weights.
            typedef float OUTTYPE;            // DDR output interface type.
        #else
            typedef ap_ufixed<8, 1> INTYPE;
            typedef ap_fixed<8, 1>  INTYPES;
            typedef ap_fixed<32, 16> OUTTYPE;
        #endif

        typedef ap_ufixed<2, 1> ATYPE;        // Unsigned adjacency/internal activation type.
        typedef ap_fixed<1, 1>  BTYPE;        // 1-bit weight type.
        typedef ap_ufixed<2, 1> FTYPE;        // Feature type.
        typedef ap_fixed<qbitsl, 1> BLTYPE;   // Linear weight type.
        typedef ap_fixed<qbitsl, 1> LTYPE;    // Linear operator type.
        typedef ap_fixed<32, 16> DTYPE;       // Accumulation/output compute type.
        typedef ap_fixed<18, 16> ITYPE;       // Intermediate accumulation type.
        typedef ap_fixed<18, 16> TTYPE;       // Attention compute type.

        typedef ap_fixed<1, 1, AP_TRN_ZERO, AP_SAT> QTYPE;
        typedef ap_fixed<1, 1, AP_TRN_ZERO, AP_SAT> QTYPE1;
        typedef ap_fixed<2, 1, AP_TRN_ZERO, AP_SAT> QTYPE2;
        typedef ap_fixed<4, 1, AP_TRN_ZERO, AP_SAT> QTYPE4;
        typedef ap_fixed<8, 1, AP_TRN_ZERO, AP_SAT> QTYPE8;
        typedef ap_fixed<qbitsl, 1, AP_TRN_ZERO, AP_SAT> QLTYPE;

        typedef ap_int<8> STYPE;              // Scaling-factor type.

        #define zero_point 0.0

    #endif

    #define FTYPE_LATENCY_ADJ 1
    #define FTYPE_LATENCY_FEA 1

#endif

#ifdef HALF

    typedef half ATYPE;
    typedef half BTYPE;
    typedef half DTYPE;
    typedef half FTYPE;
    typedef half ITYPE;
    typedef half STYPE;
    typedef half QTYPE;

    #define FTYPE_LATENCY_ADJ 4
    #define FTYPE_LATENCY_FEA 4
    #define frac_bits 0

#endif

#ifdef FLOAT

    typedef float ATYPE;
    typedef float BTYPE;
    typedef float DTYPE;
    typedef float FTYPE;
    typedef float ITYPE;

    #define FTYPE_LATENCY_ADJ 6
    #define FTYPE_LATENCY_FEA 6

#endif

/*-------------------------------------------------------------------------------
 * Dataflow and blocking configuration
 *-------------------------------------------------------------------------------
 * The compute unit always uses sparse blocks. USE_SBLOCKS only controls whether
 * the write unit also uses sparse blocks and reads multiple FIFO channels.
 *
 * A_HEIGHT_BLOCK is used for software-side data partitioning due to limitations
 * in Xilinx kernel sds_alloc. It should normally be A_HEIGHT divided by the
 * number of software-managed blocks.
 *-------------------------------------------------------------------------------
 */
#define USE_SBLOCKS    0
#define A_HEIGHT_BLOCK 1
#define C_HEIGHT_BLOCK A_HEIGHT_BLOCK

#define FAST_ATTENTION 1
#define B_BLOCK_PARALLEL 1

/*-------------------------------------------------------------------------------
 * Kernel feature enables
 *-------------------------------------------------------------------------------
 */
#define ENABLE_GEMM
#define ENABLE_SPMM

/*-------------------------------------------------------------------------------
 * Common aliases
 *-------------------------------------------------------------------------------
 */
typedef std::vector<int> vi;

#endif  // __MATRIX_MULT_H__