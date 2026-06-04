/*===============================================================================
* This file is part of the SGRACE GNN accelerator
* has been written at Linkoping/UPM University
* Author : Jose Nunez-Yanez
*Copyright (C) 2026 Jose Nunez-Yanez
*Licensed under the MIT license. See LICENSE file in the project root for details
===============================================================================
*/

#ifndef __MATRIX_MULT_H__
#define __MATRIX_MULT_H__

#include <vector>

//WARNING: this accerator uses a writec loop that ignores possible tails. This means
//that the width of the WEIGHT matrix has to be a multiple of the number of cores.
//For example 64 for P_w and then 16 cores so a total of 4 EXACT tiles are processed.
//use_sblocks is configurable to use SPMM_BLOCK or not. This optimizes the writec loop
//without SPMM_BLOCK so that the writing to memory is more efficient. This is important
//if the amount of work is so little that the bottleneck is the write loop

// 64 * 16(values in 32-bit word (ternary)) = 1024
//1536 and 16 in a word so 96 words of 32-bit
//1536 and 4 in a word so 384 words of 8-bit

//if you want to generate CSR then set this DTYPE_LENGTH to 8 since spmm uses packing to 8-bit for A matrix
// if you set it to 32 then A matrix will be formed with 32-bits words instead of 8-bit
//8-bit packs
//32-bit packs
//quad
//8-bit packs
//32-bit packs

//#define simulation

#define MAX_N    8192 //32768 //8192 // 4096 //16384 //8192 // 4096 //4096 //20480  //4096 //38000 //4096 //20480 //32768 //8192 //20480  //4096 //8192 //4096 //1024 //98304 //4096 //20480 // 4096 //256 // 6144 //20480//16384 //4096 //32768 //20480 //64
#define MAX_M    2048 //4096 //4096 //20480  //4096 //38000 ///4096 //20480 //32768 //8192 //20480  //4096 //8192 //4096 //1024 //98304 //4096 //20480 // 4096 // 256 // 6144 //20480//16384 //4096 //24576 //2048 //384 //1536 //384 //1536 // 384 //48 //768 //96 //384 //96 //384 //96 //384// 96
#define MAX_P    64 //2048 //512 //64//1//64//1

#define FIXEDPOINT
#define qbits 8
#define qbitsl 8

#define MAX_FIFO 16
#define PIPO_BLOCKS 2 // use a PIPO with only one block to save memory instead of standard 2

//48 for WL 64
//96 for WL 32
// 384 for WL 8
// 768 for WL 4

#define A_HEIGHT   MAX_N
//#define A_WIDTH    MAX_N

#define F_HEIGHT   MAX_N
//#define F_WIDTH    MAX_M

#define B_HEIGHT   MAX_M
#define B_WIDTH    MAX_P

//#define C_HEIGHT   MAX_N
#define C_WIDTH    MAX_P

//#define TWOBIT_TERN
//#define TWOBIT_SIX
//#define TWOBIT_OPT
//#define TWOBIT_POS
//#define TWOBIT_NAIVE
//#define TWOBIT
//#define FOURBIT_NAIVE
//#define FOURBIT
//#define EIGHTBIT
//#define FOURBIT_XIL
//#define ONEBIT
//#define HALF
//#define FLOAT
//#define TEST

//#define TRAINING_MODE 1 //hardware generates data for backpropagation (inference + training)
//(TRAINING_MODE SWITCH not in used, training mode write output to memory, inference mode write output to stream)
// training mode one layer at a time, inference mode pack all layers together.

#define SIGNED_MODE 0 //use when training configuration so all inputs are signed
#define COO_MODE 1
//#define BINARY_MODE 1 no in used //use for 1-bit precission mode
#define INT_QUANT 1 //internal quantization so interface to DDR set to float
#define INT_QUANT_A 1 //internal hardware quantization of adjacency (can be done in software if adjacency is fixed).
#define INT_QUANT_F 1 //internal hardware quantization of features (important since input/features is not constant).
#define INT_QUANT_W 1 //internal hardware quantization of weights (can be done in software if weights are fixed/inference only).
#define INT_DEQUANT 1 //remove the scaling of the quantization and the internal scaling in the output (needed and done in hardware)
#define FEA_THREADS 1
#define ADJ_THREADS 1
#define GAT_ENABLE 0 //implement support for GAT
#define LINEAR_ENABLE 1 //implement support for LINEAR
#define GATV2 0 //usegatv2 modificatino or standard gat
#define SPMM_BLOCK 1 //Fused row execution optimization to improve sparse performance. 1 no fusion or 4 for 4-row fusion.
#define ATEN_BLOCK 1 //leave aten block at 1 since dataflow issues otherwise
#define OPT_ATTN 1 ////OPT_ATTN control how much sparsity expected in adj (e.g. OPT_ATTN = 1 worse case with fully dense adjacency,OPT_ATTN = 8 12.5% non zeros in adj) (this is per row, some rows have quite a few nonzeros)
#define B_WIDTH_BLOCK MAX_P //96 //attention cannot do blocks so this value has to match the W width

//with OPT_ATTN 8 and 4096 adj you have buffer for 512 nonzeros
//#ifdef EIGHTBIT

    //binary mode

    //ternary mode

    //8-bit signed mode

    //16-bit unsigned mode

    //8-bit unsigned mode normal

    //8-bit signed mode

#if (SIGNED_MODE == 1)

    typedef ap_axis<32,0,0,0> ASTYPE; //axi stream type

    #if (INT_QUANT == 1)
       typedef float INTYPE; //interface to DDR type set to float
    #else
       typedef ap_fixed<8, 1> INTYPE;
    #endif
    typedef ap_fixed<8, 1> ATYPE;
    #if (INT_QUANT == 1)
       typedef float INTYPES; //interface to DDR type set to float (weights)
    #else
       typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to float (weights)
    #endif
    typedef ap_fixed<8, 1> BTYPE;
	typedef ap_fixed<8, 1> FTYPE;
	typedef ap_fixed<32, 16> DTYPE;

    #if (INT_QUANT == 1)
	   typedef float OUTTYPE;
    #else
		typedef ap_fixed<32, 16> OUTTYPE;
    #endif
	typedef ap_fixed<32, 16> ITYPE;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<1, 1, AP_TRN_ZERO,AP_SAT>  QTYPE1;
	typedef ap_fixed<2, 1, AP_TRN_ZERO,AP_SAT>  QTYPE2;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE4;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE8;

	typedef ap_fixed<32, 16> TTYPE; //attention computing type
    typedef ap_int<8> STYPE; //scaling factor type
    #define zero_point  0.0

    //8-bit unsigned mode
#else

 #if (qbits == 8)
    union fp_int{
    int i;
    float f;
    };

    typedef ap_axis<32,0,0,0> ASTYPE; //axi stream type
    #if (INT_QUANT == 1)
       typedef float INTYPE; //interface to DDR type set to float
    #else
       typedef ap_ufixed<8, 1> INTYPE;
    #endif
    typedef ap_ufixed<8, 1> ATYPE;
    #if (INT_QUANT == 1)
       typedef float INTYPES; //interface to DDR type set to float (weights)
    #else
       typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to float (weights)
    #endif

    typedef ap_fixed<8, 1> BTYPE;
    typedef ap_fixed<8, 1> BLTYPE;
	typedef ap_ufixed<8, 1> FTYPE;
	typedef ap_fixed<8, 1> LTYPE; //linear operator type
	typedef ap_fixed<32, 16> DTYPE;

    #if (INT_QUANT == 1)
	   typedef float OUTTYPE;
    #else
		typedef ap_fixed<32, 16> OUTTYPE;
    #endif
	typedef ap_fixed<32, 16> ITYPE;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<1, 1, AP_TRN_ZERO,AP_SAT>  QTYPE1;
	typedef ap_fixed<2, 1, AP_TRN_ZERO,AP_SAT>  QTYPE2;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE4;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE8;

	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QLTYPE;

	typedef ap_fixed<32, 16> TTYPE; //attention computing type
    typedef ap_int<8> STYPE; //scaling factor type
    //#defineh beta_qu  255
    #define zero_point  0.0
    //linear EX special precision

    //#define f_alignl 0
 #endif

    //4-bit unsigned mode

      //2-bit unsigned mode
/*
    typedef ap_axiu<8,0,0,0> ASTYPE; //axi stream type
    #if (INT_QUANT == 1)
       typedef float INTYPE; //interface to DDR type set to float
    #else
       typedef ap_ufixed<8, 1> INTYPE;
    #endif
    typedef ap_ufixed<2, 1> ATYPE;
    #if (INT_QUANT == 1)
       typedef float INTYPES; //interface to DDR type set to float (weights)
    #else
       typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to float (weights)
    #endif
    typedef ap_fixed<2, 1> BTYPE;
	typedef ap_ufixed<2, 1> FTYPE;
	typedef ap_fixed<32, 16> DTYPE;

    #if (INT_QUANT == 1)
	   typedef float OUTTYPE;
    #else
		typedef ap_fixed<32, 16> OUTTYPE;
    #endif
	typedef ap_fixed<22, 16> ITYPE;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<2, 1, AP_TRN_ZERO,AP_SAT>  QTYPE2;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE4;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE8;
	typedef ap_fixed<22, 16> TTYPE; //attention computing type
    typedef ap_int<8> STYPE; //scaling factor type
    #define zero_point  0.0
    #define qbits 2
*/

    //1-bit unsigned mode

 #if (qbits==1)

 union fp_int{
    int i;
    float f;
 };

 typedef ap_axis<32,0,0,0> ASTYPE; //axi stream type
    #if (INT_QUANT == 1)
       typedef float INTYPE; //interface to DDR type set to float
    #else
       typedef ap_ufixed<8, 1> INTYPE;
    #endif
    typedef ap_ufixed<2, 1> ATYPE;
    #if (INT_QUANT == 1)
       typedef float INTYPES; //interface to DDR type set to float (weights)
    #else
       typedef ap_fixed<8, 1> INTYPES; //interface to DDR type set to float (weights)
    #endif
    typedef ap_fixed<1, 1> BTYPE;
	typedef ap_ufixed<2, 1> FTYPE;
    typedef ap_fixed<qbitsl, 1> BLTYPE;
	typedef ap_fixed<qbitsl, 1> LTYPE; //linear operator type
	typedef ap_fixed<32, 16> DTYPE;

    #if (INT_QUANT == 1)
	   typedef float OUTTYPE;
    #else
		typedef ap_fixed<32, 16> OUTTYPE;
    #endif
	typedef ap_fixed<18, 16> ITYPE;
	typedef ap_fixed<1, 1, AP_TRN_ZERO,AP_SAT>  QTYPE;
	typedef ap_fixed<1, 1, AP_TRN_ZERO,AP_SAT>  QTYPE1;
	typedef ap_fixed<2, 1, AP_TRN_ZERO,AP_SAT>  QTYPE2;
	typedef ap_fixed<4, 1, AP_TRN_ZERO,AP_SAT>  QTYPE4;
	typedef ap_fixed<8, 1, AP_TRN_ZERO,AP_SAT>  QTYPE8;

	typedef ap_fixed<qbitsl, 1, AP_TRN_ZERO,AP_SAT>  QLTYPE;

	typedef ap_fixed<18, 16> TTYPE; //attention computing type
    typedef ap_int<8> STYPE; //scaling factor type

    #define zero_point  0.0

 #endif

    //2-bit unsigned mode

    //1-bit unsigned mode

    //normal mode

    /*typedef ap_fixed<32, 16> ATYPE;
    typedef ap_fixed<32, 16> BTYPE;
	typedef ap_fixed<32, 16> DTYPE;
	typedef ap_fixed<32, 16> FTYPE;
	typedef ap_fixed<32, 16, AP_RND,AP_SAT> ITYPE;
	typedef ap_fixed<32, 16> QTYPE;
    typedef ap_ufixed<32, 1> STYPE; //scaling factor type
    #define frac_bits 16*/

    #define FTYPE_LATENCY_ADJ 1
    #define FTYPE_LATENCY_FEA 1

#endif

#ifdef HALF

	typedef half ATYPE;
	typedef half BTYPE;
	typedef half DTYPE;
	typedef half FTYPE;
	typedef half ITYPE;
    typedef half STYPE; //scaling factor type
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
	//#define FTYPE_LATENCY 4 //100 MHZ
	#define FTYPE_LATENCY_ADJ 6
	#define FTYPE_LATENCY_FEA 6 //200 MHZ
#endif

//#ifdef FLOAT
//  	#define FTYPE_LATENCY 6   //optimal latency is 6 for 200 MHz with 3 II increases but we reduce pressure on logic
//#endif

//#ifdef HALF
//        #define FTYPE_LATENCY 4
//#endif

#define USE_SBLOCKS 0

/*the compute unit always uses sblocks, the USE_SBLOCKS controls if the write unit also uses sblocks and reads multiple FIFO channels
 * or reads only one FIFO channel per core and this is generally better because it optimizes the loop
 * that writes data to memory
 */

/* spmm block controls how many rows of the sparse matrix are processed in a single for loop. In principle only one
 * row is processed and then a matrix mult output is written into the C buffer memory. If only a few elements in the row
 * are nonzero then the overhead is significant since the loop needs to start again for the next row. The loop
 * achieves II 1 but if the number of elements of the row is small the flushing the pipeline and restarting the row
 * is an overhead. By grouping several rows in a single loop it is possible to alleviate this problem and have more nonzeros to process
 */

#define A_HEIGHT_BLOCK  1// 4096 //(512/4)

#define FAST_ATTENTION 1

#define B_BLOCK_PARALLEL 1
//#define PES_ADJ 2 // Number of PEs for ADJ processing
//#define PES_FEA 2 // Number iof PEs for FEA processing

#define ENABLE_GEMM
#define ENABLE_SPMM
//#define ENABLE_SCALING
//#define ENABLE_TRANSPOSE

//how many rows of B are computed in parallel in multiplication loop
//for example a couple of B rows are multiplied for A 1 row in each loop iteration
//it basically reduces how the loop iterations by 2 if it is 2.
/*
 A_HEIGHT_BLOCK  is for software part data partitioning due to the limitation in 
 the Xilinx kernel sds_alloc so A_HEIGHT_BLOCK should be A_HEIGHT divided by 
 the number of considered blocks
*/
#define C_HEIGHT_BLOCK  A_HEIGHT_BLOCK 

typedef std::vector<int> vi;

#endif //__MATRIX_MULT_H__

