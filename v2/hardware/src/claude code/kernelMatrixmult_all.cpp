

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <hls_math.h>

#include <string>
#include <fstream>
#include <sstream> // ////std::stringstream

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"

#include "matrix_mult.h"

#include "hls_streamofblocks.h"

typedef QTYPE buf[F_HEIGHT/FEA_THREADS][B_WIDTH_BLOCK];
typedef QLTYPE bufl[F_HEIGHT/FEA_THREADS][B_WIDTH_BLOCK];

// note that  BLOCK shoudl be less than B_WIDTH_gmmBLOCK

const int BLOCK=B_WIDTH_BLOCK;   //BLOCK should be less than B_WIDTH_BLOCK
const int SBLOCK=SPMM_BLOCK;   //BLOCK should be less than B_WIDTH_BLOCK
const int SBLOCK_LIN=1;   //BLOCK should be less than B_WIDTH_BLOCK

const int PARALLEL_ROW = B_BLOCK_PARALLEL;
const int FIFO_DEPTH = MAX_FIFO;

const int LINEAR_DEPTH=B_WIDTH_BLOCK*B_HEIGHT;

//worst case attention fifo
//attention FIFO with adj up to 12.5% nonzeros
//attention FIFO with adj up to 12.5% nonzeros

const int FIFO_DEPTH_ATTN = A_HEIGHT/OPT_ATTN;
const int FIFO_DEPTH_ATTN2 = A_HEIGHT*ATEN_BLOCK/OPT_ATTN;

const int FADD_LATENCY_ADJ = FTYPE_LATENCY_ADJ;
const int FADD_LATENCY_FEA = FTYPE_LATENCY_FEA;

static ap_int<64> fifo_full_0;
static ap_int<64> fifo_full_1;
static ap_int<64> fifo_full_2;
static ap_int<64> fifo_empty_0;
static ap_int<64> fifo_empty_1;
static ap_int<64> fifo_empty_2;
static ap_int<64> fifo_read_0;
static ap_int<64> fifo_read_1;
static ap_int<64> fifo_read_2;
static ap_int<64> fifo_write_0;
static ap_int<64> fifo_write_1;
static ap_int<64> fifo_write_2;
static ap_int<64> fifo_cycle_0;
static ap_int<64> fifo_cycle_1;
static ap_int<64> fifo_cycle_2;

#ifdef simulation
extern float max_adj;
extern float min_adj;
extern float max_fea;
extern float min_fea;
extern float acc2_fea_min;
extern float acc2_fea_max;
extern float acc2_adj_min;
extern float acc2_adj_max;
#endif

void quanta(ATYPE &BW,float B,float quantization_scale,int f_align, int beta_qu)
{

	float vfloat = quantization_scale*B+zero_point;
	float vround = hls::round(vfloat);

	ITYPE vquant = ITYPE(vround);

    #if (SIGNED_MODE==0)
	ITYPE ibeta_q = (ITYPE)beta_qu;
	ITYPE ialpha_q = (ITYPE)(0.0);
    #else
	ITYPE beta_q = ITYPE(beta_qu>>1);
	ITYPE ibeta_q = (ITYPE)beta_q;
	ITYPE ialpha_q = -(ITYPE)beta_q;
    #endif

    //clippping
	if (vquant>ibeta_q)
		vquant = ibeta_q;
	else if (vquant<ialpha_q)
		vquant = ialpha_q;

	if(f_align==7) //MODEL BINARY MODE
		f_align = 6;

    #if(qbits==1) //DEVICE BINARY MODE
	 ITYPE vnorm = vquant >> (1);
    #else
	 ITYPE vnorm = vquant >> (qbits-f_align-1);
    #endif

 	ATYPE fval = ATYPE(vnorm);

	BW = fval;

}

void quantf(FTYPE &BW,float B,float quantization_scale[5],int f_align, int beta_qu, int B_index)
{

	float vfloat = quantization_scale[B_index]*B+zero_point;
	float vround = hls::round(vfloat);

	ITYPE vquant = ITYPE(vround);

    #if (SIGNED_MODE==0)
    ITYPE ibeta_q = (ITYPE)beta_qu;
    ITYPE ialpha_q = (ITYPE)(0.0);
    #else
    ITYPE beta_q = ITYPE(beta_qu>>1);
    ITYPE ibeta_q = (ITYPE)beta_q;
    ITYPE ialpha_q = -(ITYPE)beta_q;
    #endif

	//clippping
	if (vquant>ibeta_q)
		vquant = ibeta_q;
	else if (vquant<ialpha_q)
		vquant = ialpha_q;

	if(f_align==7) //MODEL BINARY MODE
		f_align = 6;

    #if(qbits==1) //DEVICE BINARY MODE
       ITYPE vnorm = vquant >> (1);
    #else
       ITYPE vnorm = vquant >> (qbits-f_align-1);
    #endif

 	FTYPE fval = FTYPE(vnorm);

	BW = fval;

}

void quantl(LTYPE &BW,float B,float quantization_scale[5],int f_align, int beta_qu, int B_index)
{

	float vfloat = quantization_scale[B_index]*B+zero_point;

	float vround;

	ITYPE ibeta_q,ialpha_q,beta_q;

    if(f_align==7)
	{
	    ibeta_q = 1;
	    ialpha_q = -1;
		if(vfloat < 0.0) //BINARY MODE
		 vround = -1.0;
	    else
		 vround = 1.0;
	}
    else
    {
		beta_q = ITYPE(beta_qu>>1);
	    ibeta_q = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
    	vround = hls::round(vfloat);
    }

	ITYPE vquant = ITYPE(vround);

	//clippping
	if (vquant>ibeta_q)
		vquant = ibeta_q;
	else if (vquant<ialpha_q)
		vquant = ialpha_q;

	if(f_align==7) //BINARY MODE
		f_align = 6;
	ITYPE vnorm = vquant >> (qbitsl-f_align-1);
 	LTYPE lval = LTYPE(vnorm);

	BW = lval;

}

void quantw(BTYPE &BW,float B,float quantization_scale[5],int f_align, int beta_qu, int B_index)
{

	float vfloat = quantization_scale[B_index]*B+zero_point;

	float vround;

	ITYPE ibeta_q,ialpha_q,beta_q;

    #if (qbits==1)
     ibeta_q = 1;
     ialpha_q = 0;
	 if(vfloat < 0.0) //BINARY MODE
	  vround = 1.0;
     else
	  vround = 0.0;
    #else
	 if(f_align==7)
	 {
	    ibeta_q = 1;
	    ialpha_q = -1;
		if(vfloat < 0.0) //BINARY MODE
		 vround = -1.0;
	    else
		 vround = 1.0;
	 }
     else
     {
		beta_q = ITYPE(beta_qu>>1);
	    ibeta_q = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
    	vround = hls::round(vfloat);
     }
    #endif

	ITYPE vquant = ITYPE(vround);

	//clippping
	if (vquant>ibeta_q)
		vquant = ibeta_q;
	else if (vquant<ialpha_q)
		vquant = ialpha_q;

	if(f_align==7) //MODEL BINARY MODE
		f_align = 6;

    ITYPE vnorm = vquant >> (qbits-f_align-1);

 	BTYPE fval = BTYPE(vnorm);

	BW = fval;

}

void quantwl(BLTYPE &BW,float B,float quantization_scale[5],int f_align, int beta_qu, int B_index)
{

	float vfloat = quantization_scale[B_index]*B+zero_point;

	float vround;

	ITYPE ibeta_q,ialpha_q,beta_q;

    if(f_align==7)
	{
	    ibeta_q = 1;
	    ialpha_q = -1;
		if(vfloat < 0.0) //BINARY MODE
		 vround = -1.0;
	    else
		 vround = 1.0;
	}
    else
    {
		beta_q = ITYPE(beta_qu>>1);
	    ibeta_q = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
    	vround = hls::round(vfloat);
    }

	ITYPE vquant = ITYPE(vround);

	//clippping
	if (vquant>ibeta_q)
		vquant = ibeta_q;
	else if (vquant<ialpha_q)
		vquant = ialpha_q;

	if(f_align==7) //BINARY MODE
		f_align = 6;
	ITYPE vnorm = vquant >> (qbitsl-f_align-1);
 	BLTYPE fval = BLTYPE(vnorm);

	BW = fval;

}

void dsp_kernel_float_adj_1(ATYPE a_value,BTYPE b_block[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	#pragma HLS INLINE

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			BTYPE b_val = b_block[b_row][j];
 	  		ATYPE a_val = a_value;
			acc[j] = (ITYPE)a_val*(ITYPE)b_val;

	} // j loop

}
void dsp_kernel_float_adj_2(int block_size,ATYPE a_value,BTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block2[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	#pragma HLS INLINE

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

	  		ATYPE a_val = a_value;
	  		BTYPE b_val;

	  		int sel_block; // = (b_row>>(log2N-2))&0x3;
	  		int b_row_block;

	  		if (b_row < block_size)
	  		{
	  			b_row_block = b_row;
	  			sel_block = 0;
	  		}
	  		if (b_row > (block_size-1))
	  		{
	  			b_row_block = b_row-block_size;
	  			sel_block = 1;
	  		}

	  		BTYPE b_val1 = b_block1[b_row_block][j];
			BTYPE b_val2 = b_block2[b_row_block][j];

	  		switch(sel_block)
	  		{
	  			case 0:
	  				b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
	  				break;
	  			case 1:
	  				b_val = b_val2;
  				break;
	  			//case 2:
	  			//case 3:
	  		}
			acc[j] = (ITYPE)a_val*(ITYPE)b_val;

	} // j loop

}

void dsp_kernel_float_adj_4(int block_size,ATYPE a_value,BTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block2[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block3[B_HEIGHT][B_WIDTH_BLOCK],BTYPE b_block4[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	#pragma HLS INLINE

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

	  		ATYPE a_val = a_value;
	  		BTYPE b_val;

	  		int sel_block; // = (b_row>>(log2N-2))&0x3;
	  		int b_row_block;

	  		if (b_row < block_size)
	  		{
	  			b_row_block = b_row;
	  			sel_block = 0;
	  		}
	  		if (b_row > (block_size-1))
	  		{
	  			b_row_block = b_row-block_size;
	  			sel_block = 1;
	  		}

  		    if (b_row > (2*block_size-1) && b_row < 3*block_size)
	  		{
  		  	    b_row_block = b_row-2*block_size;
	  			sel_block = 2;
	  		}
  		    if (b_row > 3*block_size-1)
	  		{
  			    b_row_block = b_row-3*block_size;
	  			sel_block = 3;
	  		}

	  		BTYPE b_val1 = b_block1[b_row_block][j];
			BTYPE b_val2 = b_block2[b_row_block][j];
			BTYPE b_val3 = b_block3[b_row_block][j];
			BTYPE b_val4 = b_block4[b_row_block][j];

	  		switch(sel_block)
	  		{
	  			case 0:
	  				b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
	  				break;
	  			case 1:
	  				b_val = b_val2;
  				break;
	  			case 2:
	  				b_val = b_val3;
  				break;
	  			case 3:
	  				b_val = b_val4;
  				break;
	  		}
			acc[j] = (ITYPE)a_val*(ITYPE)b_val;

	} // j loop

}

void dsp_kernel_float_fea(ATYPE a_value,BTYPE b_block[B_HEIGHT][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{
	#pragma HLS INLINE

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			BTYPE b_val = b_block[b_row][j];
 	  		ATYPE a_val = a_value;
 	  		acc[j] = (ITYPE)a_val*(ITYPE)b_val;

	} // j loop

}

void dsp_kernel_int_adj_1(int block_size,TTYPE a_value,QTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],
		//ITYPE b_block2[B_HEIGHT/2][B_WIDTH_BLOCK],
		//ITYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],
		//ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
		ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
 	  		TTYPE a_val = a_value;
 	  		QTYPE b_val;

 	  		int sel_block; // = (b_row>>(log2N-2))&0x3;
 	  		int b_row_block;

 	  		if (b_row < block_size)
 	  		{
 	  			b_row_block = b_row;
 	  			sel_block = 0;
 	  		}

 	  		QTYPE b_val1 = b_block1[b_row_block][j];

 	  		switch(sel_block)
 	  		{
 	  			case 0:
 	  				b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
 	  				break;
 	  			//case 1:
 	  			//case 2:
 	  			//case 3:
 	  		}

 	  		 ITYPE a_val_i = (ITYPE)a_val;

 	  		 ITYPE b_val_i;

             #if (qbits==1)
 	  		    if(b_val==0)
 	  		     b_val_i = (ITYPE)(0.5);
 	  		    else
 	  		     b_val_i = (ITYPE)(-0.5);
             #else

                 b_val_i = (ITYPE)b_val;

             #endif

  			 ITYPE acc_i = a_val_i*b_val_i;

			acc[j] = acc_i;
	} // j loop

}

void dsp_kernel_int_adj_2(int block_size,ITYPE a_value,QTYPE b_block1[B_HEIGHT/2][B_WIDTH_BLOCK],
		QTYPE b_block2[B_HEIGHT/2][B_WIDTH_BLOCK],
		//ITYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],
		//ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
		ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

			#pragma HLS UNROLL

				acc[j] = 0;
        }

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
 	  		ATYPE a_val = a_value;
 	  		BTYPE b_val;

 	  		int sel_block; // = (b_row>>(log2N-2))&0x3;
 	  		int b_row_block;

 	  		if (b_row < block_size)
 	  		{
 	  			b_row_block = b_row;
 	  			sel_block = 0;
 	  		}
 	  		if (b_row > (block_size-1))
 	  		{
 	  			b_row_block = b_row-block_size;
 	  			sel_block = 1;
 	  		}

 	  		BTYPE b_val1 = b_block1[b_row_block][j];
 			BTYPE b_val2 = b_block2[b_row_block][j];

 	  		switch(sel_block)
 	  		{
 	  			case 0:
 	  				b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
 	  				break;
 	  			case 1:
 	  				b_val = b_val2;
	  				break;
 	  			//case 2:
 	  			//case 3:
 	  		}

			ITYPE a_val_i = (ITYPE)a_val;
			ITYPE b_val_i = (ITYPE)b_val;

			ITYPE acc_i = a_val_i*b_val_i;
			acc[j] += acc_i;
	} // j loop

}

void dsp_kernel_int_adj_4(int block_size,TTYPE a_value,QTYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],
		QTYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],
		QTYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

			#pragma HLS UNROLL

				acc[j] = 0;
        }

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
 	  		TTYPE a_val = a_value;
 	  		QTYPE b_val;

 	  		int sel_block; // = (b_row>>(log2N-2))&0x3;
 	  		int b_row_block;

 	  		if (b_row < block_size)
 	  		{
 	  			b_row_block = b_row;
 	  			sel_block = 0;
 	  		}
 	  		if (b_row > (block_size-1) && b_row < 2*block_size)
 	  		{
 	  			b_row_block = b_row-block_size;
 	  			sel_block = 1;
 	  		}
	  		if (b_row > (2*block_size-1) && b_row < 3*block_size)
 	  		{
	  			b_row_block = b_row-2*block_size;
 	  			sel_block = 2;
 	  		}
	  		if (b_row > 3*block_size-1)
 	  		{
	  			b_row_block = b_row-3*block_size;
 	  			sel_block = 3;
 	  		}

 	  		QTYPE b_val1 = b_block1[b_row_block][j];
 			QTYPE b_val2 = b_block2[b_row_block][j];
 			QTYPE b_val3 = b_block3[b_row_block][j];
 			QTYPE b_val4 = b_block4[b_row_block][j];

 	  		switch(sel_block)
 	  		{
 	  			case 0:
 	  				b_val = b_val1; //only one value of B in each row. This is the result of the first matrix mult.
 	  				break;
 	  			case 1:
 	  				b_val = b_val2;
	  				break;
 	  			case 2:
 	  				b_val = b_val3;
	  				break;
 	  			case 3:
 	  				b_val = b_val4;
	  				break;
 	  		}

			ITYPE a_val_i = (ITYPE)a_val;
			ITYPE b_val_i = (ITYPE)b_val;

			ITYPE acc_i = a_val_i*b_val_i;
			acc[j] += acc_i;
	} // j loop

}

void dsp_kernel_int_fea(FTYPE a_value,BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
 	  		FTYPE a_val = a_value;
			BTYPE b_val = b_block[b_row][j]; //only one value of B in each row. This is the result of the first matrix mult.

	        ITYPE b_val_i;
	        ITYPE a_val_i;
//	           else
//	           //else
//	        else

             #if (qbits==1)
	           if(b_val==0)
	              b_val_i = (ITYPE)(0.5);
	           else
	               b_val_i = (ITYPE)(-0.5);
             #else
                   b_val_i = (ITYPE)b_val;
             #endif
			   a_val_i = (ITYPE)a_val;

		    ITYPE acc_i = a_val_i*b_val_i;
			acc[j] = acc_i;

	} // j loop

}

void dsp_kernel_int_lin(LTYPE a_value,BLTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<32> b_row,ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc[B_WIDTH_BLOCK])
{

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
 	  		LTYPE a_val = a_value;
			BLTYPE b_val = b_block[b_row][j]; //only one value of B in each row. This is the result of the first matrix mult.

	        ITYPE b_val_i;
	        ITYPE a_val_i;
//	           else
//	           //else
//	        else
			   b_val_i = (ITYPE)b_val;
			   a_val_i = (ITYPE)a_val;

		    ITYPE acc_i = a_val_i*b_val_i;
			acc[j] = acc_i;

	} // j loop

}

void writec(float deq_factor[5],ap_uint<1> model[5][8],int first_row, int row_count,int N_adj,ap_uint<8> P[5], hls::stream<ITYPE> write_fifo[B_WIDTH_BLOCK],QLTYPE linear_pipo[B_HEIGHT][B_WIDTH_BLOCK],hls::stream<OUTTYPE>& CS, int B_index, int layer_loop)
{
		int B_WIDTH_INT;

	    bool linear_mode;
	    bool sage_mode;

		int WL;

		#if defined FLOAT
			WL = row_count;
		#endif

		#if defined HALF
			WL = row_count;
		#endif

		#ifdef FIXEDPOINT
			WL = row_count;

		#endif

        linear_mode = model[B_index][6];
        sage_mode = model[B_index][7];
		bool gcn_path = !(linear_mode^sage_mode);

		DTYPE C_out = DTYPE(0.0);
		DTYPE residual;

        LOOP_WRITE42:    for (int i = 0; i < WL; i++) {
		    LOOP_WRITE52: for (int j = 0; j <  B_WIDTH_BLOCK; j++) {
					 #pragma HLS PIPELINE II=1
		    	        if (gcn_path==1)
						 C_out =  DTYPE(write_fifo[j].read());
                        #if LINEAR_ENABLE == 1
						if (linear_mode==1)
						 residual = DTYPE(linear_pipo[i][j]);
						else
						 residual = 0;
                        #else
						residual = DTYPE(0.0);
                        #endif

						//dequant
                        #if (INT_DEQUANT == 1)
						     OUTTYPE C_float = (OUTTYPE)C_out*deq_factor[B_index]+(OUTTYPE)residual*deq_factor[B_index];

						#else
						     OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif

                        if(j<P[B_index]) //when relu only write non-zeros for sparse computation
                         CS.write(C_float);

					}
         }

}

void writeout(ap_uint<1> model[5][8],int first_row, int row_count,int N_adj,ap_uint<8> P[5], hls::stream<OUTTYPE>& write_fifo,OUTTYPE* C,hls::stream<ASTYPE>& CS, hls::stream<ASTYPE>& CSR, hls::stream<ASTYPE>& CSC, int B_index, int layer_loop)
{
		int B_WIDTH_INT;

		int WL;

		WL = row_count;

		B_WIDTH_INT = P[B_index];
        ap_uint<1> stream_mode = model[B_index][2];
        ap_uint<1> gemm_mode = model[B_index+1][1]; //check if next layer wants sparse features

		std::cout << "stream mode in adj is " << stream_mode << std::endl;

		if (stream_mode==1) //write to stream
		{

			bool last=0;
		    std::cout << "write to stream count " << WL*B_WIDTH_BLOCK << std::endl;

           		 LOOP_WRITE42:    for (int i = 0; i < WL; i++) {
			       LOOP_WRITE52: for (int j = 0; j <  B_WIDTH_INT; j++) {
					 #pragma HLS PIPELINE II=1

                     //terminate stream
			   	     if(i*j==(WL-1)*(B_WIDTH_INT-1))
			   	       last = 1;

			    	 OUTTYPE C_float =  OUTTYPE(write_fifo.read());
					 ASTYPE temp;
					 fp_int C_float_int;
    	  			 C_float_int.f = C_float;
  	  			     temp.data = C_float_int.i;

  					 if(gemm_mode==1)
                     {
	  			      temp.last = last;
                      CS.write(temp);
                     }
                     else
                     {
   	  			      if(j==0 or C_float!=0 or last==1) //do not write zero but always write if last=1 or if first element of row
   	  			      {
   	  			       temp.last = last;
                       CS.write(temp);
                       temp.data = i;
                       temp.last = last;
                       CSR.write(temp);
                       temp.data = j;
                       temp.last = last;
					   CSC.write(temp);
                      }
                     }

			       }
                  }

		}
		else  // write to memory
		{
	        	LOOP_WRITE45:    for (int i = 0; i < WL; i++) {
		 	     LOOP_WRITE55: for (int j = 0; j <  B_WIDTH_INT; j++) {
					#pragma HLS PIPELINE II=1
				 	OUTTYPE C_float =  OUTTYPE(write_fifo.read());
					C[i*B_WIDTH_INT+j+first_row*B_WIDTH_BLOCK] = C_float;
					}
	               }

		}

}

void writes(float deq_factor[5],ap_uint<1> model[5][8], int first_row, int row_count,int N_adj,ap_uint<8> P[5], hls::stream<TTYPE> &write_fifo, hls::stream<int> &rnnz_fifo,  OUTTYPE* C,int B_index)
{
		int B_WIDTH_INT;

		int WL;

		WL = row_count;

		B_WIDTH_INT = B_WIDTH_BLOCK;

		bool linear_mode = model[B_index][6];
		bool gat_mode = model[B_index][5];
		bool sage_mode = model[B_index][7];

		bool gcn_path = !(linear_mode^sage_mode);

		if (gcn_path == 1)
		{

		 if (gat_mode == 1)
		 {

			 int rnnz = rnnz_fifo.read();

		  

			     DTYPE C_out;

			     LOOP_WRITE5: for (int i = 0; i <  rnnz; i++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo.read();
                        #if (INT_DEQUANT == 1)
				           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor[B_index];
                        #else
				           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
						C[i] = C_float;
					}

		  }
		} //linear mode

}

void writesx4(float deq_factor,bool gat_mode,
		int row_count1,int row_count2,int row_count3,int row_count4,
		hls::stream<TTYPE> &write_fifo1,
		hls::stream<TTYPE> &write_fifo2,
		hls::stream<TTYPE> &write_fifo3,
		hls::stream<TTYPE> &write_fifo4,
		hls::stream<int> &rnnz_fifo1,
		hls::stream<int> &rnnz_fifo2,
		hls::stream<int> &rnnz_fifo3,
		hls::stream<int> &rnnz_fifo4,
		OUTTYPE* C,int B_index)
{
		int B_WIDTH_INT;

		DTYPE C_out;

		B_WIDTH_INT = B_WIDTH_BLOCK;

        int rnnz_total = 0;
        int rnnz1,rnnz2,rnnz3,rnnz4;

		rnnz1 = rnnz_fifo1.read();
		rnnz2 = rnnz_fifo2.read();
		rnnz3 = rnnz_fifo3.read();
		rnnz4 = rnnz_fifo4.read();

		if (gat_mode == 1)
		{

			     LOOP_WRITE51: for (int j = 0; j <  rnnz1; j++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo1.read();
                        #if (INT_DEQUANT == 1)
				           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor;
                        #else
				           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
						C[j+rnnz_total] = C_float;
					}
					rnnz_total+=rnnz1;

			     LOOP_WRITE52: for (int j = 0; j <  rnnz2; j++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo2.read();
                        #if (INT_DEQUANT == 1)
				           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor;
                        #else
				           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
						C[j+rnnz_total] = C_float;
					}
					rnnz_total+=rnnz2;

			     LOOP_WRITE53: for (int j = 0; j <  rnnz3; j++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo3.read();
                        #if (INT_QUANT == 1)
				           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor;
                        #else
				           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
						C[j+rnnz_total] = C_float;
					}
					rnnz_total+=rnnz3;

			     LOOP_WRITE54: for (int j = 0; j <  rnnz4; j++) {
					#pragma HLS PIPELINE
						C_out =  write_fifo4.read();
                        #if (INT_DEQUANT == 1)
				           OUTTYPE C_float = (OUTTYPE)C_out*deq_factor;
                        #else
				           OUTTYPE C_float = (OUTTYPE)C_out;
                        #endif
						C[j+rnnz_total] = C_float;
					}
					rnnz_total+=rnnz4;

		 

		}

}

void readptr_csr_fea(bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    #pragma HLS inline off
	int rnnz,current_index,next_index;

	current_index= rowPtr[0];

	if (gemm_mode==0)
	{
		LOOP_A_INDEX_SPMM1 : for (int A_index = 0; A_index < N; A_index++) {
			#pragma HLS PIPELINE
					next_index=rowPtr[A_index+1];
					rnnz = next_index-current_index;
					current_index = next_index;
					rnnz_fifo << rnnz;
			}
	}
	else
	{
		LOOP_A_INDEX_SPMM2 : for (int A_index = 0; A_index < N; A_index++) {
			#pragma HLS PIPELINE
				rnnz = M;
				rnnz_fifo << rnnz;
		 }
	} //end else

}

void read_ptr2(int nnz_fea,int *rowPtr, hls::stream<int> &index_fifo)
{
	int next_index1;
	//ADJ dataflow
	LOOP_A_INDEX0 : for (int A_index = 0; A_index <nnz_fea+1 ; A_index++)
	{
		#pragma HLS PIPELINE
  	    next_index1=rowPtr[A_index];
  	    index_fifo << next_index1;
	}

}

void read_ptr(bool stream_mode,int nnz_fea,int *rowPtr,  hls::stream<int> &index_fifo)
{
	int next_index1;
	//FEA dataflow
	if (stream_mode == 0)
	{
	  LOOP_A_INDEX0 : for (int A_index = 0; A_index <nnz_fea+1 ; A_index++)
	  {
		#pragma HLS PIPELINE
  	    next_index1=rowPtr[A_index];
  	    index_fifo << next_index1;
	  }
	}

}

void proc_ptr(int nnz_fea,hls::stream<int> &index_fifo,hls::stream<int> &rnnz_fifo)
{
	int next_index2;
	int rnnz = 0;
	int current_index;
	int B_index = 0;
	int first_read = 1;

	current_index =index_fifo.read();
	rnnz++;

	LOOP_A_INDEX1 : while(B_index < nnz_fea-1) {
	#pragma HLS PIPELINE
	next_index2=index_fifo.read();
	B_index++;
	if(next_index2 == current_index)
	{
	   rnnz++;

    }
    else
    {

	  rnnz_fifo << rnnz;
	  current_index=next_index2;
	  rnnz = 1;
	}

   }

    rnnz_fifo << rnnz;

   //else

    next_index2=index_fifo.read();
}

void proc_ptr2(bool gcn_path,bool linear_mode,bool stream_mode,int nnz_fea,hls::stream<int> &index_fifo,hls::stream<ASTYPE>&  rowPtrs,hls::stream<int> &rnnz_fifo,hls::stream<int> &rnnz_fifo_sage)
{
	int next_index2;
	int rnnz = 0;
	int current_index;
	ASTYPE  temp;
	int B_index = 0;
	int first_read = 1;

    if(stream_mode==0)
    {

     current_index = index_fifo.read();
     rnnz++;
	 LOOP_A_INDEX1 : while(B_index < nnz_fea-1) {

	 #pragma HLS PIPELINE
	 next_index2 =index_fifo.read();
	 B_index++;
	 if(next_index2 == current_index)
	 {
	   rnnz++;

     }
     else
     {

	   if(gcn_path==1)
	    rnnz_fifo << rnnz;

       #if (LINEAR_ENABLE == 1)
	   if(linear_mode==1)
	    rnnz_fifo_sage << rnnz;
       #endif

	   current_index=next_index2;
	   rnnz = 1;
	 }

   }

   if(gcn_path==1)
    rnnz_fifo << rnnz;

   #if (LINEAR_ENABLE == 1)
   if(linear_mode==1)
    rnnz_fifo_sage << rnnz;
   #endif
   next_index2=index_fifo.read();
   }
   else //stream mode on
   {

	 temp=rowPtrs.read();
     rnnz=1;
	 current_index= temp.data;

	 if(temp.last!=1)
	 {
	  LOOP_A_INDEX2 : do {
		 #pragma HLS PIPELINE
		 temp=rowPtrs.read();
		 next_index2= temp.data;
		 if(next_index2 == current_index)
		 {
		   rnnz++;

	     }
	     else
	     {

		   if(gcn_path==1)
		    rnnz_fifo << rnnz;

	       #if (LINEAR_ENABLE == 1)
		   if(linear_mode==1)
		    rnnz_fifo_sage << rnnz;
	       #endif

		   current_index=next_index2;
		   rnnz = 1;
		 }
	   }while(temp.last!=1);
	 }

   if(gcn_path==1)
    rnnz_fifo << rnnz;

   #if (LINEAR_ENABLE == 1)
   if(linear_mode==1)
    rnnz_fifo_sage << rnnz;
   #endif

   }
   //else

}

void read_dataflow2(bool gcn_path,bool linear_mode,bool stream_mode,int nnz_fea,int *rowPtr,hls::stream<ASTYPE>&  rowPtrs,hls::stream<int> &rnnz_fifo,hls::stream<int> &rnnz_fifo_sage)
{

    hls::stream<int>  index_fifo("index fifo");
	#pragma HLS STREAM variable= index_fifo depth=FIFO_DEPTH

    //FEA READDATAFLOW

    #pragma HLS DATAFLOW
    read_ptr(stream_mode,nnz_fea,rowPtr,index_fifo);
	proc_ptr2(gcn_path,linear_mode,stream_mode,nnz_fea,index_fifo,rowPtrs,rnnz_fifo,rnnz_fifo_sage);

}

void read_dataflow(int nnz_fea,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

	hls::stream<int>  index_fifo("index fifo");
	#pragma HLS STREAM variable= index_fifo depth=FIFO_DEPTH

    //ADJ READDATAFLOW

    #pragma HLS DATAFLOW
    read_ptr2(nnz_fea,rowPtr,index_fifo);
	proc_ptr(nnz_fea,index_fifo,rnnz_fifo);

}

void readptr_coo_fea(int nnz_fea,bool sage_mode,bool linear_mode,bool stream_mode,bool gemm_mode,int N,int M,int *rowPtr,hls::stream<ASTYPE>&  rowPtrs,hls::stream<int> &rnnz_fifo,hls::stream<int> &rnnz_fifo_sage)
{

    #pragma HLS inline off

    bool gcn_path = !(linear_mode^sage_mode);

	if (gemm_mode==0)
	{

		read_dataflow2(gcn_path,linear_mode,stream_mode,nnz_fea,rowPtr,rowPtrs,rnnz_fifo,rnnz_fifo_sage);

	}
	else
	{
		int rnnz;
		LOOP_A_INDEX2 : for (int A_index = 0; A_index <N ; A_index++) {
			#pragma HLS PIPELINE
				rnnz = M;

				if(gcn_path==1)
				 rnnz_fifo << rnnz;

                #if (LINEAR_ENABLE == 1)
				if(linear_mode==1)
				 rnnz_fifo_sage << rnnz;
                #endif
		 }
	} //end else

}

void readptr_csr_adj(bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{
	#pragma HLS inline off
	int rnnz,current_index,next_index;

	current_index= rowPtr[0];

	if (gemm_mode==0)
	{
		LOOP_A_INDEX_SPMM1 : for (int A_index = 0; A_index < N; A_index++) {
				next_index=rowPtr[A_index+1];
				rnnz = next_index-current_index;
				current_index = next_index;
				rnnz_fifo << rnnz;

        }
	}
	else
	{
		LOOP_A_INDEX_SPMM2 : for (int A_index = 0; A_index < N; A_index++) {
			#pragma HLS PIPELINE
			rnnz = M;
			rnnz_fifo << rnnz;

       }
	} //end else

}

void readptr_coo_adj(int nnz_adj,bool sage_mode,bool linear_mode,bool gemm_mode,int N,int M,int *rowPtr,hls::stream<int> &rnnz_fifo)
{

    #pragma HLS inline off

    bool gcn_path = !(linear_mode^sage_mode);

    if(gcn_path==1)
    {
	 if (gemm_mode==0)
	 {

		read_dataflow(nnz_adj,rowPtr,rnnz_fifo);

	 }
	 else
	 {
		int rnnz;
		LOOP_A_INDEX2 : for (int A_index = 0; A_index <N ; A_index++) {
			#pragma HLS PIPELINE
				rnnz = M;

				rnnz_fifo << rnnz;

		 }
	 } //end else
    }//end linear
}

void readval_csr_adj(int beta_qu,int f_align,float quantization_scale_fea,bool gemm_mode,int ccount,int last_index,hls::stream<ATYPE> &A_fifo,hls::stream<int> &col_indices_fifo,INTYPE *values,int *columnIndex)
{

		#pragma HLS inline off
	    if (gemm_mode==0)
	    {
		  LOOP_J_SPMM : for (int j = 0; j < last_index; j++) {
			#pragma HLS PIPELINE

			INTYPE value_temp;
			ATYPE value_temp2;
			value_temp = (INTYPE)values[j];

            #if (INT_QUANT_A == 1)
	            quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
            #else
	            value_temp2 = value_temp;
            #endif

			A_fifo <<  value_temp2;
			col_indices_fifo << columnIndex[j];
		  }
		}
		else
		{
				int c=0;
				LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
				#pragma HLS PIPELINE

				   INTYPE value_temp;
				   ATYPE value_temp2;

				   value_temp = (INTYPE)values[j];

                   #if (INT_QUANT_A == 1)
	                 quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
                   #else
	                 value_temp2 = value_temp;
                   #endif

	                A_fifo <<  value_temp2;
					col_indices_fifo << c;
					if (c == (ccount-1)) //column count
						c=0;
					else
						c++;
				}
		}

}

void readval_coo_adj(int beta_qu,int f_align,float quantization_scale_fea,bool sage_mode,bool linear_mode,bool gemm_mode,int ccount,int last_index,hls::stream<ATYPE> &A_fifo,hls::stream<int> &col_indices_fifo,INTYPE *values,int *columnIndex)
{

		#pragma HLS inline off

	    bool cgn_path = !(linear_mode^sage_mode);

	    if(cgn_path==1)
	    {
	     if (gemm_mode==0)
	     {
		  LOOP_J_SPMM : for (int j = 0; j < last_index; j++) {
			#pragma HLS PIPELINE

			INTYPE value_temp;
			ATYPE value_temp2;
			value_temp = (INTYPE)values[j];

            #if (INT_QUANT_A == 1)
	            quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
            #else
	            value_temp2 = value_temp;
            #endif

			A_fifo <<  value_temp2;
			col_indices_fifo << columnIndex[j];
		  }
		}
		else
		{
				int c=0;
				LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
				#pragma HLS PIPELINE

				   INTYPE value_temp;
				   ATYPE value_temp2;

				   value_temp = (INTYPE)values[j];

                   #if (INT_QUANT_A == 1)
	                 quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
                   #else
	                 value_temp2 = value_temp;
                   #endif

	                A_fifo <<  value_temp2;
					col_indices_fifo << c;
					if (c == (ccount-1)) //column count
						c=0;
					else
						c++;
				}
		 }
	    }

}

void readval_csr_adj2(int beta_qu,int f_align,float quantization_scale_fea,bool gemm_mode,int ccount,int last_index,hls::stream<ITYPE> &A_fifo,hls::stream<int> &col_indices_fifo,INTYPE *values,int *columnIndex)
{

		#pragma HLS inline off
        if (gemm_mode==0)
        {
		  LOOP_J_SPMM : for (int j = 0; j < last_index; j++) {
			#pragma HLS PIPELINE

			INTYPE value_temp;
			ATYPE value_temp2;
			value_temp = (INTYPE)values[j];

	        #if (INT_QUANT_A == 1)
		        quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
	        #else
		        value_temp2 = value_temp;
	        #endif

			A_fifo <<   (ITYPE)value_temp2;
			col_indices_fifo << columnIndex[j];
		 }
   	   }
	   else
	   {
				int c=0;
				LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
				#pragma HLS PIPELINE

					INTYPE value_temp;
					ATYPE value_temp2;
					value_temp = (INTYPE)values[j];

			        #if (INT_QUANT_A == 1)
				        quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
			        #else
				        value_temp2 = value_temp;
			        #endif

				    A_fifo <<   (ITYPE)value_temp2;
					col_indices_fifo << c;
					if (c == (ccount-1)) //column count
						c=0;
					else
						c++;
				}
	   }

}

void readval_coo_adj2(int beta_qu,int f_align,float quantization_scale_fea,bool sage_mode,bool linear_mode,bool gemm_mode,int ccount,int last_index,hls::stream<ITYPE> &A_fifo,hls::stream<int> &col_indices_fifo,INTYPE *values,int *columnIndex)
{

		#pragma HLS inline off

	   bool cgn_path = !(linear_mode^sage_mode);

	   if(cgn_path==1)
	   {
        if (gemm_mode==0)
        {
		  LOOP_J_SPMM : for (int j = 0; j < last_index; j++) {
			#pragma HLS PIPELINE

			INTYPE value_temp;
			ATYPE value_temp2;
			value_temp = (INTYPE)values[j];

	        #if (INT_QUANT == 1)
		        quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
	        #else
		        value_temp2 = value_temp;
	        #endif

			A_fifo <<   (ITYPE)value_temp2;
			col_indices_fifo << columnIndex[j];

		 }
   	   }
	   else
	   {
				int c=0;
				LOOP_J_SPMM2 : for (int j = 0; j < last_index; j++) {
				#pragma HLS PIPELINE

					INTYPE value_temp;
					ATYPE value_temp2;
					value_temp = (INTYPE)values[j];

			        #if (INT_QUANT_A == 1)
				        quanta(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu);
			        #else
				        value_temp2 = value_temp;
			        #endif

				    A_fifo <<   (ITYPE)value_temp2;

					col_indices_fifo << c;

					if (c == (ccount-1)) //column count
						c=0;
					else
						c++;
				}
	      }
	   }

}

void readval_coo_fea(int beta_qu,int f_align,int beta_qul,int f_alignl,
		float quantization_scale_fea[5],float quantization_scale_lin[5],bool sage_mode,bool linear_mode,bool stream_mode,bool gemm_mode,int ccount,int last_index,
		hls::stream<FTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,
		hls::stream<LTYPE> &A_fifo_sage,hls::stream<int> &col_indices_fifo_sage,
		INTYPE *values,hls::stream<ASTYPE>&  valuess,int *columnIndex,hls::stream<ASTYPE>&  columnIndex_feas, int B_index)
{

	#pragma HLS inline off

	std::cout << "gemm mode " <<  gemm_mode << std::endl;
	std::cout << "stream mode " <<  stream_mode << std::endl;
	std::cout << "read count " <<  last_index << std::endl;

    bool gcn_path = !(linear_mode^sage_mode);

	if (gemm_mode==0)
	{

    	fp_int C_float_int;

      	if(stream_mode==1)
      	{

            bool last_index1;
      		LOOP_J_SPMM11 : do{
  			#pragma HLS PIPELINE
  			INTYPE value_temp;
  		    FTYPE value_temp2;
  		    LTYPE value_temp3;

  		     ASTYPE temp = valuess.read();

     	     C_float_int.i = temp.data;

			 value_temp = (INTYPE)C_float_int.f;

			 temp = columnIndex_feas.read();

			 last_index1=temp.last;

			 int c = temp.data;

            #if (INT_QUANT_F == 1)
  			   quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
  			   quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
            #else
  			   value_temp2 = value_temp;
  			   value_temp3 = value_temp;
            #endif

		      if(gcn_path)
			  {

                A_fifo << value_temp2;

		        col_indices_fifo << c;
			  }

              #if (LINEAR_ENABLE == 1)

		      if(linear_mode)
		      {
		       col_indices_fifo_sage << c;
		       A_fifo_sage << value_temp3;
		      }
              #endif

      	  }while(last_index1==0);

      	}
      	else
      	{
		  LOOP_J_SPMM12 : for (int j = 0; j < last_index; j++) {
			#pragma HLS PIPELINE
			INTYPE value_temp;
		    FTYPE value_temp2;
		    LTYPE value_temp3;
		    int col_temp;

			  value_temp = (INTYPE)values[j];
			  col_temp = columnIndex[j];

            #if (INT_QUANT_F == 1)
			  quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
			  quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
            #else
			  value_temp2 = value_temp;
			  value_temp3 = value_temp;
            #endif

		    if(gcn_path)
			{
			  A_fifo << value_temp2;
			  col_indices_fifo << col_temp;
			}

            #if (LINEAR_ENABLE == 1)

		    if(linear_mode)
		    {
			 col_indices_fifo_sage << col_temp;
			 A_fifo_sage << value_temp3;
		    }

            #endif
		 }
      	}
	}
	else//gemm mode
	{

			int c=0;
        	fp_int C_float_int;

        	bool last_index1=0;

        	if(stream_mode==1) {

			LOOP_J_SPMM21 : for (int j = 0; j < last_index; j++) {

			#pragma HLS PIPELINE
				   INTYPE value_temp;
				   FTYPE value_temp2;
				   LTYPE value_temp3;

				    ASTYPE temp = valuess.read();

               	    C_float_int.i = temp.data;

			        value_temp = (INTYPE)C_float_int.f;

					last_index1=temp.last;

                   #if (INT_QUANT_F == 1)
				     quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
					 quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
                   #else
				     value_temp2 = value_temp;
				     value_temp3 = value_temp;
	                #endif

				 if(gcn_path)
				 {
				   A_fifo <<  value_temp2;
				   col_indices_fifo << c;
				 }

                #if (LINEAR_ENABLE == 1)
				if(linear_mode)
				{
				 col_indices_fifo_sage << c;
				 A_fifo_sage <<  value_temp3;
				}
				#endif
				if (c == (ccount-1)) //column count
					c=0;
				else
					c++;
			  }//while(last_index1==0);
        	}
			else
			{

			LOOP_J_SPMM22 : for (int j = 0; j < last_index; j++) {

			#pragma HLS PIPELINE
				   INTYPE value_temp;
				   FTYPE value_temp2;
				   LTYPE value_temp3;

			         value_temp = (INTYPE)values[j];

                   #if (INT_QUANT_F == 1)
				     quantf(value_temp2,value_temp,quantization_scale_fea,f_align,beta_qu,B_index);
					 quantl(value_temp3,value_temp,quantization_scale_lin,f_alignl,beta_qul,B_index);
                   #else
				     value_temp2 = value_temp;
	                #endif

				 if(gcn_path)
				 {
				  A_fifo <<  value_temp2;
				  col_indices_fifo << c;
				 }

                #if (LINEAR_ENABLE == 1)
				if(linear_mode)
				{
				 col_indices_fifo_sage << c;
				 A_fifo_sage <<  value_temp3;
				}
				#endif

				if (c == (ccount-1)) //column count
					c=0;
				else
					c++;
			}
		}
	}
}

void check_fifo_0(int a_values, hls::stream<ITYPE> &A_fifo, hls::stream<ITYPE> &A_fifo_out)
{
	ITYPE data_buffer;
	int data_count=0;
	bool loop_done = 0;
	bool data_in_buffer = 0; //data exits in data_buffer
	while((data_count < a_values) || data_in_buffer == 0)
	{
		#pragma HLS PIPELINE
			fifo_cycle_0++;
  	       	if (data_in_buffer == 0) //data_buffer empty
			{
				if(A_fifo.read_nb(data_buffer) == 1)
				{
					fifo_read_0++;
					data_count++;
					if(A_fifo_out.write_nb(data_buffer) == 0)
					{
						fifo_full_0++;
						data_in_buffer = 1; //fifo full and data stored in data_in_buffer
					}
					else
					{
						fifo_write_0++;
					}
				}
				else
				{
				}
			}
			else //data_buffer not empty
			{
				if (A_fifo_out.write_nb(data_buffer) == 1)
				{
					fifo_write_0++;
					if(A_fifo.read_nb(data_buffer) == 0)
					{
						data_in_buffer = 0; //data_buffer empty
					}
					else
					{
						fifo_read_0++;
						data_count++;
					}

				}
				else
				{
					fifo_full_0++;
				}
			}
	}

}

void check_fifo_2(int N, hls::stream<ITYPE> &C_fifo, hls::stream<ITYPE> &C_fifo_out)
{
	ITYPE data_buffer;
	int data_count=0;
	bool data_in_buffer= 0; //data exits in data_buffer

	while(data_count < N)
	{
		#pragma HLS PIPELINE
			fifo_cycle_2++;
  	       		if (data_in_buffer == 0) //data_buffer empty
			{
				if(C_fifo.read_nb(data_buffer) == 1)
				{

					fifo_read_2++;
					if(C_fifo_out.write_nb(data_buffer) == 0)
					{
						fifo_full_2++;
						data_in_buffer = 1; //fifo full and data stored in data_in_buffer
					}
					else
					{
						data_count++;
						fifo_write_2++;
					}
				}
				else
				{
					fifo_empty_2++;
				}

			}
			else //data_buffer not empty
			{
				if (C_fifo_out.write_nb(data_buffer) == 1)
				{
					fifo_write_2++;
					if(C_fifo.read_nb(data_buffer) == 0)
					{
						fifo_empty_2++;
						data_in_buffer = 0; //data_buffer empty
					}
					else
					{
						fifo_read_2++;
					}
					data_count++;

				}
				else
				{
					fifo_full_2++;
				}
			}
		   //} // j < B_WIDTH_INT
		//} //LOOP_CHECK_2
	} //while

}

void check_fifo_1(int N, int B_index, int B_index_loop, int tail, hls::stream<ITYPE> &C_fifo, hls::stream<ITYPE> &C_fifo_out)
{
	ITYPE data_buffer;
	int data_count=0;
	bool data_in_buffer= 0; //data exits in data_buffer

       

	while(data_count < N)
	{
		#pragma HLS PIPELINE
			fifo_cycle_1++;
  	       		if (data_in_buffer == 0) //data_buffer empty
			{
				if(C_fifo.read_nb(data_buffer) == 1)
				{

					fifo_read_1++;
					if(C_fifo_out.write_nb(data_buffer) == 0)
					{
						fifo_full_1++;
						data_in_buffer = 1; //fifo full and data stored in data_in_buffer
					}
					else
					{
						data_count++;
						fifo_write_1++;
					}
				}
				else
				{
				}

			}
			else //data_buffer not empty
			{
				if (C_fifo_out.write_nb(data_buffer) == 1)
				{
					fifo_write_1++;
					if(C_fifo.read_nb(data_buffer) == 0)
					{
						data_in_buffer = 0; //data_buffer empty
					}
					else
					{
						fifo_read_1++;
					}
					data_count++;

				}
				else
				{
					fifo_full_1++;
				}
			}
		   //} // j < B_WIDTH_INT
		//} //LOOP_CHECK_2
	} //while

}

void reada1_coo(int nnz_fea,int	beta_qu,int f_align,int	beta_qul,int f_alignl,float quantization_scale_fea[5],float quantization_scale_lin[5],
		int &last_index,ap_uint<1> model[5][8],int M, int first_row, int row_count,
		hls::stream<FTYPE> &A_fifo_fea,hls::stream<int> &col_indices_fifo_fea, hls::stream<int> &rnnz_fifo_fea,
		hls::stream<LTYPE> &A_fifo_fea_sage,hls::stream<int> &col_indices_fifo_fea_sage, hls::stream<int> &rnnz_fifo_fea_sage,
int *rowPtr_fea,int *columnIndex_fea,INTYPE *values_fea,
hls::stream<ASTYPE>&  rowPtr_feas,hls::stream<ASTYPE>&  columnIndex_feas,hls::stream<ASTYPE>&  values_feas,
int B_index, int layer_loop)
{

	int last_index_fea;
    bool gemm_mode,stream_mode,linear_mode,sage_mode;
    int M_int;

    gemm_mode = model[B_index][1];
    stream_mode = model[B_index][3];
	linear_mode = model[B_index][6];
	sage_mode = model[B_index][7];

    if (B_index == 0) //first layer
	      M_int = M;
	else
	      M_int = B_WIDTH_BLOCK; //in hidden layers the input width (number of features) is the B_WDITH BLOCK

	if (gemm_mode==0)
	{
	    columnIndex_fea += first_row;
	    values_fea += first_row;
	    rowPtr_fea += first_row;
	    last_index_fea = nnz_fea;
	}
	else
	{
		values_fea+=first_row*M_int;
		last_index_fea=row_count*M_int;
	}

	std::cout << "Last_index_fea " << last_index_fea << std::endl;

	readptr_coo_fea(nnz_fea,sage_mode,linear_mode,stream_mode,gemm_mode,row_count,M_int,rowPtr_fea,rowPtr_feas,rnnz_fifo_fea,rnnz_fifo_fea_sage);
	readval_coo_fea(beta_qu,f_align,beta_qul,f_alignl,quantization_scale_fea,quantization_scale_lin,sage_mode,linear_mode,stream_mode,gemm_mode,M_int,last_index_fea,
			A_fifo_fea,col_indices_fifo_fea,
			A_fifo_fea_sage,col_indices_fifo_fea_sage,
			values_fea,values_feas,columnIndex_fea,columnIndex_feas,B_index);

}

void reada2_csr(int beta_qu,int f_align,float quantization_scale_adj,bool gemm_mode,int M,int first_row, int row_count,  hls::stream<ATYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj_total_e, hls::stream<int> &rnnz_fifo_adj_total_s,hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj)
{

      //feature sparse matrix

	int last_index_adj;

	if (gemm_mode==0)
	{

	 last_index_adj=rowPtr_adj[first_row+row_count]-rowPtr_adj[first_row];

	 columnIndex_adj += rowPtr_adj[first_row];
	 values_adj += rowPtr_adj[first_row];
	 rowPtr_adj += first_row;
	}
	else
	{

		last_index_adj=row_count*M;
		values_adj+=first_row*M;
	}

	rnnz_fifo_adj_total_e << last_index_adj;
	rnnz_fifo_adj_total_s << last_index_adj;

        //feature sparse matrix

     //adjacency sparse matrix

	readptr_csr_adj(gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);

	readval_csr_adj(beta_qu,f_align,quantization_scale_adj,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

}

void reada2_coo(int nnz_adj,int beta_qu,int f_align,float quantization_scale_adj,ap_uint<1> model[5][8],int M,int first_row, int row_count,  hls::stream<ATYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj_total_e, hls::stream<int> &rnnz_fifo_adj_total_s,hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj,int B_index)
{

      //feature sparse matrix

	bool gemm_mode;

	gemm_mode = model[B_index][0];

	bool linear_mode;

	linear_mode =  model[B_index][6];

	bool gat_mode;

	gat_mode = model[B_index][5];

	bool sage_mode;

    sage_mode = model[B_index][7];

	int last_index_adj;

	if (gemm_mode==0)
	{

	 columnIndex_adj += rowPtr_adj[first_row];
	 values_adj += rowPtr_adj[first_row];
	 rowPtr_adj += first_row;
	 last_index_adj = nnz_adj;
	}
	else
	{

		values_adj+=first_row*M;
		last_index_adj = row_count*M;
	}

	if(gat_mode==1)
	{
	 rnnz_fifo_adj_total_e << nnz_adj;
	 rnnz_fifo_adj_total_s << nnz_adj;
	}

	readptr_coo_adj(nnz_adj,sage_mode,linear_mode,gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);

	readval_coo_adj(beta_qu,f_align,quantization_scale_adj,sage_mode,linear_mode,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

}

//CHECK THIS A_FIFO ITYPE should be ATYPE

void reada22_coo(int nnz_adj,int beta_qu,int f_align,float quantization_scale_adj,ap_uint<1> model[5][8],int M,int first_row, int row_count, hls::stream<ITYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj, int B_index)
{

	int last_index_adj;
	bool gemm_mode;

	gemm_mode = model[B_index][0];

	bool linear_mode;

	linear_mode =  model[B_index][6];

	bool sage_mode;

	sage_mode =  model[B_index][7];

	if (gemm_mode==0)
	{

	   columnIndex_adj += rowPtr_adj[first_row];
	   values_adj += rowPtr_adj[first_row];
	   rowPtr_adj += first_row;
	   last_index_adj = nnz_adj;
	}
	else
	{
		values_adj+=first_row*M;
		last_index_adj = row_count*M;
	}

        //feature sparse matrix

     //adjacency sparse matrix

	readptr_coo_adj(nnz_adj,sage_mode,linear_mode,gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);
	readval_coo_adj2(beta_qu,f_align,quantization_scale_adj,sage_mode,linear_mode,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

}

void reada22_csr(int beta_qu,int f_align,float quantization_scale_adj,bool gemm_mode,int M,int first_row, int row_count, hls::stream<ITYPE> &A_fifo_adj,hls::stream<int> &col_indices_fifo_adj, hls::stream<int> &rnnz_fifo_adj,
int *rowPtr_adj,int *columnIndex_adj,INTYPE *values_adj)
{

        //feature sparse matrix

	int last_index_adj;

	if (gemm_mode==0)
	{

       last_index_adj=rowPtr_adj[first_row+row_count]-rowPtr_adj[first_row];
	   columnIndex_adj += rowPtr_adj[first_row];
	   values_adj += rowPtr_adj[first_row];
	   rowPtr_adj += first_row;
	}
	else
	{
		last_index_adj=row_count*M;
		values_adj+=first_row*M;
	}

        //feature sparse matrix

     //adjacency sparse matrix

	readptr_csr_adj(gemm_mode,row_count,M,rowPtr_adj,rnnz_fifo_adj);
	readval_csr_adj2(beta_qu,f_align,quantization_scale_adj,gemm_mode,M,last_index_adj,A_fifo_adj,col_indices_fifo_adj,values_adj,columnIndex_adj);

}

void dsp_kernel_wrapper_adj_4(int block_size,int M,hls::stream<ITYPE> &A_fifo,hls::stream<int> &col_indices_fifo,QTYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],
		QTYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
		ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK])
{

#if defined FLOAT || defined HALF

FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
#pragma HLS ARRAY_PARTITION variable=acc_part complete

FTYPE acc_float[B_WIDTH_BLOCK];
    #pragma HLS ARRAY_PARTITION variable=acc_float complete

for (int j = 0; j < B_WIDTH_BLOCK; j++) {

	#pragma HLS UNROLL

		acc_float[j] = 0;
}

		RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
		#pragma HLS UNROLL
			for (int l = 0; l < FADD_LATENCY_ADJ; l++) {
			#pragma HLS UNROLL
				for (int z = 0; z < SPMM_BLOCK; z++){
					#pragma HLS UNROLL
						acc_part[l][j][z] = 0;
				}
		}
 	}

		  int BM = M[SPMM_BLOCK-1];

		  int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
		  M_aux[0] = 0;
		  for (int j = 1; j < SPMM_BLOCK+1; j++)
		  {
			 #pragma HLS UNROLL
			 M_aux[j] = M[j-1];
		  }
		  //print

		DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_ADJ) {
		#pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

		DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_ADJ; i++) {

			DTYPE v;
			int ci;
			if ((k+i) < BM) //avoid trying to read empty FIFO that only contains M elements
			{
				v = A_fifo.read();
				ci = col_indices_fifo.read();
			}
		        else
			{
				v=0;
				ci=0;
			}

			dsp_kernel_float_adj_4(block_size,v,b_block1,b_block2,b_block3,b_block4,ci,zero_point_lhs,zero_point_rhs,acc_float);

		        for (int j = 0; j < B_WIDTH_BLOCK; j++) {
		        	for (int z = 0; z < SPMM_BLOCK; z++)
		        	{
		        	 		#pragma HLS UNROLL
		        	  		if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
		        				        			acc_part[i][j][z] += acc_float[j];
		        	}
				#ifdef simulation
				if (acc_part[i][j] > max_adj)
							max_adj = acc_part[i][j];
						if (acc_part[i][j] < min_adj)
							min_adj = acc_part[i][j];
				#endif
			}

			      } //i loop

	} // k loop

for (int j = 0; j < B_WIDTH_BLOCK; j++) {
        #pragma HLS UNROLL
		for (int l = 1; l < FADD_LATENCY_ADJ; l++) {
		    for (int z = 0; z < SPMM_BLOCK; z++)
	        {
    		     acc_part[0][j][z] += acc_part[l][j][z];
	        }
		}
	}

   	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
   	        for (int z = 0; z < SPMM_BLOCK; z++)
   			{
			  #pragma HLS UNROLL
   			  FTYPE acc_part_float = acc_part[0][j][z];
   			  acc2[j][z] = acc_part_float;
   			}
	}

#endif

	#ifdef FIXEDPOINT

	ITYPE acc[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc complete

	 DSP_LOOP_SPMM: for (int i = 0; i < M; i++) {
		 	 #pragma HLS PIPELINE
			DTYPE v = A_fifo.read();

			int ci = col_indices_fifo.read();

			dsp_kernel_int_adj_4(block_size,v,b_block1,b_block2,
					b_block3,b_block4,
					ci,zero_point_lhs,zero_point_rhs,acc);

			for (int j = 0; j < B_WIDTH_BLOCK; j++) {

				#pragma HLS UNROLL
				acc2[j] += acc[j];
			}//j loop

	     	} //i loop

	#endif

}

void dsp_kernel_wrapper_adj_2(int block_size,int M[SPMM_BLOCK],hls::stream<ITYPE> &A_fifo,hls::stream<int> &col_indices_fifo,QTYPE b_block1[B_HEIGHT/4][B_WIDTH_BLOCK],
		QTYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],
		//ITYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
		ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK])
{

		#if defined FLOAT || defined HALF

	FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc_part complete

	FTYPE acc_float[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc_float complete

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

		#pragma HLS UNROLL

			acc_float[j] = 0;
	}

 		RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			#pragma HLS UNROLL
				for (int l = 0; l < FADD_LATENCY_ADJ; l++) {
				#pragma HLS UNROLL
					for (int z = 0; z < SPMM_BLOCK; z++){
								    acc_part[l][j][z] = 0;
				}
			}
	 	}

        int BM = M[SPMM_BLOCK-1];

 		 int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
 		 M_aux[0] = 0;
 	     for (int j = 1; j < SPMM_BLOCK+1; j++)
 		 {
 			#pragma HLS UNROLL
 			M_aux[j] = M[j-1];
 		 }
         //print

   		DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_ADJ) {
			#pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

			DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_ADJ; i++) {

				DTYPE v;
				int ci;
				if ((k+i) < BM) //avoid trying to read empty FIFO that only contains M elements
				{
					v = A_fifo.read();
					ci = col_indices_fifo.read();
				}
			        else
				{
					v=0;
					ci=0;
				}

				dsp_kernel_float_adj_2(block_size,v,b_block1,b_block2,ci,zero_point_lhs,zero_point_rhs,acc_float);

			        for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			          	for (int z = 0; z < SPMM_BLOCK; z++)
			          	{
			    		    #pragma HLS UNROLL
			    			if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
			    			  			acc_part[i][j][z] += acc_float[j];
			    			}//#pragma HLS UNROLL
					#ifdef simulation
					if (acc_part[i][j] > max_adj)
								max_adj = acc_part[i][j];
							if (acc_part[i][j] < min_adj)
								min_adj = acc_part[i][j];
					#endif
				}

    			      } //i loop

		} // k loop

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
			for (int l = 1; l < FADD_LATENCY_ADJ; l++) {
			    #pragma HLS unroll
				  for (int z = 0; z < SPMM_BLOCK; z++)
				  {
					acc_part[0][j][z] += acc_part[l][j][z];
				  }

			}
		}

       	for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
		        for (int z = 0; z < SPMM_BLOCK; z++)
		        {
				   FTYPE acc_part_float = acc_part[0][j][z];
				   acc2[j][z] = acc_part_float;
		        }
		}

	#endif

		#ifdef FIXEDPOINT

				ITYPE acc[B_WIDTH_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc complete

        int BM = M[SPMM_BLOCK-1];

		int M_aux[SPMM_BLOCK+1];
		M_aux[0] = 0;
		for (int j = 1; j < SPMM_BLOCK+1; j++)
		{
				#pragma HLS UNROLL
				M_aux[j] = M[j-1];
		}

		 DSP_LOOP_SPMM: for (int i = 0; i < BM; i+=1) {
 		 	 #pragma HLS PIPELINE
    				DTYPE v = A_fifo.read();

				int ci = col_indices_fifo.read();

				dsp_kernel_int_adj_2(block_size,v,b_block1,b_block2,
						//b_block3,b_block4,
						ci,zero_point_lhs,zero_point_rhs,acc);

				for (int j = 0; j < B_WIDTH_BLOCK; j++) {

					#pragma HLS UNROLL
					for (int z = 0; z < SPMM_BLOCK; z++)
					{
							#pragma HLS UNROLL
							if (i>=M_aux[z]&&i<M_aux[z+1])
									acc2[j][z] += acc[j];
					}//z loop
				}//j loop

		     	} //i loop

		#endif

}

void dsp_kernel_wrapper_adj_1(int block_size,int M,hls::stream<TTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,QTYPE b_block1[B_HEIGHT][B_WIDTH_BLOCK],
		//ITYPE b_block2[B_HEIGHT/4][B_WIDTH_BLOCK],
		//ITYPE b_block3[B_HEIGHT/4][B_WIDTH_BLOCK],ITYPE b_block4[B_HEIGHT/4][B_WIDTH_BLOCK],
		ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK])
{

		#if defined FLOAT || defined HALF

	FTYPE acc_part[FADD_LATENCY_ADJ][B_WIDTH_BLOCK][SPMM_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0 //partition all dimensions

	FTYPE acc_float[B_WIDTH_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc_float complete

	for (int j = 0; j < B_WIDTH_BLOCK; j++) {

		#pragma HLS UNROLL

			acc_float[j] = 0;
	}

 		RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			#pragma HLS UNROLL
				for (int l = 0; l < FADD_LATENCY_ADJ; l++) {
				#pragma HLS UNROLL
					for (int z = 0; z < SPMM_BLOCK; z++){
				    acc_part[l][j][z] = 0;
				}
			}
	 	}

        int BM = M[SPMM_BLOCK-1];

		 int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
		 M_aux[0] = 0;
	     for (int j = 1; j < SPMM_BLOCK+1; j++)
		 {
			#pragma HLS UNROLL
			M_aux[j] = M[j-1];
		 }
        //print

   		DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_ADJ) {
			#pragma HLS PIPELINE II=FADD_LATENCY_ADJ rewind

			DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_ADJ; i++) {

				DTYPE v;
				int ci;
				if ((k+i) < BM) //avoid trying to read empty FIFO that only contains M elements
				{

					v = A_fifo.read();
					ci = col_indices_fifo.read();
				}
			        else
				{
					v=0;
					ci=0;
				}

				dsp_kernel_float_adj_1(v,b_block1,ci,zero_point_lhs,zero_point_rhs,acc_float);

			        for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			        	for (int z = 0; z < SPMM_BLOCK; z++)
			            {
			        		#pragma HLS UNROLL
			        		if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
			        			acc_part[i][j][z] += acc_float[j];
			            }
					#ifdef simulation
					if (acc_part[i][j] > max_adj)
								max_adj = acc_part[i][j];
							if (acc_part[i][j] < min_adj)
								min_adj = acc_part[i][j];
					#endif
				}

    			      } //i loop

		} // k loop

	ACC_PART1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
            #pragma HLS UNROLL
		ACC_PART2 : for (int z = 0; z < SPMM_BLOCK; z++)
            {
				#pragma HLS UNROLL
			    ACC_PART3 : for (int l = 1; l < FADD_LATENCY_ADJ; l++) {
					#pragma HLS PIPELINE=1
	    		     acc_part[0][j][z] += acc_part[l][j][z];
		        }
			}
		}

	FLOAT_PART1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
                #pragma HLS UNROLL
		FLOAT_PART2 : for (int z = 0; z < SPMM_BLOCK; z++)
		        {
					#pragma HLS UNROLL
				   FTYPE acc_part_float = acc_part[0][j][z];
				   acc2[j][z] = acc_part_float;
		        }
		}

	#endif

		#ifdef FIXEDPOINT

       		ITYPE acc[B_WIDTH_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc complete

		 DSP_LOOP_SPMM: for (int i = 0; i < M; i++) {
 		 	 #pragma HLS PIPELINE
    				TTYPE v = A_fifo.read();

				int ci = col_indices_fifo.read();

				dsp_kernel_int_adj_1(block_size,v,b_block1,//b_block2,
						//b_block3,b_block4,
						ci,zero_point_lhs,zero_point_rhs,acc);

				for (int j = 0; j < B_WIDTH_BLOCK; j++) {
						acc2[j] += acc[j];
				}//j loop

		  } //i loop

		#endif

}

void dsp_kernel_wrapper_fea(bool gemm_mode,int M[SPMM_BLOCK],hls::stream<FTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,BTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK])
{

#if defined FLOAT || defined HALF

		ITYPE acc_part[FADD_LATENCY_FEA][B_WIDTH_BLOCK][SPMM_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0 //partition all dimensions

		ITYPE acc_float[B_WIDTH_BLOCK];
	    #pragma HLS ARRAY_PARTITION variable=acc_float complete

		for (int j = 0; j < B_WIDTH_BLOCK; j++) {

			#pragma HLS UNROLL

				acc_float[j] = 0;
		}

	 		RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
				#pragma HLS UNROLL
					 for (int l = 0; l < FADD_LATENCY_FEA; l++) {
					 #pragma HLS UNROLL
			     		for (int z = 0; z < SPMM_BLOCK; z++){
							#pragma HLS UNROLL
			     			acc_part[l][j][z] = 0;
				}
			  }
		 	}

	         int BM = M[SPMM_BLOCK-1];

			 int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
			 M_aux[0] = 0;
		     for (int j = 1; j < SPMM_BLOCK+1; j++)
			 {
				#pragma HLS UNROLL
				M_aux[j] = M[j-1];
			 }
	         //print

	   		DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_FEA) {
	   			#pragma HLS PIPELINE II=FADD_LATENCY_FEA

				DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_FEA; i++) {
					DTYPE v;
					int ci;
					if ((k+i) < BM) //avoid trying to read empty FIFO that only contains BM elements
					{
						v = A_fifo.read();
							ci = col_indices_fifo.read();
						//else
					}
				        else
					{
						v=0;
						ci=0;
					}

					dsp_kernel_float_fea(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc_float);

					SPMM_BLOCK_LOOP1 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
					  	#pragma HLS UNROLL
				        	SPMM_BLOCK_LOOP2 : for (int z = 0; z < SPMM_BLOCK; z++)
				        	{
							#pragma HLS PIPELINE II=1
				        	if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
				        			acc_part[i][j][z] += acc_float[j];
				        	}//z loop
					} //j loop

        	    } //i loop

			} // k loop

			ACC_PART1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
	            #pragma HLS UNROLL
				ACC_PART2: for (int z = 0; z < SPMM_BLOCK; z++)
			    {
					#pragma HLS UNROLL
				    ACC_PART3 : for (int l = 1; l < FADD_LATENCY_FEA; l++)
				    {
					    #pragma HLS PIPELINE II=1
		    		     acc_part[0][j][z] += acc_part[l][j][z];
		            }
				}
			}

			ACC_PART_FLOAT1 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
	            #pragma HLS UNROLL
				ACC_PART_FLOAT2 : for (int z = 0; z < SPMM_BLOCK; z++)
			    {
				#pragma HLS UNROLL
				FTYPE acc_part_float = acc_part[0][j][z];
		    		acc2[j][z] = acc_part_float;
			    }

			}

		#endif

#ifdef FIXEDPOINT

       	ITYPE acc[B_WIDTH_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS bind_op variable=acc op=add impl=dsp

         int BM = M[SPMM_BLOCK-1];

		 int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
		 M_aux[0] = 0;
	     for (int j = 1; j < SPMM_BLOCK+1; j++)
		 {
			#pragma HLS UNROLL
			M_aux[j] = M[j-1];
		 }

		 DSP_LOOP_SPMM: for (int i = 0; i < BM; i++) {
 		 	 #pragma HLS PIPELINE

    			FTYPE v = A_fifo.read();

    			int ci;
    			ci = col_indices_fifo.read();
    			//else

				dsp_kernel_int_fea(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc);

				for (int j = 0; j < B_WIDTH_BLOCK; j++) {

					#pragma HLS UNROLL
					for (int z = 0; z < SPMM_BLOCK; z++)
					{
						//critical #pragma HLS UNROLL
                        #pragma HLS UNROLL
						if (i>=M_aux[z]&&i<M_aux[z+1])
						 acc2[j][z] += acc[j];
					}//z loop

				 }//j loop

		     	} //i loop

		#endif

}

void dsp_kernel_wrapper_lin(bool gemm_mode,int M,hls::stream<LTYPE> &A_fifo,hls::stream<int> &col_indices_fifo,BLTYPE b_block[B_HEIGHT/4][B_WIDTH_BLOCK],ap_int<8> zero_point_lhs,ap_int<8> zero_point_rhs,ITYPE acc2[B_WIDTH_BLOCK])
{

#if defined FLOAT || defined HALF

		ITYPE acc_part[FADD_LATENCY_FEA][B_WIDTH_BLOCK][SPMM_BLOCK];
        #pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0 //partition all dimensions

		ITYPE acc_float[B_WIDTH_BLOCK];
	    #pragma HLS ARRAY_PARTITION variable=acc_float complete

		for (int j = 0; j < B_WIDTH_BLOCK; j++) {

			#pragma HLS UNROLL

				acc_float[j] = 0;
		}

	 		RESET_ACC_LOOP_SPMM: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
				#pragma HLS UNROLL
					 for (int l = 0; l < FADD_LATENCY_FEA; l++) {
					 #pragma HLS UNROLL
			     		for (int z = 0; z < SPMM_BLOCK; z++){
							#pragma HLS UNROLL
			     			acc_part[l][j][z] = 0;
				}
			  }
		 	}

	         int BM = M[SPMM_BLOCK-1];

			 int M_aux[SPMM_BLOCK+1]; //store the different number of nonzeros intervals
			 M_aux[0] = 0;
		     for (int j = 1; j < SPMM_BLOCK+1; j++)
			 {
				#pragma HLS UNROLL
				M_aux[j] = M[j-1];
			 }
	         //print

	   		DSP_LOOP_SPMM: for(int k = 0; k < BM; k+=FADD_LATENCY_FEA) {
	   			#pragma HLS PIPELINE II=FADD_LATENCY_FEA

				DSP_LOOP_SPMM2: for(int i = 0; i < FADD_LATENCY_FEA; i++) {
					DTYPE v;
					int ci;
					if ((k+i) < BM) //avoid trying to read empty FIFO that only contains BM elements
					{
						v = A_fifo.read();
							ci = col_indices_fifo.read();
						//else
					}
				        else
					{
						v=0;
						ci=0;
					}

					dsp_kernel_float_fea(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc_float);

					SPMM_BLOCK_LOOP1 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
					  	#pragma HLS UNROLL
				        	SPMM_BLOCK_LOOP2 : for (int z = 0; z < SPMM_BLOCK; z++)
				        	{
							#pragma HLS PIPELINE II=1
				        	if ((k+i)>=M_aux[z]&&(k+i)<M_aux[z+1])
				        			acc_part[i][j][z] += acc_float[j];
				        	}//z loop
					} //j loop

        	    } //i loop

			} // k loop

			ACC_PART1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
	            #pragma HLS UNROLL
				ACC_PART2: for (int z = 0; z < SPMM_BLOCK; z++)
			    {
					#pragma HLS UNROLL
				    ACC_PART3 : for (int l = 1; l < FADD_LATENCY_FEA; l++)
				    {
					    #pragma HLS PIPELINE II=1
		    		     acc_part[0][j][z] += acc_part[l][j][z];
		            }
				}
			}

			ACC_PART_FLOAT1 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
	            #pragma HLS UNROLL
				ACC_PART_FLOAT2 : for (int z = 0; z < SPMM_BLOCK; z++)
			    {
				#pragma HLS UNROLL
				FTYPE acc_part_float = acc_part[0][j][z];
		    		acc2[j][z] = acc_part_float;
			    }

			}

		#endif

#ifdef FIXEDPOINT

       	ITYPE acc[B_WIDTH_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS bind_op variable=acc op=add impl=dsp

		 DSP_LOOP_SPMM: for (int i = 0; i < M; i++) {
 		 	 #pragma HLS PIPELINE

    			LTYPE v = A_fifo.read();

    			int ci;
    				ci = col_indices_fifo.read();
    			//else

				dsp_kernel_int_lin(v,b_block,ci,zero_point_lhs,zero_point_rhs,acc);

				for (int j = 0; j < B_WIDTH_BLOCK; j++) {

					#pragma HLS UNROLL
					acc2[j] += acc[j];

				 }//j loop

		     	} //i loop

		#endif

}

void compute2_4(bool relu, float relu_t, int block_size,ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<ITYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo, QTYPE B_accel1[B_HEIGHT/2][B_WIDTH_BLOCK],QTYPE B_accel2[B_HEIGHT/2][B_WIDTH_BLOCK],
QTYPE B_accel3[B_HEIGHT/4][B_WIDTH_BLOCK],QTYPE B_accel4[B_HEIGHT/4][B_WIDTH_BLOCK],
hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK])
{

		ITYPE acc[B_WIDTH_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc complete

		ITYPE acc2[B_WIDTH_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

	      int B_WIDTH_INT;
	      ITYPE C_fifo_val;

		  B_WIDTH_INT = B_WIDTH_BLOCK;
	      //else

		for (int A_index = 0; A_index < row_count; A_index++) {

			//computing

			LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
				#pragma HLS UNROLL
					acc2[j] = 0;
				}

			int rnnz;
			rnnz = rnnz_fifo.read();

			dsp_kernel_wrapper_adj_4(block_size,rnnz,A_fifo,col_indices_fifo,B_accel1,
					B_accel2,
					B_accel3,B_accel4,
					zero_point_lhs,zero_point_rhs,acc2);

			LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
	                #pragma HLS UNROLL
					if (j < B_WIDTH_INT)
					{

						if (relu == 0)
							C_fifo_val = acc2[j];
						else
							if (acc2[j] < (ITYPE)relu_t)
							   C_fifo_val = 0.0;
							else
							   C_fifo_val = acc2[j];

						C_fifo[j].write(C_fifo_val);

					}
			   }

	          } // A_index loop

}

void compute2_2(int block_size,ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<ITYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> rnnz_fifo[SPMM_BLOCK], QTYPE B_accel1[B_HEIGHT/2][B_WIDTH_BLOCK],QTYPE B_accel2[B_HEIGHT/2][B_WIDTH_BLOCK],
		//ITYPE B_accel3[B_HEIGHT/4][B_WIDTH_BLOCK],ITYPE B_accel4[B_HEIGHT/4][B_WIDTH_BLOCK],
hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK][SPMM_BLOCK],int B_index, int B_index_loop, int tail)
{

		ITYPE acc[B_WIDTH_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc complete

		ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];
		#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

	      int B_WIDTH_INT;

	      if (B_index < (B_index_loop-1))
			B_WIDTH_INT = B_WIDTH_BLOCK;
	      else
			B_WIDTH_INT = tail;

		for (int A_index = 0; A_index < row_count; A_index+=SPMM_BLOCK) {

			//computing

			LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
				#pragma HLS UNROLL
				LOOP_ACC22 : for (int i = 0; i < SPMM_BLOCK; i++) {
					#pragma HLS UNROLL
					acc2[j][i] = 0;
				}
			}

			int rnnz[SPMM_BLOCK];
			int crows = 0;
			LOOP_RNNZ :for (int i = 0; i < SPMM_BLOCK; i++) {
	            #pragma HLS UNROLL
				rnnz[i] = rnnz_fifo[i].read();
				if ((A_index+i)<row_count)
				    crows++;

			}

			dsp_kernel_wrapper_adj_2(block_size,rnnz,A_fifo,col_indices_fifo,B_accel1,
					B_accel2,
					//B_accel3,B_accel4,
					zero_point_lhs,zero_point_rhs,acc2);

			LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
	                #pragma HLS UNROLL
					if (j < B_WIDTH_INT)
					{
						#ifdef simulation
						if (acc2[j] < acc2_adj_min)
							acc2_adj_min = acc2[j];
						else if (acc2[j] > acc2_adj_max)
							acc2_adj_max = acc2[j];
						#endif
						LOOP_C_BUF2 : for (int i = 0; i < SPMM_BLOCK; i++) {
							#pragma HLS UNROLL
							if (i<crows)
								C_fifo[j][i].write(acc2[j][i]);
						}

					}
			}

	          } // A_index loop

}

void compute2_1(ap_uint<1> model[5][8],float srelu[5],int block_size,ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<TTYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo, QTYPE B_accel1[B_HEIGHT/2][B_WIDTH_BLOCK],
		//ITYPE B_accel2[B_HEIGHT/2][B_WIDTH_BLOCK],
		//ITYPE B_accel3[B_HEIGHT/4][B_WIDTH_BLOCK],ITYPE B_accel4[B_HEIGHT/4][B_WIDTH_BLOCK],
hls::stream<ITYPE> C_fifo[B_WIDTH_BLOCK],int B_index)
{

	ITYPE acc[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc complete

	ITYPE acc2[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

      int B_WIDTH_INT;
      ITYPE C_fifo_val;

		B_WIDTH_INT = B_WIDTH_BLOCK;

    bool relu;

    relu = model[B_index][4];

    float relu_t = srelu[B_index];

    bool linear_mode;

	linear_mode = model[B_index][6];

    bool sage_mode;

	sage_mode = model[B_index][7];

	 bool gcn_path = !(linear_mode^sage_mode);

    if (gcn_path==1)
    {
	 for (int A_index = 0; A_index < row_count; A_index++) {

		LOOP_ACC21: for (int j = 0; j < B_WIDTH_BLOCK; j++) {
			#pragma HLS UNROLL
				acc2[j] = 0;
		}

		int rnnz;
		rnnz = rnnz_fifo.read();

		dsp_kernel_wrapper_adj_1(block_size,rnnz,A_fifo,col_indices_fifo,B_accel1,
				//B_accel2,
				//B_accel3,B_accel4,
				zero_point_lhs,zero_point_rhs,acc2);

		LOOP_C_BUF1: for (int j = 0; j < B_WIDTH_BLOCK; j++)
		{
                #pragma HLS UNROLL
				if (j < B_WIDTH_INT)
				{

					if(relu==0)
							C_fifo_val = acc2[j];
					else
						if(acc2[j] < (ITYPE)relu_t)
							C_fifo_val = 0.0;
						else
							C_fifo_val = acc2[j];

					C_fifo[j].write(C_fifo_val);
				}
		}

          } // A_index loop
    }

}

QTYPE8 float_to_fix(float f_in,int n_bits)
{
	float f=(1<<n_bits);
	QTYPE8 i_out = (f_in*f)*(1.0/f);
	return i_out;
}

void compute1_1(STYPE scale_fea[5],ITYPE* max_fea,int quantized_multiplier,ap_uint<1> model[5][8],ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs, int first_row, int row_count,hls::stream<FTYPE> &A_fifo, hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo,BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],QTYPE C_buf1[B_HEIGHT][B_WIDTH_BLOCK],
		//ITYPE C_buf2[B_HEIGHT][B_WIDTH_BLOCK],
		//ITYPE C_buf3[B_HEIGHT/4][B_WIDTH_BLOCK],ITYPE C_buf4[B_HEIGHT/4][B_WIDTH_BLOCK],
		QTYPE A_buf1[B_HEIGHT][B_WIDTH_BLOCK], int B_index)
{

	ITYPE acc[B_WIDTH_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc complete

	ITYPE acc2[B_WIDTH_BLOCK][SPMM_BLOCK];
	#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0 //all dimensions are partitioned

  

      int B_WIDTH_INT;

      		B_WIDTH_INT = B_WIDTH_BLOCK;

     bool gemm_mode;
	 gemm_mode = model[B_index][1];

     bool linear_mode;

	 linear_mode = model[B_index][6];

	 bool sage_mode;

	 sage_mode = model[B_index][7];

	 bool gcn_path = !(linear_mode^sage_mode);

	 if(gcn_path==1)
	 {

      for (int A_index = 0; A_index < row_count; A_index+=SPMM_BLOCK) {

         #pragma HLS dataflow

		//computing

    	LOOP_ACC21 :for (int j = 0; j < B_WIDTH_BLOCK; j++) {
    	      #pragma HLS UNROLL
    	    	LOOP_ACC22 : for (int i = 0; i < SPMM_BLOCK; i++) {
    	               #pragma HLS UNROLL
    	  		acc2[j][i] = 0;
    	  	}
    	}

    	int rnnz[SPMM_BLOCK];

    	rnnz[0] = rnnz_fifo.read();

    	int rnnz_temp;

		LOOP_RNNZ_SPMM :for (int i = 1; i < SPMM_BLOCK; i++) {
			#pragma HLS PIPELINE II=2
			if((A_index+i) < row_count)
			    rnnz_temp = rnnz_fifo.read();
			else
				rnnz_temp = 0;
		    rnnz[i] = rnnz_temp+rnnz[i-1];

			//else
		}

		dsp_kernel_wrapper_fea(gemm_mode,rnnz,A_fifo,col_indices_fifo,B_accel,zero_point_lhs,zero_point_rhs,acc2);

		LOOP_C_BUF1 : for (int j = 0; j < B_WIDTH_BLOCK; j++) {
	        #pragma HLS UNROLL
			if (j < B_WIDTH_INT)
			{
				#ifdef simulation
				if (acc2[j] < acc2_fea_min)
					acc2_fea_min = acc2[j];
				else if (acc2[j] > acc2_fea_max)
					acc2_fea_max = acc2[j];
				#endif
				LOOP_C_BUF2 : for (int i = 0; i < SPMM_BLOCK; i++) {
                    #pragma HLS UNROLL
					ITYPE cur_val = ITYPE(acc2[j][i]);

					*max_fea = 0;

					ap_fixed<32, 16>  acc2_temp_1 = acc2[j][i];

					QTYPE1 acc2_temp_1_1 = QTYPE1(acc2_temp_1 >> scale_fea[B_index]);
					QTYPE2 acc2_temp_1_2 = QTYPE2(acc2_temp_1 >> scale_fea[B_index]);
					QTYPE4 acc2_temp_1_4 = QTYPE4(acc2_temp_1 >> scale_fea[B_index]);
					QTYPE8 acc2_temp_1_8 = QTYPE8(acc2_temp_1 >> scale_fea[B_index]);
					QTYPE acc2_temp_1_16 = QTYPE(acc2_temp_1 >> scale_fea[B_index]);

                   #if GAT_ENABLE == 1

					 if(quantized_multiplier==1)
					 {
                             #if(qbits==1)
						      C_buf1[A_index+i][j].range(0,0)=acc2_temp_1_2[1]; //need to use range to avoid rounding otherwise a bit 1 becomes 0 since only 0 and -1 are possible in the target.
						      A_buf1[A_index+i][j].range(0,0)=acc2_temp_1_2[1];
                             #else
						      acc2_temp_1_2[0] = 1;
						      C_buf1[A_index+i][j]=acc2_temp_1_2;
						      A_buf1[A_index+i][j]=acc2_temp_1_2;
                             #endif
					 }
			         else if(quantized_multiplier==2)
					 {
							 C_buf1[A_index+i][j]=acc2_temp_1_2;
					         A_buf1[A_index+i][j]=acc2_temp_1_2;
					 }
					 else if(quantized_multiplier==4)
					 {
						     C_buf1[A_index+i][j]=acc2_temp_1_4;
			                 A_buf1[A_index+i][j]=acc2_temp_1_4;
					 }
			         else if(quantized_multiplier==8)
			         {

			        	     C_buf1[A_index+i][j]=acc2_temp_1_8;
					         A_buf1[A_index+i][j]=acc2_temp_1_8;
			         }
					 else
					 {
						    C_buf1[A_index+i][j]=acc2_temp_1_16;
				            A_buf1[A_index+i][j]=acc2_temp_1_16;
					 }

                   #else

					 if(quantized_multiplier==1)
					 {
                      #if(qbits==1)
	                     C_buf1[A_index+i][j]=acc2_temp_1_1;
                      #else
				         acc2_temp_1_2[0] = 1;
				     	 C_buf1[A_index+i][j]=acc2_temp_1_2;
                      #endif
					 }
					 else if(quantized_multiplier==2)
						 C_buf1[A_index+i][j]=acc2_temp_1_2;
					 else if(quantized_multiplier==4)
						 C_buf1[A_index+i][j]=acc2_temp_1_4;
					 else if(quantized_multiplier==8)
						 C_buf1[A_index+i][j]=acc2_temp_1_8;

					 else
					 {
						    C_buf1[A_index+i][j]=acc2_temp_1_16;

					 }

                    #endif

				} //c_buf loop2
			}	//c_buf loop1

		   } // j < B_WIDTH_BLOCK

          } // A_index loop
     } // linear_mode
         /

void func_rnnz(int i,int N_adj,hls::stream<ATYPE> &max_fifo,hls::stream<int> rnnz_fifo[SPMM_BLOCK],hls::stream<int>  rnnz_f[ATEN_BLOCK],hls::stream<ATYPE> val_f[ATEN_BLOCK])
{
  int rnnz_old = 0;
  int rnnz_val=0;
  ATYPE max_val;

  LOOP_RNNZ :for (int z = 0; z < ATEN_BLOCK; z++) {
		#pragma  HLS PIPELINE II=1

		if ((i+z) < N_adj)
		{
			rnnz_val = rnnz_fifo[0].read();
		    rnnz_f[z] << rnnz_val;
		    max_val = max_fifo.read();
		    val_f[z] << max_val;

			std::cout << "rnnz val 1 " << rnnz_val << std::endl;
		 }
		 else
		 {
			std::cout << "rnnz val 2 " << rnnz_val << std::endl;
		    rnnz_f[z] << rnnz_val;
		    val_f[z] << max_val;
		 }
  }
}

void func_exp(hls::stream<int> rnnz_f[ATEN_BLOCK],hls::stream<ATYPE> val_f[ATEN_BLOCK],hls::stream<FTYPE> &E_fifo,
hls::stream<ATYPE>  sum_f[ATEN_BLOCK],hls::stream<ATYPE>  val_f2[ATEN_BLOCK],hls::stream<int>  rnnz_f2[ATEN_BLOCK],
hls::stream<ATYPE>  &support_f)
{
    std::cout << "loop 2 " << std::endl;
  	int val_rnnz[ATEN_BLOCK+1];
  	ATYPE val_max[ATEN_BLOCK];
	ATYPE support;
	ATYPE attention_candidate;
    ATYPE sum[ATEN_BLOCK];

    val_rnnz[0]=0;
  	LOOP_1 :for (int z = 0; z < ATEN_BLOCK; z++)
  	{
			#pragma HLS UNROLL
  		    val_rnnz[z+1] = rnnz_f[z].read();
  		    val_max[z] = val_f[z].read();
  		    sum[z]=0;
    }

  	LOOP_SOFTMAX2 : for (int j = 0; j < val_rnnz[ATEN_BLOCK]; j++) {
  			 		int row_index1;
  	                #pragma HLS PIPELINE II=1
  				         attention_candidate = E_fifo.read();
  			  	         for(int k = 0; k<ATEN_BLOCK; k++)
  			    			  if ((j >= val_rnnz[k])&&(j < val_rnnz[k+1]))
  			    			  	 row_index1 = k;
  				         
  	                     #ifdef FIXEDPOINT
  				              support = hls::exp(attention_candidate- val_max[row_index1]);
  	                     #else
  				              support= hls::half_exp(attention_candidate- val_max[row_index1]);
  	                     #endif
  				         sum[row_index1] += support;
  				         support_f << support;

  	}

  	LOOP_2 :for (int z = 0; z < ATEN_BLOCK; z++)
  	{
       #pragma HLS UNROLL
  	   sum_f[z] << sum[z];
       val_f2[z] << val_max[z];
  	   rnnz_f2[z] << val_rnnz[z+1];
  	}

}

void func_fixed(int N_adj,hls::stream<ATYPE>  sum_f[ATEN_BLOCK],hls::stream<ATYPE>  val_f2[ATEN_BLOCK],hls::stream<int>  rnnz_f2[ATEN_BLOCK],
hls::stream<ATYPE>  sum_f2[ATEN_BLOCK],hls::stream<int>  rnnz_f3[ATEN_BLOCK])
{
	 std::cout << "loop 3 " << std::endl;
     int val_rnnz2[ATEN_BLOCK+1];
     ATYPE val_max2[ATEN_BLOCK];
     ATYPE val_sum[ATEN_BLOCK];
  	 ATYPE fixed_val = ATYPE(-9e3);
 	 ATYPE fixed_support;

     val_rnnz2[0] = 0;
  	 LOOP_3 :for (int z = 0; z < ATEN_BLOCK; z++)
  	 {
             #pragma HLS UNROLL
             val_rnnz2[z+1] = rnnz_f2[z].read();
             val_max2[z] = val_f2[z].read();
             val_sum[z] =  sum_f[z].read();
  	 }

   	 LOOP_FIXED :for (int z = 0; z < ATEN_BLOCK; z++) {
           #pragma HLS PIPELINE II=1
   	  	   ATYPE sum_local;
   	       #ifdef FIXEDPOINT
   			     fixed_support = hls::exp(fixed_val- val_max2[z]);
   	       #else
   			     fixed_support = hls::half_exp(fixed_val- val_max2[z]);
   	       #endif
   		   fixed_support = (N_adj-val_rnnz2[z+1]+val_rnnz2[z])*fixed_support;
   		   sum_local = val_sum[z]+fixed_support;
   		   sum_f2[z] << sum_local;

   		}
   	LOOP_4 :for (int z = 0; z < ATEN_BLOCK; z++)
   	{
           #pragma HLS UNROLL
   	       rnnz_f3[z] << val_rnnz2[z+1];
   	}

}

void func_div(hls::stream<int> rnnz_att_fifo[SPMM_BLOCK],hls::stream<ATYPE> &A_fifo,hls::stream<ATYPE>  &support_f,
hls::stream<int> &col_indices_fifo,hls::stream<ATYPE> sum_f2[ATEN_BLOCK],hls::stream<int>  rnnz_f3[ATEN_BLOCK],
hls::stream<ATYPE> &val_att_fifo,hls::stream<int> &col_att_fifo)
{
	int col;
	ATYPE val;
	int rnnz_old = 0;
	int rnnz_val=0;

    std::cout << "loop 4 " << std::endl;
 	int val_rnnz3[ATEN_BLOCK+1];
 	ATYPE val_sum2[ATEN_BLOCK];

    val_rnnz3[0]=0;
	LOOP_5 :for (int z = 0; z < ATEN_BLOCK; z++)
	{
       #pragma HLS UNROLL
 	  rnnz_val = rnnz_f3[z].read();
 	  val_rnnz3[z+1] = rnnz_val;
 	  val_sum2[z] = sum_f2[z].read();
 	  rnnz_att_fifo[0] << rnnz_val-rnnz_old;
 	  rnnz_old = rnnz_val;
    }

	LOOP_SOFTMAX4 : for (int j = 0; j < val_rnnz3[ATEN_BLOCK]; j++) {
	  		 int row_index2;
	 		 #pragma HLS PIPELINE II=1
	  	     for(int k = 0; k<ATEN_BLOCK; k++)
	    	   if ((j >= val_rnnz3[k])&&(j < val_rnnz3[k+1]))
	    		  	 row_index2 = k;

	         
	    	  val = A_fifo.read();
		      col = col_indices_fifo.read();
			  col_att_fifo << col;

			  ATYPE out_val = support_f.read()/val_sum2[row_index2];

	    	   val_att_fifo << out_val;

	}
}

#ifdef func_loops

void compute_attention2(bool gat_mode,int N_adj,
		hls::stream<ATYPE> &A_fifo,
		hls::stream<int> &col_indices_fifo, hls::stream<int> rnnz_fifo[SPMM_BLOCK],hls::stream<ATYPE> &E_fifo,
		hls::stream<ATYPE> &max_fifo,hls::stream<ATYPE> &val_att_fifo,hls::stream<int> &col_att_fifo,hls::stream<int> rnnz_att_fifo[SPMM_BLOCK])

    	hls::stream<int>  rnnz_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f depth=FIFO_DEPTH dim=1

    	hls::stream<int>  rnnz_f2[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f2 depth=FIFO_DEPTH dim=1

    	hls::stream<int>  rnnz_f3[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f3 depth=FIFO_DEPTH dim=1

    	hls::stream<ATYPE>  val_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= val_f depth=FIFO_DEPTH dim=1

    	hls::stream<ATYPE>  val_f2[ATEN_BLOCK];
    	#pragma HLS STREAM variable= val_f2 depth=FIFO_DEPTH dim=1

    	hls::stream<ATYPE>  sum_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= sum_f depth=FIFO_DEPTH dim=1

    	hls::stream<ATYPE>  sum_f2[ATEN_BLOCK];
       	#pragma HLS STREAM variable= sum_f2 depth=FIFO_DEPTH dim=1

       	hls::stream<ATYPE>  support_f;
        #pragma HLS STREAM variable= support_f depth=FIFO_DEPTH_ATTN2 dim=1

    	 if (gat_mode==1)
      	 {

    	   ATEN_LOOP:for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    	   {

                   #pragma HLS DATAFLOW
    		        func_rnnz(i,N_adj,max_fifo,rnnz_fifo,rnnz_f,val_f);

    		        func_exp(rnnz_f,val_f,E_fifo,sum_f,val_f2,rnnz_f2,support_f);

    		        func_fixed(N_adj,sum_f,val_f2,rnnz_f2,sum_f2,rnnz_f3);

    		        func_div(rnnz_att_fifo,A_fifo,support_f,col_indices_fifo,sum_f2,rnnz_f3,val_att_fifo,col_att_fifo);
    	   }

      	 }
    	 else
    	  {
    		     int col;
    		     ATYPE val;

    	         LOOP_GCN : for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    		     {

    	             #pragma HLS DATAFLOW
    		         int rnnz_old=0;
					 int rnnz_val;
    			     LOOP_RNNZ2 :for (int z = 0; z < ATEN_BLOCK; z++) {
						#pragma HLS PIPELINE II=1
    			 		if ((i+z) < N_adj)
    			 		{
    			          rnnz_val = rnnz_fifo[0].read();
    			          ATYPE max_val = max_fifo.read();
    			          rnnz_att_fifo[0] << rnnz_val-rnnz_old;
    			          rnnz_old = rnnz_val;
    			 		}
    			 		//else
    			     }
    			 	 LOOP_SOFTMAX5 : for (int j = 0; j < rnnz_old; j++) {
    			         #pragma HLS PIPELINE II=1
    			    	   val = A_fifo.read();
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;
    			    	   val_att_fifo << val;
    			 	 }
    			    }
    	}

}

#endif

#ifndef func_loops

// good 218K cycles with ATEN BLOCK 1

void compute_attention2(ap_uint<1> model[5][8],int N_adj,
		hls::stream<ATYPE> &A_fifo,
		hls::stream<int> &col_indices_fifo, hls::stream<int> &rnnz_fifo,hls::stream<TTYPE> &E_fifo,
		hls::stream<TTYPE> &max_fifo,hls::stream<TTYPE> &val_att_fifo,hls::stream<int> &col_att_fifo,hls::stream<int> &rnnz_att_fifo,hls::stream<TTYPE> &SO_fifo,hls::stream<int> &SO_rnnz_fifo,int B_index)
{
		int col;
		TTYPE val,attention_candidate;
		TTYPE fixed_val = TTYPE(-9e3);
	    TTYPE sum[ATEN_BLOCK];
    	TTYPE support;
    	TTYPE fixed_support;
    	TTYPE div_val;
    	TTYPE const_one = TTYPE(1);

    	hls::stream<TTYPE>  support_f;
    	#pragma HLS STREAM variable= support_f depth= FIFO_DEPTH_ATTN2
        #pragma HLS bind_storage variable = support_f type=FIFO impl=URAM

    	hls::stream<int>  rnnz_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f depth=FIFO_DEPTH

    	hls::stream<int>  rnnz_f2[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f2 depth=FIFO_DEPTH

    	hls::stream<int>  rnnz_f3[ATEN_BLOCK];
    	#pragma HLS STREAM variable= rnnz_f3 depth=FIFO_DEPTH

    	hls::stream<TTYPE>  val_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= val_f depth=FIFO_DEPTH

    	hls::stream<TTYPE>  val_f2[ATEN_BLOCK];
    	#pragma HLS STREAM variable= val_f2 depth=FIFO_DEPTH

    	hls::stream<TTYPE>  sum_f[ATEN_BLOCK];
    	#pragma HLS STREAM variable= sum_f depth=FIFO_DEPTH

    	hls::stream<TTYPE>  sum_f2[ATEN_BLOCK];
       	#pragma HLS STREAM variable= sum_f2 depth=FIFO_DEPTH

         bool gat_mode = model[B_index][5];

         bool linear_mode = model[B_index][6];
 		 bool sage_mode = model[B_index][7];
 		 bool gcn_path = !(linear_mode^sage_mode);

    	 if (gcn_path==1)
      	 {

    	  if (gat_mode==1)
      	  {

    	   ATEN_LOOP:for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    	   {

                   #pragma HLS DATAFLOW
    		       int rnnz_old = 0;
    		       ITYPE max_old = 0;
    		       LOOP_RNNZ :for (int z = 0; z < ATEN_BLOCK; z++) {
    		           #pragma  HLS PIPELINE II=1
    		    	   int rnnz_val;
    		    	   ITYPE max_val;
    		  		   if ((i+z) < N_adj)
    		  		   {
    		        	  rnnz_val = rnnz_fifo.read();
    		        	  rnnz_f[z] << rnnz_val;
    		        	  max_val = max_fifo.read();
    		        	  val_f[z] << max_val;
			        	  rnnz_att_fifo << rnnz_val-rnnz_old;
			        	  rnnz_old = rnnz_val;
			        	  max_old = max_val;

    		  		   }
    		  		   else
    		  		   {
    		  			 rnnz_f[z] << rnnz_old; //rnnz_val;
    		        	 val_f[z] << max_old; // max_val;
    		  		   }

    		       	 }

    		       	  int val_rnnz[4];
    		       	  ATYPE val_max[4];
    		       	  LOOP_1 :for (int z = 0; z < ATEN_BLOCK; z++)
    		       	  {
						#pragma HLS UNROLL
    		            val_rnnz[z] = rnnz_f[z].read();
    		            val_max[z] = val_f[z].read();
    		            sum[z] = 0;

    		       	  }

    			 	  LOOP_SOFTMAX2 : for (int j = 0; j < val_rnnz[ATEN_BLOCK-1]; j++) {
    			 		int row_index1;
    	                #pragma HLS PIPELINE II=1
    				         attention_candidate = E_fifo.read();
    				         if(j < val_rnnz[0])
    				        	 row_index1 = 0;
    				         else if (j < val_rnnz[1])
    	    				     row_index1 = 1;
    				         else if (j < val_rnnz[2])
        	    				 row_index1 = 2;
    				         else
        	    				 row_index1 = 3;
    	                     #ifdef FIXEDPOINT
    				              support = hls::exp(attention_candidate- val_max[row_index1]);
    	                     #else
    				              support= hls::half_exp(attention_candidate- val_max[row_index1]);
    	                     #endif
    				         sum[row_index1] += support;
    				         support_f << support;

    					 }

    		     	  LOOP_2 :for (int z = 0; z < ATEN_BLOCK; z++)
    		     	  {
                       #pragma HLS UNROLL
    			 	   sum_f[z] << sum[z];
       			 	   val_f2[z] << val_max[z];
    			 	   rnnz_f2[z] << val_rnnz[z];
    		     	  }

        		     int val_rnnz2[ATEN_BLOCK+1];
        		     TTYPE val_max2[ATEN_BLOCK];
        			 TTYPE val_sum[ATEN_BLOCK];

        		     val_rnnz2[0] = 0;
   		     	     LOOP_3 :for (int z = 0; z < ATEN_BLOCK; z++)
   		     	     {
                         #pragma HLS UNROLL
       			         val_rnnz2[z+1] = rnnz_f2[z].read();
       			         val_max2[z] = val_f2[z].read();
                         val_sum[z] =  sum_f[z].read();
   		     	     }

    			      LOOP_FIXED :for (int z = 0; z < ATEN_BLOCK; z++) {
    	  			  TTYPE sum_local;
                      #pragma HLS PIPELINE II=1
    	              #ifdef FIXEDPOINT
    			 	        fixed_support = hls::exp(fixed_val- val_max2[z]);
    	              #else
    			 	        fixed_support = hls::half_exp(fixed_val- val_max2[z]);
    	              #endif
    			 	  fixed_support = (N_adj-val_rnnz2[z+1]+val_rnnz2[z])*fixed_support;
    			      sum_local = val_sum[z]+fixed_support;
    			      sum_f2[z] << sum_local;

    			      }
    		     	  LOOP_4 :for (int z = 0; z < ATEN_BLOCK; z++)
    	   		      {
                         #pragma HLS UNROLL
    			 	     rnnz_f3[z] << val_rnnz2[z+1];
    	   		      }

    		    	int val_rnnz3[4];
    		    	TTYPE val_sum2[4];
   	     	        LOOP_5 :for (int z = 0; z < ATEN_BLOCK; z++)
      		     	{
                      #pragma HLS UNROLL
    		    	  val_rnnz3[z] = rnnz_f3[z].read();
    		    	  val_sum2[z] = sum_f2[z].read();
      		     	}

    			 	 LOOP_SOFTMAX4 : for (int j = 0; j < val_rnnz3[ATEN_BLOCK-1]; j++) {
    			  		 int row_index2;
    			 		 #pragma HLS PIPELINE II=1
    			         if(j < val_rnnz3[0])
    		    		      	 row_index2 = 0;
    		    		 else if (j < val_rnnz3[1])
    		    	           row_index2 = 1;
    		    		 else if (j < val_rnnz3[2])
    		        	  	 row_index2 = 2;
    		    		 else
    		        	     row_index2 = 3;
    			    	   val = (ATYPE)(A_fifo.read());
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;

    					   TTYPE out_val = support_f.read()/val_sum2[row_index2];
    			    	   //val_att_fifo << (ATYPE)(out_val); //cast to A type NO CASTING KEEP PRECISON FOR ATTENTION
    			    	   val_att_fifo << out_val; //NO CASTING KEEP PRECISON FOR ATTENTION
    			    	   //val_att_fifo << out_val; //cast to A type
    			    	   SO_fifo << out_val;

    			 	 }
    		       }
      	 }
    	 else

    	  {

    	         LOOP_GCN : for (int i = 0; i < N_adj; i+=ATEN_BLOCK)
    		     {

    	             #pragma HLS DATAFLOW
    		         int rnnz[ATEN_BLOCK+1];
    			     rnnz[0]=0;
    			     int crows = 0;
    			     LOOP_RNNZ2 :for (int z = 0; z < ATEN_BLOCK; z++) {
						#pragma HLS PIPELINE II=1
    			        rnnz[z+1] = rnnz_fifo.read();
    			        rnnz_att_fifo << rnnz[z+1]-rnnz[z];
    			       	if ((z+i)<N_adj)
    			       	  crows++;
    			     }
    			 	 LOOP_SOFTMAX5 : for (int j = 0; j < rnnz[ATEN_BLOCK]; j++) {
    			         #pragma HLS PIPELINE II=1
    			    	   val = A_fifo.read();
    				       col = col_indices_fifo.read();
    					   col_att_fifo << col;
    			    	   val_att_fifo << (TTYPE)(val);
    			 	 }
    			    }
    		}

         } //linear mode

}
#endif

void loop_attention(float deq_factor[5],int beta_qu,int f_align,float quantization_scale_adj,float quantization_scale_w[5],
		ap_uint<1> model[5][8],
		int nnz_adj1,int nnz_adj2,int nnz_adj3,int nnz_adj4,
		int * rowPtr_adj1,int * rowPtr_adj2,int * rowPtr_adj3,int * rowPtr_adj4,
		int *columnIndex_adj1, int *columnIndex_adj2, int *columnIndex_adj3, int *columnIndex_adj4,
		INTYPE *values_adj1, 	INTYPE *values_adj2,	INTYPE *values_adj3,	INTYPE *values_adj4,
		int N_adj, int M_adj, ap_uint<8> P_w[5],
		INTYPE *A,
        #if(PIPO_BLOCKS>=2)
		 hls::stream_of_blocks<buf> &A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
        #else
		 buf A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
        #endif
		hls::stream_of_blocks<buf> &A_buffer31,hls::stream_of_blocks<buf> &A_buffer41,
		OUTTYPE* E1,
		OUTTYPE* S1,
		hls::stream<int> &rnnz_att_fifo1,hls::stream<int> &col_att_fifo1,hls::stream<TTYPE> &val_att_fifo1,
		hls::stream<int> &rnnz_att_fifo2,hls::stream<int> &col_att_fifo2,hls::stream<TTYPE> &val_att_fifo2,
		hls::stream<int> &rnnz_att_fifo3,hls::stream<int> &col_att_fifo3,hls::stream<TTYPE> &val_att_fifo3,
		hls::stream<int> &rnnz_att_fifo4,hls::stream<int> &col_att_fifo4,hls::stream<TTYPE> &val_att_fifo4,
		int layer_loop)

{

        

         hls::stream<TTYPE>   EO_fifo1("EO fifo1");
         #pragma HLS STREAM variable=EO_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_fifo1 type=FIFO impl=URAM
         hls::stream<int>   EO_rnnz_fifo1("EO rnnz fifo1");;
         #pragma HLS STREAM variable=EO_rnnz_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_rnnz_fifo1 type=FIFO impl=URAM
         hls::stream<TTYPE>   SO_fifo1("SO fifo1");
         #pragma HLS STREAM variable=SO_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_fifo1 type=FIFO impl=URAM
         hls::stream<int>   SO_rnnz_fifo1("SO rnnz fifo1");;
         #pragma HLS STREAM variable=SO_rnnz_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_rnnz_fifo1 type=FIFO impl=URAM

         hls::stream<TTYPE>   EO_fifo2("EO fifo2");
         #pragma HLS STREAM variable=EO_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_fifo2 type=FIFO impl=URAM
         hls::stream<int>   EO_rnnz_fifo2("EO rnnz fifo2");;
         #pragma HLS STREAM variable=EO_rnnz_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_rnnz_fifo2 type=FIFO impl=URAM
         hls::stream<TTYPE>   SO_fifo2("SO fifo2");
         #pragma HLS STREAM variable=SO_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_fifo2 type=FIFO impl=URAM
         hls::stream<int>   SO_rnnz_fifo2("SO rnnz fifo2");;
         #pragma HLS STREAM variable=SO_rnnz_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_rnnz_fifo2 type=FIFO impl=URAM

         hls::stream<TTYPE>   EO_fifo3("EO fifo3");
         #pragma HLS STREAM variable=EO_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_fifo3 type=FIFO impl=URAM
         hls::stream<int>   EO_rnnz_fifo3("EO rnnz fifo3");;
         #pragma HLS STREAM variable=EO_rnnz_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_rnnz_fifo3 type=FIFO impl=URAM
         hls::stream<TTYPE>   SO_fifo3("SO fifo3");
         #pragma HLS STREAM variable=SO_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_fifo3 type=FIFO impl=URAM
         hls::stream<int>   SO_rnnz_fifo3("SO rnnz fifo3");;
         #pragma HLS STREAM variable=SO_rnnz_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_rnnz_fifo3 type=FIFO impl=URAM

         hls::stream<TTYPE>   EO_fifo4("EO fifo4");
         #pragma HLS STREAM variable=EO_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_fifo4 type=FIFO impl=URAM
         hls::stream<int>   EO_rnnz_fifo4("EO rnnz fifo4");;
         #pragma HLS STREAM variable=EO_rnnz_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = EO_rnnz_fifo4 type=FIFO impl=URAM
         hls::stream<TTYPE>   SO_fifo4("SO fifo4");
         #pragma HLS STREAM variable=SO_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_fifo4 type=FIFO impl=URAM
         hls::stream<int>   SO_rnnz_fifo4("SO rnnz fifo4");;
         #pragma HLS STREAM variable=SO_rnnz_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = SO_rnnz_fifo4 type=FIFO impl=URAM

         hls::stream<TTYPE>   max_fifo1("max_fifo1");
         #pragma HLS STREAM variable=max_fifo1 depth=FIFO_DEPTH
         hls::stream<TTYPE>   E_fifo1("E fifo1");
         #pragma HLS STREAM variable=E_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_fifo1 type=FIFO impl=URAM
         hls::stream<ATYPE>   A_fifo1("A fifo1");
         //This seems the adj size which is problematic to store
         #pragma HLS STREAM variable=A_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = A_fifo1 type=FIFO impl=URAM
         hls::stream<int>  E_col_indices_fifo1("E col fifo1");
         #pragma HLS STREAM variable=E_col_indices_fifo1 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_col_indices_fifo1 type=FIFO impl=URAM
         hls::stream<int> E_rnnz_fifo1;
         #pragma HLS STREAM variable=E_rnnz_fifo1 depth=FIFO_DEPTH

         hls::stream<TTYPE>   max_fifo2("max_fifo2");
         #pragma HLS STREAM variable=max_fifo2 depth=FIFO_DEPTH
         hls::stream<TTYPE>   E_fifo2("E fifo2");
         #pragma HLS STREAM variable=E_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_fifo2 type=FIFO impl=URAM
         hls::stream<ATYPE>   A_fifo2("A fifo2");
         //This seems the adj size which is problematic to store
         #pragma HLS STREAM variable=A_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = A_fifo2 type=FIFO impl=URAM
         hls::stream<int>  E_col_indices_fifo2("E col fifo2");
         #pragma HLS STREAM variable=E_col_indices_fifo2 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_col_indices_fifo2 type=FIFO impl=URAM
         hls::stream<int> E_rnnz_fifo2;
         #pragma HLS STREAM variable=E_rnnz_fifo2 depth=FIFO_DEPTH

         hls::stream<TTYPE>   max_fifo3("max_fifo3");
         #pragma HLS STREAM variable=max_fifo3 depth=FIFO_DEPTH
         hls::stream<TTYPE>   E_fifo3("E fifo3");
         #pragma HLS STREAM variable=E_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_fifo3 type=FIFO impl=URAM
         hls::stream<ATYPE>   A_fifo3("A fifo3");
         //This seems the adj size which is problematic to store
         #pragma HLS STREAM variable=A_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = A_fifo3 type=FIFO impl=URAM
         hls::stream<int>  E_col_indices_fifo3("E col fifo3");
         #pragma HLS STREAM variable=E_col_indices_fifo3 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_col_indices_fifo3 type=FIFO impl=URAM
         hls::stream<int> E_rnnz_fifo3;
         #pragma HLS STREAM variable=E_rnnz_fifo3 depth=FIFO_DEPTH

         hls::stream<TTYPE>   max_fifo4("max_fifo4");
         #pragma HLS STREAM variable=max_fifo4 depth=FIFO_DEPTH
         hls::stream<TTYPE>   E_fifo4("E fifo4");
         #pragma HLS STREAM variable=E_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_fifo4 type=FIFO impl=URAM
         hls::stream<ATYPE>   A_fifo4("A fifo4");
         //This seems the adj size which is problematic to store
         #pragma HLS STREAM variable=A_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = A_fifo4 type=FIFO impl=URAM
         hls::stream<int>  E_col_indices_fifo4("E col fifo4");
         #pragma HLS STREAM variable=E_col_indices_fifo4 depth=FIFO_DEPTH_ATTN
         #pragma HLS bind_storage variable = E_col_indices_fifo4 type=FIFO impl=URAM
         hls::stream<int> E_rnnz_fifo4;
         #pragma HLS STREAM variable=E_rnnz_fifo4 depth=FIFO_DEPTH

         hls::stream<int> rnnz_fifo_adj1;
         #pragma HLS STREAM variable=rnnz_fifo_adj1 depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj1_total_e;
         #pragma HLS STREAM variable=rnnz_fifo_adj1_total_e depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj1_total_s;
         #pragma HLS STREAM variable=rnnz_fifo_adj1_total_s depth=FIFO_DEPTH
         hls::stream<ATYPE> A_fifo_adj1("A fifo adj1");
         #pragma HLS STREAM variable=A_fifo_adj1 depth=FIFO_DEPTH
         hls::stream<int>  col_indices_fifo_adj1("col fifo1");
         #pragma HLS STREAM variable=col_indices_fifo_adj1 depth=FIFO_DEPTH

         hls::stream<int> rnnz_fifo_adj2;
         #pragma HLS STREAM variable=rnnz_fifo_adj2 depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj2_total_e;
         #pragma HLS STREAM variable=rnnz_fifo_adj2_total_e depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj2_total_s;
         #pragma HLS STREAM variable=rnnz_fifo_adj2_total_s depth=FIFO_DEPTH
         hls::stream<ATYPE> A_fifo_adj2("A fifo adj2");
         #pragma HLS STREAM variable=A_fifo_adj2 depth=FIFO_DEPTH
         hls::stream<int>  col_indices_fifo_adj2("col fifo2");
         #pragma HLS STREAM variable=col_indices_fifo_adj2 depth=FIFO_DEPTH

         hls::stream<int> rnnz_fifo_adj3;
         #pragma HLS STREAM variable=rnnz_fifo_adj3 depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj3_total_e;
         #pragma HLS STREAM variable=rnnz_fifo_adj3_total_e depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj3_total_s;
         #pragma HLS STREAM variable=rnnz_fifo_adj3_total_s depth=FIFO_DEPTH
         hls::stream<ATYPE> A_fifo_adj3("A fifo adj3");
         #pragma HLS STREAM variable=A_fifo_adj3 depth=FIFO_DEPTH
         hls::stream<int>  col_indices_fifo_adj3("col fifo3");
         #pragma HLS STREAM variable=col_indices_fifo_adj3 depth=FIFO_DEPTH

         hls::stream<int> rnnz_fifo_adj4;
         #pragma HLS STREAM variable=rnnz_fifo_adj4 depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj4_total_e;
         #pragma HLS STREAM variable=rnnz_fifo_adj4_total_e depth=FIFO_DEPTH
         hls::stream<int> rnnz_fifo_adj4_total_s;
         #pragma HLS STREAM variable=rnnz_fifo_adj4_total_s depth=FIFO_DEPTH
         hls::stream<ATYPE> A_fifo_adj4("A fifo adj4");
         #pragma HLS STREAM variable=A_fifo_adj4 depth=FIFO_DEPTH
         hls::stream<int>  col_indices_fifo_adj4("col fifo4");
         #pragma HLS STREAM variable=col_indices_fifo_adj4 depth=FIFO_DEPTH

         BTYPE ate_m1[2*C_WIDTH];

         #if (PIPO_BLOCKS>=2)

	 	   LOOP_ATTN : for (int B_index = 0; B_index < layer_loop; B_index++) {
         #else
	       int B_index = 0;
         #endif

	 	std::cout << "attention layer " << B_index << std::endl;

	    #pragma HLS DATAFLOW

        #if ADJ_THREADS == 1

		  for (int j = 0; j < 2*B_WIDTH_BLOCK; j++) {
								#pragma HLS PIPELINE
	     	                    BTYPE ate_temp;
	   	                        INTYPE AF = A[j];
	                            #if (INT_QUANT_W == 1)
		  			        	   quantw(ate_temp,AF,quantization_scale_w,f_align,beta_qu,B_index);
	                            #else
		  			        	   ate_temp = AF;
                                #endif
			  			        ate_m1[j] = ate_temp;
		 }

	  	 //stream adj

		int first_row1;//,first_row2;//,first_row3,first_row4;
		int row_count1;//,row_count2;//,row_count3,row_count4;

		int N_adj_block = N_adj/ADJ_THREADS;
		int N_adj_block_compute = N_adj/FEA_THREADS; // in compute2 each block only contains  N_adj/FEA_THREADS elements
		row_count1 = N_adj_block;
		first_row1 = 0;

       #if GAT_ENABLE == 1
	  	 std::cout << "Read ADJ data" << std::endl;

         #if (COO_MODE == 0)
	 	   reada2_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1_total_e,rnnz_fifo_adj1_total_s,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1);
         #else
	 	   reada2_coo(nnz_adj1,beta_qu,f_align,quantization_scale_adj,model,M_adj,first_row1,row_count1,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1_total_e,rnnz_fifo_adj1_total_s,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1,B_index);
	 	 #endif

	 	 //generate stream with attention

         #if (FAST_ATTENTION==1)

	  	 std::cout << "prepare fast attention mechanism" << std::endl;

	     prepare_attentional_mechanism_input2(model,row_count1, P_w,A_buffer11,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1,E_col_indices_fifo1,E_rnnz_fifo1,A_fifo1,E_fifo1,max_fifo1,ate_m1,EO_fifo1,EO_rnnz_fifo1,B_index);

	     std::cout << "write e out" << std::endl;
	     //write e out

	     writes(deq_factor,model,first_row1,row_count1,N_adj,P_w, EO_fifo1,rnnz_fifo_adj1_total_e, E1,B_index);

		 //compute attention

	 	 std::cout << "compute fast attention" << std::endl;

	     compute_attention2(model,row_count1,A_fifo1, E_col_indices_fifo1, E_rnnz_fifo1,E_fifo1,max_fifo1,val_att_fifo1,col_att_fifo1, rnnz_att_fifo1,SO_fifo1,SO_rnnz_fifo1,B_index);

	 	 std::cout << "done fast attention" << std::endl;

	     //write s out
	 	 std::cout << "write s out" << std::endl;
	     float deq_dummy[5] = {1.0};
	 	 writes(deq_dummy,model,first_row1,row_count1,N_adj,P_w, SO_fifo1, rnnz_fifo_adj1_total_s,S1,B_index);

         #else

	     prepare_attentional_mechanism_input(N_adj, P_w,C_buffer,E_fifo,ate_m);

		 //compute attention

		 std::cout << "compute attention" << std::endl;

		 compute_attention(N_adj,A_fifo_adj, col_indices_fifo_adj, rnnz_fifo_adj,E_fifo,val_att_fifo,col_att_fifo, rnnz_att_fifo);

		 std::cout << "done attention" << std::endl;

         #endif

         #else //GAT DISABLE
		   std::cout << "Read ADJ data" << std::endl;
		   hls::stream<ATYPE> val_att_fifo1_int;
		   #if (COO_MODE == 0)
		     reada22_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,val_att_fifo1,col_att_fifo1,rnnz_att_fifo1,rowPtr_adj1,columnIndex_adj1,values_adj1);
           #else
             reada22_coo(nnz_adj1,beta_qu,f_align,quantization_scale_adj,model,M_adj,first_row1,row_count1,val_att_fifo1,col_att_fifo1,rnnz_att_fifo1,rowPtr_adj1,columnIndex_adj1,values_adj1,B_index);
		   #endif

         #endif

         #endif

         #if ADJ_THREADS == 4

			  for (int j = 0; j < 2*B_WIDTH_BLOCK; j++) {
										#pragma HLS PIPELINE
					  			        ate_m1[j] = A[j];
				 }

			  	 //stream adj

				int first_row1,first_row2,first_row3,first_row4;
				int row_count1,row_count2,row_count3,row_count4;

				int N_adj_block = N_adj/ADJ_THREADS;
				int N_adj_block_compute = N_adj/FEA_THREADS; // in compute2 each block only contains  N_adj/FEA_THREADS elements
				int N_adj_rest = N_adj%4;
				row_count1 = N_adj_block;
				row_count2 = N_adj_block;
				row_count3 = N_adj_block;
				row_count4 = N_adj_block+N_adj_rest;
				first_row1 = 0;
				first_row2 = N_adj_block;
				first_row3 = 2*N_adj_block;
				first_row4 = 3*N_adj_block;

                #if GAT_ENABLE == 1
			  	 std::cout << "Read ADJ data" << std::endl;

                 #if (COO_MODE == 0)
			 	  reada2_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1_total_e,rnnz_fifo_adj1_total_s,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1);
			 	  reada2_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row2,row_count2,A_fifo_adj2,col_indices_fifo_adj2,rnnz_fifo_adj2_total_e,rnnz_fifo_adj2_total_s,rnnz_fifo_adj2,rowPtr_adj2,columnIndex_adj2,values_adj2);
			 	  reada2_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row3,row_count3,A_fifo_adj3,col_indices_fifo_adj3,rnnz_fifo_adj3_total_e,rnnz_fifo_adj3_total_s,rnnz_fifo_adj3,rowPtr_adj3,columnIndex_adj3,values_adj3);
			 	  reada2_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row4,row_count4,A_fifo_adj4,col_indices_fifo_adj4,rnnz_fifo_adj4_total_e,rnnz_fifo_adj4_total_s,rnnz_fifo_adj4,rowPtr_adj4,columnIndex_adj4,values_adj4);
                 #else
			 	  reada2_coo(nnz_adj1,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1_total_e,rnnz_fifo_adj1_total_s,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1);
		    	  reada2_coo(nnz_adj2,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row2,row_count2,A_fifo_adj2,col_indices_fifo_adj2,rnnz_fifo_adj2_total_e,rnnz_fifo_adj2_total_s,rnnz_fifo_adj2,rowPtr_adj2,columnIndex_adj2,values_adj2);
			 	  reada2_coo(nnz_adj3,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row3,row_count3,A_fifo_adj3,col_indices_fifo_adj3,rnnz_fifo_adj3_total_e,rnnz_fifo_adj3_total_s,rnnz_fifo_adj3,rowPtr_adj3,columnIndex_adj3,values_adj3);
			 	  reada2_coo(nnz_adj4,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row4,row_count4,A_fifo_adj4,col_indices_fifo_adj4,rnnz_fifo_adj4_total_e,rnnz_fifo_adj4_total_s,rnnz_fifo_adj4,rowPtr_adj4,columnIndex_adj4,values_adj4);
                 #endif

			 	 //generate stream with attention

		         #if (FAST_ATTENTION==1)

			  	 std::cout << "prepare fast attention mechanism" << std::endl;

			     prepare_attentional_mechanism_inputx4(gat_mode,row_count1, row_count2, row_count3, row_count4, P_w,A_buffer11,A_buffer21,A_buffer31,A_buffer41,
			     A_fifo_adj1,A_fifo_adj2,A_fifo_adj3,A_fifo_adj4,
				 col_indices_fifo_adj1, col_indices_fifo_adj2, col_indices_fifo_adj3, col_indices_fifo_adj4,
				 rnnz_fifo_adj1,rnnz_fifo_adj2,rnnz_fifo_adj3,rnnz_fifo_adj4,
				 E_col_indices_fifo1, E_col_indices_fifo2, E_col_indices_fifo3, E_col_indices_fifo4,
				 E_rnnz_fifo1,E_rnnz_fifo2,E_rnnz_fifo3,E_rnnz_fifo4,
				 A_fifo1, A_fifo2, A_fifo3, A_fifo4,
				 E_fifo1, E_fifo2, E_fifo3, E_fifo4,
				 max_fifo1, max_fifo2, max_fifo3, max_fifo4,
				 ate_m1,
				 EO_fifo1,EO_fifo2,EO_fifo3,EO_fifo4,
				 EO_rnnz_fifo1,EO_rnnz_fifo2,EO_rnnz_fifo3,EO_rnnz_fifo4);

			     writesx4(deq_factor,gat_mode,row_count1,row_count2,row_count3,row_count4,
			    	    EO_fifo1,EO_fifo2,EO_fifo3,EO_fifo4,
						rnnz_fifo_adj1_total_e,rnnz_fifo_adj2_total_e,rnnz_fifo_adj3_total_e,rnnz_fifo_adj4_total_e,
			     		E1,B_index);

				 //compute attention

			 	 std::cout << "compute fast attention" << std::endl;

			     compute_attention2(gat_mode,row_count1,A_fifo1, E_col_indices_fifo1, E_rnnz_fifo1,E_fifo1,max_fifo1,val_att_fifo1,col_att_fifo1, rnnz_att_fifo1,SO_fifo1,SO_rnnz_fifo1);
			     compute_attention2(gat_mode,row_count2,A_fifo2, E_col_indices_fifo2, E_rnnz_fifo2,E_fifo2,max_fifo2,val_att_fifo2,col_att_fifo2, rnnz_att_fifo2,SO_fifo2,SO_rnnz_fifo2);
			     compute_attention2(gat_mode,row_count3,A_fifo3, E_col_indices_fifo3, E_rnnz_fifo3,E_fifo3,max_fifo3,val_att_fifo3,col_att_fifo3, rnnz_att_fifo3,SO_fifo3,SO_rnnz_fifo3);
			     compute_attention2(gat_mode,row_count4,A_fifo4, E_col_indices_fifo4, E_rnnz_fifo4,E_fifo4,max_fifo4,val_att_fifo4,col_att_fifo4, rnnz_att_fifo4,SO_fifo4,SO_rnnz_fifo4);

			     float deq_dummy = 1.0;
			     writesx4(deq_dummy,gat_mode,row_count1,row_count2,row_count3,row_count4,
				 SO_fifo1,SO_fifo2,SO_fifo3,SO_fifo4,
				 rnnz_fifo_adj1_total_s,rnnz_fifo_adj2_total_s,rnnz_fifo_adj3_total_s,rnnz_fifo_adj4_total_s,
				 S1,B_index);

			 	 std::cout << "done fast attention" << std::endl;

		         #else

			     prepare_attentional_mechanism_input(N_adj, P_w,C_buffer,E_fifo,ate_m);

				 //compute attention

				 std::cout << "compute attention" << std::endl;

				 compute_attention(N_adj,A_fifo_adj, col_indices_fifo_adj, rnnz_fifo_adj,E_fifo,val_att_fifo,col_att_fifo, rnnz_att_fifo);

				 std::cout << "done attention" << std::endl;

                  #endif

                #else //GAT DISABLE
                std::cout << "Read ADJ data" << std::endl;
                #if (COO_MODE == 0)
                  reada22_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,val_att_fifo1,col_att_fifo1,rnnz_att_fifo1,rowPtr_adj1,columnIndex_adj1,values_adj1);
                  reada22_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row2,row_count2,val_att_fifo2,col_att_fifo2,rnnz_att_fifo2,rowPtr_adj2,columnIndex_adj2,values_adj2);
                  reada22_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row3,row_count3,val_att_fifo3,col_att_fifo3,rnnz_att_fifo3,rowPtr_adj3,columnIndex_adj3,values_adj3);
                  reada22_csr(beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row4,row_count4,val_att_fifo4,col_att_fifo4,rnnz_att_fifo4,rowPtr_adj4,columnIndex_adj4,values_adj4);
               #else
                  reada22_coo(nnz_adj1,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row1,row_count1,val_att_fifo1,col_att_fifo1,rnnz_att_fifo1,rowPtr_adj1,columnIndex_adj1,values_adj1);
                  reada22_coo(nnz_adj2,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row2,row_count2,val_att_fifo2,col_att_fifo2,rnnz_att_fifo2,rowPtr_adj2,columnIndex_adj2,values_adj2);
                  reada22_coo(nnz_adj3,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row3,row_count3,val_att_fifo3,col_att_fifo3,rnnz_att_fifo3,rowPtr_adj3,columnIndex_adj3,values_adj3);
                  reada22_coo(nnz_adj4,beta_qu,f_align,quantization_scale_adj,gemm_mode,M_adj,first_row4,row_count4,val_att_fifo4,col_att_fifo4,rnnz_att_fifo4,rowPtr_adj4,columnIndex_adj4,values_adj4);
               #endif

             #endif
             #endif

         #if (PIPO_BLOCKS>=2)
	 	   }
        #endif

}

void readb(bool load_weights,ap_uint<1> model[5][8],int beta_qu,int f_align,float quantization_scale_w[5],int M_fea,ap_uint<8> P_w[5],int B_index,BTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],INTYPES* B)
{

	int B_shift;
	int M_fea_current;

	if (B_index == 0)
	{
		B_shift = 0;
	    M_fea_current = M_fea;
	}
	else
	{
		B_shift = B_WIDTH_BLOCK*M_fea+(B_index-1)*B_WIDTH_BLOCK*B_WIDTH_BLOCK; //shift the weight loading
	    M_fea_current = B_WIDTH_BLOCK;
	}

	bool linear_mode = model[B_index][6];
    bool sage_mode = model[B_index][7];
	bool gcn_path = !(linear_mode^sage_mode);

	 
	bool load_weights_gcn = load_weights & gcn_path;

	if(load_weights_gcn==1)
	  {

		 LOOP_BLOCKB1 : for (int j = 0; j < P_w[B_index]; j++) {
			        LOOP_BLOCKB2 : for (int i = 0; i < M_fea_current; i++) {
							#pragma HLS PIPELINE

			        	    INTYPE BF = (INTYPE)B[i+j*M_fea_current+B_shift];
			        	    BTYPE B_accel_temp;
                          #if (INT_QUANT_W == 1)
			        	     quantw(B_accel_temp,BF,quantization_scale_w,f_align,beta_qu,B_index);
                          #else
			        	     B_accel_temp = BF;
                          #endif

			        		B_accel[i][j] = B_accel_temp;

			          }
		  }

	  }
}

void readbl(bool load_weights,ap_uint<1> model[5][8],int beta_qu,int f_align,float quantization_scale_w[5],int M_fea,ap_uint<8> P_w[5],int B_index,BLTYPE B_accel[B_HEIGHT][B_WIDTH_BLOCK],INTYPES* B)
{

	int B_shift;
	int M_fea_current;

	if (B_index == 0)
	{
		B_shift = 0;
	    M_fea_current = M_fea;
	}
	else
	{
		B_shift = B_WIDTH_BLOCK*M_fea+(B_index-1)*B_WIDTH_BLOCK*B_WIDTH_BLOCK; //shift the weight loading
	    M_fea_current = B_WIDTH_BLOCK;
	}

	bool linear_mode = model[B_index][6];

	 
	bool load_weights_linear = load_weights & linear_mode;

	if(load_weights_linear==1)
	  {

		 LOOP_BLOCKB1 : for (int j = 0; j < P_w[B_index]; j++) {
			        LOOP_BLOCKB2 : for (int i = 0; i < M_fea_current; i++) {
							#pragma HLS PIPELINE

			        	    INTYPE BF = (INTYPE)B[i+j*M_fea_current+B_shift];
			        	    BLTYPE B_accel_temp;
                          #if (INT_QUANT_W == 1)
			        	     quantwl(B_accel_temp,BF,quantization_scale_w,f_align,beta_qu,B_index);
                          #else
			        	     B_accel_temp = BF;
                          #endif

			        		B_accel[i][j] = B_accel_temp;

			          }
		  }

	  }
}

void loop_fea(bool load_weights,int beta_qu,int f_align,int beta_qul,int f_alignl,float quantization_scale_fea[5],float quantization_scale_w[5],float quantization_scale_lin[5],
	ap_uint<1> model[5][8],
	STYPE scale_fea[5],ITYPE* max_fea,int quantized_multiplier,int quantized_multiplierl,
    int nnz_fea1,int nnz_fea2,int nnz_fea3,int nnz_fea4,
	int *rowPtr_fea1,int *rowPtr_fea2,int *rowPtr_fea3,int *rowPtr_fea4,
	int *columnIndex_fea1,int *columnIndex_fea2,int *columnIndex_fea3,int *columnIndex_fea4,
	INTYPE *values_fea1,INTYPE *values_fea2,INTYPE *values_fea3,INTYPE *values_fea4,
	hls::stream<ASTYPE>&  rowPtr_feas1,hls::stream<ASTYPE>& rowPtr_feas2,hls::stream<ASTYPE>&  rowPtr_feas3,hls::stream<ASTYPE>& rowPtr_feas4,
	hls::stream<ASTYPE>&  columnIndex_feas1,hls::stream<ASTYPE>& columnIndex_feas2,hls::stream<ASTYPE>&  columnIndex_feas3,hls::stream<ASTYPE>& columnIndex_feas4,
	hls::stream<ASTYPE>&  values_feas1,hls::stream<ASTYPE>& values_feas2,hls::stream<ASTYPE>&  values_feas3,hls::stream<ASTYPE>& values_feas4,
	INTYPES* B,INTYPES* B2,
	int N_fea, int M_fea,ap_uint<8> P_w[5],
	ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs,
    #if (PIPO_BLOCKS>=2)
	  hls::stream_of_blocks<buf> &C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
    #else
	  buf C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
	#endif
	hls::stream_of_blocks<buf> &C_buffer13,hls::stream_of_blocks<buf> &C_buffer14,
	hls::stream_of_blocks<buf> &C_buffer21,hls::stream_of_blocks<buf> &C_buffer22,
	hls::stream_of_blocks<buf> &C_buffer23,hls::stream_of_blocks<buf> &C_buffer24,
	hls::stream_of_blocks<buf> &C_buffer31,hls::stream_of_blocks<buf> &C_buffer32,
	hls::stream_of_blocks<buf> &C_buffer33,hls::stream_of_blocks<buf> &C_buffer34,
	hls::stream_of_blocks<buf> &C_buffer41,hls::stream_of_blocks<buf> &C_buffer42,
	hls::stream_of_blocks<buf> &C_buffer43,hls::stream_of_blocks<buf> &C_buffer44,
    #if (PIPO_BLOCKS>=2)
	hls::stream_of_blocks<buf> &A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
    #else
	buf A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
    #endif
	hls::stream_of_blocks<buf> &A_buffer31,hls::stream_of_blocks<buf> &A_buffer41,
    #if (PIPO_BLOCKS>=2)
	 hls::stream_of_blocks<bufl> &linear_pipo,
    #else
	 bufl linear_pipo,
    #endif
	int layer_loop)
{

     BTYPE B_accel1[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel1 block factor= BLOCK/2 dim=2
     BTYPE B_accel2[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel2 block factor= BLOCK/2 dim=2
     BTYPE B_accel3[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel3 block factor= BLOCK/2 dim=2
     BTYPE B_accel4[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel4 block factor= BLOCK/2 dim=2

     BLTYPE B_accel12[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel12 block factor= BLOCK/2 dim=2
     BLTYPE B_accel22[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel22 block factor= BLOCK/2 dim=2
     BLTYPE B_accel32[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel32 block factor= BLOCK/2 dim=2
     BLTYPE B_accel42[B_HEIGHT][B_WIDTH_BLOCK];
     #pragma HLS array_partition variable=B_accel42 block factor= BLOCK/2 dim=2

	 hls::stream<int> rnnz_fifo_fea1;
	 #pragma HLS STREAM variable=rnnz_fifo_fea1 depth=FIFO_DEPTH
	 hls::stream<int> rnnz_fifo_fea2;
	 #pragma HLS STREAM variable=rnnz_fifo_fea2 depth=FIFO_DEPTH
	 hls::stream<int> rnnz_fifo_fea3;
	 #pragma HLS STREAM variable=rnnz_fifo_fea3 depth=FIFO_DEPTH
	 hls::stream<int> rnnz_fifo_fea4;
	 #pragma HLS STREAM variable=rnnz_fifo_fea4 depth=FIFO_DEPTH

	 hls::stream<int> rnnz_fifo_fea12;
	 #pragma HLS STREAM variable=rnnz_fifo_fea12 depth=FIFO_DEPTH
	 hls::stream<int> rnnz_fifo_fea22;
	 #pragma HLS STREAM variable=rnnz_fifo_fea22 depth=FIFO_DEPTH
	 hls::stream<int> rnnz_fifo_fea32;
	 #pragma HLS STREAM variable=rnnz_fifo_fea32 depth=FIFO_DEPTH
	 hls::stream<int> rnnz_fifo_fea42;
	 #pragma HLS STREAM variable=rnnz_fifo_fea42 depth=FIFO_DEPTH

	 hls::stream<FTYPE> A_fifo_fea1;
	 #pragma HLS STREAM variable=A_fifo_fea1 depth=FIFO_DEPTH
	 hls::stream<FTYPE> A_fifo_fea2;
	 #pragma HLS STREAM variable=A_fifo_fea2 depth=FIFO_DEPTH
	 hls::stream<FTYPE> A_fifo_fea3;
	 #pragma HLS STREAM variable=A_fifo_fea3 depth=FIFO_DEPTH
	 hls::stream<FTYPE> A_fifo_fea4;
	 #pragma HLS STREAM variable=A_fifo_fea4 depth=FIFO_DEPTH

	 hls::stream<LTYPE> A_fifo_fea12;
	 #pragma HLS STREAM variable=A_fifo_fea12 depth=FIFO_DEPTH
	 hls::stream<LTYPE> A_fifo_fea22;
	 #pragma HLS STREAM variable=A_fifo_fea22 depth=FIFO_DEPTH
	 hls::stream<LTYPE> A_fifo_fea32;
	 #pragma HLS STREAM variable=A_fifo_fea32 depth=FIFO_DEPTH
	 hls::stream<LTYPE> A_fifo_fea42;
	 #pragma HLS STREAM variable=A_fifo_fea42 depth=FIFO_DEPTH

	 hls::stream<bool> exit_loop;
	 #pragma HLS STREAM variable=exit_loop depth=FIFO_DEPTH

	 hls::stream<int>  col_indices_fifo_fea1;
	 #pragma HLS STREAM variable=col_indices_fifo_fea1 depth=FIFO_DEPTH
	 hls::stream<int>  col_indices_fifo_fea2;
	 #pragma HLS STREAM variable=col_indices_fifo_fea2 depth=FIFO_DEPTH
	 hls::stream<int>  col_indices_fifo_fea3;
	 #pragma HLS STREAM variable=col_indices_fifo_fea3 depth=FIFO_DEPTH
	 hls::stream<int>  col_indices_fifo_fea4;
	 #pragma HLS STREAM variable=col_indices_fifo_fea4 depth=FIFO_DEPTH

	 hls::stream<int>  col_indices_fifo_fea12;
	 #pragma HLS STREAM variable=col_indices_fifo_fea12 depth=FIFO_DEPTH
	 hls::stream<int>  col_indices_fifo_fea22;
	 #pragma HLS STREAM variable=col_indices_fifo_fea22 depth=FIFO_DEPTH
	 hls::stream<int>  col_indices_fifo_fea32;
	 #pragma HLS STREAM variable=col_indices_fifo_fea32 depth=FIFO_DEPTH
	 hls::stream<int>  col_indices_fifo_fea42;
	 #pragma HLS STREAM variable=col_indices_fifo_fea42 depth=FIFO_DEPTH

	 int B_WIDTH_INT;

    #if (PIPO_BLOCKS>=2)
	 LOOP_FEA : for (int B_index = 0; B_index < layer_loop; B_index++) {
    #else
	  int B_index = 0;
     #endif
    	#pragma HLS DATAFLOW

		B_WIDTH_INT = B_WIDTH_BLOCK;

	 	std::cout << "fea layer " << B_index << std::endl;

        //else

		//else //SPMM

		/*these are the weights*/

		#if FEA_THREADS == 1

	    //read weights before locking buffer faster?

          	 readb(load_weights,model,beta_qu,f_align,quantization_scale_w,M_fea,P_w,B_index,B_accel1,B); //gnn weights

             #if LINEAR_ENABLE == 1

          	 readbl(load_weights,model,beta_qul,f_alignl,quantization_scale_w,M_fea,P_w,B_index,B_accel12,B2); //linear weights

             #endif

             #if (PIPO_BLOCKS>=2)
	             hls::write_lock<buf> C_fea11(C_buffer11); // one output for the ADJ_LOOP and one for attention
                 #if (LINEAR_ENABLE==1)
	              hls::write_lock<bufl> linear_fea(linear_pipo); //
                #else
		           QLTYPE linear_fea[B_HEIGHT][B_WIDTH_BLOCK];
                #endif
             #if GAT_ENABLE == 1
	             hls::write_lock<buf> A_fea11(A_buffer11); //we write the same output to two buffers
              #else
	             QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];
	         #endif
             #else
            #if GAT_ENABLE == 1
            #else
               QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];
            #endif

	        #endif

	  		  // read sparse matrices

	              int first_row1,first_row2,first_row3,first_row4;
	              int row_count1,row_count2,row_count3,row_count4;

	              int N_fea_block = N_fea;
	  			  int N_fea_rest = 0;
	  		      row_count1 = N_fea_block;
	  		      first_row1 = 0;

                  int last_index1;

	  	          #if (COO_MODE == 0)
	  	            reada1_csr(beta_qu,f_align,quantization_scale_fea,last_index1,stream_mode_int,gemm_mode_int,M_fea_int,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,
	  		        rowPtr_fea1,columnIndex_fea1,values_fea1,values_feas1);
                  #else
	  	            reada1_coo(nnz_fea1,beta_qu,f_align,beta_qul,f_alignl,quantization_scale_fea,quantization_scale_lin,last_index1,model,M_fea,first_row1,row_count1,
	  	            		A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,
							A_fifo_fea12,col_indices_fifo_fea12,rnnz_fifo_fea12,
	  		         rowPtr_fea1,columnIndex_fea1,values_fea1,
					 rowPtr_feas1,columnIndex_feas1,values_feas1,
					 B_index,layer_loop);
                   #endif

	  		  //outputs C_buffer

	  	  	    std::cout << "COMPUTE1 " << std::endl;

	  	  	  ITYPE max_fea1,max_fea2;

             #if (PIPO_BLOCKS>=2)

	  		  compute1_1(scale_fea,&max_fea1,quantized_multiplier,model,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_fea11,
	  		  A_fea11,B_index);

              #if LINEAR_ENABLE == 1
	  		  compute1_12(scale_fea,&max_fea2,quantized_multiplierl,model,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea12, col_indices_fifo_fea12, rnnz_fifo_fea12,B_accel12,linear_fea,B_index);
              #endif
	  		 #else

	  		  compute1_1(scale_fea,&max_fea1,quantized_multiplier,gemm_mode_int,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_buffer11,
	  		  A_buffer11);

              #if LINEAR_ENABLE == 1
	  		  compute1_12(scale_fea,&max_fea2,quantized_multiplier,gemm_mode_int,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea12, col_indices_fifo_fea12, rnnz_fifo_fea12,B_accel12,linear_pipo);
              #endif
	  		  #endif
	  	      *max_fea = max_fea1;

		#endif

#if FEA_THREADS == 2

	   hls::write_lock<buf> C_fea11(C_buffer11);
		#if ADJ_THREADS == 2
		    hls::write_lock<buf> C_fea12(C_buffer12);
		    hls::write_lock<buf> C_fea22(C_buffer22);
		#endif
		    hls::write_lock<buf> C_fea21(C_buffer21);

			for (int j = 0; j < B_WIDTH_INT; j++) {
				        LOOP_BLOCKB : for (int i = 0; i < M_fea; i++) {
								#pragma HLS PIPELINE
				        		BTYPE B_accel_temp = B[i+j*M_fea+B_index*B_WIDTH_BLOCK*M_fea];
				        		B_accel1[i][j] = B_accel_temp;
				        		B_accel2[i][j] = B_accel_temp;

							}
			}

			  int first_row1,first_row2,first_row3,first_row4;
			  int row_count1,row_count2,row_count3,row_count4;

      	  	  int N_fea_block = N_fea/2;
			  int N_fea_rest = N_fea%2;
		      row_count1 = N_fea_block;
		      row_count2 = N_fea_block+N_fea_rest;
		      first_row1 = 0;
		      first_row2 = N_fea_block;

              std::cout << "Thread fea 1" << std::endl;
	          reada1(gemm_mode,M_fea,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,B_index_loop,tail,
		        rowPtr_fea1,columnIndex_fea1,values_fea1);
	          std::cout << "Thread fea 2" << std::endl;
	          reada1(gemm_mode,M_fea,first_row2,row_count2,A_fifo_fea2,col_indices_fifo_fea2,rnnz_fifo_fea2,B_index_loop,tail,
	            rowPtr_fea2,columnIndex_fea2,values_fea2);

	#if ADJ_THREADS == 2
		  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_fea11, C_fea12,
				  //C_fea13, C_fea14,
				  B_index, B_index_loop, tail);
		  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,B_accel2,C_fea21, C_fea22,
				  //C_fea23, C_fea24,
				  B_index, B_index_loop, tail);
	#endif

	#if ADJ_THREADS == 1

		  compute1_1(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_fea11,
	  				  //C_fea12,
	  				  //C_fea13, C_fea14,
	  				  );

		  compute1_1(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,B_accel2,C_fea21,
	  				  //C_fea12,
	  				  //C_fea13, C_fea14,
	  				  );
	#endif

	#endif

       #if FEA_THREADS == 4

		#if ADJ_THREADS == 4

	    hls::write_lock<buf> C_fea11(C_buffer11);
	    hls::write_lock<buf> C_fea12(C_buffer12);
	    hls::write_lock<buf> C_fea13(C_buffer13);
	    hls::write_lock<buf> C_fea14(C_buffer14);
	    hls::write_lock<buf> C_fea21(C_buffer21);
	    hls::write_lock<buf> C_fea22(C_buffer22);
	    hls::write_lock<buf> C_fea23(C_buffer23);
	    hls::write_lock<buf> C_fea24(C_buffer24);
	    hls::write_lock<buf> C_fea31(C_buffer31);
	    hls::write_lock<buf> C_fea32(C_buffer32);
	    hls::write_lock<buf> C_fea33(C_buffer33);
	    hls::write_lock<buf> C_fea34(C_buffer34);
	    hls::write_lock<buf> C_fea41(C_buffer41);
	    hls::write_lock<buf> C_fea42(C_buffer42);
	    hls::write_lock<buf> C_fea43(C_buffer43);
	    hls::write_lock<buf> C_fea44(C_buffer44);

       #if GAT_ENABLE == 1
	    hls::write_lock<buf> A_fea11(A_buffer11);
	    hls::write_lock<buf> A_fea21(A_buffer21);
	    hls::write_lock<buf> A_fea31(A_buffer31);
	    hls::write_lock<buf> A_fea41(A_buffer41);
        #else
	    QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];
	    QTYPE A_fea21[B_HEIGHT][B_WIDTH_BLOCK];
	    QTYPE A_fea31[B_HEIGHT][B_WIDTH_BLOCK];
	    QTYPE A_fea41[B_HEIGHT][B_WIDTH_BLOCK];
        #endif

	    #endif

		#if ADJ_THREADS == 2

	    	hls::write_lock<buf> C_fea11(C_buffer11);
	 	    hls::write_lock<buf> C_fea12(C_buffer12);
	 	    hls::write_lock<buf> C_fea21(C_buffer21);
	 	    hls::write_lock<buf> C_fea22(C_buffer22);
	 	    hls::write_lock<buf> C_fea31(C_buffer31);
	 	    hls::write_lock<buf> C_fea32(C_buffer32);
	 	    hls::write_lock<buf> C_fea41(C_buffer41);
	 	    hls::write_lock<buf> C_fea42(C_buffer42);

		#endif

		for (int j = 0; j < B_WIDTH_INT; j++) {
			        LOOP_BLOCKB : for (int i = 0; i < M_fea; i++) {
							#pragma HLS PIPELINE
  			        	    INTYPE BF = (INTYPE)B[i+j*M_fea+B_index*B_WIDTH_BLOCK*M_fea];
  			        	    BTYPE B_accel_temp;
                            #if (INT_QUANT_W == 1)
  			        	     quantw(B_accel_temp,BF,quantization_scale_w,f_align,beta_qu);
                            #else
  			        	     B_accel_temp = BF;
                            #endif

			        		B_accel1[i][j] = B_accel_temp;
			        		B_accel2[i][j] = B_accel_temp;
			        		B_accel3[i][j] = B_accel_temp;
			        		B_accel4[i][j] = B_accel_temp;

						}
		}

		  // read sparse matrices

              int first_row1,first_row2,first_row3,first_row4;
              int row_count1,row_count2,row_count3,row_count4;

              int N_fea_block = N_fea/4;
			  int N_fea_rest = N_fea%4;
		      row_count1 = N_fea_block;
		      row_count2 = N_fea_block;
		      row_count3 = N_fea_block;
		      row_count4 = N_fea_block+N_fea_rest;
		      first_row1 = 0;
		      first_row2 = N_fea_block;
		      first_row3 = 2*N_fea_block;
		      first_row4 = 3*N_fea_block;
		      ITYPE max_fea1,max_fea2,max_fea3,max_fea4;

              int last_index1,last_index2,last_index3,last_index4;
              #if (COO_MODE == 0)
	          reada1_csr(beta_qu,f_align,quantization_scale_fea,last_index1,stream_mode,gemm_mode,M_fea,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,
		        rowPtr_fea1,columnIndex_fea1,values_fea1,values_feas1);
	          reada1_csr(beta_qu,f_align,quantization_scale_fea,last_index2,stream_mode,gemm_mode,M_fea,first_row2,row_count2,A_fifo_fea2,col_indices_fifo_fea2,rnnz_fifo_fea2,
	            rowPtr_fea2,columnIndex_fea2,values_fea2,values_feas2);
	          reada1_csr(beta_qu,f_align,quantization_scale_fea,last_index3,stream_mode,gemm_mode,M_fea,first_row3,row_count3,A_fifo_fea3,col_indices_fifo_fea3,rnnz_fifo_fea3,
	            rowPtr_fea3,columnIndex_fea3,values_fea3,values_feas3);
	          reada1_csr(beta_qu,f_align,quantization_scale_fea,last_index4,stream_mode,gemm_mode,M_fea,first_row4,row_count4,A_fifo_fea4,col_indices_fifo_fea4,rnnz_fifo_fea4,
	            rowPtr_fea4,columnIndex_fea4,values_fea4,values_feas4);
              #else
	           reada1_coo(nnz_fea1,beta_qu,f_align,quantization_scale_fea,last_index1,stream_mode,gemm_mode,M_fea,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,
		        rowPtr_fea1,columnIndex_fea1,values_fea1,values_feas1);
	           reada1_coo(nnz_fea2,beta_qu,f_align,quantization_scale_fea,last_index2,stream_mode,gemm_mode,M_fea,first_row2,row_count2,A_fifo_fea2,col_indices_fifo_fea2,rnnz_fifo_fea2,
	            rowPtr_fea2,columnIndex_fea2,values_fea2,values_feas2);
	           reada1_coo(nnz_fea3,beta_qu,f_align,quantization_scale_fea,last_index3,stream_mode,gemm_mode,M_fea,first_row3,row_count3,A_fifo_fea3,col_indices_fifo_fea3,rnnz_fifo_fea3,
	            rowPtr_fea3,columnIndex_fea3,values_fea3,values_feas3);
	           reada1_coo(nnz_fea4,beta_qu,f_align,quantization_scale_fea,last_index4,stream_mode,gemm_mode,M_fea,first_row4,row_count4,A_fifo_fea4,col_indices_fifo_fea4,rnnz_fifo_fea4,
	            rowPtr_fea4,columnIndex_fea4,values_fea4,values_feas4);
               #endif

		  //outputs C_buffer

			#if ADJ_THREADS == 4

	          compute1_4(scale_fea,&max_fea1,quantized_multiplier,gemm_mode,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_fea11, C_fea12, C_fea13, C_fea14,A_fea11);
	          compute1_4(scale_fea,&max_fea2,quantized_multiplier,gemm_mode,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,B_accel2,C_fea21, C_fea22, C_fea23, C_fea24,A_fea21);
	          compute1_4(scale_fea,&max_fea3,quantized_multiplier,gemm_mode,zero_point_lhs,  zero_point_rhs, first_row3,row_count3,A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3,B_accel3,C_fea31, C_fea32, C_fea33, C_fea34,A_fea31);
	          compute1_4(scale_fea,&max_fea4,quantized_multiplier,gemm_mode,zero_point_lhs,  zero_point_rhs, first_row4,row_count4,A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4,B_accel4,C_fea41, C_fea42, C_fea43, C_fea44,A_fea41);

		      *max_fea = max_fea1;

	        #endif

			#if ADJ_THREADS == 2

			  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,B_accel1,C_fea11, C_fea12, //C_fea13, C_fea14,
					  B_index, B_index_loop, tail);
			  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,B_accel2,C_fea21, C_fea22, //C_fea23, C_fea24,
					  B_index, B_index_loop, tail);
			  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row3,row_count3,A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3,B_accel3,C_fea31, C_fea32, //C_fea33, C_fea34,
					  B_index, B_index_loop, tail);
			  compute1_2(gemm_mode,zero_point_lhs,  zero_point_rhs, first_row4,row_count4,A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4,B_accel4,C_fea41, C_fea42, //C_fea43, C_fea44,
					  B_index, B_index_loop, tail);

			#endif
#endif

       #if (PIPO_BLOCKS>=2)
		   }
        #endif

}

void loop_adj(float deq_factor[5],ap_uint<1> model[5][8],float srelu[5],hls::stream<ITYPE> &A_fifo_adj1,hls::stream<int> &col_indices_fifo_adj1,hls::stream<int> &rnnz_fifo_adj1,
	hls::stream<TTYPE> &A_fifo_adj2,hls::stream<int> &col_indices_fifo_adj2,hls::stream<int> &rnnz_fifo_adj2,
	hls::stream<TTYPE> &A_fifo_adj3,hls::stream<int> &col_indices_fifo_adj3,hls::stream<int> &rnnz_fifo_adj3,
	hls::stream<TTYPE> &A_fifo_adj4,hls::stream<int> &col_indices_fifo_adj4,hls::stream<int> &rnnz_fifo_adj4,
	int N_adj, int M_adj,ap_uint<8> P_w[5], ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs,
    #if (PIPO_BLOCKS>=2)
	  hls::stream_of_blocks<buf> &C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
    #else
	  buf C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
    #endif
	hls::stream_of_blocks<buf> &C_buffer13,hls::stream_of_blocks<buf> &C_buffer14,
	hls::stream_of_blocks<buf> &C_buffer21,hls::stream_of_blocks<buf> &C_buffer22,
	hls::stream_of_blocks<buf> &C_buffer23,hls::stream_of_blocks<buf> &C_buffer24,
	hls::stream_of_blocks<buf> &C_buffer31,hls::stream_of_blocks<buf> &C_buffer32,
	hls::stream_of_blocks<buf> &C_buffer33,hls::stream_of_blocks<buf> &C_buffer34,
	hls::stream_of_blocks<buf> &C_buffer41,hls::stream_of_blocks<buf> &C_buffer42,
	hls::stream_of_blocks<buf> &C_buffer43,hls::stream_of_blocks<buf> &C_buffer44,
    #if (PIPO_BLOCKS>=2)
	 hls::stream_of_blocks<bufl> &linear_pipo,
    #else
	 bufl linear_pipo,
    #endif
	int layer_loop,OUTTYPE* D1,OUTTYPE* D2,OUTTYPE* D3,OUTTYPE* D4,hls::stream<ASTYPE>& DS1,hls::stream<ASTYPE>& DS1R, hls::stream<ASTYPE>& DS1C,
	hls::stream<ASTYPE>&  DS2, hls::stream<ASTYPE>& DS3,hls::stream<ASTYPE>&  DS4)
{

       hls::stream<ITYPE>       D_fifo1[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=D_fifo1 depth=FIFO_DEPTH
       hls::stream<ITYPE>       D_fifo2[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=D_fifo2 depth=FIFO_DEPTH
       hls::stream<ITYPE>       D_fifo3[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=D_fifo3 depth=FIFO_DEPTH
       hls::stream<ITYPE>       D_fifo4[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=D_fifo4 depth=FIFO_DEPTH

       hls::stream<OUTTYPE>   out_fifo1;
       #pragma HLS STREAM variable=out_fifo1 depth=FIFO_DEPTH

       hls::stream<ITYPE>       write_fifo1[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=write_fifo1 depth=FIFO_DEPTH
       hls::stream<ITYPE>       write_fifo2[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=write_fifo2 depth=FIFO_DEPTH
       hls::stream<ITYPE>       write_fifo3[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=write_fifo3 depth=FIFO_DEPTH
       hls::stream<ITYPE>       write_fifo4[B_WIDTH_BLOCK];
       #pragma HLS STREAM variable=write_fifo4 depth=FIFO_DEPTH

    #if (PIPO_BLOCKS>=2)

	   LOOP_ADJ : for (int B_index = 0; B_index < layer_loop; B_index++) {

    #else

		   int B_index = 0;

    #endif

        #pragma HLS DATAFLOW
	      //else

	        //else

		 	  std::cout << "adj layer " << B_index << std::endl;

	#if ADJ_THREADS == 1

         #if (PIPO_BLOCKS>=2)
		     hls::read_lock<buf> C_adj11(C_buffer11);
             #if (LINEAR_ENABLE==1)
		      hls::read_lock<bufl> linear_adj(linear_pipo);
             #else
		      QLTYPE linear_adj[B_HEIGHT][B_WIDTH_BLOCK];
             #endif
		    #endif
			#if FEA_THREADS == 2
		    	hls::read_lock<buf> C_adj21(C_buffer21);
			#endif
		 		    

		 	    int first_row1;//,first_row2;//,first_row3,first_row4;
		 	    int row_count1;//,row_count2;//,row_count3,row_count4;

		         int N_adj_block = N_adj/ADJ_THREADS;
		         int N_adj_block_compute = N_adj/FEA_THREADS; // in compute2 each block only contains  N_adj/FEA_THREADS elements
		 		row_count1 = N_adj_block;
		 		first_row1 = 0;

				#if FEA_THREADS == 1
                   #if(PIPO_BLOCKS>=2)

		 	    	 compute2_1(model,srelu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,
		 	    		//C_adj21,
		 	    		//C_adj31,C_adj41,
		 	    		D_fifo1,B_index);

                   #else

		 	    	 compute2_1(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_buffer11,
			 	    		//C_adj21,
			 	    		//C_adj31,C_adj41,
			 	    		D_fifo1);
                   #endif
				#endif
				#if FEA_THREADS == 2
		 	    	compute2_2(N_adj_block_compute,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,C_adj21,
			   	    		//C_adj31,C_adj41,
			   	    		D_fifo1);

				#endif
		         		//C_adj32,C_adj42,

		 		 // write write _fifo into D

                      #if(PIPO_BLOCKS>=2)
		 	           writec(deq_factor,model,first_row1,row_count1,N_adj,P_w, D_fifo1,linear_adj,out_fifo1,B_index,layer_loop);
                      #else
		 	           writec(deq_factor,model,first_row1,row_count1,N_adj,P_w, D_fifo1,linear_pipo,D1,DS1,B_index,layer_loop);
                      #endif

		 	          writeout(model,first_row1,row_count1,N_adj,P_w, out_fifo1,D1,DS1,DS1R, DS1C,B_index,layer_loop);

	#endif

#if ADJ_THREADS == 2

	    hls::read_lock<buf> C_adj11(C_buffer11);
	    hls::read_lock<buf> C_adj12(C_buffer12);
	   	hls::read_lock<buf> C_adj21(C_buffer21);
	   	hls::read_lock<buf> C_adj22(C_buffer22);

		#if FEA_THREADS == 4
	    	hls::read_lock<buf> C_adj31(C_buffer31);
		    hls::read_lock<buf> C_adj32(C_buffer32);
		    hls::read_lock<buf> C_adj41(C_buffer41);
		    hls::read_lock<buf> C_adj42(C_buffer42);
		#endif

	   	int first_row1,first_row2;//,first_row3,first_row4;
	   	int row_count1,row_count2;//,row_count3,row_count4;

	    int N_adj_block = N_adj/ADJ_THREADS;
	   	int N_adj_rest = N_adj%ADJ_THREADS;
	    int N_adj_block_compute = N_adj/FEA_THREADS; // in compute2 each block only contains  N_adj/FEA_THREADS elements
	   	row_count1 = N_adj_block;
	   	row_count2 = N_adj_block+N_adj_rest;;
	   	first_row1 = 0;
	   	first_row2 = N_adj_block;

        std::cout << "Thread adj 1" << std::endl;
	   	reada2(first_row1,row_count1,B_index_loop,tail,A_fifo_adj1,col_indices_fifo_adj1,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1);
        std::cout << "Thread adj 2" << std::endl;
	   	reada2(first_row2,row_count2,B_index_loop,tail,A_fifo_adj2,col_indices_fifo_adj2,rnnz_fifo_adj2,rowPtr_adj2,columnIndex_adj2,values_adj2);

		#if FEA_THREADS == 2

	   	compute2_2(N_adj_block_compute,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,C_adj21,
	   	    		//C_adj31,C_adj41,
	   	    		D_fifo1, B_index, B_index_loop, tail);

	    compute2_2(N_adj_block_compute,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,C_adj12,C_adj22,
	           		//C_adj32,C_adj42,
	           		D_fifo2, B_index, B_index_loop, tail);

		#endif

		#if FEA_THREADS == 4

	   	compute2_4(N_adj_block_compute,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,C_adj21,
	   	    		C_adj31,C_adj41,
	   	    		D_fifo1, B_index, B_index_loop, tail);

	    compute2_4(N_adj_block_compute,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,C_adj12,C_adj22,
	           		C_adj32,C_adj42,
	           		D_fifo2, B_index, B_index_loop, tail);

		#endif

	   	 writec(first_row1,row_count1,P_w, D_fifo1, D1,B_index,B_index_loop, tail);
	   	 writec(first_row2,row_count2,P_w, D_fifo2, D2,B_index,B_index_loop, tail);

		#endif

	#if ADJ_THREADS == 4

		    hls::read_lock<buf> C_adj11(C_buffer11);
		    hls::read_lock<buf> C_adj12(C_buffer12);
		    hls::read_lock<buf> C_adj13(C_buffer13);
		    hls::read_lock<buf> C_adj14(C_buffer14);
		    hls::read_lock<buf> C_adj21(C_buffer21);
		    hls::read_lock<buf> C_adj22(C_buffer22);
		    hls::read_lock<buf> C_adj23(C_buffer23);
		    hls::read_lock<buf> C_adj24(C_buffer24);
		    hls::read_lock<buf> C_adj31(C_buffer31);
		    hls::read_lock<buf> C_adj32(C_buffer32);
		    hls::read_lock<buf> C_adj33(C_buffer33);
		    hls::read_lock<buf> C_adj34(C_buffer34);
		    hls::read_lock<buf> C_adj41(C_buffer41);
		    hls::read_lock<buf> C_adj42(C_buffer42);
		    hls::read_lock<buf> C_adj43(C_buffer43);
		    hls::read_lock<buf> C_adj44(C_buffer44);

	    int first_row1,first_row2,first_row3,first_row4;
	    int row_count1,row_count2,row_count3,row_count4;

        int N_adj_block = N_adj/4;
	    int N_adj_rest = N_adj%4;
		row_count1 = N_adj_block;
		row_count2 = N_adj_block;
		row_count3 = N_adj_block;
		row_count4 = N_adj_block+N_adj_rest;
		first_row1 = 0;
		first_row2 = N_adj_block;
		first_row3 = 2*N_adj_block;
		first_row4 = 3*N_adj_block;

	    compute2_4(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row1,row_count1,A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,C_adj11,C_adj21,C_adj31,C_adj41,D_fifo1);
        compute2_4(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row2,row_count2,A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,C_adj12,C_adj22,C_adj32,C_adj42,D_fifo2);
        compute2_4(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row3,row_count3,A_fifo_adj3, col_indices_fifo_adj3, rnnz_fifo_adj3,C_adj13,C_adj23,C_adj33,C_adj43,D_fifo3);
        compute2_4(relu,N_adj_block,zero_point_lhs,  zero_point_rhs, first_row4,row_count4,A_fifo_adj4, col_indices_fifo_adj4, rnnz_fifo_adj4,C_adj14,C_adj24,C_adj34,C_adj44,D_fifo4);

	    writec(deq_factor,stream_mode,first_row1,row_count1,N_adj,P_w, D_fifo1, D1,DS1,B_index);
	    writec(deq_factor,stream_mode,first_row2,row_count2,N_adj,P_w, D_fifo2, D2,DS2,B_index);
	    writec(deq_factor,stream_mode,first_row3,row_count3,N_adj,P_w, D_fifo3, D3,DS3,B_index);
	    writec(deq_factor,stream_mode,first_row4,row_count4,N_adj,P_w, D_fifo4, D4,DS4,B_index);

		#endif

     #if (PIPO_BLOCKS>=2)

	     }

     #endif

}

void loop_adj2(
int nnz_adj1,int nnz_adj2,int nnz_adj3,int nnz_adj4,
int beta_qu,int f_align,float quantization_scale_adj,float quantization_scale_w[5],
float deq_factor[5],
ap_uint<1> model[5][8],float srelu[5],
int * rowPtr_adj1,int * rowPtr_adj2,int * rowPtr_adj3,int * rowPtr_adj4,
int *columnIndex_adj1, int *columnIndex_adj2, int *columnIndex_adj3, int *columnIndex_adj4,
INTYPE *values_adj1, 	INTYPE *values_adj2,	INTYPE *values_adj3,	INTYPE *values_adj4,
int N_adj, int M_adj, ap_uint<8> P_w[5], ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs,
INTYPE *A,
#if (PIPO_BLOCKS>=2)
hls::stream_of_blocks<buf> &A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
#else
buf A_buffer11,hls::stream_of_blocks<buf> &A_buffer21,
#endif
hls::stream_of_blocks<buf> &A_buffer31,hls::stream_of_blocks<buf> &A_buffer41,
OUTTYPE* E1,
OUTTYPE* S1,
#if (PIPO_BLOCKS>=2)
hls::stream_of_blocks<buf> &C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
#else
buf C_buffer11,hls::stream_of_blocks<buf> &C_buffer12,
#endif
hls::stream_of_blocks<buf> &C_buffer13,hls::stream_of_blocks<buf> &C_buffer14,
hls::stream_of_blocks<buf> &C_buffer21,hls::stream_of_blocks<buf> &C_buffer22,
hls::stream_of_blocks<buf> &C_buffer23,hls::stream_of_blocks<buf> &C_buffer24,
hls::stream_of_blocks<buf> &C_buffer31,hls::stream_of_blocks<buf> &C_buffer32,
hls::stream_of_blocks<buf> &C_buffer33,hls::stream_of_blocks<buf> &C_buffer34,
hls::stream_of_blocks<buf> &C_buffer41,hls::stream_of_blocks<buf> &C_buffer42,
hls::stream_of_blocks<buf> &C_buffer43,hls::stream_of_blocks<buf> &C_buffer44,
#if (PIPO_BLOCKS>=2)
 hls::stream_of_blocks<bufl> &linear_pipo,
#else
 bufl linear_pipo,
#endif
int layer_loop,OUTTYPE* D1,OUTTYPE* D2,OUTTYPE* D3,OUTTYPE* D4,hls::stream<ASTYPE>& DS1,
hls::stream<ASTYPE>& DS1R, hls::stream<ASTYPE>& DS1C,
hls::stream<ASTYPE>&  DS2, hls::stream<ASTYPE>& DS3,hls::stream<ASTYPE>&  DS4)
{

    hls::stream<int>  rnnz_att1("rnnz_att1 stream");
    #pragma HLS STREAM variable= rnnz_att1 depth=FIFO_DEPTH
    hls::stream<ITYPE>  values_att1("values_att1 stream");
    #pragma HLS STREAM variable= values_att1 depth=FIFO_DEPTH
    hls::stream<int>  columnIndex_att1("columnIndex_att1 stream");
    #pragma HLS STREAM variable= columnIndex_att1 depth=FIFO_DEPTH

    hls::stream<int>  rnnz_att2;
    #pragma HLS STREAM variable= rnnz_att2 depth=FIFO_DEPTH
    hls::stream<ITYPE>  values_att2;
    #pragma HLS STREAM variable= values_att2 depth=FIFO_DEPTH
    hls::stream<int>  columnIndex_att2;
    #pragma HLS STREAM variable= columnIndex_att2 depth=FIFO_DEPTH

    hls::stream<int>  rnnz_att3;
    #pragma HLS STREAM variable= rnnz_att3 depth=FIFO_DEPTH
    hls::stream<ITYPE>  values_att3;
    #pragma HLS STREAM variable= values_att3 depth=FIFO_DEPTH
    hls::stream<int>  columnIndex_att3;
    #pragma HLS STREAM variable= columnIndex_att3 depth=FIFO_DEPTH

    hls::stream<int>  rnnz_att4;
    #pragma HLS STREAM variable= rnnz_att4 depth=FIFO_DEPTH
    hls::stream<ITYPE>  values_att4;
    #pragma HLS STREAM variable= values_att4 depth=FIFO_DEPTH
    hls::stream<int>  columnIndex_att4;
    #pragma HLS STREAM variable= columnIndex_att4 depth=FIFO_DEPTH

   #pragma HLS DATAFLOW

    loop_attention(deq_factor,beta_qu,f_align,quantization_scale_adj,quantization_scale_w,
    model,
    nnz_adj1,nnz_adj2,nnz_adj3,nnz_adj4,
    rowPtr_adj1,rowPtr_adj2,rowPtr_adj3,rowPtr_adj4,
    columnIndex_adj1,columnIndex_adj2,columnIndex_adj3,columnIndex_adj4,
	values_adj1,values_adj2,values_adj3,values_adj4,
    N_adj,M_adj,P_w,A,A_buffer11,A_buffer21,A_buffer31,A_buffer41,
	E1,
	S1,
    rnnz_att1,columnIndex_att1,values_att1,
	rnnz_att2,columnIndex_att2,values_att2,
	rnnz_att3,columnIndex_att3,values_att3,
	rnnz_att4,columnIndex_att4,values_att4,
    layer_loop);

    std::cout << "Done loop attention" << std::endl;

	loop_adj(deq_factor,model,srelu,
	values_att1,columnIndex_att1,rnnz_att1,
	values_att2,columnIndex_att2,rnnz_att2,
	values_att3,columnIndex_att3,rnnz_att3,
	values_att4,columnIndex_att4,rnnz_att4,
	N_adj, M_adj,P_w,zero_point_lhs,zero_point_rhs,
    C_buffer11,C_buffer12,C_buffer13,C_buffer14,
    C_buffer21,C_buffer22,C_buffer23,C_buffer24,
    C_buffer31,C_buffer32,C_buffer33,C_buffer34,
    C_buffer41,C_buffer42,C_buffer43,C_buffer44,
    	linear_pipo,
	layer_loop,D1,D2,D3,D4,DS1,DS1R,DS1C,DS2,DS3,DS4);

	  std::cout << "Done loop adj" << std::endl;

}

void mmult_wrapper(bool load_weights,int beta_qu,int f_align,int beta_qul,int f_alignl,float quantization_scale_adj,float quantization_scale_fea[5],float quantization_scale_w[5],float quantization_scale_lin[5],
	float deq_factor[5],
	ap_uint<1> model[5][8],float srelu[5],
    STYPE scale_fea[5],ITYPE* max_fea,int quantized_multiplier,int quantized_multiplierl,
	ap_int<32> *shift, ap_int<32> *bias,
	ap_int<32> bias_count, ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs,
	ap_int<8> zero_point_dst, ap_int<8> clamp_max,ap_int<8> clamp_min,int N_adj, int M_adj, int M_fea,
	ap_uint<8> P_w[5], INTYPES* B, INTYPES* B2,
	 OUTTYPE* D1, OUTTYPE* D2, OUTTYPE* D3,OUTTYPE* D4,
	 hls::stream<ASTYPE>& DS1, hls::stream<ASTYPE>& DS1R, hls::stream<ASTYPE>& DS1C,
	 hls::stream<ASTYPE>&  DS2, hls::stream<ASTYPE>& DS3,hls::stream<ASTYPE>&  DS4,
	 OUTTYPE* E1,
	 OUTTYPE* S1,
	 INTYPE *ate_m,
	 int array_c_adjust, ap_int<32>  layer_loop,
	 int nnz_fea1,int nnz_fea2,int nnz_fea3,int nnz_fea4,
	 int *rowPtr_fea1,int *rowPtr_fea2,int *rowPtr_fea3,int *rowPtr_fea4,
	 int *columnIndex_fea1, int *columnIndex_fea2, int *columnIndex_fea3, int *columnIndex_fea4,
	 INTYPE *values_fea1,INTYPE *values_fea2,INTYPE *values_fea3,INTYPE *values_fea4,
	 hls::stream<ASTYPE>&  rowPtr_feas1,hls::stream<ASTYPE>& rowPtr_feas2,hls::stream<ASTYPE>&  rowPtr_feas3,hls::stream<ASTYPE>& rowPtr_feas4,
	 hls::stream<ASTYPE>&  columnIndex_feas1,hls::stream<ASTYPE>& columnIndex_feas2,hls::stream<ASTYPE>&  columnIndex_feas3,hls::stream<ASTYPE>& columnIndex_feas4,
	 hls::stream<ASTYPE>&  values_feas1,hls::stream<ASTYPE>& values_feas2,hls::stream<ASTYPE>&  values_feas3,hls::stream<ASTYPE>& values_feas4,
	 int nnz_adj1, int nnz_adj2, int nnz_adj3, int nnz_adj4,
	 int *rowPtr_adj1,int *rowPtr_adj2,int *rowPtr_adj3,int *rowPtr_adj4,
	 int *columnIndex_adj1,int *columnIndex_adj2,int *columnIndex_adj3,int *columnIndex_adj4,
	 INTYPE *values_adj1,INTYPE *values_adj2,INTYPE *values_adj3,INTYPE *values_adj4)
{

      #if (PIPO_BLOCKS>=2)
	    hls::stream_of_blocks<bufl,PIPO_BLOCKS> linear_pipo;
      #else
	    bufl linear_pipo;
      #endif
	  #pragma HLS array_partition variable=linear_pipo block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=linear_pipo cyclic factor= SBLOCK_LIN dim=1

      #if (PIPO_BLOCKS>=2)
       hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer11;
      #else
       buf C_buffer11;
      #endif
      #pragma HLS array_partition variable=C_buffer11 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer11 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer12;
	  #pragma HLS array_partition variable=C_buffer12 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer12 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer13;
      #pragma HLS array_partition variable=C_buffer13 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer13 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer14;
      #pragma HLS array_partition variable=C_buffer14 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer14 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer21;
	  #pragma HLS array_partition variable=C_buffer21 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer21 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer22;
      #pragma HLS array_partition variable=C_buffer22 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer22 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer23;
      #pragma HLS array_partition variable=C_buffer23 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer23 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer24;
      #pragma HLS array_partition variable=C_buffer24 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer24 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer31;
      #pragma HLS array_partition variable=C_buffer31 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer31 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer32;
      #pragma HLS array_partition variable=C_buffer32 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer32 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer33;
      #pragma HLS array_partition variable=C_buffer33 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer33 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer34;
      #pragma HLS array_partition variable=C_buffer34 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer34 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer41;
      #pragma HLS array_partition variable=C_buffer41 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer41 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer42;
      #pragma HLS array_partition variable=C_buffer42 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer42 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer43;
      #pragma HLS array_partition variable=C_buffer43 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer43 cyclic factor= SBLOCK dim=1

      hls::stream_of_blocks<buf,PIPO_BLOCKS> C_buffer44;
      #pragma HLS array_partition variable=C_buffer44 block factor= BLOCK/2 dim=2
      #pragma HLS array_partition variable=C_buffer44 cyclic factor= SBLOCK dim=1

      #if (PIPO_BLOCKS>=2)
      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer11;
      #else
      buf A_buffer11;
      #endif

      #pragma HLS array_partition variable=A_buffer11 block factor= BLOCK/2 dim=2

      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer21;
	  #pragma HLS array_partition variable=A_buffer21 block factor= BLOCK/2 dim=2

      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer31;
      #pragma HLS array_partition variable=A_buffer31 block factor= BLOCK/2 dim=2

      hls::stream_of_blocks<buf,PIPO_BLOCKS> A_buffer41;
      #pragma HLS array_partition variable=A_buffer41 block factor= BLOCK/2 dim=2

      int B_WIDTH_INT,a_values;

      #if (PIPO_BLOCKS>=2)
        #pragma HLS DATAFLOW
      #endif

      loop_fea(load_weights,beta_qu,f_align,beta_qul,f_alignl,quantization_scale_fea,quantization_scale_w,quantization_scale_lin,
      model,
      scale_fea,max_fea,quantized_multiplier,quantized_multiplierl,
      nnz_fea1,nnz_fea2,nnz_fea3,nnz_fea4,
      rowPtr_fea1,rowPtr_fea2,rowPtr_fea3,rowPtr_fea4,
      columnIndex_fea1,columnIndex_fea2,columnIndex_fea3,columnIndex_fea4,
      values_fea1,values_fea2,values_fea3,values_fea4,
      rowPtr_feas1,rowPtr_feas2,rowPtr_feas3,rowPtr_feas4,
      columnIndex_feas1,columnIndex_feas2,columnIndex_feas3,columnIndex_feas4,
	  values_feas1,values_feas2,values_feas3,values_feas4,
      B,B2,
	  M_adj, M_fea, P_w,
	  zero_point_lhs, zero_point_rhs,
      C_buffer11,C_buffer12,C_buffer13,C_buffer14,
      C_buffer21,C_buffer22,C_buffer23,C_buffer24,
      C_buffer31,C_buffer32,C_buffer33,C_buffer34,
      C_buffer41,C_buffer42,C_buffer43,C_buffer44,
	  A_buffer11,A_buffer21,A_buffer31,A_buffer41,
	  linear_pipo,
	  layer_loop);

      std::cout << "Done loop fea" << std::endl;

	  loop_adj2(nnz_adj1,nnz_adj2,nnz_adj3,nnz_adj4,
	  beta_qu,f_align,quantization_scale_adj,quantization_scale_w,
	  deq_factor,
	  model,srelu,
	  rowPtr_adj1,rowPtr_adj2,rowPtr_adj3,rowPtr_adj4,
      columnIndex_adj1,columnIndex_adj2,columnIndex_adj3,columnIndex_adj4,
  	  values_adj1,values_adj2,values_adj3,values_adj4,
      N_adj,M_adj, P_w,zero_point_lhs,zero_point_rhs,
	  ate_m,
	  A_buffer11,A_buffer21,A_buffer31,A_buffer41,
	  E1,S1,
      C_buffer11,C_buffer12,C_buffer13,C_buffer14,
      C_buffer21,C_buffer22,C_buffer23,C_buffer24,
      C_buffer31,C_buffer32,C_buffer33,C_buffer34,
      C_buffer41,C_buffer42,C_buffer43,C_buffer44,
      linear_pipo,
	  layer_loop,D1,D2,D3,D4,DS1,DS1R,DS1C,DS2,DS3,DS4);

}

typedef unsigned long u32;

/*
 The amount of data saved in the FPGA is B_HEIGHT*B_WIDTH_BLOCK+A_WIDTH+B_WIDTH_BLOCK which should be less than FPGA BRAM size
*/

//gemm_mode fea adj
// 0 0 0 dense dense not in used in graph layers
// 1 0 1 dense sparse normal mode for layer 2
// 2 1 0 sparse dense used in training
// 3 1 1 sparse sparse normal mode for layer 1
void mmult_top(bool load_weights,int beta_qu,int f_align,int beta_qul,int f_alignl,float quantization_scale_adj,
float quantization_scale_fea[5],float quantization_scale_w[5],float quantization_scale_lin[5],
float deq_factor[5],
ap_uint<8> model[5],float srelu[5],
STYPE scale_fea[5], ITYPE* max_fea,ap_int<32>  layer_count,int quantized_multiplier,int quantized_multiplierl, ap_int<32> *shift, ap_int<32> *bias,
ap_int<32> bias_count,ap_int<64> *profiling, ap_int<8> zero_point_lhs,  ap_int<8> zero_point_rhs,
ap_int<8> zero_point_dst,
ap_int<8> clamp_max,ap_int<8> clamp_min,int N_adj, int M_adj, int M_fea, ap_uint<8> P_w[5],
INTYPES* B,INTYPES* B2,
OUTTYPE* D1, OUTTYPE* D2, OUTTYPE* D3,OUTTYPE* D4,
hls::stream<ASTYPE>& DS1, hls::stream<ASTYPE>& DS1R,hls::stream<ASTYPE>& DS1C,
hls::stream<ASTYPE>&  DS2, hls::stream<ASTYPE>& DS3,hls::stream<ASTYPE>&  DS4,
OUTTYPE* E1,
OUTTYPE* S1,
INTYPE *ate_m,
int array_c_adjust,
int nnz_fea1,int nnz_fea2,int nnz_fea3,int nnz_fea4,
int *rowPtr_fea1,int *rowPtr_fea2,int *rowPtr_fea3,int *rowPtr_fea4,
int *columnIndex_fea1, int *columnIndex_fea2, int *columnIndex_fea3, int *columnIndex_fea4,
INTYPE *values_fea1,INTYPE *values_fea2,INTYPE *values_fea3,INTYPE *values_fea4,
hls::stream<ASTYPE>&  rowPtr_feas1,hls::stream<ASTYPE>& rowPtr_feas2,hls::stream<ASTYPE>&  rowPtr_feas3,hls::stream<ASTYPE>& rowPtr_feas4,
hls::stream<ASTYPE>&  columnIndex_feas1,hls::stream<ASTYPE>& columnIndex_feas2,hls::stream<ASTYPE>&  columnIndex_feas3,hls::stream<ASTYPE>& columnIndex_feas4,
hls::stream<ASTYPE>&  values_feas1,hls::stream<ASTYPE>& values_feas2,hls::stream<ASTYPE>&  values_feas3,hls::stream<ASTYPE>& values_feas4,
int nnz_adj1,int nnz_adj2,int nnz_adj3,int nnz_adj4,
int *rowPtr_adj1,int *rowPtr_adj2,int *rowPtr_adj3,int *rowPtr_adj4,
int *columnIndex_adj1,int *columnIndex_adj2,int *columnIndex_adj3,int *columnIndex_adj4,
INTYPE *values_adj1,INTYPE *values_adj2,INTYPE *values_adj3,INTYPE *values_adj4)
{

     #pragma HLS INTERFACE s_axilite port = return bundle = control
     #pragma HLS INTERFACE s_axilite port = load_weights bundle = control
     #pragma HLS INTERFACE s_axilite port = beta_qu bundle = control
     #pragma HLS INTERFACE s_axilite port = f_align bundle = control
     #pragma HLS INTERFACE s_axilite port = beta_qul bundle = control
     #pragma HLS INTERFACE s_axilite port = f_alignl bundle = control
     #pragma HLS INTERFACE s_axilite port = deq_factor bundle = control
	 #pragma HLS INTERFACE s_axilite port = nnz_fea1 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_fea2 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_fea3 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_fea4 bundle = control
	 #pragma HLS INTERFACE s_axilite port = nnz_adj1 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_adj2 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_adj3 bundle = control
     #pragma HLS INTERFACE s_axilite port = nnz_adj4 bundle = control
     #pragma HLS INTERFACE s_axilite port = quantization_scale_adj bundle = control
     #pragma HLS INTERFACE s_axilite port = quantization_scale_fea bundle = control
	 #pragma HLS INTERFACE s_axilite port = quantization_scale_w bundle = control
     #pragma HLS INTERFACE s_axilite port = quantization_scale_lin bundle = control
	 #pragma HLS INTERFACE s_axilite port = bias_count bundle = control
     #pragma HLS INTERFACE s_axilite port = zero_point_lhs bundle = control
     #pragma HLS INTERFACE s_axilite port = zero_point_rhs bundle = control
     #pragma HLS INTERFACE s_axilite port = zero_point_dst bundle = control
     #pragma HLS INTERFACE s_axilite port = clamp_max bundle = control
     #pragma HLS INTERFACE s_axilite port = clamp_min bundle = control
	 #pragma HLS INTERFACE s_axilite port = N_adj bundle = control
     #pragma HLS INTERFACE s_axilite port = M_adj bundle = control
     #pragma HLS INTERFACE s_axilite port = M_fea bundle = control
     #pragma HLS INTERFACE s_axilite port = P_w bundle = control
	 #pragma HLS INTERFACE s_axilite port = array_c_adjust bundle = control
     #pragma HLS INTERFACE s_axilite port = model bundle = control

     #pragma HLS INTERFACE s_axilite port = layer_count bundle = control
     #pragma HLS INTERFACE s_axilite port = quantized_multiplier bundle = control
     #pragma HLS INTERFACE s_axilite port = quantized_multiplierl bundle = control

     #pragma HLS INTERFACE axis port = DS1 depth=64000
     #pragma HLS INTERFACE axis port = DS1R depth=64000
     #pragma HLS INTERFACE axis port = DS1C depth=64000
     #pragma HLS INTERFACE axis port = DS2 depth=4096
     #pragma HLS INTERFACE axis port = DS3 depth=4096
     #pragma HLS INTERFACE axis port = DS4 depth=4096

     #pragma HLS INTERFACE axis port = columnIndex_feas1 depth=4096
     #pragma HLS INTERFACE axis port = columnIndex_feas2 depth=4096
     #pragma HLS INTERFACE axis port = columnIndex_feas3 depth=4096
     #pragma HLS INTERFACE axis port = columnIndex_feas4 depth=4096

     #pragma HLS INTERFACE axis port = rowPtr_feas1 depth=4096
     #pragma HLS INTERFACE axis port = rowPtr_feas2 depth=4096
     #pragma HLS INTERFACE axis port = rowPtr_feas3 depth=4096
     #pragma HLS INTERFACE axis port = rowPtr_feas4 depth=4096

     #pragma HLS INTERFACE axis port = values_feas1 depth=64000
     #pragma HLS INTERFACE axis port = values_feas2 depth=4096
     #pragma HLS INTERFACE axis port = values_feas3 depth=4096
     #pragma HLS INTERFACE axis port = values_feas4 depth=4096

     #pragma HLS INTERFACE m_axi port = profiling depth=16 offset=slave bundle = profiling
     #pragma HLS INTERFACE m_axi port=rowPtr_fea1 depth=64000 offset=slave bundle = rowPtr_fea1
	 #pragma HLS INTERFACE m_axi port=rowPtr_fea2 depth=4096 offset=slave bundle = rowPtr_fea2
     #pragma HLS INTERFACE m_axi port=rowPtr_fea3 depth=4096 offset=slave bundle = rowPtr_fea3
     #pragma HLS INTERFACE m_axi port=rowPtr_fea4 depth=4096 offset=slave bundle = rowPtr_fea4
     #pragma HLS INTERFACE m_axi port=columnIndex_fea1 depth=64000 offset=slave bundle = columnIndex_fea1
     #pragma HLS INTERFACE m_axi port=columnIndex_fea2 depth=4096 offset=slave bundle = columnIndex_fea2
     #pragma HLS INTERFACE m_axi port=columnIndex_fea3 depth=4096 offset=slave bundle = columnIndex_fea3
     #pragma HLS INTERFACE m_axi port=columnIndex_fea4 depth=4096 offset=slave bundle = columnIndex_fea4
	 #pragma HLS INTERFACE m_axi port=values_fea1 depth=64000 offset=slave bundle = values_fea1
     #pragma HLS INTERFACE m_axi port=values_fea2 depth=4096 offset=slave bundle = values_fea2
     #pragma HLS INTERFACE m_axi port=values_fea3 depth=4096 offset=slave bundle = values_fea3
     #pragma HLS INTERFACE m_axi port=values_fea4 depth=4096 offset=slave bundle = values_fea4
	 #pragma HLS INTERFACE m_axi port=rowPtr_adj1 depth=64000 offset=slave bundle = rowPtr_adj1
     #pragma HLS INTERFACE m_axi port=rowPtr_adj2 depth=4096 offset=slave bundle = rowPtr_adj2
     #pragma HLS INTERFACE m_axi port=rowPtr_adj3 depth=4096 offset=slave bundle = rowPtr_adj3
     #pragma HLS INTERFACE m_axi port=rowPtr_adj4 depth=4096 offset=slave bundle = rowPtr_adj4
     #pragma HLS INTERFACE m_axi port=columnIndex_adj1 depth=64000 offset=slave bundle = columnIndex_adj1
     #pragma HLS INTERFACE m_axi port=columnIndex_adj2 depth=4096 offset=slave bundle = columnIndex_adj2
     #pragma HLS INTERFACE m_axi port=columnIndex_adj3 depth=4096 offset=slave bundle = columnIndex_adj3
     #pragma HLS INTERFACE m_axi port=columnIndex_adj4 depth=4096 offset=slave bundle = columnIndex_adj4
	 #pragma HLS INTERFACE m_axi port=values_adj1 depth=64000 offset=slave bundle = values_adj1
	 #pragma HLS INTERFACE m_axi port=values_adj2 depth=4096 offset=slave bundle = values_adj2
     #pragma HLS INTERFACE m_axi port=values_adj3 depth=4096 offset=slave bundle = values_adj3
     #pragma HLS INTERFACE m_axi port=values_adj4 depth=4096 offset=slave bundle = values_adj4
	 #pragma HLS INTERFACE m_axi port=B depth=32000 offset=slave bundle=B
     #pragma HLS INTERFACE m_axi port=B2 depth=32000 offset=slave bundle=B2
     #pragma HLS INTERFACE m_axi port=D1 depth=64000 offset=slave  bundle=D1
	 #pragma HLS INTERFACE m_axi port=D2 depth=1000 offset=slave bundle=D2
     #pragma HLS INTERFACE m_axi port=D3 depth=1000 offset=slave bundle=D3
     #pragma HLS INTERFACE m_axi port=D4 depth=1000 offset=slave bundle=D4
     #pragma HLS INTERFACE m_axi port=E1 depth=64000 offset=slave bundle=E1
     #pragma HLS INTERFACE m_axi port=S1 depth=64000 offset=slave bundle=S1
     #pragma HLS INTERFACE m_axi port=ate_m depth=1000 offset=slave bundle=ate_m
     #pragma HLS INTERFACE m_axi port=shift offset=slave depth=1024 bundle=shift
     #pragma HLS INTERFACE m_axi port=bias offset=slave depth=1024 bundle=bias
     #pragma HLS INTERFACE m_axi port=model offset=slave depth=1024 bundle=model
     #pragma HLS INTERFACE m_axi port=quantization_scale_fea offset=slave bundle=quantization_scale_fea
     #pragma HLS INTERFACE m_axi port=quantization_scale_w offset=slave bundle=quantization_scale_w
     #pragma HLS INTERFACE m_axi port=quantization_scale_lin offset=slave bundle=quantization_scale_lin
     #pragma HLS INTERFACE m_axi port=deq_factor offset=slave bundle=deq_factor
     #pragma HLS INTERFACE m_axi port=scale_fea offset=slave bundle=scale_fea
     #pragma HLS INTERFACE m_axi port=P_w offset=slave bundle=P_w
     #pragma HLS INTERFACE m_axi port=srelu offset=slave bundle=srelu

     #pragma HLS INTERFACE s_axilite port=columnIndex_fea1 bundle = control
	 #pragma HLS INTERFACE s_axilite port=columnIndex_fea2 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_fea3 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_fea4 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_fea1 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_fea2 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_fea3 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_fea4 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_adj1 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_adj2 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_adj3 bundle = control
     #pragma HLS INTERFACE s_axilite port=columnIndex_adj4 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_adj1 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_adj2 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_adj3 bundle = control
     #pragma HLS INTERFACE s_axilite port=rowPtr_adj4 bundle = control
     #pragma HLS INTERFACE s_axilite port=values_adj1  bundle = control
	 #pragma HLS INTERFACE s_axilite port=values_adj2  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_adj3  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_adj4  bundle = control
	 #pragma HLS INTERFACE s_axilite port=values_fea1  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_fea2  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_fea3  bundle = control
     #pragma HLS INTERFACE s_axilite port=values_fea4  bundle = control
     #pragma HLS INTERFACE s_axilite port=B  bundle = control
     #pragma HLS INTERFACE s_axilite port=B2  bundle = control
     #pragma HLS INTERFACE s_axilite port=D1  bundle = control
	 #pragma HLS INTERFACE s_axilite port=D2  bundle = control
     #pragma HLS INTERFACE s_axilite port=D3  bundle = control
     #pragma HLS INTERFACE s_axilite port=D4  bundle = control
     #pragma HLS INTERFACE s_axilite port=E1  bundle = control
     #pragma HLS INTERFACE s_axilite port=S1  bundle = control
     #pragma HLS INTERFACE s_axilite port=profiling  bundle = control
     #pragma HLS INTERFACE s_axilite port=quantized_multiplier  bundle = control
     #pragma HLS INTERFACE s_axilite port=shift  bundle = control
     #pragma HLS INTERFACE s_axilite port=bias  bundle = control
     #pragma HLS INTERFACE s_axilite port=ate_m  bundle = control
     #pragma HLS INTERFACE s_axilite port=scale_fea  bundle = control
     #pragma HLS INTERFACE s_axilite port=max_fea  bundle = control
     #pragma HLS INTERFACE s_axilite port=srelu  bundle = control

	 ap_int<32> bias_data[1024];
	 ap_int<32> shift_data[1024];

	 float srelu_int[5];

	 ap_uint<8> P_w_int[5];
     #pragma HLS ARRAY_PARTITION variable=P_w_int complete

	 ap_uint<1> model_int[5][8];
     #pragma HLS ARRAY_PARTITION variable=model_int complete

	 float quantization_scale_lin_int[5];
	 #pragma HLS ARRAY_PARTITION variable=quantization_scale_lin_int complete

	 float quantization_scale_w_int[5];
	 #pragma HLS ARRAY_PARTITION variable=quantization_scale_w_int complete

	 float quantization_scale_fea_int[5];
     #pragma HLS ARRAY_PARTITION variable=quantization_scale_fea_int complete

	 float deq_factor_int[5];
     #pragma HLS ARRAY_PARTITION variable=deq_factor_int complete

	 STYPE scale_fea_int[5];
     #pragma HLS ARRAY_PARTITION variable=scale_fea_int complete

	 //hls::stream<ASTYPE>&  values_feas1,hls::stream<ASTYPE>& values_feas2,hls::stream<ASTYPE>&  values_feas3,hls::stream<ASTYPE>& values_feas4,

	 //load bias
         //preloading bias and param data seems to be a good idea but in practice performance is the same and we save preloading overhead
         //param data is loaded in demand in this case
         //preloading is important for certain matrix configurations with small A and large B so I am going to leave it

     //else
	 {

	 fifo_empty_0 = 0;
 	 fifo_empty_1 = 0;
 	 fifo_empty_2 = 0;
	 fifo_full_0 = 0;
	 fifo_full_1 = 0;
	 fifo_full_2 = 0;
	 fifo_read_0 = 0;
	 fifo_read_1 = 0;
	 fifo_read_2 = 0;
	 fifo_write_0 = 0;
	 fifo_write_1 = 0;
	 fifo_write_2 = 0;
	 fifo_cycle_0 = 0;
	 fifo_cycle_1 = 0;
	 fifo_cycle_2 = 0;

     ap_int<32> layer_loop = layer_count;

     //load model
     for(int i=0;i<layer_loop;i++)
     {
    	 model_int[i][0] = model[i][0];
    	 model_int[i][1] = model[i][1];
    	 model_int[i][2] = model[i][2];
    	 model_int[i][3] = model[i][3];
    	 model_int[i][4] = model[i][4];
    	 model_int[i][5] = model[i][5];
    	 model_int[i][6] = model[i][6];
    	 model_int[i][7] = model[i][7];
    	 srelu_int[i] = srelu[i],
         quantization_scale_lin_int[i] = quantization_scale_lin[i];
         quantization_scale_w_int[i] = quantization_scale_w[i];
    	 quantization_scale_fea_int[i] = quantization_scale_fea[i];
    	 deq_factor_int[i] = deq_factor[i];
    	 scale_fea_int[i] = scale_fea[i];
    	 P_w_int[i] = P_w[i];

    	 std::cout << " Instruction is "<< model_int[i][7] <<  model_int[i][6] <<  model_int[i][5] <<  model_int[i][4] <<
    	 model_int[i][3] <<  model_int[i][1] <<  model_int[i][1] <<  model_int[i][0] << std::endl;
     }

	 //else
	  /*simulation run short, remove in normal synthesis*/

	  /

     }
}

void kernelmult1(
bool load_weights,
int	beta_qu,
int f_align,
float quantization_scale_adj,
float quantization_scale_fea[5],
float quantization_scale_w[5],
float quantization_scale_lin[5],
float deq_factor[5],
int layer_count,
ap_uint<8> model[5],
STYPE scale_fea[5],
ITYPE* max_fea,
int quantized_multiplier,
ap_int<32> *shift,
ap_int<32> *bias,
ap_int<32> bias_count,
ap_int<64> *profiling,
ap_int<8> zero_point_lhs,
ap_int<8> zero_point_rhs,
ap_int<8> zero_point_dst,
ap_int<8> clamp_max,
ap_int<8> clamp_min,
INTYPES *array_b,
INTYPES *array_b2,
OUTTYPE *array_d1,
OUTTYPE *array_d2,
OUTTYPE *array_d3,
OUTTYPE *array_d4,
hls::stream<ASTYPE>& stream_d1,
hls::stream<ASTYPE>& stream_d1r,
hls::stream<ASTYPE>& stream_d1c,
hls::stream<ASTYPE>& stream_d2,
hls::stream<ASTYPE>& stream_d3,
hls::stream<ASTYPE>& stream_d4,
OUTTYPE *array_e1,
OUTTYPE *array_s1,
INTYPE *ate_m,
INTYPE *values_fea1,
INTYPE *values_fea2,
INTYPE *values_fea3,
INTYPE *values_fea4,
hls::stream<ASTYPE>& values_feas1,
hls::stream<ASTYPE>& values_feas2,
hls::stream<ASTYPE>& values_feas3,
hls::stream<ASTYPE>& values_feas4,
int *colIndices_fea1,
int *colIndices_fea2,
int *colIndices_fea3,
int *colIndices_fea4,
hls::stream<ASTYPE>& columnIndex_feas1,
hls::stream<ASTYPE>& columnIndex_feas2,
hls::stream<ASTYPE>& columnIndex_feas3,
hls::stream<ASTYPE>& columnIndex_feas4,
int nnz_fea1,
int nnz_fea2,
int nnz_fea3,
int nnz_fea4,
int *rowPtr_fea1,
int *rowPtr_fea2,
int *rowPtr_fea3,
int *rowPtr_fea4,
hls::stream<ASTYPE>& rowPtr_feas1,
hls::stream<ASTYPE>& rowPtr_feas2,
hls::stream<ASTYPE>& rowPtr_feas3,
hls::stream<ASTYPE>& rowPtr_feas4,
INTYPE *values_adj1,
INTYPE *values_adj2,
INTYPE *values_adj3,
INTYPE *values_adj4,
int *colIndices_adj1,
int *colIndices_adj2,
int *colIndices_adj3,
int *colIndices_adj4,
int nnz_adj1,
int nnz_adj2,
int nnz_adj3,
int nnz_adj4,
int *rowPtr_adj1,
int *rowPtr_adj2,
int *rowPtr_adj3,
int *rowPtr_adj4,
int N_adj,
int M_adj,
int M_fea,
int P_w
)
{

    int array_c_adjust=N_adj;
    ap_uint<8> P_w_int[5];

    P_w_int[0]=P_w;
    float srelu[5];
	srelu[0]=0.0;

    std::cout << " kernel starting " << std::endl;

    int quantized_multiplierl = 8;
    int beta_qul = 255;
    int f_alignl = 0;

    mmult_top(load_weights,beta_qu,f_align,beta_qul,f_alignl,quantization_scale_adj,quantization_scale_fea,quantization_scale_w,quantization_scale_lin,deq_factor,
    model,srelu,scale_fea,max_fea,layer_count,quantized_multiplier,quantized_multiplierl,shift,bias,bias_count,profiling,zero_point_lhs,zero_point_rhs, zero_point_dst,clamp_max,clamp_min,
    N_adj, M_adj, M_fea, P_w_int,
    array_b, array_b2,
    array_d1,array_d2,array_d3,array_d4,
	stream_d1,stream_d1r,stream_d1c,
	stream_d2,stream_d3,stream_d4,
    array_e1,
	array_s1,
    ate_m,
    array_c_adjust,
    nnz_fea1,nnz_fea2,nnz_fea3,nnz_fea4,
    rowPtr_fea1,rowPtr_fea2,rowPtr_fea3,rowPtr_fea4,
    colIndices_fea1,colIndices_fea2,colIndices_fea3,colIndices_fea4,
    values_fea1,values_fea2,values_fea3,values_fea4,
	rowPtr_feas1,rowPtr_feas2,rowPtr_feas3,rowPtr_feas4,
	columnIndex_feas1, columnIndex_feas2, columnIndex_feas3,columnIndex_feas4,
	values_feas1,values_feas2,values_feas3,values_feas4,
    nnz_adj1,nnz_adj2,nnz_adj3,nnz_adj4,
    rowPtr_adj1,rowPtr_adj2,rowPtr_adj3,rowPtr_adj4,
    colIndices_adj1,colIndices_adj2,colIndices_adj3,colIndices_adj4,
    values_adj1,values_adj2,values_adj3,values_adj4);

    std::cout << " 0 output " << array_d1[0] << std::endl;

    std::cout << " 3 output " << array_d1[3] << std::endl;

    std::cout << " 7 output " << array_d1[7] << std::endl;

    std::cout << " 9 output " << array_d1[9] << std::endl;

    std::cout << " 13 output " << array_d1[13] << std::endl;

    std::cout << " 20 output " << array_d1[20] << std::endl;

    std::cout << " 33 output " << array_d1[33] << std::endl;

    std::cout << " kernel done " << std::endl;
}
