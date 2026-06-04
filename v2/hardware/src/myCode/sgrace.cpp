typedef unsigned long u32;













// =============================================================================================
// =============================================================================================
// MMULT WRAPPER
// =============================================================================================
// =============================================================================================

void mmult_wrapper(
    bool load_weights,
    int beta_qu,
    int f_align,
    int beta_qul,
    int f_alignl,
    float quantization_scale_adj,
    float quantization_scale_fea[5],
    float quantization_scale_w[5],
    float quantization_scale_lin[5],
    float deq_factor[5],
    ap_uint<1> model[5][8],
    float srelu[5],
    STYPE scale_fea[5],
    ITYPE* max_fea,
    int quantized_multiplier,
    int quantized_multiplierl,
    ap_int<32>* shift,
    ap_int<32>* bias,
    ap_int<32> bias_count,
    ap_int<8> zero_point_lhs,
    ap_int<8> zero_point_rhs,
    ap_int<8> zero_point_dst,
    ap_int<8> clamp_max,
    ap_int<8> clamp_min,
    int N_adj,
    int M_adj,
    int M_fea,
    ap_uint<8> P_w[5],

    INTYPES* B,
    INTYPES* B2,

    OUTTYPE* D1,
    OUTTYPE* D2,
    OUTTYPE* D3,
    OUTTYPE* D4,

    hls::stream<ASTYPE>& DS1,
    hls::stream<ASTYPE>& DS1R,
    hls::stream<ASTYPE>& DS1C,
    hls::stream<ASTYPE>& DS2,
    hls::stream<ASTYPE>& DS3,
    hls::stream<ASTYPE>& DS4,

    OUTTYPE* E1,
    OUTTYPE* S1,
    INTYPE* ate_m,

    int array_c_adjust,
    ap_int<32> layer_loop,

    int nnz_fea1,
    int nnz_fea2,
    int nnz_fea3,
    int nnz_fea4,

    int* rowPtr_fea1,
    int* rowPtr_fea2,
    int* rowPtr_fea3,
    int* rowPtr_fea4,

    int* columnIndex_fea1,
    int* columnIndex_fea2,
    int* columnIndex_fea3,
    int* columnIndex_fea4,

    INTYPE* values_fea1,
    INTYPE* values_fea2,
    INTYPE* values_fea3,
    INTYPE* values_fea4,

    hls::stream<ASTYPE>& rowPtr_feas1,
    hls::stream<ASTYPE>& rowPtr_feas2,
    hls::stream<ASTYPE>& rowPtr_feas3,
    hls::stream<ASTYPE>& rowPtr_feas4,

    hls::stream<ASTYPE>& columnIndex_feas1,
    hls::stream<ASTYPE>& columnIndex_feas2,
    hls::stream<ASTYPE>& columnIndex_feas3,
    hls::stream<ASTYPE>& columnIndex_feas4,

    hls::stream<ASTYPE>& values_feas1,
    hls::stream<ASTYPE>& values_feas2,
    hls::stream<ASTYPE>& values_feas3,
    hls::stream<ASTYPE>& values_feas4,

    int nnz_adj1,
    int nnz_adj2,
    int nnz_adj3,
    int nnz_adj4,

    int* rowPtr_adj1,
    int* rowPtr_adj2,
    int* rowPtr_adj3,
    int* rowPtr_adj4,

    int* columnIndex_adj1,
    int* columnIndex_adj2,
    int* columnIndex_adj3,
    int* columnIndex_adj4,

    INTYPE* values_adj1,
    INTYPE* values_adj2,
    INTYPE* values_adj3,
    INTYPE* values_adj4
)
{
    // ---------------------------------------------------------------------
    // Linear operator PIPO buffer
    // ---------------------------------------------------------------------

#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<bufl, PIPO_BLOCKS> linear_pipo;
#else
    bufl linear_pipo;
#endif

    #pragma HLS array_partition variable=linear_pipo block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=linear_pipo cyclic factor=SBLOCK_LIN dim=1

    // ---------------------------------------------------------------------
    // Feature-stage output buffers
    //
    // C_bufferXY naming:
    //   X = output/thread group
    //   Y = feature/adjacency partition
    // ---------------------------------------------------------------------

#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer11;
#else
    buf C_buffer11;
#endif

    #pragma HLS array_partition variable=C_buffer11 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer11 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer12;
    #pragma HLS array_partition variable=C_buffer12 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer12 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer13;
    #pragma HLS array_partition variable=C_buffer13 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer13 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer14;
    #pragma HLS array_partition variable=C_buffer14 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer14 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer21;
    #pragma HLS array_partition variable=C_buffer21 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer21 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer22;
    #pragma HLS array_partition variable=C_buffer22 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer22 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer23;
    #pragma HLS array_partition variable=C_buffer23 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer23 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer24;
    #pragma HLS array_partition variable=C_buffer24 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer24 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer31;
    #pragma HLS array_partition variable=C_buffer31 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer31 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer32;
    #pragma HLS array_partition variable=C_buffer32 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer32 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer33;
    #pragma HLS array_partition variable=C_buffer33 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer33 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer34;
    #pragma HLS array_partition variable=C_buffer34 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer34 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer41;
    #pragma HLS array_partition variable=C_buffer41 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer41 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer42;
    #pragma HLS array_partition variable=C_buffer42 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer42 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer43;
    #pragma HLS array_partition variable=C_buffer43 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer43 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> C_buffer44;
    #pragma HLS array_partition variable=C_buffer44 block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=C_buffer44 cyclic factor=SBLOCK dim=1

    // ---------------------------------------------------------------------
    // Adjacency-stage input buffers
    // ---------------------------------------------------------------------

#if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer11;
#else
    buf A_buffer11;
#endif

    #pragma HLS array_partition variable=A_buffer11 block factor=BLOCK/2 dim=2

    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer21;
    #pragma HLS array_partition variable=A_buffer21 block factor=BLOCK/2 dim=2

    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer31;
    #pragma HLS array_partition variable=A_buffer31 block factor=BLOCK/2 dim=2

    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer41;
    #pragma HLS array_partition variable=A_buffer41 block factor=BLOCK/2 dim=2

    // ---------------------------------------------------------------------
    // Dataflow region
    //
    // loop_fea:
    //   Loads/processes feature data and weights.
    //   Produces intermediate C buffers and A buffers.
    //
    // loop_adj2:
    //   Consumes adjacency data and intermediate feature results.
    //   Produces final output buffers/streams.
    // ---------------------------------------------------------------------

#if (PIPO_BLOCKS >= 2)
    #pragma HLS DATAFLOW
#endif

    loop_fea(
        load_weights,
        beta_qu,
        f_align,
        beta_qul,
        f_alignl,
        quantization_scale_fea,
        quantization_scale_w,
        quantization_scale_lin,
        model,
        scale_fea,
        max_fea,
        quantized_multiplier,
        quantized_multiplierl,

        nnz_fea1,
        nnz_fea2,
        nnz_fea3,
        nnz_fea4,

        rowPtr_fea1,
        rowPtr_fea2,
        rowPtr_fea3,
        rowPtr_fea4,

        columnIndex_fea1,
        columnIndex_fea2,
        columnIndex_fea3,
        columnIndex_fea4,

        values_fea1,
        values_fea2,
        values_fea3,
        values_fea4,

        rowPtr_feas1,
        rowPtr_feas2,
        rowPtr_feas3,
        rowPtr_feas4,

        columnIndex_feas1,
        columnIndex_feas2,
        columnIndex_feas3,
        columnIndex_feas4,

        values_feas1,
        values_feas2,
        values_feas3,
        values_feas4,

        B,
        B2,

        M_adj,
        M_fea,
        P_w,

        zero_point_lhs,
        zero_point_rhs,

        C_buffer11,
        C_buffer12,
        C_buffer13,
        C_buffer14,

        C_buffer21,
        C_buffer22,
        C_buffer23,
        C_buffer24,

        C_buffer31,
        C_buffer32,
        C_buffer33,
        C_buffer34,

        C_buffer41,
        C_buffer42,
        C_buffer43,
        C_buffer44,

        A_buffer11,
        A_buffer21,
        A_buffer31,
        A_buffer41,

        linear_pipo,
        layer_loop
    );

    std::cout << "Done loop fea" << std::endl;

    loop_adj2(
        nnz_adj1,
        nnz_adj2,
        nnz_adj3,
        nnz_adj4,

        beta_qu,
        f_align,
        quantization_scale_adj,
        quantization_scale_w,
        deq_factor,
        model,
        srelu,

        rowPtr_adj1,
        rowPtr_adj2,
        rowPtr_adj3,
        rowPtr_adj4,

        columnIndex_adj1,
        columnIndex_adj2,
        columnIndex_adj3,
        columnIndex_adj4,

        values_adj1,
        values_adj2,
        values_adj3,
        values_adj4,

        N_adj,
        M_adj,
        P_w,

        zero_point_lhs,
        zero_point_rhs,

        ate_m,

        A_buffer11,
        A_buffer21,
        A_buffer31,
        A_buffer41,

        E1,
        S1,

        C_buffer11,
        C_buffer12,
        C_buffer13,
        C_buffer14,

        C_buffer21,
        C_buffer22,
        C_buffer23,
        C_buffer24,

        C_buffer31,
        C_buffer32,
        C_buffer33,
        C_buffer34,

        C_buffer41,
        C_buffer42,
        C_buffer43,
        C_buffer44,

        linear_pipo,
        layer_loop,

        D1,
        D2,
        D3,
        D4,

        DS1,
        DS1R,
        DS1C,
        DS2,
        DS3,
        DS4
    );
}

// =============================================================================================
// =============================================================================================
// MMULT TOP
// =============================================================================================
// =============================================================================================
/*
 * FPGA BRAM usage note:
 *
 * The amount of data stored locally in the FPGA is approximately:
 *
 *     B_HEIGHT * B_WIDTH_BLOCK + A_WIDTH + B_WIDTH_BLOCK
 *
 * This should be smaller than the available FPGA BRAM capacity.
 *
 * gemm_mode encoding:
 *
 *   gemm_mode | fea | adj | Meaning
 *   ----------|-----|-----|----------------------------------------
 *      0      |  0  |  0  | Dense feature, dense adjacency
 *      1      |  0  |  1  | Dense feature, sparse adjacency
 *      2      |  1  |  0  | Sparse feature, dense adjacency
 *      3      |  1  |  1  | Sparse feature, sparse adjacency
 */

void mmult_top(
    bool load_weights,
    int beta_qu,
    int f_align,
    int beta_qul,
    int f_alignl,
    float quantization_scale_adj,
    float quantization_scale_fea[5],
    float quantization_scale_w[5],
    float quantization_scale_lin[5],
    float deq_factor[5],
    ap_uint<8> model[5],
    float srelu[5],
    STYPE scale_fea[5],
    ITYPE* max_fea,
    ap_int<32> layer_count,
    int quantized_multiplier,
    int quantized_multiplierl,
    ap_int<32>* shift,
    ap_int<32>* bias,
    ap_int<32> bias_count,
    ap_int<64>* profiling,
    ap_int<8> zero_point_lhs,
    ap_int<8> zero_point_rhs,
    ap_int<8> zero_point_dst,
    ap_int<8> clamp_max,
    ap_int<8> clamp_min,
    int N_adj,
    int M_adj,
    int M_fea,
    ap_uint<8> P_w[5],

    INTYPES* B,
    INTYPES* B2,

    OUTTYPE* D1,
    OUTTYPE* D2,
    OUTTYPE* D3,
    OUTTYPE* D4,

    hls::stream<ASTYPE>& DS1,
    hls::stream<ASTYPE>& DS1R,
    hls::stream<ASTYPE>& DS1C,
    hls::stream<ASTYPE>& DS2,
    hls::stream<ASTYPE>& DS3,
    hls::stream<ASTYPE>& DS4,

    OUTTYPE* E1,
    OUTTYPE* S1,
    INTYPE* ate_m,

    int array_c_adjust,

    int nnz_fea1,
    int nnz_fea2,
    int nnz_fea3,
    int nnz_fea4,

    int* rowPtr_fea1,
    int* rowPtr_fea2,
    int* rowPtr_fea3,
    int* rowPtr_fea4,

    int* columnIndex_fea1,
    int* columnIndex_fea2,
    int* columnIndex_fea3,
    int* columnIndex_fea4,

    INTYPE* values_fea1,
    INTYPE* values_fea2,
    INTYPE* values_fea3,
    INTYPE* values_fea4,

    hls::stream<ASTYPE>& rowPtr_feas1,
    hls::stream<ASTYPE>& rowPtr_feas2,
    hls::stream<ASTYPE>& rowPtr_feas3,
    hls::stream<ASTYPE>& rowPtr_feas4,

    hls::stream<ASTYPE>& columnIndex_feas1,
    hls::stream<ASTYPE>& columnIndex_feas2,
    hls::stream<ASTYPE>& columnIndex_feas3,
    hls::stream<ASTYPE>& columnIndex_feas4,

    hls::stream<ASTYPE>& values_feas1,
    hls::stream<ASTYPE>& values_feas2,
    hls::stream<ASTYPE>& values_feas3,
    hls::stream<ASTYPE>& values_feas4,

    int nnz_adj1,
    int nnz_adj2,
    int nnz_adj3,
    int nnz_adj4,

    int* rowPtr_adj1,
    int* rowPtr_adj2,
    int* rowPtr_adj3,
    int* rowPtr_adj4,

    int* columnIndex_adj1,
    int* columnIndex_adj2,
    int* columnIndex_adj3,
    int* columnIndex_adj4,

    INTYPE* values_adj1,
    INTYPE* values_adj2,
    INTYPE* values_adj3,
    INTYPE* values_adj4
)
{
    // ---------------------------------------------------------------------
    // AXI-Lite control interface
    // ---------------------------------------------------------------------

    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE s_axilite port=load_weights bundle=control
    #pragma HLS INTERFACE s_axilite port=beta_qu bundle=control
    #pragma HLS INTERFACE s_axilite port=f_align bundle=control
    #pragma HLS INTERFACE s_axilite port=beta_qul bundle=control
    #pragma HLS INTERFACE s_axilite port=f_alignl bundle=control
    #pragma HLS INTERFACE s_axilite port=deq_factor bundle=control

    #pragma HLS INTERFACE s_axilite port=nnz_fea1 bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_fea2 bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_fea3 bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_fea4 bundle=control

    #pragma HLS INTERFACE s_axilite port=nnz_adj1 bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_adj2 bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_adj3 bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz_adj4 bundle=control

    #pragma HLS INTERFACE s_axilite port=quantization_scale_adj bundle=control
    #pragma HLS INTERFACE s_axilite port=quantization_scale_fea bundle=control
    #pragma HLS INTERFACE s_axilite port=quantization_scale_w bundle=control
    #pragma HLS INTERFACE s_axilite port=quantization_scale_lin bundle=control

    #pragma HLS INTERFACE s_axilite port=bias_count bundle=control
    #pragma HLS INTERFACE s_axilite port=zero_point_lhs bundle=control
    #pragma HLS INTERFACE s_axilite port=zero_point_rhs bundle=control
    #pragma HLS INTERFACE s_axilite port=zero_point_dst bundle=control
    #pragma HLS INTERFACE s_axilite port=clamp_max bundle=control
    #pragma HLS INTERFACE s_axilite port=clamp_min bundle=control

    #pragma HLS INTERFACE s_axilite port=N_adj bundle=control
    #pragma HLS INTERFACE s_axilite port=M_adj bundle=control
    #pragma HLS INTERFACE s_axilite port=M_fea bundle=control
    #pragma HLS INTERFACE s_axilite port=P_w bundle=control
    #pragma HLS INTERFACE s_axilite port=array_c_adjust bundle=control
    #pragma HLS INTERFACE s_axilite port=model bundle=control

    #pragma HLS INTERFACE s_axilite port=layer_count bundle=control
    #pragma HLS INTERFACE s_axilite port=quantized_multiplier bundle=control
    #pragma HLS INTERFACE s_axilite port=quantized_multiplierl bundle=control

    // ---------------------------------------------------------------------
    // AXI-stream output interfaces
    // ---------------------------------------------------------------------

    #pragma HLS INTERFACE axis port=DS1 depth=64000
    #pragma HLS INTERFACE axis port=DS1R depth=64000
    #pragma HLS INTERFACE axis port=DS1C depth=64000
    #pragma HLS INTERFACE axis port=DS2 depth=4096
    #pragma HLS INTERFACE axis port=DS3 depth=4096
    #pragma HLS INTERFACE axis port=DS4 depth=4096

    // ---------------------------------------------------------------------
    // AXI-stream feature CSR interfaces
    // ---------------------------------------------------------------------

    #pragma HLS INTERFACE axis port=columnIndex_feas1 depth=4096
    #pragma HLS INTERFACE axis port=columnIndex_feas2 depth=4096
    #pragma HLS INTERFACE axis port=columnIndex_feas3 depth=4096
    #pragma HLS INTERFACE axis port=columnIndex_feas4 depth=4096

    #pragma HLS INTERFACE axis port=rowPtr_feas1 depth=4096
    #pragma HLS INTERFACE axis port=rowPtr_feas2 depth=4096
    #pragma HLS INTERFACE axis port=rowPtr_feas3 depth=4096
    #pragma HLS INTERFACE axis port=rowPtr_feas4 depth=4096

    #pragma HLS INTERFACE axis port=values_feas1 depth=64000
    #pragma HLS INTERFACE axis port=values_feas2 depth=4096
    #pragma HLS INTERFACE axis port=values_feas3 depth=4096
    #pragma HLS INTERFACE axis port=values_feas4 depth=4096

    // ---------------------------------------------------------------------
    // AXI memory interfaces: profiling
    // ---------------------------------------------------------------------

    #pragma HLS INTERFACE m_axi port=profiling depth=16 offset=slave bundle=profiling

    // ---------------------------------------------------------------------
    // AXI memory interfaces: feature CSR matrix
    // ---------------------------------------------------------------------

    #pragma HLS INTERFACE m_axi port=rowPtr_fea1 depth=64000 offset=slave bundle=rowPtr_fea1
    #pragma HLS INTERFACE m_axi port=rowPtr_fea2 depth=4096 offset=slave bundle=rowPtr_fea2
    #pragma HLS INTERFACE m_axi port=rowPtr_fea3 depth=4096 offset=slave bundle=rowPtr_fea3
    #pragma HLS INTERFACE m_axi port=rowPtr_fea4 depth=4096 offset=slave bundle=rowPtr_fea4

    #pragma HLS INTERFACE m_axi port=columnIndex_fea1 depth=64000 offset=slave bundle=columnIndex_fea1
    #pragma HLS INTERFACE m_axi port=columnIndex_fea2 depth=4096 offset=slave bundle=columnIndex_fea2
    #pragma HLS INTERFACE m_axi port=columnIndex_fea3 depth=4096 offset=slave bundle=columnIndex_fea3
    #pragma HLS INTERFACE m_axi port=columnIndex_fea4 depth=4096 offset=slave bundle=columnIndex_fea4

    #pragma HLS INTERFACE m_axi port=values_fea1 depth=64000 offset=slave bundle=values_fea1
    #pragma HLS INTERFACE m_axi port=values_fea2 depth=4096 offset=slave bundle=values_fea2
    #pragma HLS INTERFACE m_axi port=values_fea3 depth=4096 offset=slave bundle=values_fea3
    #pragma HLS INTERFACE m_axi port=values_fea4 depth=4096 offset=slave bundle=values_fea4

    // ---------------------------------------------------------------------
    // AXI memory interfaces: adjacency CSR matrix
    // ---------------------------------------------------------------------

    #pragma HLS INTERFACE m_axi port=rowPtr_adj1 depth=64000 offset=slave bundle=rowPtr_adj1
    #pragma HLS INTERFACE m_axi port=rowPtr_adj2 depth=4096 offset=slave bundle=rowPtr_adj2
    #pragma HLS INTERFACE m_axi port=rowPtr_adj3 depth=4096 offset=slave bundle=rowPtr_adj3
    #pragma HLS INTERFACE m_axi port=rowPtr_adj4 depth=4096 offset=slave bundle=rowPtr_adj4

    #pragma HLS INTERFACE m_axi port=columnIndex_adj1 depth=64000 offset=slave bundle=columnIndex_adj1
    #pragma HLS INTERFACE m_axi port=columnIndex_adj2 depth=4096 offset=slave bundle=columnIndex_adj2
    #pragma HLS INTERFACE m_axi port=columnIndex_adj3 depth=4096 offset=slave bundle=columnIndex_adj3
    #pragma HLS INTERFACE m_axi port=columnIndex_adj4 depth=4096 offset=slave bundle=columnIndex_adj4

    #pragma HLS INTERFACE m_axi port=values_adj1 depth=64000 offset=slave bundle=values_adj1
    #pragma HLS INTERFACE m_axi port=values_adj2 depth=4096 offset=slave bundle=values_adj2
    #pragma HLS INTERFACE m_axi port=values_adj3 depth=4096 offset=slave bundle=values_adj3
    #pragma HLS INTERFACE m_axi port=values_adj4 depth=4096 offset=slave bundle=values_adj4

    // ---------------------------------------------------------------------
    // AXI memory interfaces: weights and outputs
    // ---------------------------------------------------------------------

    #pragma HLS INTERFACE m_axi port=B depth=32000 offset=slave bundle=B
    #pragma HLS INTERFACE m_axi port=B2 depth=32000 offset=slave bundle=B2

    #pragma HLS INTERFACE m_axi port=D1 depth=64000 offset=slave bundle=D1
    #pragma HLS INTERFACE m_axi port=D2 depth=1000 offset=slave bundle=D2
    #pragma HLS INTERFACE m_axi port=D3 depth=1000 offset=slave bundle=D3
    #pragma HLS INTERFACE m_axi port=D4 depth=1000 offset=slave bundle=D4

    #pragma HLS INTERFACE m_axi port=E1 depth=64000 offset=slave bundle=E1
    #pragma HLS INTERFACE m_axi port=S1 depth=64000 offset=slave bundle=S1
    #pragma HLS INTERFACE m_axi port=ate_m depth=1000 offset=slave bundle=ate_m

    // ---------------------------------------------------------------------
    // AXI memory interfaces: model and quantization parameters
    // ---------------------------------------------------------------------

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

    // ---------------------------------------------------------------------
    // AXI-Lite pointer control registers
    // ---------------------------------------------------------------------

    #pragma HLS INTERFACE s_axilite port=columnIndex_fea1 bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_fea2 bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_fea3 bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_fea4 bundle=control

    #pragma HLS INTERFACE s_axilite port=rowPtr_fea1 bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea2 bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea3 bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_fea4 bundle=control

    #pragma HLS INTERFACE s_axilite port=columnIndex_adj1 bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj2 bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj3 bundle=control
    #pragma HLS INTERFACE s_axilite port=columnIndex_adj4 bundle=control

    #pragma HLS INTERFACE s_axilite port=rowPtr_adj1 bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj2 bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj3 bundle=control
    #pragma HLS INTERFACE s_axilite port=rowPtr_adj4 bundle=control

    #pragma HLS INTERFACE s_axilite port=values_adj1 bundle=control
    #pragma HLS INTERFACE s_axilite port=values_adj2 bundle=control
    #pragma HLS INTERFACE s_axilite port=values_adj3 bundle=control
    #pragma HLS INTERFACE s_axilite port=values_adj4 bundle=control

    #pragma HLS INTERFACE s_axilite port=values_fea1 bundle=control
    #pragma HLS INTERFACE s_axilite port=values_fea2 bundle=control
    #pragma HLS INTERFACE s_axilite port=values_fea3 bundle=control
    #pragma HLS INTERFACE s_axilite port=values_fea4 bundle=control

    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=B2 bundle=control

    #pragma HLS INTERFACE s_axilite port=D1 bundle=control
    #pragma HLS INTERFACE s_axilite port=D2 bundle=control
    #pragma HLS INTERFACE s_axilite port=D3 bundle=control
    #pragma HLS INTERFACE s_axilite port=D4 bundle=control

    #pragma HLS INTERFACE s_axilite port=E1 bundle=control
    #pragma HLS INTERFACE s_axilite port=S1 bundle=control
    #pragma HLS INTERFACE s_axilite port=profiling bundle=control

    #pragma HLS INTERFACE s_axilite port=quantized_multiplier bundle=control
    #pragma HLS INTERFACE s_axilite port=shift bundle=control
    #pragma HLS INTERFACE s_axilite port=bias bundle=control
    #pragma HLS INTERFACE s_axilite port=ate_m bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_fea bundle=control
    #pragma HLS INTERFACE s_axilite port=max_fea bundle=control
    #pragma HLS INTERFACE s_axilite port=srelu bundle=control

    // ---------------------------------------------------------------------
    // Local parameter storage
    //
    // These arrays cache host-provided parameters locally.
    // Complete partitioning allows parallel access during the computation.
    // ---------------------------------------------------------------------

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

    // ---------------------------------------------------------------------
    // Profiling counters
    // ---------------------------------------------------------------------

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

    // ---------------------------------------------------------------------
    // Load per-layer configuration
    // ---------------------------------------------------------------------

    ap_int<32> layer_loop = layer_count;

    for (int i = 0; i < layer_loop; i++)
    {
        model_int[i][0] = model[i][0];
        model_int[i][1] = model[i][1];
        model_int[i][2] = model[i][2];
        model_int[i][3] = model[i][3];
        model_int[i][4] = model[i][4];
        model_int[i][5] = model[i][5];
        model_int[i][6] = model[i][6];
        model_int[i][7] = model[i][7];

        srelu_int[i] = srelu[i];

        quantization_scale_lin_int[i] = quantization_scale_lin[i];
        quantization_scale_w_int[i]   = quantization_scale_w[i];
        quantization_scale_fea_int[i] = quantization_scale_fea[i];

        deq_factor_int[i] = deq_factor[i];
        scale_fea_int[i]  = scale_fea[i];
        P_w_int[i]        = P_w[i];

#ifndef __SYNTHESIS__
        std::cout
            << "Layer " << i
            << " instruction is "
            << model_int[i][7]
            << model_int[i][6]
            << model_int[i][5]
            << model_int[i][4]
            << model_int[i][3]
            << model_int[i][2]
            << model_int[i][1]
            << model_int[i][0]
            << std::endl;
#endif
    }

    // ---------------------------------------------------------------------
    // Execute accelerator pipeline
    // ---------------------------------------------------------------------

    ITYPE max_fea_val = 0;

    mmult_wrapper(
        load_weights,
        beta_qu,
        f_align,
        beta_qul,
        f_alignl,
        quantization_scale_adj,
        quantization_scale_fea_int,
        quantization_scale_w_int,
        quantization_scale_lin_int,
        deq_factor_int,
        model_int,
        srelu_int,
        scale_fea_int,
        &max_fea_val,
        quantized_multiplier,
        quantized_multiplierl,
        shift_data,
        bias_data,
        bias_count,
        zero_point_lhs,
        zero_point_rhs,
        zero_point_dst,
        clamp_max,
        clamp_min,
        N_adj,
        M_adj,
        M_fea,
        P_w_int,

        B,
        B2,

        D1,
        D2,
        D3,
        D4,

        DS1,
        DS1R,
        DS1C,
        DS2,
        DS3,
        DS4,

        E1,
        S1,
        ate_m,

        array_c_adjust,
        layer_loop,

        nnz_fea1,
        nnz_fea2,
        nnz_fea3,
        nnz_fea4,

        rowPtr_fea1,
        rowPtr_fea2,
        rowPtr_fea3,
        rowPtr_fea4,

        columnIndex_fea1,
        columnIndex_fea2,
        columnIndex_fea3,
        columnIndex_fea4,

        values_fea1,
        values_fea2,
        values_fea3,
        values_fea4,

        rowPtr_feas1,
        rowPtr_feas2,
        rowPtr_feas3,
        rowPtr_feas4,

        columnIndex_feas1,
        columnIndex_feas2,
        columnIndex_feas3,
        columnIndex_feas4,

        values_feas1,
        values_feas2,
        values_feas3,
        values_feas4,

        nnz_adj1,
        nnz_adj2,
        nnz_adj3,
        nnz_adj4,

        rowPtr_adj1,
        rowPtr_adj2,
        rowPtr_adj3,
        rowPtr_adj4,

        columnIndex_adj1,
        columnIndex_adj2,
        columnIndex_adj3,
        columnIndex_adj4,

        values_adj1,
        values_adj2,
        values_adj3,
        values_adj4
    );

    // ---------------------------------------------------------------------
    // Return result metadata
    // ---------------------------------------------------------------------

    *max_fea = max_fea_val;

#ifndef __SYNTHESIS__
    std::cout << "Done mmult wrapper" << std::endl;
#endif
}


// =============================================================================================
// =============================================================================================
// KERNEL MULT
// =============================================================================================
// =============================================================================================
void kernelmult1(
    bool load_weights,
    int beta_qu,
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
    ap_int<32>* shift,
    ap_int<32>* bias,
    ap_int<32> bias_count,
    ap_int<64>* profiling,
    ap_int<8> zero_point_lhs,
    ap_int<8> zero_point_rhs,
    ap_int<8> zero_point_dst,
    ap_int<8> clamp_max,
    ap_int<8> clamp_min,

    // Weight buffers
    INTYPES* array_b,
    INTYPES* array_b2,

    // Output buffers
    OUTTYPE* array_d1,
    OUTTYPE* array_d2,
    OUTTYPE* array_d3,
    OUTTYPE* array_d4,

    // Output streams
    hls::stream<ASTYPE>& stream_d1,
    hls::stream<ASTYPE>& stream_d1r,
    hls::stream<ASTYPE>& stream_d1c,
    hls::stream<ASTYPE>& stream_d2,
    hls::stream<ASTYPE>& stream_d3,
    hls::stream<ASTYPE>& stream_d4,

    OUTTYPE* array_e1,
    OUTTYPE* array_s1,
    INTYPE* ate_m,

    // Feature matrix values
    INTYPE* values_fea1,
    INTYPE* values_fea2,
    INTYPE* values_fea3,
    INTYPE* values_fea4,

    hls::stream<ASTYPE>& values_feas1,
    hls::stream<ASTYPE>& values_feas2,
    hls::stream<ASTYPE>& values_feas3,
    hls::stream<ASTYPE>& values_feas4,

    // Feature matrix column indices
    int* colIndices_fea1,
    int* colIndices_fea2,
    int* colIndices_fea3,
    int* colIndices_fea4,

    hls::stream<ASTYPE>& columnIndex_feas1,
    hls::stream<ASTYPE>& columnIndex_feas2,
    hls::stream<ASTYPE>& columnIndex_feas3,
    hls::stream<ASTYPE>& columnIndex_feas4,

    // Feature matrix non-zero counts
    int nnz_fea1,
    int nnz_fea2,
    int nnz_fea3,
    int nnz_fea4,

    // Feature matrix row pointers
    int* rowPtr_fea1,
    int* rowPtr_fea2,
    int* rowPtr_fea3,
    int* rowPtr_fea4,

    hls::stream<ASTYPE>& rowPtr_feas1,
    hls::stream<ASTYPE>& rowPtr_feas2,
    hls::stream<ASTYPE>& rowPtr_feas3,
    hls::stream<ASTYPE>& rowPtr_feas4,

    // Adjacency matrix values
    INTYPE* values_adj1,
    INTYPE* values_adj2,
    INTYPE* values_adj3,
    INTYPE* values_adj4,

    // Adjacency matrix column indices
    int* colIndices_adj1,
    int* colIndices_adj2,
    int* colIndices_adj3,
    int* colIndices_adj4,

    // Adjacency matrix non-zero counts
    int nnz_adj1,
    int nnz_adj2,
    int nnz_adj3,
    int nnz_adj4,

    // Adjacency matrix row pointers
    int* rowPtr_adj1,
    int* rowPtr_adj2,
    int* rowPtr_adj3,
    int* rowPtr_adj4,

    // Matrix dimensions
    int N_adj,
    int M_adj,
    int M_fea,
    int P_w
)
{
    int array_c_adjust = N_adj;

    ap_uint<8> P_w_int[5];
    P_w_int[0] = P_w;

    float srelu[5];
    srelu[0] = 0.0;

    std::cout << "kernel starting" << std::endl;

    const int quantized_multiplierl = 8;
    const int beta_qul = 255;
    const int f_alignl = 0;

    mmult_top(
        load_weights,
        beta_qu,
        f_align,
        beta_qul,
        f_alignl,
        quantization_scale_adj,
        quantization_scale_fea,
        quantization_scale_w,
        quantization_scale_lin,
        deq_factor,
        model,
        srelu,
        scale_fea,
        max_fea,
        layer_count,
        quantized_multiplier,
        quantized_multiplierl,
        shift,
        bias,
        bias_count,
        profiling,
        zero_point_lhs,
        zero_point_rhs,
        zero_point_dst,
        clamp_max,
        clamp_min,

        N_adj,
        M_adj,
        M_fea,
        P_w_int,

        array_b,
        array_b2,

        array_d1,
        array_d2,
        array_d3,
        array_d4,

        stream_d1,
        stream_d1r,
        stream_d1c,
        stream_d2,
        stream_d3,
        stream_d4,

        array_e1,
        array_s1,
        ate_m,
        array_c_adjust,

        nnz_fea1,
        nnz_fea2,
        nnz_fea3,
        nnz_fea4,

        rowPtr_fea1,
        rowPtr_fea2,
        rowPtr_fea3,
        rowPtr_fea4,

        colIndices_fea1,
        colIndices_fea2,
        colIndices_fea3,
        colIndices_fea4,

        values_fea1,
        values_fea2,
        values_fea3,
        values_fea4,

        rowPtr_feas1,
        rowPtr_feas2,
        rowPtr_feas3,
        rowPtr_feas4,

        columnIndex_feas1,
        columnIndex_feas2,
        columnIndex_feas3,
        columnIndex_feas4,

        values_feas1,
        values_feas2,
        values_feas3,
        values_feas4,

        nnz_adj1,
        nnz_adj2,
        nnz_adj3,
        nnz_adj4,

        rowPtr_adj1,
        rowPtr_adj2,
        rowPtr_adj3,
        rowPtr_adj4,

        colIndices_adj1,
        colIndices_adj2,
        colIndices_adj3,
        colIndices_adj4,

        values_adj1,
        values_adj2,
        values_adj3,
        values_adj4
    );

#ifndef __SYNTHESIS__
    std::cout << "0  output " << array_d1[0]  << std::endl;
    std::cout << "3  output " << array_d1[3]  << std::endl;
    std::cout << "7  output " << array_d1[7]  << std::endl;
    std::cout << "9  output " << array_d1[9]  << std::endl;
    std::cout << "13 output " << array_d1[13] << std::endl;
    std::cout << "20 output " << array_d1[20] << std::endl;
    std::cout << "33 output " << array_d1[33] << std::endl;
#endif

    std::cout << "kernel done" << std::endl;
}