
typedef unsigned long u32;


void loop_fea(
    bool load_weights,
    int beta_qu,
    int f_align,
    int beta_qul,
    int f_alignl,
    float quantization_scale_fea[5],
    float quantization_scale_w[5],
    float quantization_scale_lin[5],
    ap_uint<1> model[5][8],
    STYPE scale_fea[5],
    ITYPE* max_fea,
    int quantized_multiplier,
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
    INTYPES* B,
    INTYPES* B2,
    int N_fea,
    int M_fea,
    ap_uint<8> P_w[5],
    ap_int<8> zero_point_lhs,
    ap_int<8> zero_point_rhs,
    #if (PIPO_BLOCKS>=2)
    hls::stream_of_blocks<buf>& C_buffer11,
    hls::stream_of_blocks<buf>& C_buffer12,
    #else
    buf C_buffer11,
    hls::stream_of_blocks<buf>& C_buffer12,
    #endif
    hls::stream_of_blocks<buf>& C_buffer13,
    hls::stream_of_blocks<buf>& C_buffer14,
    hls::stream_of_blocks<buf>& C_buffer21,
    hls::stream_of_blocks<buf>& C_buffer22,
    hls::stream_of_blocks<buf>& C_buffer23,
    hls::stream_of_blocks<buf>& C_buffer24,
    hls::stream_of_blocks<buf>& C_buffer31,
    hls::stream_of_blocks<buf>& C_buffer32,
    hls::stream_of_blocks<buf>& C_buffer33,
    hls::stream_of_blocks<buf>& C_buffer34,
    hls::stream_of_blocks<buf>& C_buffer41,
    hls::stream_of_blocks<buf>& C_buffer42,
    hls::stream_of_blocks<buf>& C_buffer43,
    hls::stream_of_blocks<buf>& C_buffer44,
    #if (PIPO_BLOCKS>=2)
    hls::stream_of_blocks<buf>& A_buffer11,
    hls::stream_of_blocks<buf>& A_buffer21,
    #else
    buf A_buffer11,
    hls::stream_of_blocks<buf>& A_buffer21,
    #endif
    hls::stream_of_blocks<buf>& A_buffer31,
    hls::stream_of_blocks<buf>& A_buffer41,
    #if (PIPO_BLOCKS>=2)
    hls::stream_of_blocks<buf>& linear_pipo,
    #else
    buf linear_pipo,
    #endif
    int layer_loop
)
{
    BTYPE B_accel1[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel1 block factor=BLOCK/2 dim=2

    BTYPE B_accel2[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel2 block factor=BLOCK/2 dim=2

    BTYPE B_accel3[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel3 block factor=BLOCK/2 dim=2

    BTYPE B_accel4[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel4 block factor=BLOCK/2 dim=2

    BTYPE B_accel12[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel12 block factor=BLOCK/2 dim=2

    BTYPE B_accel22[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel22 block factor=BLOCK/2 dim=2

    BTYPE B_accel32[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel32 block factor=BLOCK/2 dim=2

    BTYPE B_accel42[B_HEIGHT][B_WIDTH_BLOCK];
    #pragma HLS array_partition variable=B_accel42 block factor=BLOCK/2 dim=2

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

    // hls::stream<FTYPE> A_fifo_feaq1;
    // #pragma HLS STREAM variable=A_fifo_feaq1 depth=FIFO_DEPTH
    // hls::stream<FTYPE> A_fifo_feaq2;
    // #pragma HLS STREAM variable=A_fifo_feaq2 depth=FIFO_DEPTH
    // hls::stream<FTYPE> A_fifo_feaq3;
    // #pragma HLS STREAM variable=A_fifo_feaq3 depth=FIFO_DEPTH
    // hls::stream<FTYPE> A_fifo_feaq4;
    // #pragma HLS STREAM variable=A_fifo_feaq4 depth=FIFO_DEPTH

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

    hls::stream<int> col_indices_fifo_fea1;
    #pragma HLS STREAM variable=col_indices_fifo_fea1 depth=FIFO_DEPTH

    hls::stream<int> col_indices_fifo_fea2;
    #pragma HLS STREAM variable=col_indices_fifo_fea2 depth=FIFO_DEPTH

    hls::stream<int> col_indices_fifo_fea3;
    #pragma HLS STREAM variable=col_indices_fifo_fea3 depth=FIFO_DEPTH

    hls::stream<int> col_indices_fifo_fea4;
    #pragma HLS STREAM variable=col_indices_fifo_fea4 depth=FIFO_DEPTH

    hls::stream<int> col_indices_fifo_fea12;
    #pragma HLS STREAM variable=col_indices_fifo_fea12 depth=FIFO_DEPTH

    hls::stream<int> col_indices_fifo_fea22;
    #pragma HLS STREAM variable=col_indices_fifo_fea22 depth=FIFO_DEPTH

    hls::stream<int> col_indices_fifo_fea32;
    #pragma HLS STREAM variable=col_indices_fifo_fea32 depth=FIFO_DEPTH

    hls::stream<int> col_indices_fifo_fea42;
    #pragma HLS STREAM variable=col_indices_fifo_fea42 depth=FIFO_DEPTH

    int B_WIDTH_INT;

    // bool gemm_mode_int, stream_mode_int;
    // int M_fea_int;
    // int last_M_fea_int = 0;

    #if (PIPO_BLOCKS>=2)
    LOOP_FEA : for (int B_index = 0; B_index < layer_loop; B_index++) {
    #else
    int B_index = 0;
    #endif

        #pragma HLS DATAFLOW

        B_WIDTH_INT = B_WIDTH_BLOCK;

        std::cout << "fea layer " << B_index << std::endl;

        // if (layer_loop == 1)
        // {
        //     gemm_mode_int = gemm_mode;
        //     stream_mode_int = stream_mode;
        //     M_fea_int = M_fea;
        // }
        // else
        // {

        // a_values = N * M;
        // else // SPMM
        //     a_values = nnz_fea;

        /* These are the weights. */

        #if FEA_THREADS == 1

        // Read weights before locking the buffer for better performance?

        // hls::write_lock<buf> C_fea13(C_buffer13);
        // hls::write_lock<buf> C_fea14(C_buffer14);
        // hls::write_lock<buf> C_fea21(C_buffer21);
        // hls::write_lock<buf> C_fea22(C_buffer22);
        // hls::write_lock<buf> C_fea23(C_buffer23);
        // hls::write_lock<buf> C_fea24(C_buffer24);
        // hls::write_lock<buf> C_fea31(C_buffer31);
        // hls::write_lock<buf> C_fea32(C_buffer32);
        // hls::write_lock<buf> C_fea33(C_buffer33);
        // hls::write_lock<buf> C_fea34(C_buffer34);
        // hls::write_lock<buf> C_fea41(C_buffer41);
        // hls::write_lock<buf> C_fea42(C_buffer42);
        // hls::write_lock<buf> C_fea43(C_buffer43);
        // hls::write_lock<buf> C_fea44(C_buffer44);

        // std::cout << "Loop FEA " << std::endl;

        std::cout << "load weights " << std::endl;

        #if LINEAR_ONLY == 0
        readb(load_weights, model, beta_qu, f_align, quantization_scale_w, M_fea, P_w, B_index, B_accel1, B); // GNN weights
        #endif

        #if GNN_ONLY == 0
        readbl(load_weights, model, beta_qu, f_align, quantization_scale_w, M_fea, P_w, B_index, B_accel12, B2); // Linear weights
        #endif

        #if (PIPO_BLOCKS>=2)
        hls::write_lock<buf> C_fea11(C_buffer11); // One output for ADJ_LOOP and one for attention.
        hls::write_lock<buf> linear_fea(linear_pipo);

            #if GAT_ENABLE == 1
            hls::write_lock<buf> A_fea11(A_buffer11); // The same output is written to two buffers.
            #else
            QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];
            #endif
        #else
            #if GAT_ENABLE == 1
            #else
            QTYPE A_fea11[B_HEIGHT][B_WIDTH_BLOCK];
            #endif
        #endif

        // Read sparse matrices.

        // std::cout << "reada " << std::endl;

        // int max_fea1;
        int first_row1, first_row2, first_row3, first_row4;
        int row_count1, row_count2, row_count3, row_count4;

        int N_fea_block = N_fea;
        int N_fea_rest = 0;
        row_count1 = N_fea_block;
        // row_count2 = N_fea_block;
        // row_count3 = N_fea_block;
        // row_count2 = N_fea_block + N_fea_rest;
        first_row1 = 0;
        // first_row2 = N_fea_block;
        // first_row3 = 2 * N_fea_block;
        // first_row4 = 3 * N_fea_block;

        // std::cout << "gemm_mode_int " << gemm_mode_int << std::endl;
        int last_index1;

        // reada1(exit_loop,first_row1,row_count1,A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,B_index_loop,tail,
        //        rowPtr_fea1,columnIndex_fea1,values_fea1);

        #if (COO_MODE == 0)
        reada1_csr(
            beta_qu, f_align, quantization_scale_fea, last_index1, stream_mode_int, gemm_mode_int, M_fea_int,
            first_row1, row_count1,
            A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
            rowPtr_fea1, columnIndex_fea1, values_fea1, values_feas1
        );
        #else
        reada1_coo(
            nnz_fea1, beta_qu, f_align, beta_qul, f_alignl, quantization_scale_fea, quantization_scale_lin, last_index1,
            model, M_fea, first_row1, row_count1,
            A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
            A_fifo_fea12, col_indices_fifo_fea12, rnnz_fifo_fea12,
            rowPtr_fea1, columnIndex_fea1, values_fea1,
            rowPtr_feas1, columnIndex_feas1, values_feas1,
            B_index, layer_loop
        );
        #endif

        // reada1(first_row2,row_count2,A_fifo_fea2,col_indices_fifo_fea2,rnnz_fifo_fea2,B_index_loop,tail,
        //        rowPtr_fea2,columnIndex_fea2,values_fea2);
        // reada1(first_row3,row_count3,A_fifo_fea3,col_indices_fifo_fea3,rnnz_fifo_fea3,B_index_loop,tail,
        //        rowPtr_fea3,columnIndex_fea3,values_fea3);
        // reada1(first_row4,row_count4,A_fifo_fea4,col_indices_fifo_fea4,rnnz_fifo_fea4,B_index_loop,tail,
        //        rowPtr_fea4,columnIndex_fea4,values_fea4);

        // quant1(A_fifo_fea1,col_indices_fifo_fea1,rnnz_fifo_fea1,
        //        A_fifo_feaq1,col_indices_fifo_feaq1,rnnz_fifo_feaq1,last_index1,quantization_scale_fea);

        // check_fifo_0(105165, A_fifo_fea1, A_fifo_fea1_out);

        // Inputs: A_fifo_fea, col_indices_fifo_fea, rnnz_fifo_fea, and B_accel
        // Output: C_buffer
        // compute1 performs FEA * W = C

        // std::cout << "compute1" << std::endl;

        std::cout << "COMPUTE1 " << std::endl;

        ITYPE max_fea1, max_fea2;

        #if (PIPO_BLOCKS>=2)
            #if LINEAR_ONLY == 0
            compute1_1(
                scale_fea, &max_fea1, quantized_multiplier, model, zero_point_lhs, zero_point_rhs,
                first_row1, row_count1,
                A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                B_accel1, C_fea11,
                A_fea11, B_index
            );
            #endif

            #if GNN_ONLY == 0
            compute1_12(
                scale_fea, &max_fea2, quantized_multiplier, model, zero_point_lhs, zero_point_rhs,
                first_row1, row_count1,
                A_fifo_fea12, col_indices_fifo_fea12, rnnz_fifo_fea12,
                B_accel12, linear_fea, B_index
            );
            #endif
        #else
            #if LINEAR_ONLY == 0
            compute1_1(
                scale_fea, &max_fea1, quantized_multiplier, gemm_mode_int, zero_point_lhs, zero_point_rhs,
                first_row1, row_count1,
                A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                B_accel1, C_buffer11,
                A_buffer11
            );
            #endif

            #if GNN_ONLY == 0
            compute1_12(
                scale_fea, &max_fea2, quantized_multiplier, gemm_mode_int, zero_point_lhs, zero_point_rhs,
                first_row1, row_count1,
                A_fifo_fea12, col_indices_fifo_fea12, rnnz_fifo_fea12,
                B_accel12, linear_pipo
            );
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
        // hls::write_lock<buf> C_fea13(C_buffer13);
        // hls::write_lock<buf> C_fea14(C_buffer14);
        hls::write_lock<buf> C_fea21(C_buffer21);

        // hls::write_lock<buf> C_fea23(C_buffer23);
        // hls::write_lock<buf> C_fea24(C_buffer24);
        // hls::write_lock<buf> C_fea31(C_buffer31);
        // hls::write_lock<buf> C_fea32(C_buffer32);
        // hls::write_lock<buf> C_fea33(C_buffer33);
        // hls::write_lock<buf> C_fea34(C_buffer34);
        // hls::write_lock<buf> C_fea41(C_buffer41);
        // hls::write_lock<buf> C_fea42(C_buffer42);
        // hls::write_lock<buf> C_fea43(C_buffer43);
        // hls::write_lock<buf> C_fea44(C_buffer44);

        // std::cout << "Loop FEA " << std::endl;

        for (int j = 0; j < B_WIDTH_INT; j++) {
            LOOP_BLOCKB : for (int i = 0; i < M_fea; i++) {
                // #pragma HLS loop_tripcount min=84 max=84 avg=84
                #pragma HLS PIPELINE
                // #pragma HLS loop_tripcount min=16 max=16 avg=16
                BTYPE B_accel_temp = B[i + j * M_fea + B_index * B_WIDTH_BLOCK * M_fea];
                B_accel1[i][j] = B_accel_temp;
                B_accel2[i][j] = B_accel_temp;
                // B_accel3[i][j] = B_accel_temp;
                // B_accel4[i][j] = B_accel_temp;

                // std::cout << " " << i << " " << j << " " << B_accel[i][j] << std::endl;
            }
        }

        int first_row1, first_row2, first_row3, first_row4;
        int row_count1, row_count2, row_count3, row_count4;

        int N_fea_block = N_fea / 2;
        int N_fea_rest = N_fea % 2;
        row_count1 = N_fea_block;
        row_count2 = N_fea_block + N_fea_rest;
        first_row1 = 0;
        first_row2 = N_fea_block;

        std::cout << "Thread fea 1" << std::endl;
        reada1(gemm_mode, M_fea, first_row1, row_count1, A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1, B_index_loop, tail,
               rowPtr_fea1, columnIndex_fea1, values_fea1);

        std::cout << "Thread fea 2" << std::endl;
        reada1(gemm_mode, M_fea, first_row2, row_count2, A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2, B_index_loop, tail,
               rowPtr_fea2, columnIndex_fea2, values_fea2);

        #if ADJ_THREADS == 2
        compute1_2(
            gemm_mode, zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
            B_accel1, C_fea11, C_fea12,
            // C_fea13, C_fea14,
            B_index, B_index_loop, tail
        );

        compute1_2(
            gemm_mode, zero_point_lhs, zero_point_rhs,
            first_row2, row_count2,
            A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
            B_accel2, C_fea21, C_fea22,
            // C_fea23, C_fea24,
            B_index, B_index_loop, tail
        );
        #endif

        #if ADJ_THREADS == 1
        compute1_1(
            gemm_mode, zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
            B_accel1, C_fea11
            // C_fea12,
            // C_fea13, C_fea14,
        );

        compute1_1(
            gemm_mode, zero_point_lhs, zero_point_rhs,
            first_row2, row_count2,
            A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
            B_accel2, C_fea21
            // C_fea12,
            // C_fea13, C_fea14,
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
            // hls::write_lock<buf> C_fea13(C_buffer13);
            // hls::write_lock<buf> C_fea14(C_buffer14);
            hls::write_lock<buf> C_fea21(C_buffer21);
            hls::write_lock<buf> C_fea22(C_buffer22);
            // hls::write_lock<buf> C_fea23(C_buffer23);
            // hls::write_lock<buf> C_fea24(C_buffer24);
            hls::write_lock<buf> C_fea31(C_buffer31);
            hls::write_lock<buf> C_fea32(C_buffer32);
            // hls::write_lock<buf> C_fea33(C_buffer33);
            // hls::write_lock<buf> C_fea34(C_buffer34);
            hls::write_lock<buf> C_fea41(C_buffer41);
            hls::write_lock<buf> C_fea42(C_buffer42);
            // hls::write_lock<buf> C_fea43(C_buffer43);
            // hls::write_lock<buf> C_fea44(C_buffer44);

            #endif

        // std::cout << "Loop FEA " << std::endl;

        for (int j = 0; j < B_WIDTH_INT; j++) {
            LOOP_BLOCKB : for (int i = 0; i < M_fea; i++) {
                // #pragma HLS loop_tripcount min=84 max=84 avg=84
                #pragma HLS PIPELINE
                // #pragma HLS loop_tripcount min=16 max=16 avg=16
                INTYPE BF = (INTYPE)B[i + j * M_fea + B_index * B_WIDTH_BLOCK * M_fea];
                BTYPE B_accel_temp;
                #if (INT_QUANT_W == 1)
                quantw(B_accel_temp, BF, quantization_scale_w, f_align, beta_qu);
                #else
                B_accel_temp = BF;
                #endif

                B_accel1[i][j] = B_accel_temp;
                B_accel2[i][j] = B_accel_temp;
                B_accel3[i][j] = B_accel_temp;
                B_accel4[i][j] = B_accel_temp;

                // std::cout << " " << i << " " << j << " " << B_accel[i][j] << std::endl;
            }
        }

        // Read sparse matrices.

        // std::cout << "reada " << std::endl;

        int first_row1, first_row2, first_row3, first_row4;
        int row_count1, row_count2, row_count3, row_count4;

        int N_fea_block = N_fea / 4;
        int N_fea_rest = N_fea % 4;
        row_count1 = N_fea_block;
        row_count2 = N_fea_block;
        row_count3 = N_fea_block;
        row_count4 = N_fea_block + N_fea_rest;
        first_row1 = 0;
        first_row2 = N_fea_block;
        first_row3 = 2 * N_fea_block;
        first_row4 = 3 * N_fea_block;

        ITYPE max_fea1, max_fea2, max_fea3, max_fea4;

        // std::cout << "READA1 " << std::endl;

        int last_index1, last_index2, last_index3, last_index4;

        #if (COO_MODE == 0)
        reada1_csr(beta_qu, f_align, quantization_scale_fea, last_index1, stream_mode, gemm_mode, M_fea, first_row1, row_count1, A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                  rowPtr_fea1, columnIndex_fea1, values_fea1, values_feas1);
        reada1_csr(beta_qu, f_align, quantization_scale_fea, last_index2, stream_mode, gemm_mode, M_fea, first_row2, row_count2, A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
                  rowPtr_fea2, columnIndex_fea2, values_fea2, values_feas2);
        reada1_csr(beta_qu, f_align, quantization_scale_fea, last_index3, stream_mode, gemm_mode, M_fea, first_row3, row_count3, A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3,
                  rowPtr_fea3, columnIndex_fea3, values_fea3, values_feas3);
        reada1_csr(beta_qu, f_align, quantization_scale_fea, last_index4, stream_mode, gemm_mode, M_fea, first_row4, row_count4, A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4,
                  rowPtr_fea4, columnIndex_fea4, values_fea4, values_feas4);
        #else
        reada1_coo(nnz_fea1, beta_qu, f_align, quantization_scale_fea, last_index1, stream_mode, gemm_mode, M_fea, first_row1, row_count1, A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1,
                   rowPtr_fea1, columnIndex_fea1, values_fea1, values_feas1);
        reada1_coo(nnz_fea2, beta_qu, f_align, quantization_scale_fea, last_index2, stream_mode, gemm_mode, M_fea, first_row2, row_count2, A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2,
                   rowPtr_fea2, columnIndex_fea2, values_fea2, values_feas2);
        reada1_coo(nnz_fea3, beta_qu, f_align, quantization_scale_fea, last_index3, stream_mode, gemm_mode, M_fea, first_row3, row_count3, A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3,
                   rowPtr_fea3, columnIndex_fea3, values_fea3, values_feas3);
        reada1_coo(nnz_fea4, beta_qu, f_align, quantization_scale_fea, last_index4, stream_mode, gemm_mode, M_fea, first_row4, row_count4, A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4,
                   rowPtr_fea4, columnIndex_fea4, values_fea4, values_feas4);
        #endif

        // check_fifo_0(a_values, A_fifo, A_fifo_out);

        // Inputs: A_fifo_fea, col_indices_fifo_fea, rnnz_fifo_fea, and B_accel
        // Outputs: C_buffer
        // compute1 performs FEA * W = C

        // std::cout << "compute1" << std::endl;
        // std::cout << "COMPUTE1 " << std::endl;

            #if ADJ_THREADS == 4

            compute1_4(scale_fea, &max_fea1, quantized_multiplier, gemm_mode, zero_point_lhs, zero_point_rhs, first_row1, row_count1, A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1, B_accel1, C_fea11, C_fea12, C_fea13, C_fea14, A_fea11);
            compute1_4(scale_fea, &max_fea2, quantized_multiplier, gemm_mode, zero_point_lhs, zero_point_rhs, first_row2, row_count2, A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2, B_accel2, C_fea21, C_fea22, C_fea23, C_fea24, A_fea21);
            compute1_4(scale_fea, &max_fea3, quantized_multiplier, gemm_mode, zero_point_lhs, zero_point_rhs, first_row3, row_count3, A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3, B_accel3, C_fea31, C_fea32, C_fea33, C_fea34, A_fea31);
            compute1_4(scale_fea, &max_fea4, quantized_multiplier, gemm_mode, zero_point_lhs, zero_point_rhs, first_row4, row_count4, A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4, B_accel4, C_fea41, C_fea42, C_fea43, C_fea44, A_fea41);

            *max_fea = max_fea1;

            #endif

            #if ADJ_THREADS == 2

            compute1_2(gemm_mode, zero_point_lhs, zero_point_rhs, first_row1, row_count1, A_fifo_fea1, col_indices_fifo_fea1, rnnz_fifo_fea1, B_accel1, C_fea11, C_fea12, // C_fea13, C_fea14,
                       B_index, B_index_loop, tail);
            compute1_2(gemm_mode, zero_point_lhs, zero_point_rhs, first_row2, row_count2, A_fifo_fea2, col_indices_fifo_fea2, rnnz_fifo_fea2, B_accel2, C_fea21, C_fea22, // C_fea23, C_fea24,
                       B_index, B_index_loop, tail);
            compute1_2(gemm_mode, zero_point_lhs, zero_point_rhs, first_row3, row_count3, A_fifo_fea3, col_indices_fifo_fea3, rnnz_fifo_fea3, B_accel3, C_fea31, C_fea32, // C_fea33, C_fea34,
                       B_index, B_index_loop, tail);
            compute1_2(gemm_mode, zero_point_lhs, zero_point_rhs, first_row4, row_count4, A_fifo_fea4, col_indices_fifo_fea4, rnnz_fifo_fea4, B_accel4, C_fea41, C_fea42, // C_fea43, C_fea44,
                       B_index, B_index_loop, tail);

            #endif
        #endif

    #if (PIPO_BLOCKS>=2)
    }
    #endif
}

void loop_adj(
    float deq_factor[5],
    ap_uint<1> model[5][8],
    float srelu[5],
    hls::stream<ITYPE>& A_fifo_adj1,    
    hls::stream<int>& col_indices_fifo_adj1,
    hls::stream<int>& rnnz_fifo_adj1,
    hls::stream<TTYPE>& A_fifo_adj2,
    hls::stream<int>& col_indices_fifo_adj2,
    hls::stream<int>& rnnz_fifo_adj2,
    hls::stream<TTYPE>& A_fifo_adj3,
    hls::stream<int>& col_indices_fifo_adj3,
    hls::stream<int>& rnnz_fifo_adj3,
    hls::stream<TTYPE>& A_fifo_adj4,
    hls::stream<int>& col_indices_fifo_adj4,
    hls::stream<int>& rnnz_fifo_adj4,
    int N_adj,
    int M_adj,
    ap_uint<8> P_w[5],
    ap_int<8> zero_point_lhs,
    ap_int<8> zero_point_rhs,
    #if (PIPO_BLOCKS>=2)
    hls::stream_of_blocks<buf>& C_buffer11,
    hls::stream_of_blocks<buf>& C_buffer12,
    #else
    buf C_buffer11,
    hls::stream_of_blocks<buf>& C_buffer12,
    #endif
    hls::stream_of_blocks<buf>& C_buffer13,
    hls::stream_of_blocks<buf>& C_buffer14,
    hls::stream_of_blocks<buf>& C_buffer21,
    hls::stream_of_blocks<buf>& C_buffer22,
    hls::stream_of_blocks<buf>& C_buffer23,
    hls::stream_of_blocks<buf>& C_buffer24,
    hls::stream_of_blocks<buf>& C_buffer31,
    hls::stream_of_blocks<buf>& C_buffer32,
    hls::stream_of_blocks<buf>& C_buffer33,
    hls::stream_of_blocks<buf>& C_buffer34,
    hls::stream_of_blocks<buf>& C_buffer41,
    hls::stream_of_blocks<buf>& C_buffer42,
    hls::stream_of_blocks<buf>& C_buffer43,
    hls::stream_of_blocks<buf>& C_buffer44,
    #if (PIPO_BLOCKS>=2)
    hls::stream_of_blocks<buf>& linear_pipo,
    #else
    buf linear_pipo,
    #endif
    int layer_loop,
    OUTTYPE* D1,
    OUTTYPE* D2,
    OUTTYPE* D3,
    OUTTYPE* D4,
    hls::stream<ASTYPE>& DS1,
    hls::stream<ASTYPE>& DS1R,
    hls::stream<ASTYPE>& DS1C,
    hls::stream<ASTYPE>& DS2,
    hls::stream<ASTYPE>& DS3,
    hls::stream<ASTYPE>& DS4
)
{
    hls::stream<ITYPE> D_fifo1[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=D_fifo1 depth=FIFO_DEPTH

    hls::stream<ITYPE> D_fifo2[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=D_fifo2 depth=FIFO_DEPTH

    hls::stream<ITYPE> D_fifo3[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=D_fifo3 depth=FIFO_DEPTH

    hls::stream<ITYPE> D_fifo4[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=D_fifo4 depth=FIFO_DEPTH

    hls::stream<OUTTYPE> out_fifo1;
    #pragma HLS STREAM variable=out_fifo1 depth=FIFO_DEPTH

    hls::stream<ITYPE> write_fifo1[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=write_fifo1 depth=FIFO_DEPTH

    hls::stream<ITYPE> write_fifo2[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=write_fifo2 depth=FIFO_DEPTH

    hls::stream<ITYPE> write_fifo3[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=write_fifo3 depth=FIFO_DEPTH

    hls::stream<ITYPE> write_fifo4[B_WIDTH_BLOCK];
    #pragma HLS STREAM variable=write_fifo4 depth=FIFO_DEPTH

    // int B_WIDTH_INT;

    #if (PIPO_BLOCKS>=2)
    LOOP_ADJ : for (int B_index = 0; B_index < layer_loop; B_index++) {
    #else
    int B_index = 0;
    #endif

        #pragma HLS DATAFLOW

        // if (B_index < (B_index_loop - 1))
        //     B_WIDTH_INT = B_WIDTH_BLOCK;
        // else
        //     B_WIDTH_INT = tail;

        // if (layer_loop == 1)
        // {
        //     stream_mode_int = stream_mode;
        // }
        // else
        // {
        std::cout << "adj layer " << B_index << std::endl;
        // }

    #if ADJ_THREADS == 1

        // while (C_buffer11.empty()); // Execute only when the producer has generated valid data.
        #if (PIPO_BLOCKS>=2)
        hls::read_lock<buf> C_adj11(C_buffer11);
        hls::read_lock<buf> linear_adj(linear_pipo);
        #endif

        // hls::read_lock<buf> C_adj12(C_buffer12);
        // hls::read_lock<buf> C_adj13(C_buffer13);
        // hls::read_lock<buf> C_adj14(C_buffer14);

        #if FEA_THREADS == 2
        hls::read_lock<buf> C_adj21(C_buffer21);
        #endif

        // hls::read_lock<buf> C_adj22(C_buffer22);
        /*
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
        */

        int first_row1;
        int row_count1;

        int N_adj_block = N_adj / ADJ_THREADS;
        int N_adj_block_compute = N_adj / FEA_THREADS; // In compute2, each block contains N_adj / FEA_THREADS elements.
        // int N_adj_rest = N_adj % 2;

        row_count1 = N_adj_block;
        // row_count2 = N_adj_block;
        // row_count3 = N_adj_block;
        // row_count4 = N_adj_block + N_adj_rest;

        first_row1 = 0;
        // first_row2 = N_adj_block;
        // first_row3 = 2 * N_adj_block;
        // first_row4 = 3 * N_adj_block;

        // std::cout << "READA2 " << std::endl;

        // reada2(first_row1,row_count1,B_index_loop,tail,A_fifo_adj1,coindices_fifo_adj1,rnnz_fifo_adj1,rowPtr_adj1,columnIndex_adj1,values_adj1);

        // std::cout << "COMPUTE2 " << std::endl;

        #if FEA_THREADS == 1
            #if (PIPO_BLOCKS>=2)
                #if LINEAR_ONLY == 0
                compute2_1(
                    model, srelu, N_adj_block, zero_point_lhs, zero_point_rhs,
                    first_row1, row_count1,
                    A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
                    C_adj11,
                    // C_adj21,
                    // C_adj31, C_adj41,
                    D_fifo1, B_index
                );
                #endif
            #else
                #if LINEAR_ONLY == 0
                compute2_1(
                    relu, N_adj_block, zero_point_lhs, zero_point_rhs,
                    first_row1, row_count1,
                    A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
                    C_buffer11,
                    // C_adj21,
                    // C_adj31, C_adj41,
                    D_fifo1
                );
                #endif
            #endif
        #endif

        #if FEA_THREADS == 2
        compute2_2(
            N_adj_block_compute, zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            C_adj11, C_adj21,
            // C_adj31, C_adj41,
            D_fifo1
        );
        #endif

        // compute2(N_adj_block,zero_point_lhs,zero_point_rhs,first_row2,row_count2,A_fifo_adj2,col_indices_fifo_adj2,rnnz_fifo_adj2,C_adj12,C_adj22,
        //          // C_adj32,C_adj42,
        //          D_fifo2, B_index, B_index_loop, tail);
        // compute2(N_adj_block,zero_point_lhs,zero_point_rhs,first_row3,row_count3,A_fifo_adj3,col_indices_fifo_adj3,rnnz_fifo_adj3,C_adj13,C_adj23,C_adj33,C_adj43,D_fifo3, B_index, B_index_loop, tail);
        // compute2(N_adj_block,zero_point_lhs,zero_point_rhs,first_row4,row_count4,A_fifo_adj4,col_indices_fifo_adj4,rnnz_fifo_adj4,C_adj14,C_adj24,C_adj34,C_adj44,D_fifo4, B_index, B_index_loop, tail);

        // compute2(zero_point_lhs, zero_point_rhs, N_adj, M_fea, A_fifo_adj, col_indices_fifo_adj, rnnz_fifo_adj, B, D_fifo, B_index_loop, tail);

        // scale(quantized_multiplier, shift, bias, zero_point_dst, clamp_max, clamp_min, N_adj, M_adj, P_w, D_fifo, B_index, B_index_loop, tail, write_fifo);

        // check_fifo_2(N/4, write_fifo_0, write_fifo_out_0);

        // Write write_fifo into D.
        // std::cout << "write matrix size " << N_adj << "," << P_w << std::endl;
        // std::cout << "WRITEC " << std::endl;

        // relupipe(first_row1,row_count1,N_adj,P_w, D_fifo1, D1, B_index);

        #if (PIPO_BLOCKS>=2)
        writec(deq_factor, model, first_row1, row_count1, N_adj, P_w, D_fifo1, linear_adj, out_fifo1, B_index, layer_loop);
        #else
        writec(deq_factor, model, first_row1, row_count1, N_adj, P_w, D_fifo1, linear_pipo, D1, DS1, B_index, layer_loop);
        #endif

        writeout(model, first_row1, row_count1, N_adj, P_w, out_fifo1, D1, DS1, DS1R, DS1C, B_index, layer_loop);
        // writec_transpose(deq_factor,stream_mode,first_row1,row_count1,N_adj,P_w, D_fifo1, D1,DS1,B_index);

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

        int first_row1, first_row2;
        int row_count1, row_count2;

        int N_adj_block = N_adj / ADJ_THREADS;
        int N_adj_rest = N_adj % ADJ_THREADS;
        int N_adj_block_compute = N_adj / FEA_THREADS; // In compute2, each block contains N_adj / FEA_THREADS elements.

        row_count1 = N_adj_block;
        row_count2 = N_adj_block + N_adj_rest;
        first_row1 = 0;
        first_row2 = N_adj_block;

        std::cout << "Thread adj 1" << std::endl;
        reada2(first_row1, row_count1, B_index_loop, tail, A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1, rowPtr_adj1, columnIndex_adj1, values_adj1);

        std::cout << "Thread adj 2" << std::endl;
        reada2(first_row2, row_count2, B_index_loop, tail, A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2, rowPtr_adj2, columnIndex_adj2, values_adj2);

        #if FEA_THREADS == 2

        compute2_2(
            N_adj_block_compute, zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            C_adj11, C_adj21,
            // C_adj31,C_adj41,
            D_fifo1, B_index, B_index_loop, tail
        );

        compute2_2(
            N_adj_block_compute, zero_point_lhs, zero_point_rhs,
            first_row2, row_count2,
            A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,
            C_adj12, C_adj22,
            // C_adj32,C_adj42,
            D_fifo2, B_index, B_index_loop, tail
        );

        #endif

        #if FEA_THREADS == 4

        compute2_4(
            N_adj_block_compute, zero_point_lhs, zero_point_rhs,
            first_row1, row_count1,
            A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1,
            C_adj11, C_adj21, C_adj31, C_adj41,
            D_fifo1, B_index, B_index_loop, tail
        );

        compute2_4(
            N_adj_block_compute, zero_point_lhs, zero_point_rhs,
            first_row2, row_count2,
            A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2,
            C_adj12, C_adj22, C_adj32, C_adj42,
            D_fifo2, B_index, B_index_loop, tail
        );

        #endif

        writec(first_row1, row_count1, P_w, D_fifo1, D1, B_index, B_index_loop, tail);
        writec(first_row2, row_count2, P_w, D_fifo2, D2, B_index, B_index_loop, tail);

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

        int first_row1, first_row2, first_row3, first_row4;
        int row_count1, row_count2, row_count3, row_count4;

        int N_adj_block = N_adj / 4;
        int N_adj_rest = N_adj % 4;

        row_count1 = N_adj_block;
        row_count2 = N_adj_block;
        row_count3 = N_adj_block;
        row_count4 = N_adj_block + N_adj_rest;

        first_row1 = 0;
        first_row2 = N_adj_block;
        first_row3 = 2 * N_adj_block;
        first_row4 = 3 * N_adj_block;

        // std::cout << "READA2 " << std::endl;
        // std::cout << "COMPUTE2 " << std::endl;

        compute2_4(relu, N_adj_block, zero_point_lhs, zero_point_rhs, first_row1, row_count1, A_fifo_adj1, col_indices_fifo_adj1, rnnz_fifo_adj1, C_adj11, C_adj21, C_adj31, C_adj41, D_fifo1);
        compute2_4(relu, N_adj_block, zero_point_lhs, zero_point_rhs, first_row2, row_count2, A_fifo_adj2, col_indices_fifo_adj2, rnnz_fifo_adj2, C_adj12, C_adj22, C_adj32, C_adj42, D_fifo2);
        compute2_4(relu, N_adj_block, zero_point_lhs, zero_point_rhs, first_row3, row_count3, A_fifo_adj3, col_indices_fifo_adj3, rnnz_fifo_adj3, C_adj13, C_adj23, C_adj33, C_adj43, D_fifo3);
        compute2_4(relu, N_adj_block, zero_point_lhs, zero_point_rhs, first_row4, row_count4, A_fifo_adj4, col_indices_fifo_adj4, rnnz_fifo_adj4, C_adj14, C_adj24, C_adj34, C_adj44, D_fifo4);

        writec(deq_factor, stream_mode, first_row1, row_count1, N_adj, P_w, D_fifo1, D1, DS1, B_index);
        writec(deq_factor, stream_mode, first_row2, row_count2, N_adj, P_w, D_fifo2, D2, DS2, B_index);
        writec(deq_factor, stream_mode, first_row3, row_count3, N_adj, P_w, D_fifo3, D3, DS3, B_index);
        writec(deq_factor, stream_mode, first_row4, row_count4, N_adj, P_w, D_fifo4, D4, DS4, B_index);

    #endif

    #if (PIPO_BLOCKS>=2)
    }
    #endif
}

void loop_adj2(
    int nnz_adj1,
    int nnz_adj2,
    int nnz_adj3,
    int nnz_adj4,
    int beta_qu,
    int f_align,
    float quantization_scale_adj,
    float quantization_scale_w[5],
    float deq_factor[5],
    ap_uint<1> model[5][8],
    float srelu[5],
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
    INTYPE* values_adj4,
    int N_adj,
    int M_adj,
    ap_uint<8> P_w[5],
    ap_int<8> zero_point_lhs,
    ap_int<8> zero_point_rhs,
    INTYPE* A,
    #if (PIPO_BLOCKS>=2)
    hls::stream_of_blocks<buf>& A_buffer11,
    hls::stream_of_blocks<buf>& A_buffer21,
    #else
    buf A_buffer11,
    hls::stream_of_blocks<buf>& A_buffer21,
    #endif
    hls::stream_of_blocks<buf>& A_buffer31,
    hls::stream_of_blocks<buf>& A_buffer41,
    OUTTYPE* E1,
    OUTTYPE* S1,
    #if (PIPO_BLOCKS>=2)
    hls::stream_of_blocks<buf>& C_buffer11,
    hls::stream_of_blocks<buf>& C_buffer12,
    #else
    buf C_buffer11,
    hls::stream_of_blocks<buf>& C_buffer12,
    #endif
    hls::stream_of_blocks<buf>& C_buffer13,
    hls::stream_of_blocks<buf>& C_buffer14,
    hls::stream_of_blocks<buf>& C_buffer21,
    hls::stream_of_blocks<buf>& C_buffer22,
    hls::stream_of_blocks<buf>& C_buffer23,
    hls::stream_of_blocks<buf>& C_buffer24,
    hls::stream_of_blocks<buf>& C_buffer31,
    hls::stream_of_blocks<buf>& C_buffer32,
    hls::stream_of_blocks<buf>& C_buffer33,
    hls::stream_of_blocks<buf>& C_buffer34,
    hls::stream_of_blocks<buf>& C_buffer41,
    hls::stream_of_blocks<buf>& C_buffer42,
    hls::stream_of_blocks<buf>& C_buffer43,
    hls::stream_of_blocks<buf>& C_buffer44,
    #if (PIPO_BLOCKS>=2)
    hls::stream_of_blocks<buf>& linear_pipo,
    #else
    buf linear_pipo,
    #endif
    int layer_loop,
    OUTTYPE* D1,
    OUTTYPE* D2,
    OUTTYPE* D3,
    OUTTYPE* D4,
    hls::stream<ASTYPE>& DS1,
    hls::stream<ASTYPE>& DS1R,
    hls::stream<ASTYPE>& DS1C,
    hls::stream<ASTYPE>& DS2,
    hls::stream<ASTYPE>& DS3,
    hls::stream<ASTYPE>& DS4
)
{
    hls::stream<int> rnnz_att1("rnnz_att1 stream");
    #pragma HLS STREAM variable=rnnz_att1 depth=FIFO_DEPTH

    hls::stream<ITYPE> values_att1("values_att1 stream");
    #pragma HLS STREAM variable=values_att1 depth=FIFO_DEPTH

    hls::stream<int> columnIndex_att1("columnIndex_att1 stream");
    #pragma HLS STREAM variable=columnIndex_att1 depth=FIFO_DEPTH

    hls::stream<int> rnnz_att2;
    #pragma HLS STREAM variable=rnnz_att2 depth=FIFO_DEPTH

    hls::stream<ITYPE> values_att2;
    #pragma HLS STREAM variable=values_att2 depth=FIFO_DEPTH

    hls::stream<int> columnIndex_att2;
    #pragma HLS STREAM variable=columnIndex_att2 depth=FIFO_DEPTH

    hls::stream<int> rnnz_att3;
    #pragma HLS STREAM variable=rnnz_att3 depth=FIFO_DEPTH

    hls::stream<ITYPE> values_att3;
    #pragma HLS STREAM variable=values_att3 depth=FIFO_DEPTH

    hls::stream<int> columnIndex_att3;
    #pragma HLS STREAM variable=columnIndex_att3 depth=FIFO_DEPTH

    hls::stream<int> rnnz_att4;
    #pragma HLS STREAM variable=rnnz_att4 depth=FIFO_DEPTH

    hls::stream<ITYPE> values_att4;
    #pragma HLS STREAM variable=values_att4 depth=FIFO_DEPTH

    hls::stream<int> columnIndex_att4;
    #pragma HLS STREAM variable=columnIndex_att4 depth=FIFO_DEPTH

    #pragma HLS DATAFLOW

    loop_attention(
        deq_factor, beta_qu, f_align, quantization_scale_adj, quantization_scale_w,
        model,
        nnz_adj1, nnz_adj2, nnz_adj3, nnz_adj4,
        rowPtr_adj1, rowPtr_adj2, rowPtr_adj3, rowPtr_adj4,
        columnIndex_adj1, columnIndex_adj2, columnIndex_adj3, columnIndex_adj4,
        values_adj1, values_adj2, values_adj3, values_adj4,
        N_adj, M_adj, P_w, A, A_buffer11, A_buffer21, A_buffer31, A_buffer41,
        E1,
        S1,
        rnnz_att1, columnIndex_att1, values_att1,
        rnnz_att2, columnIndex_att2, values_att2,
        rnnz_att3, columnIndex_att3, values_att3,
        rnnz_att4, columnIndex_att4, values_att4,
        layer_loop
    );

    std::cout << "Done loop attention" << std::endl;

    // std::cout << "Start loop adj with gemm mode " << gemm_mode << std::endl;

    loop_adj(
        deq_factor, model, srelu,
        values_att1, columnIndex_att1, rnnz_att1,
        values_att2, columnIndex_att2, rnnz_att2,
        values_att3, columnIndex_att3, rnnz_att3,
        values_att4, columnIndex_att4, rnnz_att4,
        N_adj, M_adj, P_w, zero_point_lhs, zero_point_rhs,
        C_buffer11, C_buffer12, C_buffer13, C_buffer14,
        C_buffer21, C_buffer22, C_buffer23, C_buffer24,
        C_buffer31, C_buffer32, C_buffer33, C_buffer34,
        C_buffer41, C_buffer42, C_buffer43, C_buffer44,
        linear_pipo,
        layer_loop, D1, D2, D3, D4, DS1, DS1R, DS1C, DS2, DS3, DS4
    );

    std::cout << "Done loop adj" << std::endl;
}

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
    // hls::stream<QTYPE> linear_fifo;
    // #pragma HLS STREAM variable=linear_fifo depth=LINEAR_DEPTH
    // #pragma HLS bind_storage variable=linear_fifo type=FIFO impl=URAM

    #if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf, PIPO_BLOCKS> linear_pipo;
    #else
    buf linear_pipo;
    #endif
    #pragma HLS array_partition variable=linear_pipo block factor=BLOCK/2 dim=2
    #pragma HLS array_partition variable=linear_pipo cyclic factor=SBLOCK_LIN dim=1

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

    #if (PIPO_BLOCKS >= 2)
    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer11;
    #else
    buf A_buffer11;
    #endif
    #pragma HLS array_partition variable=A_buffer11 block factor=BLOCK/2 dim=2
    // #pragma HLS array_partition variable=A_buffer11 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer21;
    #pragma HLS array_partition variable=A_buffer21 block factor=BLOCK/2 dim=2
    // #pragma HLS array_partition variable=A_buffer21 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer31;
    #pragma HLS array_partition variable=A_buffer31 block factor=BLOCK/2 dim=2
    // #pragma HLS array_partition variable=A_buffer31 cyclic factor=SBLOCK dim=1

    hls::stream_of_blocks<buf, PIPO_BLOCKS> A_buffer41;
    #pragma HLS array_partition variable=A_buffer41 block factor=BLOCK/2 dim=2
    // #pragma HLS array_partition variable=A_buffer41 cyclic factor=SBLOCK dim=1

    // hls::stream_of_blocks<buf> D_buffer11;
    // #pragma HLS array_partition variable=D_buffer11 block factor=BLOCK/2 dim=2
    // #pragma HLS array_partition variable=D_buffer11 cyclic factor=SBLOCK dim=1

    // hls::stream<ITYPE> D_fifo_0;
    // #pragma HLS STREAM variable=D_fifo_0 depth=128 dim=1

    // hls::stream<ITYPE> write_fifo_0;
    // #pragma HLS STREAM variable=write_fifo_0 depth=128 dim=1

    // hls::stream<ITYPE> write_fifo_out_0;
    // #pragma HLS STREAM variable=write_fifo_out_0 depth=8 dim=1

    int B_WIDTH_INT, a_values;

    #if (PIPO_BLOCKS >= 2)
    // LOOP_FEA: for (int B_index = 0; B_index < layer_loop; B_index++) {
    #pragma HLS DATAFLOW
    #endif

    // std::cout << "Start loop fea with gemm mode " << gemm_mode[0] << std::endl;

    loop_fea(
        load_weights, beta_qu, f_align, beta_qul, f_alignl,
        quantization_scale_fea, quantization_scale_w, quantization_scale_lin,
        model,
        scale_fea, max_fea, quantized_multiplier,
        nnz_fea1, nnz_fea2, nnz_fea3, nnz_fea4,
        rowPtr_fea1, rowPtr_fea2, rowPtr_fea3, rowPtr_fea4,
        columnIndex_fea1, columnIndex_fea2, columnIndex_fea3, columnIndex_fea4,
        values_fea1, values_fea2, values_fea3, values_fea4,
        rowPtr_feas1, rowPtr_feas2, rowPtr_feas3, rowPtr_feas4,
        columnIndex_feas1, columnIndex_feas2, columnIndex_feas3, columnIndex_feas4,
        values_feas1, values_feas2, values_feas3, values_feas4,
        B, B2,
        M_adj, M_fea, P_w,
        zero_point_lhs, zero_point_rhs,
        C_buffer11, C_buffer12, C_buffer13, C_buffer14,
        C_buffer21, C_buffer22, C_buffer23, C_buffer24,
        C_buffer31, C_buffer32, C_buffer33, C_buffer34,
        C_buffer41, C_buffer42, C_buffer43, C_buffer44,
        A_buffer11, A_buffer21, A_buffer31, A_buffer41,
        linear_pipo,
        layer_loop
    );

    std::cout << "Done loop fea" << std::endl;

    loop_adj2(
        nnz_adj1, nnz_adj2, nnz_adj3, nnz_adj4,
        beta_qu, f_align, quantization_scale_adj, quantization_scale_w,
        deq_factor,
        model, srelu,
        rowPtr_adj1, rowPtr_adj2, rowPtr_adj3, rowPtr_adj4,
        columnIndex_adj1, columnIndex_adj2, columnIndex_adj3, columnIndex_adj4,
        values_adj1, values_adj2, values_adj3, values_adj4,
        N_adj, M_adj, P_w, zero_point_lhs, zero_point_rhs,
        ate_m,
        A_buffer11, A_buffer21, A_buffer31, A_buffer41,
        E1, S1,
        C_buffer11, C_buffer12, C_buffer13, C_buffer14,
        C_buffer21, C_buffer22, C_buffer23, C_buffer24,
        C_buffer31, C_buffer32, C_buffer33, C_buffer34,
        C_buffer41, C_buffer42, C_buffer43, C_buffer44,
        linear_pipo,
        layer_loop, D1, D2, D3, D4, DS1, DS1R, DS1C, DS2, DS3, DS4
    );
    // }
}



/*
 * The amount of data stored in the FPGA is:
 *     B_HEIGHT * B_WIDTH_BLOCK + A_WIDTH + B_WIDTH_BLOCK
 * This total should remain below the available FPGA BRAM capacity.
 */

// gemm_mode / fea / adj
// 0 / 0 / 0 : dense / dense, not used in graph layers
// 1 / 0 / 1 : dense / sparse, normal mode for layer 2
// 2 / 1 / 0 : sparse / dense, used in training
// 3 / 1 / 1 : sparse / sparse, normal mode for layer 1
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

    // #pragma HLS INTERFACE ap_none port = stream_mode

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
    // #pragma HLS INTERFACE m_axi port=D1 depth=32000 offset=slave latency=0 num_write_outstanding=2048 bundle=D1
    // #pragma HLS INTERFACE m_axi port=D2 depth=32000 offset=slave latency=0 num_write_outstanding=2048 bundle=D2
    // #pragma HLS INTERFACE m_axi port=D3 depth=32000 offset=slave latency=0 num_write_outstanding=2048 bundle=D3
    // #pragma HLS INTERFACE m_axi port=D4 depth=32000 offset=slave latency=0 num_write_outstanding=2048 bundle=D4
    // #pragma HLS INTERFACE m_axi port=D1 depth=64000 offset=slave max_widen_bitwidth=512 max_write_burst_length=16 bundle=D1
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

    // c_fifo_stream_t C_fifo[B_WIDTH_BLOCK];
    // #pragma HLS STREAM variable=C_fifo depth=1024 dim=1

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

    // hls::stream<ASTYPE> DS2;
    // hls::stream<ASTYPE> DS3;
    // hls::stream<ASTYPE> DS4;

    // hls::stream<ASTYPE> OUTS1;
    // #pragma HLS STREAM variable=OUTS1 depth=FIFO_DEPTH

    // hls::stream<ASTYPE>& values_feas1,hls::stream<ASTYPE>& values_feas2,
    // hls::stream<ASTYPE>& values_feas3,hls::stream<ASTYPE>& values_feas4,

    // ap_int<32> quantized_multiplier_data[1024];

    // Load bias and parameter data.
    // Preloading bias and parameter data was considered, but performance was
    // observed to be unchanged in practice while introducing preload overhead.
    // Parameters are therefore loaded on demand in the current implementation.
    // Preloading can still be beneficial for certain matrix configurations,
    // especially with small A and large B, so the original logic is retained
    // in commented form for reference.
    // if (bias_count > 0)
    // {
    //     for (int bias_index = 0; bias_index < bias_count; bias_index++)
    //     {
    //         #pragma HLS PIPELINE
    //         bias_data[bias_index] = bias[bias_index];
    //         shift_data[bias_index] = shift[bias_index];
    //         quantized_multiplier_data[bias_index] = quantized_multiplier[bias_index];
    //     }
    // }
    // else
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

        // Load model instructions and per-layer parameters.
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
            srelu_int[i] = srelu[i],
            quantization_scale_lin_int[i] = quantization_scale_lin[i];
            quantization_scale_w_int[i] = quantization_scale_w[i];
            quantization_scale_fea_int[i] = quantization_scale_fea[i];
            deq_factor_int[i] = deq_factor[i];
            scale_fea_int[i] = scale_fea[i];
            P_w_int[i] = P_w[i];

            // quantization_scale_w_int[i] = 127.0;
            // quantization_scale_fea_int[i] = 255.0;
            // deq_factor_int[i] = 4.0631776391272885;

            std::cout << " Instruction is " << model_int[i][7] << model_int[i][6] << model_int[i][5] << model_int[i][4]
                      << model_int[i][3] << model_int[i][1] << model_int[i][1] << model_int[i][0] << std::endl;
        }

        // else
        // {
        //     // Simulation-only short run; remove for normal synthesis.
        //     B_index_loop = 2;
        //     tail = 0;
        // }

        // std::cout << " B_index_loop is " << B_index_loop << " tail is " << tail << std::endl;

        // for (int B_index = 0; B_index < B_index_loop; B_index++) {
        // *max_fea = 0;
        ITYPE max_fea_val = 0;

        // bool fixed_gat = 1;

        mmult_wrapper(
            load_weights, beta_qu, f_align, beta_qul, f_alignl,
            quantization_scale_adj, quantization_scale_fea_int,
            quantization_scale_w_int, quantization_scale_lin_int,
            deq_factor_int,
            model_int, srelu_int,
            scale_fea_int, &max_fea_val, quantized_multiplier, shift_data, bias_data,
            bias_count, zero_point_lhs, zero_point_rhs, zero_point_dst, clamp_max,
            clamp_min, N_adj, M_adj, M_fea, P_w_int,
            B, B2,
            D1, D2, D3, D4,
            DS1, DS1R, DS1C,
            DS2, DS3, DS4,
            E1,
            S1,
            ate_m,
            array_c_adjust, layer_loop,
            nnz_fea1, nnz_fea2, nnz_fea3, nnz_fea4,
            rowPtr_fea1, rowPtr_fea2, rowPtr_fea3, rowPtr_fea4,
            columnIndex_fea1, columnIndex_fea2, columnIndex_fea3, columnIndex_fea4,
            values_fea1, values_fea2, values_fea3, values_fea4,
            rowPtr_feas1, rowPtr_feas2, rowPtr_feas3, rowPtr_feas4,
            columnIndex_feas1, columnIndex_feas2, columnIndex_feas3, columnIndex_feas4,
            values_feas1, values_feas2, values_feas3, values_feas4,
            nnz_adj1, nnz_adj2, nnz_adj3, nnz_adj4,
            rowPtr_adj1, rowPtr_adj2, rowPtr_adj3, rowPtr_adj4,
            columnIndex_adj1, columnIndex_adj2, columnIndex_adj3, columnIndex_adj4,
            values_adj1, values_adj2, values_adj3, values_adj4
        );

        // std::cout << "max fea val " << max_fea_val << std::endl;

        *max_fea = max_fea_val;

        // std::cout << "max fea " << *max_fea << std::endl;

        std::cout << "Done mmult wrapper" << std::endl;

        // }

        /*
        profiling[0] = fifo_full_0;
        profiling[1] = fifo_full_1;
        profiling[2] = fifo_full_2;
        profiling[3] = fifo_empty_0;
        profiling[4] = fifo_empty_1;
        profiling[5] = fifo_empty_2;
        profiling[6] = fifo_read_0;
        profiling[7] = fifo_read_1;
        profiling[8] = fifo_read_2;
        profiling[9] = fifo_write_0;
        profiling[5] = fifo_write_1;
        profiling[11] = fifo_write_2;
        profiling[12] = fifo_cycle_0;
        profiling[13] = fifo_cycle_1;
        profiling[14] = fifo_cycle_2;
        */
    }
}


void kernelmult1(

/* ===================== config / quantization ===================== */
bool load_weights,
int beta_qu,
int f_align,
float quantization_scale_adj,
float quantization_scale_fea[5],
float quantization_scale_w[5],
float quantization_scale_lin[5],
float deq_factor[5],
STYPE scale_fea[5],
int quantized_multiplier,
ap_int<32> *shift,
ap_int<32> *bias,
ap_int<32> bias_count,
ap_int<8> zero_point_lhs,
ap_int<8> zero_point_rhs,
ap_int<8> zero_point_dst,
ap_int<8> clamp_max,
ap_int<8> clamp_min,

/* ===================== model flags ===================== */
int layer_count,
ap_uint<8> model[5],

/* ===================== weights ===================== */
INTYPES *array_b,
INTYPES *array_b2,
INTYPE *ate_m,

/* ===================== output arrays / output streams ===================== */
ITYPE* max_fea,
ap_int<64> *profiling,
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

/* ===================== feature sparse inputs ===================== */
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

/* ===================== adjacency sparse inputs ===================== */
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

/* ===================== dimensions ===================== */
int N_adj,
int M_adj,
int M_fea,
int P_w

) {
    int array_c_adjust = N_adj;
    ap_uint<8> P_w_int[5];

    P_w_int[0] = P_w;

    float srelu[5];
    srelu[0] = 0.0;

    // Reserved partition-based output offset logic.
    // Kept disabled to preserve current execution behavior.
    // int N_adj_block = N_adj / ADJ_THREADS;
    // array_d2 += N_adj_block * P_w;
    // array_d3 += 2 * N_adj_block * P_w;
    // array_d4 += 3 * N_adj_block * P_w;

    // Reserved SDS pragmas.
    // #pragma SDS resource(1)
    // #pragma SDS async(1)

    std::cout << " kernel starting " << std::endl;

    mmult_top(
        load_weights, beta_qu, f_align, beta_qu, f_align,
        quantization_scale_adj, quantization_scale_fea, quantization_scale_w,
        quantization_scale_lin, deq_factor,
        model, srelu, scale_fea, max_fea, layer_count,
        quantized_multiplier, shift, bias, bias_count, profiling,
        zero_point_lhs, zero_point_rhs, zero_point_dst, clamp_max, clamp_min,
        N_adj, M_adj, M_fea, P_w_int,
        array_b, array_b2,
        array_d1, array_d2, array_d3, array_d4,
        stream_d1, stream_d1r, stream_d1c,
        stream_d2, stream_d3, stream_d4,
        array_e1,
        array_s1,
        ate_m,
        array_c_adjust,
        nnz_fea1, nnz_fea2, nnz_fea3, nnz_fea4,
        rowPtr_fea1, rowPtr_fea2, rowPtr_fea3, rowPtr_fea4,
        colIndices_fea1, colIndices_fea2, colIndices_fea3, colIndices_fea4,
        values_fea1, values_fea2, values_fea3, values_fea4,
        rowPtr_feas1, rowPtr_feas2, rowPtr_feas3, rowPtr_feas4,
        columnIndex_feas1, columnIndex_feas2, columnIndex_feas3, columnIndex_feas4,
        values_feas1, values_feas2, values_feas3, values_feas4,
        nnz_adj1, nnz_adj2, nnz_adj3, nnz_adj4,
        rowPtr_adj1, rowPtr_adj2, rowPtr_adj3, rowPtr_adj4,
        colIndices_adj1, colIndices_adj2, colIndices_adj3, colIndices_adj4,
        values_adj1, values_adj2, values_adj3, values_adj4
    );

    std::cout << " 0 output " << array_d1[0] << std::endl;
    std::cout << " 3 output " << array_d1[3] << std::endl;
    std::cout << " 7 output " << array_d1[7] << std::endl;

    // Reserved SDS synchronization pragma.
    // #pragma SDS wait(1)

    std::cout << " kernel done " << std::endl;
}