/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include "sgrace_writers.h"

// =============================================================================================
// =============================================================================================
/**
 * writec - Dequantize and forward SpMM results to the writeout stage.
 *
 * For each output row i and active column j (j < P[B_index]):
 *   1. Reads the GNN accumulator value from write_fifo[j]    (when gcn_path == 1).
 *   2. Reads the linear residual from linear_pipo[i][j]      (when LINEAR_ENABLE and linear_mode).
 *   3. Dequantizes:
 *        output = C_out * deq_factor[B_index] + residual * deq_factor[B_index]
 *   4. Writes the result to the CS output stream for writeout.
 *
 * Columns j >= P[B_index] are skipped (sparse output: only the active
 * output width is forwarded downstream).
 *
 * @param deq_factor   Per-layer dequantization scale factors.
 * @param model        Per-layer mode flags [layer][bit].
 * @param first_row    First row index (unused in body; kept for symmetry).
 * @param row_count    Number of rows to process.
 * @param N_adj        Total adjacency rows (unused in body; kept for symmetry).
 * @param P            Per-layer output column widths.
 * @param write_fifo   Input FIFOs: GNN accumulator values [B_WIDTH_BLOCK].
 * @param linear_pipo  Input tile: linear-projection residuals [B_HEIGHT][B_WIDTH_BLOCK].
 * @param CS           Output stream: dequantized results → writeout.
 * @param B_index      Current layer index.
 * @param layer_loop   Total number of layer iterations (unused in body).
 */
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
)
{
    bool linear_mode = model[B_index][6];
    bool sage_mode   = model[B_index][7];
    bool gcn_path    = !(linear_mode ^ sage_mode);

    LOOP_WRITE42: for (int i = 0; i < row_count; i++)
    {
        LOOP_WRITE52: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            #pragma HLS PIPELINE II=1

            DTYPE C_out = DTYPE(0.0);
            DTYPE residual;

            if (gcn_path == 1)
                C_out = DTYPE(write_fifo[j].read());

#if LINEAR_ENABLE == 1
            residual = (linear_mode == 1) ? DTYPE(linear_pipo[i][j]) : DTYPE(0.0);
#else
            residual = DTYPE(0.0);
#endif

#if (INT_DEQUANT == 1)
            OUTTYPE C_float = (OUTTYPE)C_out    * deq_factor[B_index]
                            + (OUTTYPE)residual * deq_factor[B_index];
#else
            OUTTYPE C_float = (OUTTYPE)C_out;
#endif

            /* Emit only active output columns */
            if (j < P[B_index])
                CS.write(C_float);
        }
    }
}




// =============================================================================================
// =============================================================================================
/**
 * writeout - Write dequantized adjacency outputs to DDR or AXI-stream.
 *
 * Reads dequantized values from write_fifo and routes them based on stream_mode:
 *
 *   stream_mode == 1 (AXI-stream output):
 *     Writes values to CS (value stream).
 *     When gemm_mode for the next layer == 0 (sparse), also writes row/column
 *     indices to CSR and CSC streams (for downstream COO reconstruction).
 *     TLAST is asserted on the last element.
 *
 *   stream_mode == 0 (DDR output):
 *     Writes to C[i * B_WIDTH_INT + j + first_row * B_WIDTH_BLOCK].
 *
 * Note: the TLAST condition  i*j == (WL-1)*(B_WIDTH_INT-1)  is taken directly
 * from the original code and preserved intentionally.
 *
 * @param model       Per-layer mode flags [layer][bit]:
 *                      bit 2 of B_index   = stream_mode (output to AXI-stream).
 *                      bit 1 of B_index+1 = gemm_mode of the next layer.
 * @param first_row   First row offset for DDR addressing.
 * @param row_count   Number of rows to write.
 * @param N_adj       Total adjacency rows (unused in body; kept for symmetry).
 * @param P           Per-layer active output column widths.
 * @param write_fifo  Input stream: dequantized values from writec.
 * @param C           DDR output array (stream_mode == 0).
 * @param CS          AXI-stream: output values (stream_mode == 1).
 * @param CSR         AXI-stream: row indices for sparse output (stream_mode == 1, sparse next layer).
 * @param CSC         AXI-stream: column indices for sparse output.
 * @param B_index     Current layer index.
 * @param layer_loop  Total number of layer iterations (unused in body).
 */
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
)
{
    int        B_WIDTH_INT = P[B_index];
    int        WL          = row_count;
    ap_uint<1> stream_mode = model[B_index][2];
    ap_uint<1> next_gemm   = model[B_index + 1][1];   // next layer's mode

    if (stream_mode == 1)
    {
        /* ── AXI-stream output ── */
        bool last = 0;

        LOOP_WRITE42: for (int i = 0; i < WL; i++)
        {
            LOOP_WRITE52: for (int j = 0; j < B_WIDTH_INT; j++)
            {
                #pragma HLS PIPELINE II=1

                /* Assert TLAST on the final element */
                if (i * j == (WL - 1) * (B_WIDTH_INT - 1))
                    last = 1;

                OUTTYPE  C_float = OUTTYPE(write_fifo.read());
                fp_int   C_int;
                C_int.f  = C_float;

                ASTYPE temp;
                temp.data = C_int.i;
                temp.last = last;

                if (next_gemm == 1)
                {
                    /* Dense next layer: value stream only */
                    CS.write(temp);
                }
                else
                {
                    /* Sparse next layer: emit value + row + column index streams.
                     * Zero values are suppressed except: first element of each row (j==0)
                     * and the last element (last==1). */
                    if (j == 0 || C_float != 0 || last == 1)
                    {
                        CS.write(temp);

                        temp.data = i;
                        CSR.write(temp);

                        temp.data = j;
                        CSC.write(temp);
                    }
                }
            }
        }
    }
    else
    {
        /* ── DDR output ── */
        LOOP_WRITE45: for (int i = 0; i < WL; i++)
        {
            LOOP_WRITE55: for (int j = 0; j < B_WIDTH_INT; j++)
            {
                #pragma HLS PIPELINE II=1
                OUTTYPE C_float = OUTTYPE(write_fifo.read());
                C[i * B_WIDTH_INT + j + first_row * B_WIDTH_BLOCK] = C_float;
            }
        }
    }
}



// =============================================================================================
// =============================================================================================
/**
 * writec_transpose - Write transposed SpMM output to DDR or AXI-stream.
 *
 * Processes output in tiles of FIFO_DEPTH rows to amortize inner loop overhead.
 * For each (j, z) pair writes:
 *   C[i * FIFO_DEPTH + j * WL + z + first_row * B_WIDTH_BLOCK
 *     + B_index * N_adj * B_WIDTH_BLOCK] = dequantized value
 *
 * When STREAM_MODE_OUT == 1, writes to the AXI-stream CS instead of DDR.
 *
 * @param deq_factor  Scalar dequantization factor.
 * @param stream_mode Unused (routing controlled by STREAM_MODE_OUT macro).
 * @param first_row   Row offset for DDR addressing.
 * @param row_count   Number of rows.
 * @param N_adj       Total adjacency rows (used in DDR address formula).
 * @param P           Active output column width (unused in body; kept for symmetry).
 * @param write_fifo  Input FIFOs: accumulator values [B_WIDTH_BLOCK].
 * @param C           DDR output array.
 * @param CS          AXI-stream output (STREAM_MODE_OUT == 1).
 * @param B_index     Current layer index (used in DDR address formula).
 */
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
)
{
    int WL = row_count;

    LOOP_WRITE4: for (int i = 0; i < WL; i += FIFO_DEPTH)
    {
        LOOP_WRITE5: for (int j = 0; j < B_WIDTH_BLOCK; j++)
        {
            LOOP_WRITE6: for (int z = 0; z < FIFO_DEPTH; z++)
            {
                #pragma HLS PIPELINE II=1

                DTYPE C_out;
                if ((i + z) < WL)
                    C_out = DTYPE(write_fifo[j].read());
                else
                    C_out = 0.0;

#if (INT_DEQUANT == 1)
                OUTTYPE C_float = (OUTTYPE)C_out * deq_factor;
#else
                OUTTYPE C_float = (OUTTYPE)C_out;
#endif

#if (STREAM_MODE_OUT == 1)
                ASTYPE temp;
                temp.data = C_float;
                CS.write(temp);
#else
                C[i * FIFO_DEPTH + j * WL + z
                  + first_row * B_WIDTH_BLOCK
                  + B_index * N_adj * B_WIDTH_BLOCK] = C_float;
#endif
            }
        }
    }
}



// =============================================================================================
// =============================================================================================
/**
 * writes - Write edge-attention scores to DDR.
 *
 * Active only when gcn_path == 1 AND gat_mode == 1.
 * Reads the total non-zero count from rnnz_fifo, then drains write_fifo
 * and writes each dequantized value to C[i].
 *
 * @param deq_factor  Per-layer dequantization scale factors.
 * @param model       Per-layer mode flags [layer][bit].
 * @param first_row   First row offset (unused in body; kept for symmetry).
 * @param row_count   Number of rows (unused in body; kept for symmetry).
 * @param N_adj       Total adjacency rows (unused in body).
 * @param P           Per-layer output column widths (unused in body).
 * @param write_fifo  Input stream: attention scores (TTYPE).
 * @param rnnz_fifo   Input FIFO: total non-zeros to write.
 * @param C           DDR output array: edge scores.
 * @param B_index     Current layer index.
 */
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
)
{
    bool linear_mode = model[B_index][6];
    bool gat_mode    = model[B_index][5];
    bool sage_mode   = model[B_index][7];
    bool gcn_path    = !(linear_mode ^ sage_mode);

    if (gcn_path == 1 && gat_mode == 1)
    {
        int rnnz = rnnz_fifo.read();

        LOOP_WRITE5: for (int i = 0; i < rnnz; i++)
        {
            #pragma HLS PIPELINE
            DTYPE   C_out = write_fifo.read();
#if (INT_DEQUANT == 1)
            OUTTYPE C_float = (OUTTYPE)C_out * deq_factor[B_index];
#else
            OUTTYPE C_float = (OUTTYPE)C_out;
#endif
            C[i] = C_float;
        }
    }
}


// =============================================================================================
// =============================================================================================
/**
 * writesx4 - Write attention scores or softmax values for four row partitions to DDR.
 *
 * Reads total non-zero counts from rnnz_fifo1..4, then for each partition
 * drains the corresponding write_fifo and writes the values contiguously
 * into the DDR array C at offset rnnz_total (which accumulates across partitions).
 *
 * Optional dequantization (INT_DEQUANT): output = (OUTTYPE)val * deq_factor.
 *
 * The function body executes only when gat_mode == 1; non-GAT layers skip it.
 *
 * Note: write_fifo3 uses the INT_QUANT guard instead of INT_DEQUANT — this
 * preserves the original behavior and is kept intentionally.
 *
 * @param deq_factor     Scalar dequantization factor applied to each output value.
 * @param gat_mode       True when GAT attention is active; no output otherwise.
 * @param row_count1..4  Number of rows in each partition (currently unused; retained for symmetry).
 * @param write_fifo1..4 Input FIFOs: attention/softmax values for each partition.
 * @param rnnz_fifo1..4  Input FIFOs: total non-zeros for each partition.
 * @param C              DDR output array (edge scores or softmax values).
 * @param B_index        Current layer index (currently unused; retained for symmetry).
 */
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
)
{
    if (gat_mode == 1)
    {
        /* Read total nnz for all four partitions up front */
        int rnnz1 = rnnz_fifo1.read();
        int rnnz2 = rnnz_fifo2.read();
        int rnnz3 = rnnz_fifo3.read();
        int rnnz4 = rnnz_fifo4.read();

        int rnnz_total = 0;   // running DDR write offset

        /* ── Partition 1 ── */
        LOOP_WRITE51: for (int j = 0; j < rnnz1; j++)
        {
            #pragma HLS PIPELINE
            DTYPE   val = write_fifo1.read();
#if (INT_DEQUANT == 1)
            OUTTYPE out = (OUTTYPE)val * deq_factor;
#else
            OUTTYPE out = (OUTTYPE)val;
#endif
            C[j + rnnz_total] = out;
        }
        rnnz_total += rnnz1;

        /* ── Partition 2 ── */
        LOOP_WRITE52: for (int j = 0; j < rnnz2; j++)
        {
            #pragma HLS PIPELINE
            DTYPE   val = write_fifo2.read();
#if (INT_DEQUANT == 1)
            OUTTYPE out = (OUTTYPE)val * deq_factor;
#else
            OUTTYPE out = (OUTTYPE)val;
#endif
            C[j + rnnz_total] = out;
        }
        rnnz_total += rnnz2;

        /* ── Partition 3 (uses INT_QUANT guard — preserves original behavior) ── */
        LOOP_WRITE53: for (int j = 0; j < rnnz3; j++)
        {
            #pragma HLS PIPELINE
            DTYPE   val = write_fifo3.read();
#if (INT_QUANT == 1)
            OUTTYPE out = (OUTTYPE)val * deq_factor;
#else
            OUTTYPE out = (OUTTYPE)val;
#endif
            C[j + rnnz_total] = out;
        }
        rnnz_total += rnnz3;

        /* ── Partition 4 ── */
        LOOP_WRITE54: for (int j = 0; j < rnnz4; j++)
        {
            #pragma HLS PIPELINE
            DTYPE   val = write_fifo4.read();
#if (INT_DEQUANT == 1)
            OUTTYPE out = (OUTTYPE)val * deq_factor;
#else
            OUTTYPE out = (OUTTYPE)val;
#endif
            C[j + rnnz_total] = out;
        }
        rnnz_total += rnnz4;
    }
}


