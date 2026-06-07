/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include "sgrace_quant.h"



/* ════════════════════════════════════════════════════════════════════════════
 * Quantization functions for feature, adjacency, weight, and linear values.
 *
 * All four functions follow the same pattern:
 *   1. Scale and shift:  vfloat = quantization_scale * B + zero_point
 *   2. Round to integer: vround = round(vfloat)   (or binary sign for f_align==7)
 *   3. Clip to [ialpha_q, ibeta_q]
 *   4. Right-shift by (qbits - f_align - 1) to produce the fixed-point value.
 *
 * Clip range depends on SIGNED_MODE (quanta/quantf) or is always symmetric
 * ±(beta_qu>>1) (quantl/quantw).
 * ════════════════════════════════════════════════════════════════════════════ */

// =============================================================================================
// =============================================================================================
/**
 * quanta - Quantize a floating-point adjacency value to ATYPE.
 *
 * Clip range:
 *   SIGNED_MODE == 0: [0,        beta_qu]
 *   SIGNED_MODE == 1: [-beta_q,  +beta_q]   where beta_q = beta_qu >> 1
 *
 * Binary device mode (qbits == 1): right-shift = 1.
 *
 * @param BW                  Output: quantized adjacency value.
 * @param B                   Input: raw floating-point adjacency value.
 * @param quantization_scale  Single scale factor (not per-layer).
 * @param f_align             Fractional alignment bits (7 = bipolar binary).
 * @param beta_qu             Quantization range (full width).
 */
void quanta(
    ATYPE &BW,
    float  B,
    float  quantization_scale,
    int    f_align,
    int    beta_qu
)
{
    float vfloat = quantization_scale * B + zero_point;
    float vround = hls::round(vfloat);

    ITYPE vquant = ITYPE(vround);

#if (SIGNED_MODE == 0)
    ITYPE ibeta_q  = (ITYPE)beta_qu;
    ITYPE ialpha_q = (ITYPE)(0.0);
#else
    ITYPE beta_q   = ITYPE(beta_qu >> 1);
    ITYPE ibeta_q  = (ITYPE)beta_q;
    ITYPE ialpha_q = -(ITYPE)beta_q;
#endif

    /* Clip to representable range */
    if      (vquant > ibeta_q)  vquant = ibeta_q;
    else if (vquant < ialpha_q) vquant = ialpha_q;

    if (f_align == 7)   // bipolar binary: use f_align = 6 for shift
        f_align = 6;

#if (qbits == 1)
    ITYPE vnorm = vquant >> 1;
#else
    ITYPE vnorm = vquant >> (qbits - f_align - 1);
#endif

    BW = ATYPE(vnorm);
}



// =============================================================================================
// =============================================================================================
/**
 * quantf - Quantize a floating-point feature value to FTYPE (GNN branch).
 *
 * Identical to quanta but uses per-layer scale (quantization_scale[B_index])
 * and outputs FTYPE.
 *
 * @param BW                  Output: quantized feature value.
 * @param B                   Input: raw floating-point feature value.
 * @param quantization_scale  Per-layer scale factor array.
 * @param f_align             Fractional alignment bits (7 = bipolar binary).
 * @param beta_qu             Quantization range.
 * @param B_index             Current layer index.
 */
void quantf(
    FTYPE &BW,
    float  B,
    float  quantization_scale[5],
    int    f_align,
    int    beta_qu,
    int    B_index
)
{
    float vfloat = quantization_scale[B_index] * B + zero_point;
    float vround = hls::round(vfloat);

    ITYPE vquant = ITYPE(vround);

#if (SIGNED_MODE == 0)
    ITYPE ibeta_q  = (ITYPE)beta_qu;
    ITYPE ialpha_q = (ITYPE)(0.0);
#else
    ITYPE beta_q   = ITYPE(beta_qu >> 1);
    ITYPE ibeta_q  = (ITYPE)beta_q;
    ITYPE ialpha_q = -(ITYPE)beta_q;
#endif

    if      (vquant > ibeta_q)  vquant = ibeta_q;
    else if (vquant < ialpha_q) vquant = ialpha_q;

    if (f_align == 7)
        f_align = 6;

#if (qbits == 1)
    ITYPE vnorm = vquant >> 1;
#else
    ITYPE vnorm = vquant >> (qbits - f_align - 1);
#endif

    BW = FTYPE(vnorm);
}



// =============================================================================================
// =============================================================================================
/**
 * quantl - Quantize a floating-point feature value to LTYPE (linear-projection branch).
 *
 * Uses the same bipolar-binary / general-fixed-point split as quantwl.
 * Clip range is always symmetric: [-beta_q, +beta_q].
 * Output uses qbitsl instead of qbits in the final shift.
 *
 * @param BW                  Output: quantized linear-branch feature value.
 * @param B                   Input: raw floating-point feature value.
 * @param quantization_scale  Per-layer scale factor array.
 * @param f_align             Fractional alignment bits (7 = bipolar binary).
 * @param beta_qu             Quantization range.
 * @param B_index             Current layer index.
 */
void quantl(
    LTYPE &BW,
    float  B,
    float  quantization_scale[5],
    int    f_align,
    int    beta_qu,
    int    B_index
)
{
    float vfloat = quantization_scale[B_index] * B + zero_point;
    float vround;

    ITYPE ibeta_q, ialpha_q, beta_q;

    if (f_align == 7)
    {
        /* Bipolar binary mode */
        ibeta_q  = 1;
        ialpha_q = -1;
        vround   = (vfloat < 0.0f) ? -1.0f : 1.0f;
    }
    else
    {
        /* General fixed-point */
        beta_q   = ITYPE(beta_qu >> 1);
        ibeta_q  = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
        vround   = hls::round(vfloat);
    }

    ITYPE vquant = ITYPE(vround);

    if      (vquant > ibeta_q)  vquant = ibeta_q;
    else if (vquant < ialpha_q) vquant = ialpha_q;

    if (f_align == 7)
        f_align = 6;

    ITYPE vnorm = vquant >> (qbitsl - f_align - 1);
    BW = LTYPE(vnorm);
}


// =============================================================================================
// =============================================================================================
/**
 * quantw - Quantize a floating-point weight to BTYPE (GNN branch).
 *
 * Maps:  vfloat = quantization_scale[B_index] * B + zero_point
 * then clips, rounds, and right-shifts to produce a BTYPE fixed-point value.
 *
 * Three quantization modes selected at compile time:
 *
 *   qbits == 1  (binary):
 *     vfloat < 0  → vround = 1.0  (encodes negative weight as bit 1)
 *     vfloat >= 0 → vround = 0.0
 *     Clip range: [0, 1].
 *
 *   f_align == 7  (bipolar binary, called separately):
 *     vfloat < 0  → vround = -1.0
 *     vfloat >= 0 → vround =  1.0
 *     Clip range: [-1, 1].
 *     f_align is then forced to 6 before the final shift.
 *
 *   General fixed-point:
 *     vround = round(vfloat)
 *     beta_q = beta_qu >> 1  (half-range)
 *     Clip range: [-beta_q, beta_q].
 *
 * Final normalization:  BW = BTYPE( vquant >> (qbits - f_align - 1) )
 *
 * @param BW                   Output: quantized weight value.
 * @param B                    Input: raw floating-point weight.
 * @param quantization_scale   Per-layer scale factors.
 * @param f_align              Fractional alignment bits (7 = bipolar binary mode).
 * @param beta_qu              Full quantization range [-(beta_qu>>1), +(beta_qu>>1)].
 * @param B_index              Current layer index.
 */
void quantw(
    BTYPE  &BW,
    float   B,
    float   quantization_scale[5],
    int     f_align,
    int     beta_qu,
    int     B_index
)
{
    float vfloat = quantization_scale[B_index] * B + zero_point;
    float vround;

    ITYPE ibeta_q, ialpha_q, beta_q;

#if (qbits == 1)
    /* Binary mode: encode sign as 1/0 */
    ibeta_q  = 1;
    ialpha_q = 0;
    vround   = (vfloat < 0.0f) ? 1.0f : 0.0f;

#else
    if (f_align == 7)
    {
        /* Bipolar binary mode */
        ibeta_q  = 1;
        ialpha_q = -1;
        vround   = (vfloat < 0.0f) ? -1.0f : 1.0f;
    }
    else
    {
        /* General fixed-point: symmetric clip around beta_q */
        beta_q   = ITYPE(beta_qu >> 1);
        ibeta_q  = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
        vround   = hls::round(vfloat);
    }
#endif

    ITYPE vquant = ITYPE(vround);

    /* Clip to representable range */
    if      (vquant > ibeta_q)  vquant = ibeta_q;
    else if (vquant < ialpha_q) vquant = ialpha_q;

    /* In bipolar binary mode the shift uses f_align = 6 */
    if (f_align == 7)
        f_align = 6;

    ITYPE vnorm = vquant >> (qbits - f_align - 1);
    BW = BTYPE(vnorm);
}



// =============================================================================================
// =============================================================================================
/**
 * quantwl - Quantize a floating-point weight to BLTYPE (linear-projection branch).
 *
 * Identical to quantw but:
 *   - No qbits == 1 binary mode (linear branch always uses multi-bit weights).
 *   - Output type is BLTYPE.
 *   - Uses qbitsl instead of qbits in the final shift.
 *
 * @param BW                   Output: quantized weight value (BLTYPE).
 * @param B                    Input: raw floating-point weight.
 * @param quantization_scale   Per-layer scale factors.
 * @param f_align              Fractional alignment bits (7 = bipolar binary mode).
 * @param beta_qu              Full quantization range.
 * @param B_index              Current layer index.
 */
void quantwl(
    BLTYPE &BW,
    float   B,
    float   quantization_scale[5],
    int     f_align,
    int     beta_qu,
    int     B_index
)
{
    float vfloat = quantization_scale[B_index] * B + zero_point;
    float vround;

    ITYPE ibeta_q, ialpha_q, beta_q;

    if (f_align == 7)
    {
        /* Bipolar binary mode */
        ibeta_q  = 1;
        ialpha_q = -1;
        vround   = (vfloat < 0.0f) ? -1.0f : 1.0f;
    }
    else
    {
        /* General fixed-point */
        beta_q   = ITYPE(beta_qu >> 1);
        ibeta_q  = (ITYPE)beta_q;
        ialpha_q = -(ITYPE)beta_q;
        vround   = hls::round(vfloat);
    }

    ITYPE vquant = ITYPE(vround);

    /* Clip to representable range */
    if      (vquant > ibeta_q)  vquant = ibeta_q;
    else if (vquant < ialpha_q) vquant = ialpha_q;

    if (f_align == 7)
        f_align = 6;

    ITYPE vnorm = vquant >> (qbitsl - f_align - 1);
    BW = BLTYPE(vnorm);
}



// =============================================================================================
// =============================================================================================
/**
 * float_to_fix - Convert a float to a fixed-point value with n_bits fractional bits.
 *
 * Note: the multiply/divide by (1<<n_bits) is intentional — it maps the float
 * through the fixed-point grid so the result is representable in QTYPE8.
 *
 * @param f_in   Input floating-point value.
 * @param n_bits Number of fractional bits in the target format.
 * @return       Quantized fixed-point value.
 */
QTYPE8 float_to_fix(float f_in, int n_bits)
{
    float  scale  = (1 << n_bits);
    QTYPE8 i_out  = (f_in * scale) * (1.0 / scale);
    return i_out;
}


