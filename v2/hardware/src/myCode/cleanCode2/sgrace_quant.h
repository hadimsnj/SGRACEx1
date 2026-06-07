/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#ifndef __SGRACE_QUANT_H__
#define __SGRACE_QUANT_H__

#include "sgrace_common.h"

void quanta(
    ATYPE &BW,
    float  B,
    float  quantization_scale,
    int    f_align,
    int    beta_qu
);

void quantf(
    FTYPE &BW,
    float  B,
    float  quantization_scale[5],
    int    f_align,
    int    beta_qu,
    int    B_index
);

void quantl(
    LTYPE &BW,
    float  B,
    float  quantization_scale[5],
    int    f_align,
    int    beta_qu,
    int    B_index
);

void quantw(
    BTYPE  &BW,
    float   B,
    float   quantization_scale[5],
    int     f_align,
    int     beta_qu,
    int     B_index
);

void quantwl(
    BLTYPE &BW,
    float   B,
    float   quantization_scale[5],
    int     f_align,
    int     beta_qu,
    int     B_index
);

QTYPE8 float_to_fix(float f_in, int n_bits);

#endif  /* __SGRACE_QUANT_H__ */
