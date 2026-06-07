/*===============================================================================
 * SGRACE GNN Accelerator
 * Linkoping / UPM University
 * Author : Jose Nunez-Yanez
 * Copyright (C) 2026 Jose Nunez-Yanez
 * Licensed under the MIT License. See LICENSE file in the project root for details.
 *===============================================================================*/

#include "sgrace_common.h"

/* Single definition of the FIFO telemetry counters shared across modules. */
ap_int<64> fifo_full_0,  fifo_full_1,  fifo_full_2;
ap_int<64> fifo_empty_0, fifo_empty_1, fifo_empty_2;
ap_int<64> fifo_read_0,  fifo_read_1,  fifo_read_2;
ap_int<64> fifo_write_0, fifo_write_1, fifo_write_2;
ap_int<64> fifo_cycle_0, fifo_cycle_1, fifo_cycle_2;
