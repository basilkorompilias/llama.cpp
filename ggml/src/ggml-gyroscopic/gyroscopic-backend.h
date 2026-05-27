#ifndef GYROSCOPIC_BACKEND_H
#define GYROSCOPIC_BACKEND_H

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ggml.h"
#include "gyroscopic-common.h"

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_gyroscopic_active(void);

/**
 * Return true when strict fallback mode is enabled for gyroscopic matmul.
 */
bool ggml_gyroscopic_strict(void);

/**
 * Return true when trace logging is enabled for gyroscopic matmul.
 */
bool ggml_gyroscopic_trace(void);

/**
 * Validate geometry and dtype for a potential gyroscopic matmul path.
 */
bool ggml_gyroscopic_can_use(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const void * src1_batch,
    int m,
    int n,
    int k,
    int lda_bytes,
    int ldb_bytes
);

/**
 * Count one invocation of q8dot override.
 */
void ggml_gyroscopic_q8dot_override_inc(void);

/**
 * Read total q8dot override count.
 */
long long ggml_gyroscopic_q8dot_override_get(void);

/**
 * Report unsupported combination and abort execution with error.
 */
void ggml_gyroscopic_abort_unsupported(const char * site, int32_t type0, int32_t type1);

/**
 * Fast-path override for vector dot on unsupported or blocked types.
 */
bool ggml_gyroscopic_vec_dot_f32(int n, const float * x, const float * y, float * out);

/**
 * Execute the GyroLabe matmul dispatch for Q8-backed GGML matmul.
 *
 * This is the ownership handoff from the ggml cpu backend into gyroscopic
 * execution for an entire matmul operation tile partition.
 */
void ggml_gyroscopic_mul_mat_dispatch(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst,
    const struct ggml_compute_params * params,
    int64_t r2,
    int64_t r3
);

/**
 * Execute QuBEC-optimized matrix multiplication.
 *
 * @param cell_idx Cell index in [0, GYROGRAPH_MAX_CELLS).
 * @param src0 Weight tensor, must be GGML_TYPE_Q8_0.
 * @param src1 Source tensor matching vector shape.
 * @param a Pointer to source weight panel data.
 * @param wdata Pointer to source activation block used by matmul.
 * @param w_registry_base Optional registry base pointer; if null, use `a`.
 * @param row_size Bytes per output row for the activation path.
 * @param m Row count in this partition.
 * @param n Column count in this partition.
 * @param k Reduction size.
 * @param weight_col0 Global output column offset.
 * @param w_tensor Weight tensor to locate registry entries.
 * @param row_start Global row offset in weight matrix.
 * @param k_block Kernel block index.
 * @param c Output matrix block pointer.
 * @param ldc Leading dimension of c.
 * @return true when gyroscopic backend handles the operation. false means the
 * backend declined before mutating output, so the caller may use stock fallback.
 * Once a panel is claimed and handed to GyroLabe, internal kernel failures are
 * contract errors, not fallback signals.
 */
bool ggml_gyroscopic_mul_mat(
    uint32_t cell_idx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const void * a,
    const char * wdata,
    const void * w_registry_base,
    size_t row_size,
    int m,
    int n,
    int k,
    int weight_col0,
    const struct ggml_tensor * w_tensor,
    int row_start,
    int k_block,
    float * c,
    int ldc
);

/**
 * Normalize GyroGraph index to valid cell range.
 *
 * @param i12 Raw combined state index.
 * @return clamped cell index in [0, GYROGRAPH_MAX_CELLS).
 */
uint32_t ggml_gyroscopic_graph_cell_idx(uint32_t i12);

#ifdef __cplusplus
}
#endif

#endif // GYROSCOPIC_BACKEND_H
