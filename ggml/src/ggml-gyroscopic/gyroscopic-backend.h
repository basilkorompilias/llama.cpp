#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "gyrograph_policy.h"

struct ggml_compute_params;
struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_gyroscopic_active(void);
bool ggml_gyroscopic_strict(void);
bool ggml_gyroscopic_trace(void);

void ggml_gyroscopic_abort_unsupported(
    const char * site,
    int32_t type0,
    int32_t type1
);

bool ggml_gyroscopic_vec_dot_f32(
    int n,
    const float * x,
    const float * y,
    float * out
);

/*
 * Preflight for QuBEC matmul: same geometry and dtype checks as mul_mat uses
 * before calling gyrolabe_qubec_matmul_q8_0. src1_batch is the panel base in src1.
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

/*
 * src1_batch: base pointer for the current src1 panel (use src1->data when the
 * panel starts at tensor base).
 */
/*
 * w_tensor: ggml tensor used for gyrolabe registry / block lookup (the Q8_0 weights).
 * Pass NULL to use src1.
 */
bool ggml_gyroscopic_mul_mat(
    int cell_idx,
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

int ggml_gyroscopic_graph_cell_idx(int64_t i12);

void ggml_gyroscopic_q8dot_override_inc(void);
long long ggml_gyroscopic_q8dot_override_get(void);

#ifdef __cplusplus
}
#endif
