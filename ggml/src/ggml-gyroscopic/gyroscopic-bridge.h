#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_gyroscopic_active(void);
bool ggml_gyroscopic_strict(void);
bool ggml_gyroscopic_trace(void);

const char * ggml_gyroscopic_kernel_mode_cstr(void);

void ggml_gyroscopic_note_call(
    const char * tag,
    int64_t m,
    int64_t n,
    int64_t k
);

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

bool ggml_gyroscopic_vec_dot_q8_0_q8_0(
    int n,
    const void * x,
    const void * y,
    float * out
);

bool ggml_gyroscopic_gemm_f32(
    int m,
    int n,
    int k,
    const float * a,
    int lda,
    const float * b,
    int ldb,
    float * c,
    int ldc
);

/* c column-major: c[i + j*ldc]; ldc is dst leading dim in floats (>= m). */
bool ggml_gyroscopic_gemm_q8_0_q8_0(
    int m,
    int n,
    int k,
    const void * a,
    int lda_bytes,
    const void * b,
    int ldb_bytes,
    float * c,
    int ldc
);

bool ggml_gyroscopic_out_prod_f32(
    int rows,
    int cols,
    const float * x,
    const float * y,
    float * out,
    int ld_out
);

#ifdef __cplusplus
}
#endif
