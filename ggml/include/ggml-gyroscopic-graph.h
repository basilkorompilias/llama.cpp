#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * GyroGraph state (QuBEC climate) shared by llama decode and ggml-cpu mul_mat.
 * Built when GGML_GYROSCOPIC is enabled (ggml-base). Exported from ggml-base DLL.
 */
GGML_API void ggml_gyroscopic_graph_feed_token(uint32_t seq_id, uint32_t token_id);

GGML_API double ggml_gyroscopic_graph_m2(uint32_t seq_id);

GGML_API double ggml_gyroscopic_graph_m2_empirical(uint32_t seq_id);

GGML_API bool ggml_gyroscopic_graph_has_empirical_m2(uint32_t seq_id);

GGML_API uint32_t ggml_gyroscopic_graph_resonance_key(uint32_t seq_id);

/*
 * Summary of M2 over graph cells that received at least one token (g_step > 0).
 * If n_cells_seen is 0, min/max/mean are 0.
 */
GGML_API void ggml_gyroscopic_graph_m2_stats(
    double *min_m2,
    double *max_m2,
    double *mean_m2,
    int *n_cells_seen
);

#ifdef __cplusplus
}
#endif
