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
GGML_API void ggml_gyroscopic_graph_feed_word4(uint32_t cell_id, const uint8_t word4[4], uint32_t profile_id);
GGML_API void ggml_gyroscopic_graph_feed_layer_block(uint32_t tensor_key, uint32_t row_block, uint32_t k_block, uint32_t class_id);
GGML_API void ggml_gyroscopic_graph_feed_kv_event(uint32_t segment_id, uint32_t block_id, uint32_t event_kind, uint32_t token_pos);
GGML_API uint32_t ggml_gyroscopic_graph_last_request_cell(void);

GGML_API double ggml_gyroscopic_graph_m2(uint32_t seq_id);

GGML_API double ggml_gyroscopic_graph_m2_empirical(uint32_t seq_id);

GGML_API bool ggml_gyroscopic_graph_has_empirical_m2(uint32_t seq_id);

GGML_API uint32_t ggml_gyroscopic_graph_resonance_key(uint32_t seq_id);
GGML_API uint32_t ggml_gyroscopic_graph_current_resonance(uint32_t seq_id);
GGML_API uint64_t ggml_gyroscopic_graph_step(uint32_t seq_id);
GGML_API uint8_t ggml_gyroscopic_graph_last_byte(uint32_t seq_id);
GGML_API uint8_t ggml_gyroscopic_graph_family(uint32_t seq_id);
GGML_API uint8_t ggml_gyroscopic_graph_micro_ref(uint32_t seq_id);
GGML_API uint8_t ggml_gyroscopic_graph_q6(uint32_t seq_id);
GGML_API uint8_t ggml_gyroscopic_graph_q_transport6(uint32_t seq_id);
GGML_API uint32_t ggml_gyroscopic_graph_omega_sig(uint32_t seq_id);
GGML_API uint32_t ggml_gyroscopic_graph_state24(uint32_t seq_id);
GGML_API uint16_t ggml_gyroscopic_graph_horizon_distance(uint32_t seq_id);
GGML_API uint16_t ggml_gyroscopic_graph_ab_distance(uint32_t seq_id);
GGML_API uint16_t ggml_gyroscopic_graph_parity_O12(uint32_t seq_id);
GGML_API uint16_t ggml_gyroscopic_graph_parity_E12(uint32_t seq_id);
GGML_API uint8_t ggml_gyroscopic_graph_parity_bit(uint32_t seq_id);
GGML_API uint8_t ggml_gyroscopic_graph_chi6(uint32_t seq_id);
GGML_API uint8_t ggml_gyroscopic_graph_shell(uint32_t seq_id);
GGML_API double ggml_gyroscopic_graph_shell_concentration(uint32_t seq_id);
GGML_API double ggml_gyroscopic_graph_kv_eviction_priority(uint32_t seq_id);
GGML_API bool ggml_gyroscopic_graph_near_horizon(uint32_t seq_id);

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
