#include "ggml.h"
#include "ggml-gyroscopic-graph.h"
#include "gyroscopic-common.h"

#include "ggml-threading.h"
#include "gyrograph_api.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

static int gyroscopic_base_trace(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_GYROSCOPIC_TRACE");
        cached = (v != NULL && v[0] != '\0' && v[0] != '0') ? 1 : 0;
    }
    return cached;
}

typedef struct gyroscopic_cell_state {
    int32_t omega12;
    uint64_t step;
    uint8_t last_byte;
    uint8_t has_closed_word;
    uint8_t chi_ring64[64];
    uint8_t chi_ring_pos;
    uint8_t chi_valid_len;
    uint16_t chi_hist64[64];
    uint16_t shell_hist7[7];
    uint8_t family_ring64[64];
    uint16_t family_hist4[4];
    int32_t omega_sig;
    uint16_t parity_O12;
    uint16_t parity_E12;
    uint8_t parity_bit;
    uint32_t resonance_key;
    uint8_t u6;
    uint8_t v6;
    uint8_t chi6;
    uint8_t shell;
    uint8_t word4[4];
} gyroscopic_cell_state;

static gyroscopic_cell_state g_cells[GYROGRAPH_MAX_CELLS];
static int g_graph_ready;
static uint32_t g_last_request_cell;

enum {
    GYROSCOPIC_ROLE_REQUEST_BASE = 0,
    GYROSCOPIC_ROLE_REQUEST_COUNT = 128,
    GYROSCOPIC_ROLE_LAYER_BASE = 128,
    GYROSCOPIC_ROLE_LAYER_COUNT = 64,
    GYROSCOPIC_ROLE_KV_BASE = 192,
    GYROSCOPIC_ROLE_KV_COUNT = 64
};

static void gyroscopic_update_cell_coords(gyroscopic_cell_state * cell) {
    const uint32_t omega12 = (uint32_t) cell->omega12;
    cell->u6 = (uint8_t) ((omega12 >> 6u) & 0x3Fu);
    cell->v6 = (uint8_t) (omega12 & 0x3Fu);
    cell->chi6 = gyroscopic_chi6_from_omega12(omega12);
    cell->shell = (uint8_t) gyroscopic_popcount6((uint32_t) cell->chi6);
}

static inline uint32_t gyroscopic_mix32(uint32_t x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

static inline uint32_t gyroscopic_role_cell(uint32_t base, uint32_t count, uint32_t key) {
    return base + (gyroscopic_mix32(key) % count);
}

static void gyro_graph_ensure_init(void) {
    if (g_graph_ready) {
        return;
    }
    ggml_critical_section_start();
    if (!g_graph_ready) {
        gyrograph_init();
        memset(g_cells, 0, sizeof(g_cells));
        g_graph_ready = 1;
    }
    ggml_critical_section_end();
}

void ggml_gyroscopic_graph_feed_token(uint32_t seq_id, uint32_t token_id) {
    if (gyroscopic_base_trace()) {
        fprintf(stderr, "GGMO_FEED_TOKEN_ENTER seq_id=%u token_id=%u\n", (unsigned) seq_id, (unsigned) token_id);
        fflush(stderr);
    }
    const uint32_t cell = gyroscopic_role_cell(GYROSCOPIC_ROLE_REQUEST_BASE, GYROSCOPIC_ROLE_REQUEST_COUNT, seq_id);
    gyro_graph_ensure_init();

    uint8_t w4[4];
    // Little-endian word4 serialization for GyroGraph bridge input.
    w4[0] = (uint8_t)(token_id & 0xFFu);
    w4[1] = (uint8_t)((token_id >> 8) & 0xFFu);
    w4[2] = (uint8_t)((token_id >> 16) & 0xFFu);
    w4[3] = (uint8_t)((token_id >> 24) & 0xFFu);

    ggml_gyroscopic_graph_feed_word4(
        cell,
        w4,
        GYROGRAPH_PROFILE_CHIRALITY);
    if (gyroscopic_base_trace()) {
        fprintf(stderr, "GGMO_FEED_TOKEN_EXIT seq_id=%u token_id=%u cell=%u\n", (unsigned) seq_id, (unsigned) token_id, (unsigned) cell);
        fflush(stderr);
    }
    ggml_critical_section_start();
    g_last_request_cell = cell;
    ggml_critical_section_end();
}

GGML_API uint32_t ggml_gyroscopic_graph_last_request_cell(void) {
    if (!g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint32_t cell = g_last_request_cell;
    ggml_critical_section_end();
    return cell;
}

GGML_API void ggml_gyroscopic_graph_feed_word4(uint32_t cell_id, const uint8_t word4[4], uint32_t profile_id) {
    if (gyroscopic_base_trace()) {
        fprintf(stderr, "GGMO_FEED_WORD4_ENTER cell_id=%u profile=%u\n", (unsigned) cell_id, (unsigned) profile_id);
        fflush(stderr);
    }
    if (cell_id >= GYROGRAPH_MAX_CELLS || word4 == NULL) {
        if (gyroscopic_base_trace()) {
            fprintf(stderr, "GGMO_FEED_WORD4_SKIP_INVALID cell=%u\n", (unsigned) cell_id);
            fflush(stderr);
        }
        return;
    }
    gyro_graph_ensure_init();

    uint32_t profile = profile_id;
    if (profile > (uint32_t) GYROGRAPH_PROFILE_Q_TRANSPORT) {
        profile = (uint32_t) GYROGRAPH_PROFILE_CHIRALITY;
    }

    ggml_critical_section_start();
    {
        gyroscopic_cell_state previous_state = { 0 };
        gyroscopic_cell_state next_state = { 0 };
        memcpy(&previous_state, &g_cells[cell_id], sizeof(gyroscopic_cell_state));
        int64_t cell_ids[1] = { 0 };

        gyrograph_ingest_word4_batch_indexed(
            cell_ids,
            &previous_state.omega12,
            &previous_state.step,
            &previous_state.last_byte,
            &previous_state.has_closed_word,
            previous_state.word4,
            previous_state.chi_ring64,
            &previous_state.chi_ring_pos,
            &previous_state.chi_valid_len,
            previous_state.chi_hist64,
            previous_state.shell_hist7,
            previous_state.family_ring64,
            previous_state.family_hist4,
            &previous_state.omega_sig,
            &previous_state.parity_O12,
            &previous_state.parity_E12,
            &previous_state.parity_bit,
            (uint8_t *) word4,
            &previous_state.resonance_key,
            profile,
            1);
        memcpy(&next_state, &previous_state, sizeof(gyroscopic_cell_state));
        gyroscopic_update_cell_coords(&next_state);

        memcpy(&g_cells[cell_id], &next_state, sizeof(gyroscopic_cell_state));
    }
    gyroscopic_update_cell_coords(&g_cells[cell_id]);
    if (gyroscopic_base_trace()) {
        fprintf(stderr, "GGMO_FEED_WORD4_EXIT cell_id=%u\n", (unsigned) cell_id);
        fflush(stderr);
    }
    ggml_critical_section_end();
}

GGML_API void ggml_gyroscopic_graph_feed_layer_block(uint32_t tensor_key, uint32_t row_block, uint32_t k_block, uint32_t class_id) {
    uint8_t w4[4];
    const uint32_t key = tensor_key ^ (row_block * 0x9e3779b9u) ^ (k_block * 0x85ebca6bu);
    const uint32_t cell = gyroscopic_role_cell(GYROSCOPIC_ROLE_LAYER_BASE, GYROSCOPIC_ROLE_LAYER_COUNT, key);
    w4[0] = (uint8_t)(tensor_key & 0xFFu);
    w4[1] = (uint8_t)(((row_block & 0x0Fu) << 4u) | (k_block & 0x0Fu));
    w4[2] = (uint8_t)(((class_id & 0x0Fu) << 4u) | ((row_block >> 4u) & 0x0Fu));
    w4[3] = (uint8_t)((k_block >> 4u) & 0xFFu);
    ggml_gyroscopic_graph_feed_word4(cell, w4, GYROGRAPH_PROFILE_SIGNATURE);
}

GGML_API void ggml_gyroscopic_graph_feed_kv_event(uint32_t segment_id, uint32_t block_id, uint32_t event_kind, uint32_t token_pos) {
    if (gyroscopic_base_trace()) {
        fprintf(stderr, "GGMO_FEED_KV_ENTER segment=%u block=%u kind=%u pos=%u\n", (unsigned) segment_id, (unsigned) block_id, (unsigned) event_kind, (unsigned) token_pos);
        fflush(stderr);
    }
    uint8_t w4[4];
    const uint32_t key = segment_id ^ (block_id * 0x27d4eb2du);
    const uint32_t cell = gyroscopic_role_cell(GYROSCOPIC_ROLE_KV_BASE, GYROSCOPIC_ROLE_KV_COUNT, key);
    w4[0] = (uint8_t)(segment_id & 0xFFu);
    w4[1] = (uint8_t)(block_id & 0xFFu);
    w4[2] = (uint8_t)(((event_kind & 0x0Fu) << 4u) | (token_pos & 0x0Fu));
    w4[3] = (uint8_t)((token_pos >> 4u) & 0xFFu);
    ggml_gyroscopic_graph_feed_word4(cell, w4, GYROGRAPH_PROFILE_CHIRALITY);
    if (gyroscopic_base_trace()) {
        fprintf(stderr, "GGMO_FEED_KV_EXIT segment=%u block=%u cell=%u\n", (unsigned) segment_id, (unsigned) block_id, (unsigned) cell);
        fflush(stderr);
    }
}

GGML_API double ggml_gyroscopic_graph_m2(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return GYROGRAPH_OMEGA_SIZE;
    }
    ggml_critical_section_start();
    const uint16_t * row = g_cells[seq_id].chi_hist64;
    uint64_t total = 0;
    int i;
    for (i = 0; i < 64; ++i) {
        total += (uint64_t)row[i];
    }
    double m2 = (total == 0) ? GYROGRAPH_OMEGA_SIZE : gyrograph_compute_m2_empirical(row, total);
    ggml_critical_section_end();
    return m2;
}

GGML_API double ggml_gyroscopic_graph_m2_empirical(uint32_t seq_id) {
    return ggml_gyroscopic_graph_m2(seq_id);
}

GGML_API bool ggml_gyroscopic_graph_has_empirical_m2(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return false;
    }
    ggml_critical_section_start();
    {
        const uint16_t * row = g_cells[seq_id].chi_hist64;
        uint64_t total = 0;
        int i;
        for (i = 0; i < 64; ++i) {
            total += (uint64_t) row[i];
        }
        ggml_critical_section_end();
        return total > 0;
    }
}

GGML_API uint32_t ggml_gyroscopic_graph_resonance_key(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0;
    }
    ggml_critical_section_start();
    uint32_t k = g_cells[seq_id].resonance_key;
    ggml_critical_section_end();
    return k;
}

GGML_API uint32_t ggml_gyroscopic_graph_current_resonance(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0;
    }
    ggml_critical_section_start();
    const uint8_t chi6 = g_cells[seq_id].chi6;
    const uint32_t v = (uint32_t) g_cells[seq_id].chi_hist64[chi6];
    ggml_critical_section_end();
    return v;
}

GGML_API uint64_t ggml_gyroscopic_graph_step(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0;
    }
    ggml_critical_section_start();
    const uint64_t step = g_cells[seq_id].step;
    ggml_critical_section_end();
    return step;
}

GGML_API uint8_t ggml_gyroscopic_graph_last_byte(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint8_t last_byte = g_cells[seq_id].last_byte;
    ggml_critical_section_end();
    return last_byte;
}

GGML_API uint8_t ggml_gyroscopic_graph_family(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint8_t last_byte = g_cells[seq_id].last_byte;
    ggml_critical_section_end();
    return (uint8_t)(((((last_byte ^ 0xAAu) >> 7u) & 1u) << 1u) | ((last_byte ^ 0xAAu) & 1u));
}

GGML_API uint8_t ggml_gyroscopic_graph_micro_ref(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint8_t last_byte = g_cells[seq_id].last_byte;
    ggml_critical_section_end();
    return (uint8_t)(((last_byte ^ 0xAAu) >> 1u) & 0x3Fu);
}

GGML_API uint8_t ggml_gyroscopic_graph_q6(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint8_t last_byte = g_cells[seq_id].last_byte;
    ggml_critical_section_end();
    return (uint8_t)(((last_byte ^ 0xAAu) >> 1u) & 0x3Fu);
}

GGML_API uint8_t ggml_gyroscopic_graph_q_transport6(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint8_t q = (uint8_t)(
        (g_cells[seq_id].omega_sig & 0x3Fu) ^
        ((uint32_t) g_cells[seq_id].omega_sig >> 6u)
    ) & 0x3Fu;
    ggml_critical_section_end();
    return q;
}

GGML_API uint32_t ggml_gyroscopic_graph_omega_sig(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint32_t sig = (uint32_t) g_cells[seq_id].omega_sig;
    ggml_critical_section_end();
    return sig;
}

GGML_API uint32_t ggml_gyroscopic_graph_state24(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint32_t omega12 = (uint32_t) g_cells[seq_id].omega12 & 0x0FFFu;
    ggml_critical_section_end();
    return gyroscopic_omega12_to_state24(omega12);
}

GGML_API uint16_t ggml_gyroscopic_graph_horizon_distance(uint32_t seq_id) {
    const uint32_t state24 = ggml_gyroscopic_graph_state24(seq_id);
    const uint16_t a12 = (uint16_t)((state24 >> 12u) & 0x0FFFu);
    const uint16_t b12 = (uint16_t)(state24 & 0x0FFFu);
    return (uint16_t) gyroscopic_popcount32((uint32_t)(a12 ^ (b12 ^ 0x0FFFu)));
}

GGML_API uint16_t ggml_gyroscopic_graph_ab_distance(uint32_t seq_id) {
    const uint32_t state24 = ggml_gyroscopic_graph_state24(seq_id);
    const uint16_t a12 = (uint16_t)((state24 >> 12u) & 0x0FFFu);
    const uint16_t b12 = (uint16_t)(state24 & 0x0FFFu);
    return (uint16_t) gyroscopic_popcount32((uint32_t)(a12 ^ b12));
}

GGML_API uint16_t ggml_gyroscopic_graph_parity_O12(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint16_t v = g_cells[seq_id].parity_O12;
    ggml_critical_section_end();
    return v;
}

GGML_API uint16_t ggml_gyroscopic_graph_parity_E12(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint16_t v = g_cells[seq_id].parity_E12;
    ggml_critical_section_end();
    return v;
}

GGML_API uint8_t ggml_gyroscopic_graph_parity_bit(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint8_t v = g_cells[seq_id].parity_bit;
    ggml_critical_section_end();
    return v;
}

GGML_API uint8_t ggml_gyroscopic_graph_chi6(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint8_t chi6 = g_cells[seq_id].chi6;
    ggml_critical_section_end();
    return chi6;
}

GGML_API uint8_t ggml_gyroscopic_graph_shell(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 0u;
    }
    ggml_critical_section_start();
    const uint8_t shell = g_cells[seq_id].shell;
    ggml_critical_section_end();
    return shell;
}

GGML_API double ggml_gyroscopic_graph_shell_concentration(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 1.0;
    }
    ggml_critical_section_start();
    {
        const uint16_t * row = g_cells[seq_id].shell_hist7;
        uint64_t total = 0;
        double mean = 0.0;
        double var = 0.0;
        int i;
        for (i = 0; i < 7; ++i) {
            total += (uint64_t) row[i];
        }
        if (total == 0) {
            ggml_critical_section_end();
            return 1.0;
        }
        for (i = 0; i < 7; ++i) {
            mean += (double) i * (double) row[i] / (double) total;
        }
        for (i = 0; i < 7; ++i) {
            const double d = (double) i - mean;
            var += d * d * (double) row[i] / (double) total;
        }
        ggml_critical_section_end();
        return var / GYROGRAPH_SHELL_MAX_VARIANCE;
    }
}

GGML_API bool ggml_gyroscopic_graph_near_horizon(uint32_t seq_id) {
    const double shell_conc = ggml_gyroscopic_graph_shell_concentration(seq_id);
    const double m2 = ggml_gyroscopic_graph_m2_empirical(seq_id);
    return (shell_conc < 0.25) || (m2 < 96.0);
}

GGML_API double ggml_gyroscopic_graph_kv_eviction_priority(uint32_t seq_id) {
    const double m2 = ggml_gyroscopic_graph_m2_empirical(seq_id);
    const double shell_conc = ggml_gyroscopic_graph_shell_concentration(seq_id);
    const double horizon_bias = ggml_gyroscopic_graph_near_horizon(seq_id) ? 0.75 : 1.25;
    return m2 * (1.0 + shell_conc) * horizon_bias;
}

GGML_API void ggml_gyroscopic_graph_m2_stats(
    double *min_m2,
    double *max_m2,
    double *mean_m2,
    int *n_cells_seen
) {
    if (min_m2) {
        *min_m2 = 0.0;
    }
    if (max_m2) {
        *max_m2 = 0.0;
    }
    if (mean_m2) {
        *mean_m2 = 0.0;
    }
    if (n_cells_seen) {
        *n_cells_seen = 0;
    }
    if (!g_graph_ready) {
        return;
    }
    double sum = 0.0;
    int n = 0;
    double mn = 0.0;
    double mx = 0.0;
    uint32_t sid;
    ggml_critical_section_start();
    for (sid = 0; sid < GYROGRAPH_MAX_CELLS; ++sid) {
        if (g_cells[sid].step == 0) {
            continue;
        }
        const uint16_t *row = g_cells[sid].chi_hist64;
        uint64_t total = 0;
        int i;
        for (i = 0; i < 64; ++i) {
            total += (uint64_t)row[i];
        }
        double m2 = (total == 0) ? GYROGRAPH_OMEGA_SIZE : gyrograph_compute_m2_empirical(row, total);
        if (n == 0) {
            mn = mx = m2;
        } else {
            if (m2 < mn) {
                mn = m2;
            }
            if (m2 > mx) {
                mx = m2;
            }
        }
        sum += m2;
        n++;
    }
    ggml_critical_section_end();
    if (n == 0) {
        return;
    }
    if (min_m2) {
        *min_m2 = mn;
    }
    if (max_m2) {
        *max_m2 = mx;
    }
    if (mean_m2) {
        *mean_m2 = sum / (double)n;
    }
    if (n_cells_seen) {
        *n_cells_seen = n;
    }
}
