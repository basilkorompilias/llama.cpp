#include "ggml.h"
#include "ggml-gyroscopic-graph.h"

#include "ggml-threading.h"
#include "gyrograph.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GYROGRAPH_MAX_CELLS 256

static int32_t g_omega12[GYROGRAPH_MAX_CELLS];
static uint64_t g_step[GYROGRAPH_MAX_CELLS];
static uint8_t g_last_byte[GYROGRAPH_MAX_CELLS];
static uint8_t g_has_closed_word[GYROGRAPH_MAX_CELLS];
static uint8_t g_word4[4 * GYROGRAPH_MAX_CELLS];
static uint8_t g_chi_ring64[64 * GYROGRAPH_MAX_CELLS];
static uint8_t g_chi_ring_pos[GYROGRAPH_MAX_CELLS];
static uint8_t g_chi_valid_len[GYROGRAPH_MAX_CELLS];
static uint16_t g_chi_hist64[64 * GYROGRAPH_MAX_CELLS];
static uint16_t g_shell_hist7[7 * GYROGRAPH_MAX_CELLS];
static uint8_t g_family_ring64[64 * GYROGRAPH_MAX_CELLS];
static uint16_t g_family_hist4[4 * GYROGRAPH_MAX_CELLS];
static int32_t g_omega_sig[GYROGRAPH_MAX_CELLS];
static uint16_t g_parity_O12[GYROGRAPH_MAX_CELLS];
static uint16_t g_parity_E12[GYROGRAPH_MAX_CELLS];
static uint8_t g_parity_bit[GYROGRAPH_MAX_CELLS];
static uint32_t g_resonance_key[GYROGRAPH_MAX_CELLS];
static int g_graph_ready;

static void gyro_graph_ensure_init(void) {
    if (g_graph_ready) {
        return;
    }
    ggml_critical_section_start();
    if (!g_graph_ready) {
        gyrograph_init();
        memset(g_omega12, 0, sizeof(g_omega12));
        memset(g_step, 0, sizeof(g_step));
        memset(g_last_byte, 0, sizeof(g_last_byte));
        memset(g_has_closed_word, 0, sizeof(g_has_closed_word));
        memset(g_word4, 0, sizeof(g_word4));
        memset(g_chi_ring64, 0, sizeof(g_chi_ring64));
        memset(g_chi_ring_pos, 0, sizeof(g_chi_ring_pos));
        memset(g_chi_valid_len, 0, sizeof(g_chi_valid_len));
        memset(g_chi_hist64, 0, sizeof(g_chi_hist64));
        memset(g_shell_hist7, 0, sizeof(g_shell_hist7));
        memset(g_family_ring64, 0, sizeof(g_family_ring64));
        memset(g_family_hist4, 0, sizeof(g_family_hist4));
        memset(g_omega_sig, 0, sizeof(g_omega_sig));
        memset(g_parity_O12, 0, sizeof(g_parity_O12));
        memset(g_parity_E12, 0, sizeof(g_parity_E12));
        memset(g_parity_bit, 0, sizeof(g_parity_bit));
        memset(g_resonance_key, 0, sizeof(g_resonance_key));
        g_graph_ready = 1;
    }
    ggml_critical_section_end();
}

void ggml_gyroscopic_graph_feed_token(uint32_t seq_id, uint32_t token_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS) {
        return;
    }
    gyro_graph_ensure_init();

    uint8_t w4[4];
    w4[0] = (uint8_t)(token_id & 0xFFu);
    w4[1] = (uint8_t)((token_id >> 8) & 0xFFu);
    w4[2] = (uint8_t)((token_id >> 16) & 0xFFu);
    w4[3] = (uint8_t)((token_id >> 24) & 0xFFu);

    int64_t cell_ids[1] = { (int64_t)seq_id };

    ggml_critical_section_start();
    gyrograph_ingest_word4_batch_indexed(
        cell_ids,
        g_omega12,
        g_step,
        g_last_byte,
        g_has_closed_word,
        g_word4,
        g_chi_ring64,
        g_chi_ring_pos,
        g_chi_valid_len,
        g_chi_hist64,
        g_shell_hist7,
        g_family_ring64,
        g_family_hist4,
        g_omega_sig,
        g_parity_O12,
        g_parity_E12,
        g_parity_bit,
        w4,
        g_resonance_key,
        4,
        1);
    ggml_critical_section_end();
}

GGML_API double ggml_gyroscopic_graph_m2(uint32_t seq_id) {
    if (seq_id >= GYROGRAPH_MAX_CELLS || !g_graph_ready) {
        return 4096.0;
    }
    ggml_critical_section_start();
    const uint16_t * row = g_chi_hist64 + ((size_t)seq_id * 64u);
    uint64_t total = 0;
    int i;
    for (i = 0; i < 64; ++i) {
        total += (uint64_t)row[i];
    }
    double m2 = (total == 0) ? 4096.0 : gyrograph_compute_m2_empirical(row, total);
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
        const uint16_t * row = g_chi_hist64 + ((size_t)seq_id * 64u);
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
    uint32_t k = g_resonance_key[seq_id];
    ggml_critical_section_end();
    return k;
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
        if (g_step[sid] == 0) {
            continue;
        }
        const uint16_t *row = g_chi_hist64 + ((size_t)sid * 64u);
        uint64_t total = 0;
        int i;
        for (i = 0; i < 64; ++i) {
            total += (uint64_t)row[i];
        }
        double m2 = (total == 0) ? 4096.0 : gyrograph_compute_m2_empirical(row, total);
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
