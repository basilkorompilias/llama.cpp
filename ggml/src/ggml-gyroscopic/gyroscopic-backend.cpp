#include "gyroscopic-backend.h"

#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-backend.h"
#include "ggml-gyroscopic-graph.h"

#include "ggml-impl.h"
#include "gyrograph_api.h"
#include "ggml.h"

#include "gyroscopic-common.h"

#include "gyrolabe_api.h"
#include "gyrolabe_kernel_wht.h"
#include "gyrolabe_kernel_qubec_matmul.h"
#include "gyrograph_policy.h"

#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

extern "C" bool ggml_gyroscopic_active(void);

static std::atomic<long long> g_qubec_calls{0};
static std::atomic<long long> g_qubec_attempts{0};
static std::atomic<long long> g_radial_calls{0};
static std::atomic<long long> g_chi_calls{0};
static std::atomic<long long> g_chi_gauge_calls{0};
static std::atomic<long long> g_dense_calls{0};
static std::atomic<long long> g_dispatch_scanned_blocks{0};
static std::atomic<long long> g_dispatch_no_structured_fallback{0};
static std::atomic<long long> g_dispatch_policy_residual_skipped{0};
static std::atomic<long long> g_dispatch_no_k64_blocks{0};
static std::atomic<int> g_gyro_bind_trace_printed{0};
static std::atomic<int> g_gyro_dims_trace_printed{0};
static std::atomic<long long> g_structured_rows{0};
static std::atomic<long long> g_dense_rows{0};
static std::atomic<long long> g_direct_exact_rows{0};
static std::atomic<long long> g_pq_rows{0};
static std::atomic<long long> g_dq_rows{0};
static std::atomic<long long> g_shell_radial_rows{0};
static std::atomic<long long> g_shell_gauge_rows{0};
static std::atomic<long long> g_chi_invariant_rows{0};
static std::atomic<long long> g_chi_gauge_rows{0};
static std::atomic<long long> g_generic_rows{0};
static std::atomic<long long> g_structured_attempt_rows{0};
static std::atomic<long long> g_exact_witness_rows{0};
static std::atomic<long long> g_witness_sampled_rows{0};
static std::atomic<long long> g_parity_mismatch_rows{0};
static std::atomic<long long> g_pq_est_read_bytes{0};
static std::atomic<long long> g_dq_est_read_bytes{0};
static std::atomic<long long> g_dense_est_read_bytes{0};

static std::atomic<double> g_max_abs_row_error{0.0};
static std::atomic<double> g_atlas_defect_error_sum{0.0};
static std::atomic<long long> g_atlas_defect_error_samples{0};
static std::atomic<double> g_atlas_defect_error_max{0.0};
static std::atomic<double> g_witness_error_sum{0.0};
static std::atomic<double> g_witness_error_sq_sum{0.0};
static std::atomic<long long> g_witness_error_samples{0};
static std::atomic<long long> g_shell_radial_blocks{0};
static std::atomic<long long> g_shell_gauge_blocks{0};
static std::atomic<long long> g_chi_invariant_blocks{0};
static std::atomic<long long> g_chi_gauge_blocks{0};
static std::atomic<long long> g_generic_blocks{0};
static std::atomic<long long> g_certified_shell_radial_blocks{0};
static std::atomic<long long> g_certified_shell_gauge_blocks{0};
static std::atomic<long long> g_certified_chi_invariant_blocks{0};
static std::atomic<long long> g_certified_chi_gauge_blocks{0};
static std::atomic<long long> g_certified_generic_blocks{0};
static std::atomic<long long> g_selected_shell_radial_blocks{0};
static std::atomic<long long> g_selected_shell_gauge_blocks{0};
static std::atomic<long long> g_selected_chi_invariant_blocks{0};
static std::atomic<long long> g_selected_chi_gauge_blocks{0};
static std::atomic<long long> g_selected_generic_blocks{0};
static std::atomic<long long> g_exec_shell_radial_blocks{0};
static std::atomic<long long> g_exec_shell_gauge_blocks{0};
static std::atomic<long long> g_exec_chi_invariant_blocks{0};
static std::atomic<long long> g_exec_chi_gauge_blocks{0};
static std::atomic<long long> g_exec_generic_blocks{0};
static std::atomic<long long> g_residual_none_blocks{0};
static std::atomic<long long> g_residual_k4_blocks{0};
static std::atomic<long long> g_residual_exact_q8_blocks{0};
static std::atomic<long long> g_residual_backfill_debug_blocks{0};
static std::atomic<long long> g_compiled_k4_residual_blocks{0};
static std::atomic<long long> g_exact_q8_generic_blocks{0};
static std::atomic<long long> g_direct_exact_generic_blocks{0};
static std::atomic<long long> g_direct_exact_calls{0};
static std::atomic<long long> g_chi_gauge_dq_kernel_calls{0};
static std::atomic<long long> g_chi_gauge_backfill_calls{0};
static std::atomic<long long> g_kernel_chi_gauge_k4_blocks{0};
static std::atomic<long long> g_kernel_chi_gauge_exact_q8_blocks{0};
static std::atomic<long long> g_kernel_generic_exact_q8_blocks{0};
static std::atomic<long long> g_kernel_projected_backfill_blocks{0};
static std::atomic<long long> g_dispatch_signature_index_hits{0};
static std::atomic<long long> g_dispatch_signature_index_misses{0};
static std::atomic<long long> g_q8dot_override_calls{0};
static std::atomic<int> g_decode_last_chi6{-1};
static std::atomic<int> g_decode_last_resonance_key{-1};
static std::atomic<long long> g_decode_events{0};
static std::atomic<long long> g_decode_grouped_dispatch{0};
static std::atomic<long long> g_decode_ungrouped_dispatch{0};
static std::atomic<long long> g_decode_kv_evict{0};
static std::atomic<long long> g_decode_chi_distance_sum{0};
static std::atomic<long long> g_decode_kv_priority_sum{0};
static std::atomic<int> g_decode_max_chi_distance{0};

static std::mutex g_gyrolabe_registry_q8_mutex;

static bool gyroscopic_env_truthy(const char * v) {
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    switch (v[0]) {
    case '0':
    case 'n':
    case 'N':
    case 'f':
    case 'F':
        return false;
    default:
        return true;
    }
}

static bool gyroscopic_env_pure_spectral(void) {
    return gyroscopic_env_truthy(std::getenv("GGML_GYROSCOPIC_PURE"));
}
static const bool g_pure_spectral_mode = gyroscopic_env_pure_spectral();

/* Hot-path telemetry gate.
 * Theory: instrumentation is a boundary-observability layer (Holography section 4) and
 * must not dominate the bulk computational cost it observes. When the env var
 * GGML_GYROSCOPIC_TELEMETRY is set to "0", per-panel stat accumulation in
 * gyroscopic_accumulate_dispatch_stats / gyroscopic_accumulate_call_stats is
 * skipped, eliminating ~40 contended atomic fetch_add operations per panel
 * across all worker threads. Dispatch/exec semantics are unchanged.
 * Default (unset or non-"0"): full telemetry, same as before.
 */
static bool gyroscopic_env_telemetry_enabled(void) {
    const char * v = std::getenv("GGML_GYROSCOPIC_TELEMETRY");
    if (v == nullptr || v[0] == '\0') {
        return true;
    }
    if (v[0] == '0' && v[1] == '\0') {
        return false;
    }
    return true;
}
static const bool g_hot_telemetry_enabled = gyroscopic_env_telemetry_enabled();

namespace {

const char * config_interop_mode_label(void) {
    const gyro_policy * policy = gyro_policy_get();
    if (policy == nullptr) {
        return "unknown";
    }
    switch (policy->interop_mode) {
    case GYRO_INTEROP_OBSERVABILITY:
        return "observability";
    case GYRO_INTEROP_ADVISORY:
        return "advisory";
    case GYRO_INTEROP_EXACT_SUBSTITUTION:
        return "exact_substitution";
    case GYRO_INTEROP_HYBRID_EXACT_RESIDUAL:
        return "hybrid_exact_residual";
    case GYRO_INTEROP_APPROXIMATE_DERIVED:
        return "approximate_derived";
    default:
        return "unknown";
    }
}

bool config_gyroscopic_mode_active(void) {
    return gyro_policy_get()->mode == GYRO_MODE_GYROSCOPIC;
}

bool config_strict(void) {
    return gyro_policy_get()->strict != 0;
}

static bool config_gyro_graph_approximate_mode(void) {
    const gyro_policy * policy = gyro_policy_get();
    return policy != NULL && policy->interop_mode == GYRO_INTEROP_APPROXIMATE_DERIVED;
}

static bool config_gyro_graph_owner_only_mode(void) {
    return config_gyroscopic_mode_active() && !config_gyro_graph_approximate_mode();
}

bool config_trace(void) {
    return gyro_policy_get()->trace != 0;
}

[[noreturn]] void gyroscopic_abort_fallback(
    const char * reason,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    int m,
    int n,
    int k
) {
    std::fprintf(
        stderr,
        "GyroMatMul STRICT failure: %s (src0_type=%d src1_type=%d m=%d n=%d k=%d)\n",
        reason,
        src0 ? (int) src0->type : -1,
        src1 ? (int) src1->type : -1,
        m,
        n,
        k
    );
    std::fflush(stderr);
    std::abort();
}

bool gyroscopic_mul_mat_geometry_ok(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const void * src1_batch,
    int m,
    int n,
    int k,
    int lda_bytes,
    int ldb_bytes
) {
    const bool trace_active = config_trace() && ggml_gyroscopic_active();
    if (trace_active) {
        std::fprintf(
            stderr,
            "GyroGeom enter: src0=%p src1=%p src1_batch=%p m=%d n=%d k=%d lda=%d ldb=%d\n",
            (const void *) src0,
            (const void *) src1,
            src1_batch,
            m,
            n,
            k,
            lda_bytes,
            ldb_bytes
        );
        std::fflush(stderr);
    }

    if (!ggml_gyroscopic_active() || src0 == nullptr || src1 == nullptr || src1_batch == nullptr) {
        if (trace_active) {
            std::fprintf(
                stderr,
                "GyroGeom reject: not active or null ptr (active=%d src0=%p src1=%p src1_batch=%p)\n",
                (int) ggml_gyroscopic_active(),
                (const void *) src0,
                (const void *) src1,
                src1_batch
            );
            std::fflush(stderr);
        }
        return false;
    }

    if (src0->type != GGML_TYPE_Q8_0) {
        if (trace_active) {
            std::fprintf(
                stderr,
                "GyroGeom reject: src0 type=%d (expected Q8_0)\n",
                (int) src0->type
            );
            std::fflush(stderr);
        }
        return false;
    }
    if (src0->buffer == nullptr || src0->data == nullptr) {
        if (trace_active) {
            std::fprintf(stderr, "GyroGeom reject: src0 buffer/data null\n");
            std::fflush(stderr);
        }
        return false;
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        if (trace_active) {
            std::fprintf(
                stderr,
                "GyroGeom reject: non-positive geometry m=%d n=%d k=%d\n",
                m,
                n,
                k
            );
            std::fflush(stderr);
        }
        return false;
    }
    if ((k % 64) != 0) {
        if (trace_active) {
            std::fprintf(stderr, "GyroGeom reject: k not multiple of 64 k=%d\n", k);
            std::fflush(stderr);
        }
        return false;
    }
    if (src1->data == nullptr) {
        if (trace_active) {
            std::fprintf(stderr, "GyroGeom reject: src1 data null\n");
            std::fflush(stderr);
        }
        return false;
    }
    if ((size_t) lda_bytes % sizeof(gyromatmul_block_q8_0) != 0 || (size_t) ldb_bytes % sizeof(gyromatmul_block_q8_0) != 0) {
        if (trace_active) {
            std::fprintf(
                stderr,
                "GyroGeom reject: stride bytes not aligned lda=%d ldb=%d block=%zu\n",
                lda_bytes,
                ldb_bytes,
                sizeof(gyromatmul_block_q8_0)
            );
            std::fflush(stderr);
        }
        return false;
    }
    const size_t lda_blocks = (size_t) lda_bytes / sizeof(gyromatmul_block_q8_0);
    const size_t ldb_blocks = (size_t) ldb_bytes / sizeof(gyromatmul_block_q8_0);
    if (lda_blocks == 0 || ldb_blocks == 0) {
        return false;
    }
    if (src1->type == GGML_TYPE_Q8_0) {
        if (src1->buffer == nullptr) {
            if (trace_active) {
                std::fprintf(stderr, "GyroGeom reject: src1 buffer null\n");
                std::fflush(stderr);
            }
            return false;
        }
        const ptrdiff_t start = (const char *) src1_batch - (const char *) src1->data;
        if (start < 0) {
            if (trace_active) {
                std::fprintf(stderr, "GyroGeom reject: src1_batch before src1 data start=%ld\n", (long) start);
                std::fflush(stderr);
            }
            return false;
        }
        if (ldb_bytes <= 0) {
            if (trace_active) {
                std::fprintf(stderr, "GyroGeom reject: ldb_bytes <= 0 ldb_bytes=%d\n", ldb_bytes);
                std::fflush(stderr);
            }
            return false;
        }
        if ((size_t) n > SIZE_MAX / (size_t) ldb_bytes) {
            if (trace_active) {
                std::fprintf(stderr, "GyroGeom reject: n * ldb overflow n=%d ldb_bytes=%d\n", n, ldb_bytes);
                std::fflush(stderr);
            }
            return false;
        }
        const size_t need = (size_t) start + (size_t) n * (size_t) ldb_bytes;
        if (need < (size_t) start) {
            if (trace_active) {
                std::fprintf(stderr, "GyroGeom reject: src1 need overflow\n");
                std::fflush(stderr);
            }
            return false;
        }
        if (need > ggml_nbytes(src1)) {
            if (trace_active) {
                std::fprintf(
                    stderr,
                    "GyroGeom reject: need beyond src1 bytes need=%zu ggml_nbytes=%zu\n",
                    need,
                    ggml_nbytes(src1)
                );
                std::fflush(stderr);
            }
            return false;
        }
    }

            if (trace_active) {
        std::fprintf(
            stderr,
            "GyroGeom ok: src0=%p src1=%p m=%d n=%d k=%d lda=%d ldb=%d\n",
            (const void *) src0,
            (const void *) src1,
            m,
            n,
            k,
            lda_bytes,
            ldb_bytes
        );
        std::fflush(stderr);
    }

    return true;
}

#define GYROSCOPIC_ACCUM_STAT(counter, value) ((counter).fetch_add((long long)(value)))

static void gyroscopic_accumulate_double_sum(std::atomic<double> * counter, double value) {
    if (counter == nullptr) {
        return;
    }
    if (value == 0.0) {
        return;
    }
    double cur = counter->load();
    while (true) {
        const double next = cur + value;
        if (counter->compare_exchange_weak(cur, next)) {
            return;
        }
    }
}

static void gyroscopic_accumulate_double_max(std::atomic<double> * counter, double value) {
    if (counter == nullptr) {
        return;
    }
    if (value <= 0.0) {
        return;
    }
    double cur = counter->load();
    while (value > cur) {
        if (counter->compare_exchange_weak(cur, value)) {
            return;
        }
    }
}

static int gyroscopic_decode_grouped_dispatch(uint32_t cell_idx) {
    const gyro_policy * policy = gyro_policy_get();
    if (policy == nullptr || cell_idx >= GYROGRAPH_MAX_CELLS) {
        return 1;
    }
    const uint32_t sid = ggml_gyroscopic_graph_has_empirical_m2(cell_idx) ? cell_idx : 0u;
    const uint8_t chi6 = ggml_gyroscopic_graph_chi6(sid);
    const uint8_t shell = ggml_gyroscopic_graph_shell(sid);
    const uint32_t resonance_key = ggml_gyroscopic_graph_resonance_key(sid);
    const double m2 = ggml_gyroscopic_graph_m2_empirical(sid);
    const double kv_priority = ggml_gyroscopic_graph_kv_eviction_priority(sid);
    const int near_horizon = ggml_gyroscopic_graph_near_horizon(sid) ? 1 : 0;
    const int prev_chi = g_decode_last_chi6.exchange((int) chi6);
    const int prev_resonance_key = g_decode_last_resonance_key.exchange((int) resonance_key);
    const int chi_dist = prev_chi >= 0 ? gyroscopic_popcount6((uint32_t) chi6 ^ (uint32_t) prev_chi) : 0;
    const int same_resonance = prev_resonance_key >= 0 && prev_resonance_key == (int) resonance_key;
    const int grouped_dispatch = prev_chi < 0 ? 1 : (same_resonance || chi_dist <= policy->decode_max_chirality_distance ? 1 : 0);
    const double kv_score_norm = kv_priority / GYROGRAPH_OMEGA_SIZE;
    const int kv_evict = kv_score_norm < policy->kv_eviction_threshold ? 1 : 0;
    gyro_policy_set_decode_climate(chi_dist, kv_evict, near_horizon, m2, kv_priority);

    if (g_hot_telemetry_enabled) {
        g_decode_events.fetch_add(1);
        g_decode_chi_distance_sum.fetch_add((long long) chi_dist);
        g_decode_kv_priority_sum.fetch_add((long long) kv_priority);
        if (grouped_dispatch) {
            g_decode_grouped_dispatch.fetch_add(1);
        } else {
            g_decode_ungrouped_dispatch.fetch_add(1);
        }
        if (kv_evict) {
            g_decode_kv_evict.fetch_add(1);
        }
        int cur = g_decode_max_chi_distance.load();
        while (chi_dist > cur) {
            if (g_decode_max_chi_distance.compare_exchange_weak(cur, chi_dist)) {
                break;
            }
        }
    }
    if (config_trace()) {
        std::fprintf(
            stderr,
            "GyroDecode: cell=%u chi6=%u shell=%u resonance_key=%u chi_distance=%d grouped_dispatch=%d decode_max_chi_distance=%d decode_max_shell_delta=%d kv_priority=%.6f kv_score_norm=%.6f kv_eviction_threshold=%.6f kv_evict=%d m2_empirical=%.6f\n",
            sid,
            (unsigned int) chi6,
            (unsigned int) shell,
            (unsigned int) resonance_key,
            chi_dist,
            grouped_dispatch,
            policy->decode_max_chirality_distance,
            policy->decode_max_shell_delta,
            kv_priority,
            kv_score_norm,
            policy->kv_eviction_threshold,
            kv_evict,
            m2
        );
    }
    return grouped_dispatch;
}

static void gyroscopic_accumulate_dispatch_stats(const gyrolabe_qubec_dispatch_stats * stats) {
    if (!g_hot_telemetry_enabled) {
        return;
    }
    GYROSCOPIC_ACCUM_STAT(g_dispatch_scanned_blocks, stats->scanned_blocks);
    GYROSCOPIC_ACCUM_STAT(g_dispatch_no_structured_fallback, stats->no_structured_default_route);
    GYROSCOPIC_ACCUM_STAT(g_dispatch_policy_residual_skipped, stats->policy_residual_skipped_blocks);
    GYROSCOPIC_ACCUM_STAT(g_dispatch_no_k64_blocks, stats->no_k64_blocks);
    GYROSCOPIC_ACCUM_STAT(g_shell_radial_blocks, stats->shell_radial_blocks);
    GYROSCOPIC_ACCUM_STAT(g_shell_gauge_blocks, stats->shell_gauge_blocks);
    GYROSCOPIC_ACCUM_STAT(g_chi_invariant_blocks, stats->chi_invariant_blocks);
    GYROSCOPIC_ACCUM_STAT(g_chi_gauge_blocks, stats->chi_gauge_blocks);
    GYROSCOPIC_ACCUM_STAT(g_generic_blocks, stats->generic_blocks);
    GYROSCOPIC_ACCUM_STAT(g_certified_shell_radial_blocks, stats->certified_shell_radial_blocks);
    GYROSCOPIC_ACCUM_STAT(g_certified_shell_gauge_blocks, stats->certified_shell_gauge_blocks);
    GYROSCOPIC_ACCUM_STAT(g_certified_chi_invariant_blocks, stats->certified_chi_invariant_blocks);
    GYROSCOPIC_ACCUM_STAT(g_certified_chi_gauge_blocks, stats->certified_chi_gauge_blocks);
    GYROSCOPIC_ACCUM_STAT(g_certified_generic_blocks, stats->certified_generic_blocks);
    GYROSCOPIC_ACCUM_STAT(g_selected_shell_radial_blocks, stats->selected_shell_radial_blocks);
    GYROSCOPIC_ACCUM_STAT(g_selected_shell_gauge_blocks, stats->selected_shell_gauge_blocks);
    GYROSCOPIC_ACCUM_STAT(g_selected_chi_invariant_blocks, stats->selected_chi_invariant_blocks);
    GYROSCOPIC_ACCUM_STAT(g_selected_chi_gauge_blocks, stats->selected_chi_gauge_blocks);
    GYROSCOPIC_ACCUM_STAT(g_selected_generic_blocks, stats->selected_generic_blocks);
    GYROSCOPIC_ACCUM_STAT(g_exec_shell_radial_blocks, stats->exec_shell_radial_blocks);
    GYROSCOPIC_ACCUM_STAT(g_exec_shell_gauge_blocks, stats->exec_shell_gauge_blocks);
    GYROSCOPIC_ACCUM_STAT(g_exec_chi_invariant_blocks, stats->exec_chi_invariant_blocks);
    GYROSCOPIC_ACCUM_STAT(g_exec_chi_gauge_blocks, stats->exec_chi_gauge_blocks);
    GYROSCOPIC_ACCUM_STAT(g_exec_generic_blocks, stats->exec_generic_blocks);
    GYROSCOPIC_ACCUM_STAT(g_residual_none_blocks, stats->residual_none_blocks);
    GYROSCOPIC_ACCUM_STAT(g_residual_k4_blocks, stats->residual_k4_blocks);
    GYROSCOPIC_ACCUM_STAT(g_residual_exact_q8_blocks, stats->residual_exact_q8_blocks);
    GYROSCOPIC_ACCUM_STAT(g_residual_backfill_debug_blocks, stats->residual_backfill_debug_blocks);
    GYROSCOPIC_ACCUM_STAT(g_compiled_k4_residual_blocks, stats->compiled_k4_residual_blocks);
    GYROSCOPIC_ACCUM_STAT(g_exact_q8_generic_blocks, stats->exact_q8_generic_blocks);
    GYROSCOPIC_ACCUM_STAT(g_direct_exact_generic_blocks, stats->direct_exact_generic_blocks);
    GYROSCOPIC_ACCUM_STAT(g_chi_gauge_dq_kernel_calls, stats->chi_gauge_dq_kernel_calls);
    GYROSCOPIC_ACCUM_STAT(g_chi_gauge_backfill_calls, stats->chi_gauge_backfill_calls);
    GYROSCOPIC_ACCUM_STAT(g_kernel_chi_gauge_k4_blocks, stats->kernel_chi_gauge_k4_blocks);
    GYROSCOPIC_ACCUM_STAT(g_kernel_chi_gauge_exact_q8_blocks, stats->kernel_chi_gauge_exact_q8_blocks);
    GYROSCOPIC_ACCUM_STAT(g_kernel_generic_exact_q8_blocks, stats->kernel_generic_exact_q8_blocks);
    GYROSCOPIC_ACCUM_STAT(g_kernel_projected_backfill_blocks, stats->kernel_projected_backfill_blocks);
    GYROSCOPIC_ACCUM_STAT(g_dispatch_signature_index_hits, stats->index_query_hits);
    GYROSCOPIC_ACCUM_STAT(g_dispatch_signature_index_misses, stats->index_query_misses);
}

static void gyroscopic_accumulate_call_stats(const gyrolabe_qubec_call_stats * stats) {
    if (!g_hot_telemetry_enabled) {
        return;
    }
    GYROSCOPIC_ACCUM_STAT(g_structured_rows, stats->structured_rows);
    GYROSCOPIC_ACCUM_STAT(g_structured_attempt_rows, stats->structured_attempt_rows);
    GYROSCOPIC_ACCUM_STAT(g_exact_witness_rows, stats->exact_witness_rows);
    GYROSCOPIC_ACCUM_STAT(g_witness_sampled_rows, stats->witness_sampled_rows);
    GYROSCOPIC_ACCUM_STAT(g_dense_rows, stats->dense_rows);
    GYROSCOPIC_ACCUM_STAT(g_direct_exact_rows, stats->direct_exact_rows);
    GYROSCOPIC_ACCUM_STAT(g_pq_rows, stats->pq_rows);
    GYROSCOPIC_ACCUM_STAT(g_dq_rows, stats->dq_rows);
    GYROSCOPIC_ACCUM_STAT(g_shell_radial_rows, stats->shell_radial_rows);
    GYROSCOPIC_ACCUM_STAT(g_shell_gauge_rows, stats->shell_gauge_rows);
    GYROSCOPIC_ACCUM_STAT(g_chi_invariant_rows, stats->chi_invariant_rows);
    GYROSCOPIC_ACCUM_STAT(g_chi_gauge_rows, stats->chi_gauge_rows);
    GYROSCOPIC_ACCUM_STAT(g_generic_rows, stats->generic_rows);
    GYROSCOPIC_ACCUM_STAT(g_parity_mismatch_rows, stats->parity_mismatch_rows);
    GYROSCOPIC_ACCUM_STAT(g_pq_est_read_bytes, stats->pq_est_read_bytes);
    GYROSCOPIC_ACCUM_STAT(g_dq_est_read_bytes, stats->dq_est_read_bytes);
    GYROSCOPIC_ACCUM_STAT(g_dense_est_read_bytes, stats->dense_est_read_bytes);
    GYROSCOPIC_ACCUM_STAT(g_atlas_defect_error_samples, stats->atlas_defect_error_samples);
    GYROSCOPIC_ACCUM_STAT(g_witness_error_samples, stats->witness_error_samples);
    gyroscopic_accumulate_double_sum(&g_atlas_defect_error_sum, (double) stats->atlas_defect_error_sum);
    gyroscopic_accumulate_double_max(&g_atlas_defect_error_max, (double) stats->atlas_defect_error_max);
    gyroscopic_accumulate_double_sum(&g_witness_error_sum, (double) stats->witness_error_sum);
    gyroscopic_accumulate_double_sum(&g_witness_error_sq_sum, (double) stats->witness_error_sq_sum);
}

static bool gyroscopic_register_ready(
    const struct ggml_tensor * w_tensor_runtime,
    const void * reg_base,
    const struct ggml_tensor * src0,
    int row_start,
    int m,
    int k,
    int k_block
) {
    (void) row_start;
    (void) m;
    (void) k;
    (void) k_block;

    gyrolabe_registry_entry_t reg_entry = gyrolabe_registry_find_entry(w_tensor_runtime, reg_base);
    if (reg_entry == nullptr) {
        std::lock_guard<std::mutex> lock(g_gyrolabe_registry_q8_mutex);
        reg_entry = gyrolabe_registry_find_entry(w_tensor_runtime, reg_base);
        if (reg_entry == nullptr && src0 != nullptr) {
            gyrolabe_registry_register_q8_buffer(
                reg_base,
                (int64_t) src0->ne[0],
                (int64_t) src0->ne[1],
                (int64_t) src0->ne[2],
                (int64_t) src0->ne[3],
                (size_t) src0->nb[1],
                src0->name
            );
            reg_entry = gyrolabe_registry_find_entry(w_tensor_runtime, reg_base);
        }
    }
    return reg_entry != nullptr;
}

static bool gyroscopic_mul_mat_input_ok(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const void * a,
    const void * wdata,
    int m,
    int n,
    int k,
    size_t row_size,
    int ldc,
    const float * c
) {
    const bool trace_active = config_trace() && ggml_gyroscopic_active();
    if (trace_active) {
        std::fprintf(
            stderr,
            "GyroInput enter: src0=%p src1=%p a=%p wdata=%p m=%d n=%d k=%d row_size=%zu ldc=%d c=%p\n",
            (const void *) src0,
            (const void *) src1,
            a,
            wdata,
            m,
            n,
            k,
            row_size,
            ldc,
            (const void *) c
        );
        std::fflush(stderr);
    }
    if (!ggml_gyroscopic_active() || src0 == nullptr || src1 == nullptr || a == nullptr || wdata == nullptr || c == nullptr) {
        if (trace_active) {
            std::fprintf(
                stderr,
                "GyroInput reject: bad state (active=%d src0=%p src1=%p a=%p wdata=%p c=%p)\n",
                (int) ggml_gyroscopic_active(),
                (const void *) src0,
                (const void *) src1,
                a,
                wdata,
                (const void *) c
            );
            std::fflush(stderr);
        }
        return false;
    }
    if (!gyroscopic_mul_mat_geometry_ok(src0, src1, wdata, m, n, k, (int) src0->nb[1], (int) row_size)) {
        if (trace_active) {
            std::fprintf(stderr, "GyroInput reject: geometry false\n");
            std::fflush(stderr);
        }
        if (config_strict()) {
            gyroscopic_abort_fallback("geometry rejected supported gyroscopic path", src0, src1, m, n, k);
        }
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0) {
        if (trace_active) {
            std::fprintf(stderr, "GyroInput reject: src0 type=%d\n", (int) src0->type);
            std::fflush(stderr);
        }
        return false;
    }
    if (m <= 0 || n <= 0 || k <= 0 || ldc < m) {
        if (trace_active) {
            std::fprintf(
                stderr,
                "GyroInput reject: bad shape m=%d n=%d k=%d ldc=%d\n",
                m,
                n,
                k,
                ldc
            );
            std::fflush(stderr);
        }
        return false;
    }
    if (row_size == 0 || (row_size % sizeof(gyromatmul_block_q8_0)) != 0) {
        if (trace_active) {
            std::fprintf(
                stderr,
                "GyroInput reject: bad row_size=%zu block=%zu\n",
                row_size,
                sizeof(gyromatmul_block_q8_0)
            );
            std::fflush(stderr);
        }
        return false;
    }
    if (k < 64 || (k % 64) != 0) {
        if (trace_active) {
            std::fprintf(stderr, "GyroInput reject: bad k=%d (need >=64 and mult of 64)\n", k);
            std::fflush(stderr);
        }
        if (config_strict()) {
            gyroscopic_abort_fallback("unsupported k geometry on gyroscopic path", src0, src1, m, n, k);
        }
        return false;
    }
    if (row_size > (size_t) INT_MAX) {
        if (trace_active) {
            std::fprintf(stderr, "GyroInput reject: row_size overflow row_size=%zu\n", row_size);
            std::fflush(stderr);
        }
        if (config_strict()) {
            gyroscopic_abort_fallback("row_size exceeds int on gyroscopic path", src0, src1, m, n, k);
        }
        return false;
    }
    if (trace_active) {
        std::fprintf(stderr, "GyroInput ok: src0=%p src1=%p m=%d n=%d k=%d\n", (const void *) src0, (const void *) src1, m, n, k);
        std::fflush(stderr);
    }
    return true;
}

} // namespace

static void gyroscopic_emit_trace_pulse(long long attempt_no);
static const bool g_runtime_gyroscopic_active = config_gyroscopic_mode_active();
static const bool g_runtime_gyroscopic_strict = config_strict();
static const bool g_runtime_gyroscopic_trace = config_trace();
static std::atomic<int> g_runtime_gyroscopic_env_enabled{-1};
static std::atomic<int> g_runtime_gyroscopic_clear_registry_env_enabled{-1};
static std::atomic<int> g_gyrolabe_registry_clear_done{0};

static bool gyrolabe_env_mode_enabled() {
    const char * v = getenv("GGML_GYROSCOPIC");
    if (v == nullptr || v[0] == '\0') {
        return true;
    }
    if (v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F') {
        return false;
    }
    return true;
}

static bool gyrolabe_env_mode_enabled_cached() {
    int cached = g_runtime_gyroscopic_env_enabled.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached != 0;
    }

    const int parsed = gyrolabe_env_mode_enabled() ? 1 : 0;
    g_runtime_gyroscopic_env_enabled.store(parsed, std::memory_order_release);
    return parsed != 0;
}

static bool gyrolabe_env_clear_registry_enabled() {
    const char * v = getenv("GGML_GYROSCOPIC_CLEAR_REGISTRY");
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    if (v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F') {
        return false;
    }
    return true;
}

static bool gyrolabe_env_clear_registry_enabled_cached() {
    int cached = g_runtime_gyroscopic_clear_registry_env_enabled.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached != 0;
    }

    const int parsed = gyrolabe_env_clear_registry_enabled() ? 1 : 0;
    g_runtime_gyroscopic_clear_registry_env_enabled.store(parsed, std::memory_order_release);
    return parsed != 0;
}

static void gyrolabe_maybe_clear_registry_once(void) {
    if (!gyrolabe_env_clear_registry_enabled_cached()) {
        return;
    }
    if (g_gyrolabe_registry_clear_done.exchange(1, std::memory_order_acq_rel) == 0) {
        gyrolabe_registry_clear();
    }
}

extern "C" {

bool ggml_gyroscopic_active(void) {
    return g_runtime_gyroscopic_active && gyrolabe_env_mode_enabled_cached();
}

bool ggml_gyroscopic_strict(void) {
    return g_runtime_gyroscopic_strict;
}

bool ggml_gyroscopic_trace(void) {
    return g_runtime_gyroscopic_trace;
}

bool ggml_gyroscopic_can_use(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const void * src1_batch,
    int m,
    int n,
    int k,
    int lda_bytes,
    int ldb_bytes
) {
    if (!ggml_gyroscopic_active() || src0 == nullptr || src1 == nullptr || src1_batch == nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if (k < 64 || (k % 64) != 0) {
        return false;
    }
    if (m <= 0 || n <= 0) {
        return false;
    }
    return true;
}

void ggml_gyroscopic_q8dot_override_inc(void) {
    g_q8dot_override_calls.fetch_add(1);
}

long long ggml_gyroscopic_q8dot_override_get(void) {
    return g_q8dot_override_calls.load();
}

void ggml_gyroscopic_abort_unsupported(const char * site, int32_t type0, int32_t type1) {
    std::fprintf(
        stderr,
        "Gyroscopic: unsupported live path at %s (type0=%d, type1=%d)\n",
        site,
        (int) type0,
        (int) type1
    );
    std::fflush(stderr);
    std::abort();
}

bool ggml_gyroscopic_vec_dot_f32(int n, const float * x, const float * y, float * out) {
    if (!ggml_gyroscopic_active()) {
        return false;
    }
    (void) n;
    (void) x;
    (void) y;
    (void) out;
    return false;
}

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
) {
    if (!ggml_gyroscopic_active()) {
        return false;
    }
    if (config_trace()) {
        fprintf(
            stderr,
            "GYRO_MATMUL_ENTER src0=%p src1=%p a=%p wdata=%p m=%d n=%d k=%d row_start=%d k_block=%d\n",
            (const void *) src0,
            (const void *) src1,
            a,
            (const void *) wdata,
            m,
            n,
            k,
            row_start,
            k_block
        );
        fflush(stderr);
    }
    if (!ggml_gyroscopic_can_use(src0, src1, wdata, m, n, k, (int) src0->nb[1], (int) row_size)) {
        return false;
    }
    gyrolabe_maybe_clear_registry_once();

    if (config_trace() && g_gyro_bind_trace_printed.exchange(1) == 0) {
        std::fprintf(
            stderr,
            "GyroBind: src0_type=%d src1_type=%d src0_data=%p wdata=%p\n",
            src0 ? (int) src0->type : -1,
            src1 ? (int) src1->type : -1,
            src0 ? src0->data : nullptr,
            (const void *) wdata
        );
        std::fflush(stderr);
    }
    GGML_ASSERT(row_start >= 0 && (int64_t) row_start + (int64_t) m <= (int64_t) src0->ne[1] * (int64_t) src0->ne[2]);

    /* Registry keys blocks by weight tensor pointer; w_tensor is src0 (weights, Q8_0).
     * src1 is the activation tensor (F32). */
    const struct ggml_tensor * w_tensor_runtime =
        (w_tensor != nullptr) ? w_tensor : src0;

    const void * reg_base = (w_registry_base != nullptr)
        ? w_registry_base
        : ((src0 != nullptr && src0->data != nullptr) ? src0->data : a);
    /* Registry remains optional metadata (analysis/class cache); miss does not imply
     * a routing failure as long as matmul inputs are valid.
     */
    // Optional metadata hydration only; execution must not depend on cache state.
    const bool registry_ready = gyroscopic_register_ready(w_tensor_runtime, reg_base, src0, row_start, m, k, k_block);

    const uint32_t request_cell_idx = ggml_gyroscopic_graph_last_request_cell();
    const uint32_t active_cell_idx = ggml_gyroscopic_graph_has_empirical_m2(request_cell_idx)
        ? request_cell_idx
        : (ggml_gyroscopic_graph_has_empirical_m2(cell_idx) ? cell_idx : 0u);
    uint32_t layer_class_id = 0u;
    if (registry_ready) {
        gyrolabe_registry_entry_t layer_entry = gyrolabe_registry_find_entry(w_tensor_runtime, reg_base);
        const gyrolabe_block_info_t * layer_info = gyrolabe_registry_get_block_from_entry(
            layer_entry,
            row_start >= 0 ? row_start / 64 : 0,
            k_block >= 0 ? k_block : 0);
        if (layer_info != nullptr) {
            layer_class_id = (uint32_t) layer_info->class_id;
        }
    }
    ggml_gyroscopic_graph_feed_layer_block(
        (uint32_t) ((uintptr_t) w_tensor_runtime),
        (uint32_t) (row_start >= 0 ? row_start / 64 : 0),
        (uint32_t) (k_block >= 0 ? k_block : 0),
        layer_class_id);
    const long long gyro_attempt_no = g_qubec_attempts.fetch_add(1, std::memory_order_relaxed) + 1;
    const int grouped_dispatch_for_policy = gyroscopic_decode_grouped_dispatch(active_cell_idx);
    gyro_policy_set_decode_grouped_dispatch(grouped_dispatch_for_policy);
    if (config_trace() && g_gyro_dims_trace_printed.exchange(1) == 0) {
        std::fprintf(
            stderr,
            "GyroDims: m=%d n=%d k=%d row_start=%d k_block=%d lda=%d ldb=%d row_size=%zu ldc=%d src0_ne1=%lld\n",
            m,
            n,
            k,
            row_start,
            k_block,
            (int) src0->nb[1],
            (int) row_size,
            row_size,
            ldc,
            (long long) src0->ne[1]
        );
        std::fflush(stderr);
    }

    /* QuBEC C API: a = weight panel (src0), lda = src0 row bytes; w = Q8 activations (wdata), ldb = row_size.
     * m = weight rows this thread; n = ne11 output cols; weight_col0 = global col origin; row_start = global weight row. */
    const int rc = gyrolabe_qubec_matmul_q8_0(
        m,
        n,
        k,
        (const gyromatmul_block_q8_0 *) a,
        (int) src0->nb[1],
        (const gyromatmul_block_q8_0 *) wdata,
        (int) row_size,
        c,
        ldc,
        weight_col0,
        w_tensor_runtime,
        reg_base,
        row_start,
        k_block,
        active_cell_idx
    );
    gyrolabe_qubec_dispatch_stats dispatch_stats = {};
    gyrolabe_qubec_get_last_dispatch_stats(&dispatch_stats);
    gyrolabe_qubec_call_stats call_stats = {};
    gyrolabe_qubec_get_last_call_stats(&call_stats);
    if (config_trace()) {
        std::fprintf(
            stderr,
            "GYRO_POST_STATS rc=%d structured=%d dense=%d direct=%d generic=%d pq=%d dq=%d no_structured=%d generic_exact=%d kernel_generic_q8=%d\n",
            rc,
            call_stats.structured_rows,
            call_stats.dense_rows,
            call_stats.direct_exact_rows,
            call_stats.generic_rows,
            call_stats.pq_rows,
            call_stats.dq_rows,
            dispatch_stats.no_structured_default_route,
            dispatch_stats.direct_exact_generic_blocks,
            dispatch_stats.kernel_generic_exact_q8_blocks
        );
        std::fflush(stderr);
    }
    if (rc != GYROLABE_QUBEC_MATMUL_OK) {
        gyroscopic_emit_trace_pulse(gyro_attempt_no);
        if (config_gyro_graph_owner_only_mode()) {
            gyroscopic_abort_fallback(
                "gyrolabe_qubec_matmul_q8_0 contract failure after backend claimed panel",
                src0,
                src1,
                m,
                n,
                k
            );
        }
        return false;
    }

    gyroscopic_accumulate_dispatch_stats(&dispatch_stats);
    gyroscopic_accumulate_call_stats(&call_stats);

    if (config_gyro_graph_owner_only_mode()) {
        const int total_rows = call_stats.structured_rows +
            call_stats.dense_rows +
            call_stats.direct_exact_rows +
            call_stats.generic_rows;
        const bool no_output_produced = total_rows <= 0 && call_stats.spectral_sparse_rows <= 0;
        if (no_output_produced) {
            std::fprintf(
                stderr,
                "GYRO_OWNER_CONTRACT_FAIL rc=%d structured=%d dense=%d direct=%d generic=%d pq=%d dq=%d spectral_sparse=%d\n",
                rc,
                call_stats.structured_rows,
                call_stats.dense_rows,
                call_stats.direct_exact_rows,
                call_stats.generic_rows,
                call_stats.pq_rows,
                call_stats.dq_rows,
                call_stats.spectral_sparse_rows
            );
            std::fflush(stderr);
            gyroscopic_abort_fallback(
                "gyrolabe_qubec_matmul_q8_0 owner-only contract violation after apparent success",
                src0,
                src1,
                m,
                n,
                k
            );
        }
    }

    if (config_strict() && call_stats.parity_mismatch_rows != 0) {
        gyroscopic_abort_fallback("native gyroscopic parity mismatch", src0, src1, m, n, k);
    }

    /* Count QuBEC only when the spectral chart is actually used. Direct exact
     * generic ownership is a valid GyroLabe TensorOps path, but it is not the
     * occupied Moment/chirality/WHT QuBEC path described in the SDK spec.
     */
    const bool used_qubec_operator_path =
        call_stats.pq_rows > 0 ||
        call_stats.structured_rows > 0 ||
        call_stats.spectral_sparse_rows > 0 ||
        call_stats.used_radial != 0 ||
        call_stats.used_chi != 0 ||
        call_stats.used_chi_gauge != 0 ||
        call_stats.shell_radial_rows > 0 ||
        call_stats.shell_gauge_rows > 0 ||
        call_stats.chi_invariant_rows > 0 ||
        call_stats.chi_gauge_rows > 0;
    if (used_qubec_operator_path) {
        g_qubec_calls.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_dense_calls.fetch_add(1, std::memory_order_relaxed);
    }
    gyroscopic_emit_trace_pulse(gyro_attempt_no);
    return true;
}

void ggml_gyroscopic_mul_mat_dispatch(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst,
    const struct ggml_compute_params * params,
    int64_t r2,
    int64_t r3
) {
    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(params != nullptr);
    GGML_ASSERT(r2 > 0 && r3 > 0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    const char * wdata = (src1->type == vec_dot_type) ? (const char *) src1->data : (const char *) params->wdata;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(row_size > 0);
    GGML_ASSERT(ne11 <= INT_MAX);
    GGML_ASSERT(ne01 <= INT_MAX);
    GGML_ASSERT(ne00 <= INT_MAX);

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;
            const int64_t m0 = ((int64_t) ith * ne01) / nth;
            const int64_t m1 = ((int64_t) (ith + 1) * ne01) / nth;

            if (m1 <= m0) {
                continue;
            }

            const char * a_panel = (const char *) src0->data + i02 * nb02 + i03 * nb03;
            const int64_t weight_row_base = i12 * ne11 + i13 * ne12 * ne11;
            const char * bp = (const char *) wdata + (size_t) weight_row_base * row_size;
            float * cp = (float *) ((char *) dst->data + i12 * nb2 + i13 * nb3) + m0;
            const uint32_t cell_idx = ggml_gyroscopic_graph_cell_idx((uint32_t)((i12 << 6) ^ ((int) (m0 >> 6) & 63)));

            const bool ok = ggml_gyroscopic_mul_mat(
                cell_idx,
                src0,
                src1,
                (const char *) a_panel + (size_t) m0 * (size_t) nb01,
                (const char *) bp,
                (const void *) src0->data,
                row_size,
                (int) (m1 - m0),
                (int) ne11,
                (int) ne00,
                0,
                src0,
                (int) m0,
                0,
                cp,
                (int) (nb1 / sizeof(float)));
            if (!ok) {
                gyroscopic_abort_fallback("backend dispatch could not execute requested q8 panel", src0, src1, (int) (m1 - m0), (int) ne11, (int) ne00);
            }
        }
    }
}

uint32_t ggml_gyroscopic_graph_cell_idx(uint32_t i12) {
    return i12 & (GYROGRAPH_MAX_CELLS - 1u);
}

} // extern "C"

static int gyro_trace_snapshot_interval(void) {
    static int s_cached = -1;
    if (s_cached >= 0) {
        return s_cached;
    }
    const char * e = std::getenv("GGML_GYROSCOPIC_TRACE_SNAPSHOT_EVERY");
    if (e == nullptr || e[0] == '\0') {
        s_cached = 0;
        return 0;
    }
    char * end_ptr = nullptr;
    const long n = std::strtol(e, &end_ptr, 10);
    (void) end_ptr;
    if (n < 1) {
        s_cached = 0;
        return 0;
    }
    if (n > 1000000) {
        s_cached = 1000000;
        return s_cached;
    }
    s_cached = (int) n;
    return s_cached;
}

static void gyroscopic_emit_trace_pulse(long long attempt_no) {
    if (!ggml_gyroscopic_trace()) {
        return;
    }
    const int every = gyro_trace_snapshot_interval();
    if (every <= 0 || attempt_no < 1) {
        return;
    }
    if ((attempt_no % (long long) every) != 0) {
        return;
    }
    std::fprintf(
        stderr,
        "GyroPulse: attempts=%lld qubec_calls=%lld dense_calls=%lld scanned_blocks=%lld reg=%d\n",
        (long long) attempt_no,
        (long long) g_qubec_calls.load(),
        (long long) g_dense_calls.load(),
        (long long) g_dispatch_scanned_blocks.load(),
        gyrolabe_registry_entry_count());
    std::fflush(stderr);
}

static void gyroscopic_emit_full_trace_footer(void) {
    if (!ggml_gyroscopic_trace()) {
        return;
    }
    double mn = 0.0;
    double mx = 0.0;
    double mean = 0.0;
    int cells = 0;
    ggml_gyroscopic_graph_m2_stats(&mn, &mx, &mean, &cells);
    std::fprintf(
        stderr,
        "GyroMatMul stats: qubec_calls=%lld radial_calls=%lld chi_calls=%lld chi_gauge_calls=%lld dense_calls=%lld q8dot_override_calls=%lld\n",
        (long long) g_qubec_calls.load(),
        (long long) g_radial_calls.load(),
        (long long) g_chi_calls.load(),
        (long long) g_chi_gauge_calls.load(),
        (long long) g_dense_calls.load(),
        (long long) g_q8dot_override_calls.load()
    );
    std::fprintf(
        stderr,
        "GyroRows: structured_rows=%lld dense_rows=%lld pq_rows=%lld dq_rows=%lld direct_exact_rows=%lld structured_attempt_rows=%lld exact_witness_rows=%lld witness_sampled_rows=%lld parity_mismatch_rows=%lld max_abs_row_error=%.9g shell_radial_rows=%lld shell_gauge_rows=%lld chi_invariant_rows=%lld chi_gauge_rows=%lld generic_rows=%lld pq_est_read_bytes=%lld dq_est_read_bytes=%lld dense_est_read_bytes=%lld\n",
        (long long) g_structured_rows.load(),
        (long long) g_dense_rows.load(),
        (long long) g_pq_rows.load(),
        (long long) g_dq_rows.load(),
        (long long) g_direct_exact_rows.load(),
        (long long) g_structured_attempt_rows.load(),
        (long long) g_exact_witness_rows.load(),
        (long long) g_witness_sampled_rows.load(),
        (long long) g_parity_mismatch_rows.load(),
        (double) g_max_abs_row_error.load(),
        (long long) g_shell_radial_rows.load(),
        (long long) g_shell_gauge_rows.load(),
        (long long) g_chi_invariant_rows.load(),
        (long long) g_chi_gauge_rows.load(),
        (long long) g_generic_rows.load(),
        (long long) g_pq_est_read_bytes.load(),
        (long long) g_dq_est_read_bytes.load(),
        (long long) g_dense_est_read_bytes.load()
    );
    std::fprintf(
        stderr,
        "GyroRowsFidelity: atlas_defect_error_sum=%.9g atlas_defect_error_max=%.9g atlas_defect_error_samples=%lld witness_err_sum=%.9g witness_err_sq_sum=%.9g witness_err_samples=%lld\n",
        (double) g_atlas_defect_error_sum.load(),
        (double) g_atlas_defect_error_max.load(),
        (long long) g_atlas_defect_error_samples.load(),
        (double) g_witness_error_sum.load(),
        (double) g_witness_error_sq_sum.load(),
        (long long) g_witness_error_samples.load()
    );
    std::fprintf(
        stderr,
        "GyroDispatch: attempts=%lld no_structured_fallback=%lld policy_residual_skipped=%lld kernel_error_fallback=%lld scanned_blocks=%lld no_k64_blocks=%lld dispatch_entries=%d chi_calls=%lld chi_gauge_calls=%lld shell_radial_blocks=%lld shell_gauge_blocks=%lld chi_invariant_blocks=%lld chi_gauge_blocks=%lld generic_blocks=%lld index_query_hits=%lld index_query_misses=%lld\n",
        (long long) g_qubec_attempts.load(),
        (long long) g_dispatch_no_structured_fallback.load(),
        (long long) g_dispatch_policy_residual_skipped.load(),
        0ll,
        (long long) g_dispatch_scanned_blocks.load(),
        (long long) g_dispatch_no_k64_blocks.load(),
        gyrolabe_registry_entry_count(),
        (long long) g_chi_calls.load(),
        (long long) g_chi_gauge_calls.load(),
        (long long) g_shell_radial_blocks.load(),
        (long long) g_shell_gauge_blocks.load(),
        (long long) g_chi_invariant_blocks.load(),
        (long long) g_chi_gauge_blocks.load(),
        (long long) g_generic_blocks.load(),
        (long long) g_dispatch_signature_index_hits.load(),
        (long long) g_dispatch_signature_index_misses.load()
    );
    std::fprintf(
        stderr,
        "GyroCert: certified_shell_radial_blocks=%lld certified_shell_gauge_blocks=%lld certified_chi_invariant_blocks=%lld certified_chi_gauge_blocks=%lld certified_generic_blocks=%lld\n",
        (long long) g_certified_shell_radial_blocks.load(),
        (long long) g_certified_shell_gauge_blocks.load(),
        (long long) g_certified_chi_invariant_blocks.load(),
        (long long) g_certified_chi_gauge_blocks.load(),
        (long long) g_certified_generic_blocks.load()
    );
    std::fprintf(
        stderr,
        "GyroPlan: selected_shell_radial_blocks=%lld selected_shell_gauge_blocks=%lld selected_chi_invariant_blocks=%lld selected_chi_gauge_blocks=%lld selected_generic_blocks=%lld\n",
        (long long) g_selected_shell_radial_blocks.load(),
        (long long) g_selected_shell_gauge_blocks.load(),
        (long long) g_selected_chi_invariant_blocks.load(),
        (long long) g_selected_chi_gauge_blocks.load(),
        (long long) g_selected_generic_blocks.load()
    );
    std::fprintf(
        stderr,
        "GyroExec: residual_none_blocks=%lld residual_k4_blocks=%lld residual_exact_q8_blocks=%lld residual_backfill_debug_blocks=%lld compiled_k4_residual_blocks=%lld exact_q8_generic_blocks=%lld direct_exact_generic_blocks=%lld chi_gauge_dq_kernel_calls=%lld chi_gauge_backfill_calls=%lld\n",
        (long long) g_residual_none_blocks.load(),
        (long long) g_residual_k4_blocks.load(),
        (long long) g_residual_exact_q8_blocks.load(),
        (long long) g_residual_backfill_debug_blocks.load(),
        (long long) g_compiled_k4_residual_blocks.load(),
        (long long) g_exact_q8_generic_blocks.load(),
        (long long) g_direct_exact_generic_blocks.load(),
        (long long) g_chi_gauge_dq_kernel_calls.load(),
        (long long) g_chi_gauge_backfill_calls.load()
    );
    std::fprintf(
        stderr,
        "GyroKernel: exec_shell_radial_blocks=%lld exec_shell_gauge_blocks=%lld exec_chi_invariant_blocks=%lld exec_chi_gauge_blocks=%lld exec_generic_blocks=%lld kernel_chi_gauge_k4_blocks=%lld kernel_chi_gauge_exact_q8_blocks=%lld kernel_generic_exact_q8_blocks=%lld kernel_projected_backfill_blocks=%lld\n",
        (long long) g_exec_shell_radial_blocks.load(),
        (long long) g_exec_shell_gauge_blocks.load(),
        (long long) g_exec_chi_invariant_blocks.load(),
        (long long) g_exec_chi_gauge_blocks.load(),
        (long long) g_exec_generic_blocks.load(),
        (long long) g_kernel_chi_gauge_k4_blocks.load(),
        (long long) g_kernel_chi_gauge_exact_q8_blocks.load(),
        (long long) g_kernel_generic_exact_q8_blocks.load(),
        (long long) g_kernel_projected_backfill_blocks.load()
    );
    std::fprintf(
        stderr,
        "GyroDirect: calls=%lld rows=%lld generic_exact_blocks=%lld\n",
        (long long) g_direct_exact_calls.load(),
        (long long) g_direct_exact_rows.load(),
        (long long) g_direct_exact_generic_blocks.load()
    );
    std::fprintf(
        stderr,
        "GyroResidual: mode=%s scr_min=GGML_GYROSCOPIC_SCR_MIN res_frac_max=GGML_GYROSCOPIC_RES_FRAC_MAX\n",
        gyrolabe_qubec_residual_mode_label()
    );
    {
        const gyro_policy * policy = gyro_policy_get();
        if (policy != nullptr) {
            std::fprintf(
                stderr,
                "GyroPolicy: interop_mode=%s decode_max_batch_size=%d decode_max_chi_distance=%d decode_max_shell_delta=%d kv_eviction_threshold=%.6f\n",
                config_interop_mode_label(),
                policy->decode_max_batch_size,
                policy->decode_max_chirality_distance,
                policy->decode_max_shell_delta,
                policy->kv_eviction_threshold
            );
        }
    }
    {
        const long long decode_events = g_decode_events.load();
        const long long grouped = g_decode_grouped_dispatch.load();
        const long long ungrouped = g_decode_ungrouped_dispatch.load();
        const long long kv_evict = g_decode_kv_evict.load();
        const double mean_chi_distance = decode_events > 0
            ? (double) g_decode_chi_distance_sum.load() / (double) decode_events
            : 0.0;
        const double mean_kv_priority = decode_events > 0
            ? (double) g_decode_kv_priority_sum.load() / (double) decode_events
            : 0.0;
        std::fprintf(
            stderr,
            "GyroDecodeStats: events=%lld grouped_dispatch=%lld ungrouped_dispatch=%lld kv_evict=%lld mean_chi_distance=%.6f max_chi_distance=%d mean_kv_priority=%.6f\n",
            decode_events,
            grouped,
            ungrouped,
            kv_evict,
            mean_chi_distance,
            g_decode_max_chi_distance.load(),
            mean_kv_priority
        );
    }
    std::fprintf(
        stderr,
        "GyroGraph: m2_min=%.6f m2_max=%.6f m2_mean=%.6f cells=%d\n",
        mn,
        mx,
        mean,
        cells
    );
    std::fflush(stderr);
}

struct GyroInitPrinter {
    GyroInitPrinter() {
        if (!ggml_gyroscopic_active()) {
            return;
        }
        std::fprintf(
            stderr,
            "GyroMatMul: mode=gyroscopic active=%d trace=%d strict=%d interop=%s residual=%s\n",
            (int) ggml_gyroscopic_active(),
            (int) ggml_gyroscopic_trace(),
            (int) ggml_gyroscopic_strict(),
            config_interop_mode_label(),
            gyrolabe_qubec_residual_mode_label()
        );
        std::fflush(stderr);
    }
};

static GyroInitPrinter g_gyro_init_printer;

static void gyroscopic_shutdown_pure_check(void) {
    if (!ggml_gyroscopic_active()) {
        return;
    }
    if (!g_pure_spectral_mode) {
        return;
    }
    if (g_structured_rows.load() != 0) {
        return;
    }
    std::fprintf(
        stderr,
        "GyroMatMul ERROR: GGML_GYROSCOPIC_PURE set but structured_rows=0 (no spectral QuBEC rows credited).\n"
    );
    std::fflush(stderr);
}

struct GyroTracePrinter {
    ~GyroTracePrinter() {
        gyroscopic_emit_full_trace_footer();
        gyroscopic_shutdown_pure_check();
    }
};

static GyroTracePrinter g_gyro_trace_printer;
