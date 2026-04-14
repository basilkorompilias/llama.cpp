#include "gyroscopic-backend.h"

#include "ggml-cpu-impl.h"
#include "ggml-backend.h"
#include "ggml-gyroscopic-graph.h"

#include "ggml-impl.h"
#include "gyrograph.h"
#include "ggml.h"

#include "gyrolabe.h"
#include "gyrolabe_qubec_matmul.h"
#include "gyrolabe_registry.h"
#include "gyrograph_policy.h"

#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

static bool gyroscopic_env_pure_spectral(void) {
    const char * v = std::getenv("GGML_GYROSCOPIC_PURE");
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

namespace {

bool config_gyroscopic_mode_active(void) {
    return gyro_policy_get()->mode == GYRO_MODE_GYROSCOPIC;
}

bool config_strict(void) {
    return gyro_policy_get()->strict != 0;
}

bool config_trace(void) {
    return gyro_policy_get()->trace != 0;
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
    if (!ggml_gyroscopic_active() || src0 == nullptr || src1 == nullptr || src1_batch == nullptr) {
        return false;
    }

    if (src0->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if (src0->buffer == nullptr || src0->data == nullptr) {
        return false;
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        return false;
    }
    if ((k % 32) != 0) {
        return false;
    }
    if (src1->data == nullptr) {
        return false;
    }
    if ((size_t) lda_bytes % sizeof(gyromatmul_block_q8_0) != 0 || (size_t) ldb_bytes % sizeof(gyromatmul_block_q8_0) != 0) {
        return false;
    }
    const size_t lda_blocks = (size_t) lda_bytes / sizeof(gyromatmul_block_q8_0);
    const size_t ldb_blocks = (size_t) ldb_bytes / sizeof(gyromatmul_block_q8_0);
    if (lda_blocks == 0 || ldb_blocks == 0) {
        return false;
    }
    if (src1->type == GGML_TYPE_Q8_0) {
        if (src1->buffer == nullptr) {
            return false;
        }
        const ptrdiff_t start = (const char *) src1_batch - (const char *) src1->data;
        if (start < 0) {
            return false;
        }
        const size_t need = (size_t) start + (size_t) n * (size_t) ldb_bytes;
        if (need > ggml_nbytes(src1)) {
            return false;
        }
    }

    return true;
}

} // namespace

static std::atomic<long long> g_qubec_calls{0};
static std::atomic<long long> g_qubec_attempts{0};
static std::atomic<long long> g_radial_calls{0};
static std::atomic<long long> g_chi_calls{0};
static std::atomic<long long> g_chi_gauge_calls{0};
static std::atomic<long long> g_dense_calls{0};
static std::atomic<long long> g_dispatch_scanned_blocks{0};
static std::atomic<long long> g_dispatch_no_k64_blocks{0};
static std::atomic<int> g_gyro_bind_trace_printed{0};
static std::atomic<int> g_gyro_dims_trace_printed{0};
static std::atomic<long long> g_structured_rows{0};
static std::atomic<long long> g_dense_rows{0};
static std::atomic<long long> g_structured_attempt_rows{0};
static std::atomic<long long> g_exact_witness_rows{0};
static std::atomic<long long> g_parity_mismatch_rows{0};
static std::atomic<double> g_max_abs_row_error{0.0};
static std::atomic<long long> g_q8dot_override_calls{0};

static std::mutex g_gyrolabe_registry_q8_mutex;

static void gyroscopic_emit_trace_pulse(long long attempt_no);

extern "C" {

bool ggml_gyroscopic_active(void) {
    return config_gyroscopic_mode_active();
}

bool ggml_gyroscopic_strict(void) {
    return config_strict();
}

bool ggml_gyroscopic_trace(void) {
    return config_trace();
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
    return gyroscopic_mul_mat_geometry_ok(src0, src1, src1_batch, m, n, k, lda_bytes, ldb_bytes);
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
) {
    if (!ggml_gyroscopic_active() || src0 == nullptr || src1 == nullptr || a == nullptr || wdata == nullptr || c == nullptr) {
        return false;
    }
    if (!gyroscopic_mul_mat_geometry_ok(src0, src1, wdata, m, n, k, (int) src0->nb[1], (int) row_size)) {
        g_dense_calls.fetch_add(1);
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if (m <= 0 || n <= 0 || k <= 0 || ldc < m) {
        return false;
    }
    if (row_size == 0 || (row_size % sizeof(gyromatmul_block_q8_0)) != 0) {
        return false;
    }
    if (k < 32 || (k % 32) != 0) {
        g_dense_calls.fetch_add(1);
        return false;
    }
    if (row_size > (size_t) INT_MAX) {
        g_dense_calls.fetch_add(1);
        return false;
    }

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

    /* Registry keys blocks by weight tensor pointer; w_tensor is src0 (weights, Q8_0).
     * src1 is the activation tensor (F32). */
    const struct ggml_tensor * w_tensor_runtime =
        (w_tensor != nullptr) ? w_tensor : src0;

    const void * reg_base = (w_registry_base != nullptr) ? w_registry_base : a;

    /* Hot path: skip the mutex when this weight buffer is already registered so
     * parallel graph workers do not serialize on every matmul. */
    if (gyrolabe_registry_find_entry(w_tensor_runtime, reg_base) == nullptr) {
        std::lock_guard<std::mutex> lock(g_gyrolabe_registry_q8_mutex);
        if (gyrolabe_registry_find_entry(w_tensor_runtime, reg_base) == nullptr && src0 != nullptr) {
            gyrolabe_registry_register_q8_buffer(
                reg_base,
                (int64_t) src0->ne[0],
                (int64_t) src0->ne[1],
                (int64_t) src0->ne[2],
                (int64_t) src0->ne[3],
                (size_t) src0->nb[1],
                src0->name
            );
        }
    }

    const long long gyro_attempt_no = g_qubec_attempts.fetch_add(1) + 1;
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
        cell_idx
    );
    gyrolabe_qubec_dispatch_stats dispatch_stats = {0, 0, 0, 0};
    gyrolabe_qubec_get_last_dispatch_stats(&dispatch_stats);
    gyrolabe_qubec_call_stats call_stats = {};
    gyrolabe_qubec_get_last_call_stats(&call_stats);
    g_dispatch_scanned_blocks.fetch_add(dispatch_stats.scanned_blocks);
    g_dispatch_no_k64_blocks.fetch_add(dispatch_stats.no_k64_blocks);

    g_structured_rows.fetch_add((long long) call_stats.structured_rows);
    g_structured_attempt_rows.fetch_add((long long) call_stats.structured_attempt_rows);
    g_exact_witness_rows.fetch_add((long long) call_stats.exact_witness_rows);
    g_dense_rows.fetch_add((long long) call_stats.dense_rows);
    g_parity_mismatch_rows.fetch_add((long long) call_stats.parity_mismatch_rows);
    {
        const double mx = (double) call_stats.max_abs_row_error;
        double cur = g_max_abs_row_error.load();
        while (mx > cur) {
            if (g_max_abs_row_error.compare_exchange_weak(cur, mx)) {
                break;
            }
        }
    }

    if (rc != 0) {
        g_dense_calls.fetch_add(1);
        gyroscopic_emit_trace_pulse(gyro_attempt_no);
        return false;
    }

    if (call_stats.structured_attempt_rows > 0) {
        g_qubec_calls.fetch_add(1);
    } else {
        g_dense_calls.fetch_add(1);
    }
    if (call_stats.used_radial) {
        g_radial_calls.fetch_add(1);
    }
    if (call_stats.used_chi) {
        g_chi_calls.fetch_add(1);
    }
    if (call_stats.used_chi_gauge) {
        g_chi_gauge_calls.fetch_add(1);
    }
    gyroscopic_emit_trace_pulse(gyro_attempt_no);
    return true;
}

int ggml_gyroscopic_graph_cell_idx(int64_t i12) {
    if (i12 < 0) {
        return 0;
    }
    if (i12 > 255) {
        return 255;
    }
    return (int) i12;
}

} // extern "C"

static int gyro_trace_snapshot_interval(void) {
    static int s_cached = -2;
    if (s_cached != -2) {
        return s_cached;
    }
    s_cached = 0;
    const char * e = std::getenv("GGML_GYROSCOPIC_TRACE_SNAPSHOT_EVERY");
    if (e == nullptr || e[0] == '\0') {
        return 0;
    }
    char * end_ptr = nullptr;
    const long n = std::strtol(e, &end_ptr, 10);
    (void) end_ptr;
    if (n < 1) {
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
        "GyroRows: structured_rows=%lld dense_rows=%lld structured_attempt_rows=%lld exact_witness_rows=%lld parity_mismatch_rows=%lld max_abs_row_error=%.9g\n",
        (long long) g_structured_rows.load(),
        (long long) g_dense_rows.load(),
        (long long) g_structured_attempt_rows.load(),
        (long long) g_exact_witness_rows.load(),
        (long long) g_parity_mismatch_rows.load(),
        (double) g_max_abs_row_error.load()
    );
    std::fprintf(
        stderr,
        "GyroDispatch: attempts=%lld no_structured_fallback=%lld kernel_error_fallback=%lld scanned_blocks=%lld no_k64_blocks=%lld dispatch_entries=%d\n",
        (long long) g_qubec_attempts.load(),
        0ll,
        0ll,
        (long long) g_dispatch_scanned_blocks.load(),
        (long long) g_dispatch_no_k64_blocks.load(),
        gyrolabe_registry_entry_count()
    );
    std::fprintf(
        stderr,
        "GyroResidual: mode=%s scr_min=GGML_GYROSCOPIC_SCR_MIN res_frac_max=GGML_GYROSCOPIC_RES_FRAC_MAX\n",
        gyrolabe_qubec_residual_mode_label()
    );
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
            "GyroMatMul: mode=gyroscopic active=%d trace=%d strict=%d residual=%s\n",
            (int) ggml_gyroscopic_active(),
            (int) ggml_gyroscopic_trace(),
            (int) ggml_gyroscopic_strict(),
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
    if (!gyroscopic_env_pure_spectral()) {
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
    std::abort();
}

struct GyroTracePrinter {
    ~GyroTracePrinter() {
        gyroscopic_emit_full_trace_footer();
        gyroscopic_shutdown_pure_check();
    }
};

static GyroTracePrinter g_gyro_trace_printer;
