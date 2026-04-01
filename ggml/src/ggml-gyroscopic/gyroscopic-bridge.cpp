#include "gyroscopic-bridge.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "core.h"
}

static bool env_flag(const char * name, bool defv) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return defv;
    }
    return !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
}

bool ggml_gyroscopic_active(void) {
    static const bool v = env_flag("GGML_GYROSCOPIC_MATMUL", false);
    return v;
}

bool ggml_gyroscopic_strict(void) {
    static const bool v = env_flag("GGML_GYROSCOPIC_STRICT", true);
    return v;
}

bool ggml_gyroscopic_trace(void) {
    static const bool v = env_flag("GGML_GYROSCOPIC_TRACE", false);
    return v;
}

const char * ggml_gyroscopic_kernel_mode_cstr(void) {
    const char * v = std::getenv("GGML_GYROSCOPIC_KERNEL");
    return (v && v[0] != '\0') ? v : "avx2";
}

static std::atomic<long long> g_calls_f32{0};
static std::atomic<long long> g_calls_q80{0};
static std::atomic<long long> g_calls_gemm_q8{0};

struct GyroInitPrinter {
    GyroInitPrinter() {
        if (!ggml_gyroscopic_trace()) {
            return;
        }
        std::fprintf(
            stderr,
            "GyroMatMul: compiled=1 active=%d strict=%d trace=%d kernel=%s\n",
            (int) ggml_gyroscopic_active(),
            (int) ggml_gyroscopic_strict(),
            (int) ggml_gyroscopic_trace(),
            ggml_gyroscopic_kernel_mode_cstr()
        );
        std::fflush(stderr);
    }
};

static GyroInitPrinter g_gyro_init_printer;

struct GyroTracePrinter {
    ~GyroTracePrinter() {
        if (!ggml_gyroscopic_trace()) {
            return;
        }
        std::fprintf(
            stderr,
            "GyroMatMul trace: vec_dot_f32=%lld vec_dot_q8_0_q8_0=%lld gemm_q8_0=%lld\n",
            (long long) g_calls_f32.load(),
            (long long) g_calls_q80.load(),
            (long long) g_calls_gemm_q8.load()
        );
        std::fflush(stderr);
    }
};

static GyroTracePrinter g_trace_printer;

void ggml_gyroscopic_note_call(
    const char * /*tag*/,
    int64_t /*m*/,
    int64_t /*n*/,
    int64_t /*k*/
) {
}

void ggml_gyroscopic_abort_unsupported(
    const char * site,
    int32_t type0,
    int32_t type1
) {
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

bool ggml_gyroscopic_vec_dot_f32(
    int n,
    const float * x,
    const float * y,
    float * out
) {
    if (!ggml_gyroscopic_active()) {
        return false;
    }
    if (gyromatmul_vec_dot_f32(n, x, y, out) != 0) {
        return false;
    }
    g_calls_f32.fetch_add(1);
    return true;
}

bool ggml_gyroscopic_vec_dot_q8_0_q8_0(
    int n,
    const void * x,
    const void * y,
    float * out
) {
    if (!ggml_gyroscopic_active()) {
        return false;
    }
    const int qk = 32;
    if ((n % qk) != 0) {
        return false;
    }
    const int n_blocks = n / qk;
    if (gyromatmul_vec_dot_q8_0_q8_0(
            n_blocks,
            (const gyromatmul_block_q8_0 *) x,
            (const gyromatmul_block_q8_0 *) y,
            out) != 0) {
        return false;
    }
    g_calls_q80.fetch_add(1);
    return true;
}

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
) {
    if (!ggml_gyroscopic_active()) {
        return false;
    }
    if (gyromatmul_gemm_f32(m, n, k, a, lda, b, ldb, c, ldc) != 0) {
        return false;
    }
    ggml_gyroscopic_note_call("GEMM/F32", m, n, k);
    return true;
}

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
) {
    if (!ggml_gyroscopic_active()) {
        return false;
    }
    if ((k % 32) != 0) {
        return false;
    }
    const int lda_blocks = lda_bytes / (int) sizeof(gyromatmul_block_q8_0);
    const int ldb_blocks = ldb_bytes / (int) sizeof(gyromatmul_block_q8_0);
    if (lda_blocks <= 0 || ldb_blocks <= 0) {
        return false;
    }
    if (gyromatmul_gemm_q8_0_q8_0(
            m,
            n,
            k,
            (const gyromatmul_block_q8_0 *) a,
            lda_blocks,
            (const gyromatmul_block_q8_0 *) b,
            ldb_blocks,
            c,
            ldc) != 0) {
        return false;
    }
    g_calls_gemm_q8.fetch_add(1);
    ggml_gyroscopic_note_call("GEMM/Q8_0", m, n, k);
    return true;
}

bool ggml_gyroscopic_out_prod_f32(
    int rows,
    int cols,
    const float * x,
    const float * y,
    float * out,
    int ld_out
) {
    if (!ggml_gyroscopic_active()) {
        return false;
    }
    if (gyromatmul_out_prod_f32(rows, cols, x, y, out, ld_out) != 0) {
        return false;
    }
    ggml_gyroscopic_note_call("OUT_PROD/F32", rows, cols, 1);
    return true;
}
