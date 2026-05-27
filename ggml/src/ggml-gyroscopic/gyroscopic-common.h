#ifndef GYROSCOPIC_COMMON_H
#define GYROSCOPIC_COMMON_H

#include <stdint.h>

#define GYROGRAPH_MAX_CELLS 256
#define GYROGRAPH_OMEGA_SIZE 4096.0
#define GYROGRAPH_SHELL_MAX_VARIANCE 9.0

#if defined(_MSC_VER)
#include <intrin.h>
static inline int gyroscopic_popcount32(uint32_t x) {
    return (int) __popcnt(x);
}
#else
static inline int gyroscopic_popcount32(uint32_t x) {
    return (int) __builtin_popcount(x);
}
#endif

static inline int gyroscopic_popcount6(uint32_t x) {
    return gyroscopic_popcount32(x & 0x3Fu);
}

static inline uint8_t gyroscopic_chi6_from_omega12(uint32_t omega12) {
    return (uint8_t) (((omega12 >> 6u) ^ omega12) & 0x3Fu);
}

static inline uint8_t gyroscopic_chi_from_omega12(uint32_t omega12) {
    return gyroscopic_chi6_from_omega12(omega12);
}

static inline uint32_t gyroscopic_omega12_to_state24(uint32_t omega12) {
    const uint8_t u6 = (uint8_t) ((omega12 >> 6u) & 0x3Fu);
    const uint8_t v6 = (uint8_t) (omega12 & 0x3Fu);
    uint16_t a12 = 0x0AAAu;
    uint16_t b12 = 0x0555u;
    for (uint8_t i = 0u; i < 6u; ++i) {
        const uint16_t pair = (uint16_t) (0x3u << (2u * i));
        if ((u6 & (uint8_t) (1u << i)) != 0u) {
            a12 ^= pair;
        }
        if ((v6 & (uint8_t) (1u << i)) != 0u) {
            b12 ^= pair;
        }
    }
    return ((uint32_t) a12 << 12u) | (uint32_t) b12;
}

#endif // GYROSCOPIC_COMMON_H
