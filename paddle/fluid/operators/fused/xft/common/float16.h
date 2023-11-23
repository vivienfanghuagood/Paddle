// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include <immintrin.h>

#include "bit_cast.h"

class float16_t {
public:
    float16_t() = default;
    float16_t(float val) { (*this) = val; }
    constexpr float16_t(uint16_t bits, bool) : raw_(bits) {}

    float16_t &operator=(float val);

    float16_t &operator+=(float16_t a) {
        (*this) = float(f() + a.f());
        return *this;
    }

    operator float() const;

    static void cvt_float_to_float16(const float *src, float16_t *dst, int size);
    static void cvt_float16_to_float(const float16_t *src, float *dst, int size);
    static void float_add_float16(const float *src1, const float16_t *src2, float *dst, int size);

private:
    float f() { return (float)(*this); }

    uint16_t raw_;
};

static_assert(sizeof(float16_t) == 2, "float16_t must be 2 bytes");

inline float16_t &float16_t::operator=(float f) {
    uint32_t i = bit_cast<uint32_t>(f);
    uint32_t s = i >> 31;
    uint32_t e = (i >> 23) & 0xFF;
    uint32_t m = i & 0x7FFFFF;

    uint32_t ss = s;
    uint32_t mm = m >> 13;
    uint32_t r = m & 0x1FFF;
    uint32_t ee = 0;
    int32_t eee = (e - 127) + 15;

    if (0 == e) {
        ee = 0;
        mm = 0;
    } else if (0xFF == e) {
        ee = 0x1F;
        if (0 != m && 0 == mm) mm = 1;
    } else if (0 < eee && eee < 0x1F) {
        ee = eee;
        if (r > (0x1000 - (mm & 1))) {
            mm++;
            if (mm == 0x400) {
                mm = 0;
                ee++;
            }
        }
    } else if (0x1F <= eee) {
        ee = 0x1F;
        mm = 0;
    } else {
        float ff = fabsf(f) + 0.5;
        uint32_t ii = bit_cast<uint32_t>(ff);
        ee = 0;
        mm = ii & 0x7FF;
    }

    this->raw_ = (ss << 15) | (ee << 10) | mm;
    return *this;
}

inline float16_t::operator float() const {
    uint32_t ss = raw_ >> 15;
    uint32_t ee = (raw_ >> 10) & 0x1F;
    uint32_t mm = raw_ & 0x3FF;

    uint32_t s = ss;
    uint32_t eee = ee - 15 + 127;
    uint32_t m = mm << 13;
    uint32_t e;

    if (0 == ee) {
        if (0 == mm) {
            e = 0;
        } else {
            return (ss ? -1 : 1) * std::scalbn((float)mm, -24);
        }
    } else if (0x1F == ee) {
        e = 0xFF;
    } else {
        e = eee;
    }

    uint32_t f = (s << 31) | (e << 23) | m;

    return bit_cast<float>(f);
}
