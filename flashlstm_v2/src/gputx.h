#pragma once

#if !defined(GPUTX_DISABLE_NVTX)
#if defined(__has_include)
#if __has_include(<nvToolsExt.h>)
#define GPUTX_HAS_NVTX 1
#include <nvToolsExt.h>
#endif
#endif
#endif

#ifndef GPUTX_HAS_NVTX
#define GPUTX_HAS_NVTX 0
#endif

namespace gputx {

#if GPUTX_HAS_NVTX
inline void RangePush(const char *name) {
    nvtxRangePushA(name);
}

inline void RangePop() {
    nvtxRangePop();
}

class Range {
public:
    explicit Range(const char *name) {
        RangePush(name);
    }
    Range(const Range &) = delete;
    Range &operator=(const Range &) = delete;
    Range(Range &&) = delete;
    Range &operator=(Range &&) = delete;
    ~Range() {
        RangePop();
    }
};
#else
inline void RangePush(const char *) {}
inline void RangePop() {}

class Range {
public:
    explicit Range(const char *) {}
    Range(const Range &) = delete;
    Range &operator=(const Range &) = delete;
    Range(Range &&) = delete;
    Range &operator=(Range &&) = delete;
    ~Range() = default;
};
#endif

}  // namespace gputx

#define GPUTX_CONCAT_INNER(x, y) x##y
#define GPUTX_CONCAT(x, y) GPUTX_CONCAT_INNER(x, y)

#define GPUTX_RANGE(name_literal) ::gputx::Range GPUTX_CONCAT(_gputx_range_, __COUNTER__){name_literal}
