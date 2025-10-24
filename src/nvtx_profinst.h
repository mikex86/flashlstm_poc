#pragma once

#if !defined(NVTX_DISABLE)
#if defined(__has_include)
#if __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#else
#define NVTX_DISABLE
#endif
#else
#include <nvToolsExt.h>
#endif
#endif  // !defined(NVTX_DISABLE)

#include <string>

namespace flashlstm::internal {
#ifndef NVTX_DISABLE
    class NvtxRangeGuard {
    public:
        explicit NvtxRangeGuard(const char *name) {
            if (name != nullptr) {
                nvtxRangePushA(name);
                active_ = true;
            }
        }

        explicit NvtxRangeGuard(const std::string &name)
            : NvtxRangeGuard(name.c_str()) {
        }

        NvtxRangeGuard(const NvtxRangeGuard &) = delete;

        NvtxRangeGuard &operator=(const NvtxRangeGuard &) = delete;

        ~NvtxRangeGuard() {
            if (active_) {
                nvtxRangePop();
            }
        }

    private:
        bool active_ = false;
    };
#endif  // NVTX_DISABLE
} // namespace flashlstm::internal

#define NVTX_INTERNAL_CONCAT_IMPL(a, b) a##b
#define NVTX_INTERNAL_CONCAT(a, b) NVTX_INTERNAL_CONCAT_IMPL(a, b)

#ifdef NVTX_DISABLE
#define NVTX_SCOPED_RANGE(name) do { (void)(name); } while (0)
#else
#define NVTX_SCOPED_RANGE(name) ::flashlstm::internal::NvtxRangeGuard NVTX_INTERNAL_CONCAT(_nvtx_scope_, __COUNTER__)(name)
#endif
