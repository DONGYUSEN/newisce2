// No-op pyre::journal shim for ISCE3 backprojection subset migrated to ISCE2.
// Replaces the full pyre framework dependency with silent stubs.

#pragma once

#include <string>

#ifndef __HERE__
#define __HERE__ __FILE__, __LINE__, __FUNCTION__
#endif

// ISCE_SRCINFO is defined in isce3/except/Error.h — do not duplicate here.

namespace pyre {
namespace journal {

struct _at_t {
    const char* file;
    int line;
    const char* func;
};

inline _at_t at(const char* file, int line, const char* func) {
    return {file, line, func};
}

struct _endl_t {};
struct _newline_t {};

static const _endl_t endl {};
static const _newline_t newline {};

class channel_t {
public:
    explicit channel_t(const std::string&) {}

    template <typename T>
    channel_t& operator<<(const T&) { return *this; }

    channel_t& operator<<(const _at_t&) { return *this; }
    channel_t& operator<<(const _endl_t&) { return *this; }
    channel_t& operator<<(const _newline_t&) { return *this; }
};

using error_t = channel_t;
using warning_t = channel_t;
using info_t = channel_t;
using debug_t = channel_t;
using firewall_t = channel_t;

}}
