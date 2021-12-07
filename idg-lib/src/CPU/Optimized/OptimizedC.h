// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/** @file */

/**
 * @brief Create a idg::proxy::cpu::Optimized instance
 *
 * After using the proxy
 * (see @verbatim embed:rst:inline :doc:`ProxyC.h` @endverbatim)
 * it needs to be destroyed by a call to Proxy_destroy()
 *
 * @return A pointer to a Proxy object
 */
struct Proxy* CPU_Optimized_create();
