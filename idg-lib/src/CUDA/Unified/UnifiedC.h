// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * @brief Create a idg::proxy::cuda::Unified instance
 *
 * After using the proxy
 * (see @verbatim embed:rst:inline :doc:`ProxyC.h` @endverbatim)
 * it needs to be destroyed by a call to Proxy_destroy()
 *
 * @return A pointer to a Proxy object
 */
struct Proxy* CUDA_Unified_create();
