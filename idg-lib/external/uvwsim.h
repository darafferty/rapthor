/*
 * Copyright (c) 2014, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the uvwsim library.
 * Contact: uvwsim at oerc.ox.ac.uk
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef UVWSIM_H_
#define UVWSIM_H_

#define UVWSIM_VERSION 0.9
#define UVWSIM_VERSION_MAJOR 0
#define UVWSIM_VERSION_MINOR 9

/* Platform recognition */
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
#    define UVWSIM_OS_WIN32
#endif
#if (defined(WIN64) || defined(_WIN64) || defined(__WIN64__))
#    define UVWSIM_OS_WIN64
#endif
#if (defined(UVWSIM_OS_WIN32) || defined(UVWSIM_OS_WIN64))
#    define UVWSIM_OS_WIN
#endif

/* Declare import and export macros */
#ifndef UVWSIM_DECL_EXPORT
#   ifdef UVWSIM_OS_WIN
#       define UVWSIM_DECL_EXPORT __declspec(dllexport)
#   elif __GNUC__ >= 4
#       define UVWSIM_DECL_EXPORT __attribute__ ((visibility("default")))
#   else
#       define UVWSIM_DECL_EXPORT
#   endif
#endif
#ifndef UVWSIM_DECL_IMPORT
#   ifdef UVWSIM_OS_WIN
#       define UVWSIM_DECL_IMPORT __declspec(dllimport)
#   elif __GNUC__ >= 4
#       define UVWSIM_DECL_IMPORT __attribute__ ((visibility("default")))
#   else
#       define UVWSIM_DECL_IMPORT
#   endif
#endif

/* UVWSIM_API is used to identify public functions which should be exported */
#ifdef uvwsim_EXPORTS
#   define UVWSIM_API UVWSIM_DECL_EXPORT
#else
#   define UVWSIM_API UVWSIM_DECL_IMPORT
#endif

#include <stdio.h> /* Required for definition of size_t and FILE */

#ifdef __cplusplus
extern "C" {
#endif

/* Public API */
int UVWSIM_API uvwsim_file_exists(const char* filename);
int UVWSIM_API uvwsim_get_num_stations(const char* filename);
int UVWSIM_API uvwsim_load_station_coords(const char* filename, int n,
        double* x, double*y, double* z);
void UVWSIM_API uvwsim_convert_enu_to_ecef(int n, double* x_ecef,
        double* y_ecef, double* z_ecef, const double* x_enu,
        const double* y_enu, const double* z_enu, double lon, double lat,
        double alt);
int UVWSIM_API uvwsim_num_baselines(int n);
void UVWSIM_API uvwsim_evaluate_baseline_uvw(double* uu, double* vv,
        double* ww, int nant, const double* x_ecef, const double* y_ecef,
        const double* z_ecef, double ra0, double dec0, double time_mjd);
double UVWSIM_API uvwsim_datetime_to_mjd(int year, int month, int day, int hour,
        int minute, double seconds);

/* Private functions. */
int uvwsim_getline(char** lineptr, size_t* n, FILE* stream);
size_t uvwsim_string_to_array(char* str, size_t n, double* data);
double uvwsim_convert_mjd_to_gast_fast(double mjd);
double uvwsim_convert_mjd_to_gmst(double mjd);
double uvwsim_equation_of_equinoxes_fast(double mjd);

#ifdef __cplusplus
}
#endif

#endif /* UVWSIM_H_ */
