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

#include "uvwsim.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#ifndef M_PI
#defined M_PI 3.14159265358979323846264338327950288
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DELIMITERS ", \t"

int uvwsim_get_num_stations(const char* filename)
{
    FILE* file = NULL;
    char* line = NULL;
    size_t bufsize = 0;
    int n = 0;
    file = fopen(filename, "r");
    if (!file) return -1;
    while (uvwsim_getline(&line, &bufsize, file) != EOF)
    {
        size_t read = 0;
        double coords[] = {0.0, 0.0, 0.0};
        if (line[0] == '#') continue;
        read = uvwsim_string_to_array(line, sizeof(coords)/sizeof(double), coords);
        if (read < 2) continue;
        ++n;
    }
    if (line) free(line);
    fclose(file);
    return n;
}

int uvwsim_load_station_coords(const char* filename, int nant,
        double* x, double*y, double* z)
{
    FILE* file = NULL;
    char* line = NULL;
    size_t bufsize = 0;
    int n = 0;

    if (!x || !y || !z || !filename || !*filename)
        return 0;

    file = fopen(filename, "r");
    if (!file) return -1;

    while (uvwsim_getline(&line, &bufsize, file) != EOF)
    {
        size_t read = 0;
        double coords[] = {0.0, 0.0, 0.0};

        /* Ignore comment lines (lines starting with '#') */
        if (line[0] == '#') continue;

        read = uvwsim_string_to_array(line, sizeof(coords)/sizeof(double), coords);

        /* Stop reading if a line has less than two coordinates specified */
        if (read < 2) break;

        x[n] = coords[0];
        y[n] = coords[1];
        z[n] = coords[2];

        /* Increment the coordinate counter */
        ++n;

        /* Stop if we have reached the number of antennas specified */
        if(n >= nant) break;
    }

    /* Free the line buffer and close the file */
    if (line) free(line);
    fclose(file);

    /* Return the number of co-ordinates read */
    return n;
}

void uvwsim_convert_enu_to_ecef(int nant, double* x_ecef, double* y_ecef,
        double* z_ecef, const double* x_enu, const double* y_enu,
        const double* z_enu, double lon, double lat, double alt)
{
    int i;

    /* pre-compute some trig. */
    double sinlon = sin(lon);
    double coslon = cos(lon);
    double sinlat = sin(lat);
    double coslat = cos(lat);

    /* Compute the ECEF coordinates of the reference point
     * by converting from geodetic spherical to ECEF coordinates */
    double x_r, y_r, z_r;
    const double a = 6378137.000; /* Equatorial radius (semi-major axis). */
    const double b = 6356752.314; /* Polar radius (semi-minor axis). */
    const double e2 = 1.0 - (b*b) / (a*a);
    double n_phi = a / sqrt(1.0 - e2*pow(sinlat,2.0));
    x_r = (n_phi+alt)*coslat*coslon;
    y_r = (n_phi+alt)*coslat*sinlon;
    z_r = ((1.0-e2)*n_phi+alt)*sinlat;

    for (i = 0; i < nant; ++i)
    {
        /* Copy input coordinates into temporary variables */
        double x = x_enu[i];
        double y = y_enu[i];
        double z = z_enu[i];

        /* Rotate from horizon (ENU) to offset ECEF frame */
        double xt, yt, zt;
        xt = -x * sinlon - y * sinlat * coslon + z * coslat * coslon;
        yt =  x * coslon - y * sinlat * sinlon + z * coslat * sinlon;
        zt =  y * coslat + z * sinlat;

        /* Translate from offset ECEF to ECEF by adding the coordinate of the
         * reference point */
        x_ecef[i] = xt + x_r;
        y_ecef[i] = yt + y_r;
        z_ecef[i] = zt + z_r;
    }
}

int uvwsim_num_baselines(int nant)
{
    return (nant*(nant-1))/2;
}

void uvwsim_evaluate_baseline_uvw(double* uu, double* vv,
        double* ww, int nant, const double* x_ecef, const double* y_ecef,
        const double* z_ecef, double ra0, double dec0, double time_mjd)
{
    int i, s1, s2, b;
    double gast = uvwsim_convert_mjd_to_gast_fast(time_mjd);
    double ha0  = gast - ra0;

    /* Allocate memory for station uvw */
    double* u = (double*)malloc(nant * sizeof(double));
    double* v = (double*)malloc(nant * sizeof(double));
    double* w = (double*)malloc(nant * sizeof(double));

    /* Convert to station uvw */
    double sinha0  = sin(ha0);
    double cosha0  = cos(ha0);
    double sindec0 = sin(dec0);
    double cosdec0 = cos(dec0);

    /* Evaluate baseline uvw */
    for (i = 0; i < nant; ++i)
    {
        double t = x_ecef[i] * cosha0 - y_ecef[i]*sinha0;
        u[i] = x_ecef[i]*sinha0 + y_ecef[i]*cosha0;
        v[i] = t*sindec0   + z_ecef[i]*cosdec0;
        w[i] = t*cosdec0   + z_ecef[i]*sindec0;
    }

    /* Convert from station uvw, to baseline uvw */
    for (s1 = 0, b = 0; s1 < nant; ++s1)
    {
        for (s2 = s1+1; s2 < nant; ++s2, ++b)
        {
            uu[b] = u[s2] - u[s1];
            vv[b] = v[s2] - v[s1];
            ww[b] = w[s2] - w[s1];
        }
    }

    /* cleanup */
    free(u);
    free(v);
    free(w);
}

double uvwsim_datetime_to_mjd(int year, int month, int day, int hour,
        int minute, double seconds)
{
    double day_fraction;
    int a, y, m, jdn;

    /* Compute Julian day Day Number (Note: all integer division) */
    a = (14 - month) / 12;
    y = year + 4800 - a;
    m = month + 12 * a - 3;
    jdn = day + (153*m+2)/5 + (365*y) + (y/4) - (y/100) + (y/400) - 32045;

    day_fraction = (hour + minute/60. + seconds/3600.) / 24.;
    return ((double)jdn + (day_fraction - 0.5) - 2400000.5);
}

/* http://goo.gl/LWf7Gz */
/* http://msdn.microsoft.com/en-us/library/14h5k7ff.aspx */
int UVWSIM_API uvwsim_file_exists(const char* filename)
{
#if defined(UVWSIM_OS_WIN)
    struct _stat buf;
    return (_stat(filename, &buf) == 0);
#else
    struct stat buf;
    return (stat(filename, &buf) == 0);
#endif
}

/* Private functions ------------------------------------------------------- */

int uvwsim_getline(char** lineptr, size_t* n, FILE* stream)
{
    const int ERROR_MEM_ALLOC_FAILURE = -4;

    /* Initialise the byte counter. */
    size_t size = 0;
    int c;

    /* Check if buffer is empty. */
    if (*n == 0 || *lineptr == 0)
    {
        *n = 80;
        *lineptr = (char*)malloc(*n);
        if (*lineptr == 0)
            return ERROR_MEM_ALLOC_FAILURE;
    }

    /* Read in the line. */
    for (;;)
    {
        /* Get the character. */
        c = getc(stream);

        /* Check if end-of-file or end-of-line has been reached. */
        if (c == EOF || c == '\n')
            break;

        /* Allocate space for size+1 bytes (including NULL terminator). */
        if (size + 1 >= *n)
        {
            void *t;

            /* Double the length of the buffer. */
            *n = 2 * *n + 1;
            t = realloc(*lineptr, *n);
            if (!t)
                return ERROR_MEM_ALLOC_FAILURE;
            *lineptr = (char*)t;
        }

        /* Store the character. */
        (*lineptr)[size++] = c;
    }

    /* Add a NULL terminator. */
    (*lineptr)[size] = '\0';

    /* Return the number of characters read, or EOF as appropriate. */
    if (c == EOF && size == 0)
        return EOF;
    return size;
}

size_t uvwsim_string_to_array(char* str, size_t n, double* data)
{
    size_t i = 0;
    char *save_ptr, *token;
    do
    {
        token = strtok_r(str, DELIMITERS, &save_ptr);
        /*
        if (token == NULL) {
            printf("INVALID DELIM!, '%s' %lu\n", str, strlen(str));
            break;
        }
        */
        str = NULL;
        if (!token) break;
        if (token[0] == '#') break;
        if (sscanf(token, "%lf", &data[i]) > 0) i++;
        /*printf("'%s'->[%li]%f | ", token, i, data[i-1]);*/
    }
    while (i < n);
    /*printf("\n");*/
    return i;
}

double uvwsim_convert_mjd_to_gmst(double mjd)
{
    double d, T, gmst;

    /* Days from J2000 */
    d = mjd - 51544.5;

    /* Centuries from J2000 */
    T = d / 36525.0;

    /* GMST at this time, in radians */
    gmst = 24110.54841 + 8640184.812866*T + 0.093104*T*T - 6.2e-6*T*T*T;
    gmst *= (15.0*M_PI)/(3600.*180.);
    gmst += fmod(mjd, 1.0) * 2.0 * M_PI;

    /* Range check (0, 2pi) */
    T = fmod(gmst, 2*M_PI);
    return (T >= 0.0) ? T : T + 2.0 * M_PI;
}

double uvwsim_equation_of_equinoxes_fast(double mjd)
{
    double d, omega, L, delta_psi, epsilon, eqeq;

    /* Days from J2000.0. */
    d = mjd - 51544.5;

    /* Longitude of ascending node of the Moon. */
    omega = (125.04 - 0.052954 * d) * (M_PI/180.0);

    /* Mean Longitude of the Sun. */
    L = (280.47 + 0.98565 * d) * (M_PI/180.0);

    /* eqeq = delta_psi * cos(epsilon). */
    delta_psi = -0.000319 * sin(omega) - 0.000024 * sin(2.0 * L);
    epsilon = (23.4393 - 0.0000004 * d) * (M_PI/180.0);

    /* Return equation of equinoxes, in radians. */
    eqeq = delta_psi * cos(epsilon) * ((15.0*M_PI)/180.0);
    return eqeq;
}

double uvwsim_convert_mjd_to_gast_fast(double mjd)
{
    double gmst = uvwsim_convert_mjd_to_gmst(mjd);
    double gast = gmst + uvwsim_equation_of_equinoxes_fast(mjd);
    return gast;
}

#ifdef __cplusplus
}
#endif
