inline void compute_phase(
    const int nr_timesteps,
    const int nr_channels,
    const int x,
    const int y,
    const float u_offset,
    const float v_offset,
    const float w_offset,
    const float uvw[nr_timesteps][3],
    const float wavenumbers[nr_channels],
    float phase[nr_timesteps][nr_channels]
) {
    // Compute l,m,n
    const float l = (x-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
    const float m = (y-(SUBGRIDSIZE/2)) * IMAGESIZE/SUBGRIDSIZE;
    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
    // accurately for small values of l and m
    const float tmp = (l * l) + (m * m);
    const float n = tmp / (1.0f + sqrtf(1.0f - tmp));

    for (int time = 0; time < nr_timesteps; time++) {
        // Load UVW coordinates
        float u = uvw[time][0];
        float v = uvw[time][1];
        float w = uvw[time][2];

        // Compute phase index
        float phase_index = u*l + v*m + w*n;

        // Compute phase offset
        float phase_offset = u_offset*l + v_offset*m + w_offset*n;

        for (int chan = 0; chan < nr_channels; chan++) {
            // Compute phase
            float wavenumber = wavenumbers[chan];
            phase[time][chan]  = (phase_index * wavenumber) - phase_offset;
        }
    } // end time
}

inline void compute_sincos(
    const int n,
    const float *x,
    float *sin,
    float *cos
) {
    #if defined(USE_VML)
    vmsSinCos(n, x, sin, cos, VML_PRECISION);
    #else
    for (int i = 0; i < n; i++) {
            sin[i] = sinf(phase[i]);
            cos[i] = cosf(phase[i]);
        }
    }
    #endif
}

inline void cmul_reduce(
    const int nr_timesteps,
    const int nr_channels,
    const float a_real[nr_timesteps][NR_POLARIZATIONS][nr_channels],
    const float a_imag[nr_timesteps][NR_POLARIZATIONS][nr_channels],
    const float b_real[nr_timesteps][nr_channels],
    const float b_imag[nr_timesteps][nr_channels],
    FLOAT_COMPLEX *c
) {
    // Initialize pixel for every polarization
    float c_xx_real = 0.0f;
    float c_xy_real = 0.0f;
    float c_yx_real = 0.0f;
    float c_yy_real = 0.0f;
    float c_xx_imag = 0.0f;
    float c_xy_imag = 0.0f;
    float c_yx_imag = 0.0f;
    float c_yy_imag = 0.0f;

    // Update pixel for every timestep
    for (int time = 0; time < nr_timesteps; time++) {
        // Update pixel for every channel
        #pragma omp simd reduction(+:c_xx_real,c_xx_imag,\
                                     c_xy_real,c_xy_imag,\
                                     c_yx_real,c_yx_imag,\
                                     c_yy_real,c_yy_imag)
        for (int chan = 0; chan < nr_channels; chan++) {
            // Update pixels
            c_xx_real +=  a_real[time][0][chan] * b_real[time][chan];
            c_xx_imag +=  a_real[time][0][chan] * b_imag[time][chan];
            c_xx_real += -a_imag[time][0][chan] * b_imag[time][chan];
            c_xx_imag +=  a_imag[time][0][chan] * b_real[time][chan];

            c_xy_real +=  a_real[time][1][chan] * b_real[time][chan];
            c_xy_imag +=  a_real[time][1][chan] * b_imag[time][chan];
            c_xy_real += -a_imag[time][1][chan] * b_imag[time][chan];
            c_xy_imag +=  a_imag[time][1][chan] * b_real[time][chan];

            // #pragma distribute_point

            c_yx_real +=  a_real[time][2][chan] * b_real[time][chan];
            c_yx_imag +=  a_real[time][2][chan] * b_imag[time][chan];
            c_yx_real += -a_imag[time][2][chan] * b_imag[time][chan];
            c_yx_imag +=  a_imag[time][2][chan] * b_real[time][chan];

            c_yy_real +=  a_real[time][3][chan] * b_real[time][chan];
            c_yy_imag +=  a_real[time][3][chan] * b_imag[time][chan];
            c_yy_real += -a_imag[time][3][chan] * b_imag[time][chan];
            c_yy_imag +=  a_imag[time][3][chan] * b_real[time][chan];
        }
    }

    // Combine real and imaginary parts
    c[0] = FLOAT_COMPLEX(c_xx_real, c_xx_imag);
    c[1] = FLOAT_COMPLEX(c_xy_real, c_xy_imag);
    c[2] = FLOAT_COMPLEX(c_yx_real, c_yx_imag);
    c[3] = FLOAT_COMPLEX(c_yy_real, c_yy_imag);
}

inline void apply_aterm(
    const FLOAT_COMPLEX aXX1, const FLOAT_COMPLEX aXY1,
    const FLOAT_COMPLEX aYX1, const FLOAT_COMPLEX aYY1,
    const FLOAT_COMPLEX aXX2, const FLOAT_COMPLEX aXY2,
    const FLOAT_COMPLEX aYX2, const FLOAT_COMPLEX aYY2,
    FLOAT_COMPLEX pixels[NR_POLARIZATIONS]
) {
    // Apply aterm to subgrid: P*A1
    // [ pixels[0], pixels[1];    [ aXX1, aXY1;
    //   pixels[2], pixels[3] ] *   aYX1, aYY1 ]
    FLOAT_COMPLEX pixelsXX = pixels[0];
    FLOAT_COMPLEX pixelsXY = pixels[1];
    FLOAT_COMPLEX pixelsYX = pixels[2];
    FLOAT_COMPLEX pixelsYY = pixels[3];
    pixels[0]  = (pixelsXX * aXX1);
    pixels[0] += (pixelsXY * aYX1);
    pixels[1]  = (pixelsXX * aXY1);
    pixels[1] += (pixelsXY * aYY1);
    pixels[2]  = (pixelsYX * aXX1);
    pixels[2] += (pixelsYY * aYX1);
    pixels[3]  = (pixelsYX * aXY1);
    pixels[3] += (pixelsYY * aYY1);

    // Apply aterm to subgrid: A2^H*P
    // [ aXX2, aYX1;      [ pixels[0], pixels[1];
    //   aXY1, aYY2 ]  *    pixels[2], pixels[3] ]
    pixelsXX = pixels[0];
    pixelsXY = pixels[1];
    pixelsYX = pixels[2];
    pixelsYY = pixels[3];
    pixels[0]  = (pixelsXX * aXX2);
    pixels[0] += (pixelsYX * aYX2);
    pixels[1]  = (pixelsXY * aXX2);
    pixels[1] += (pixelsYY * aYX2);
    pixels[2]  = (pixelsXX * aXY2);
    pixels[2] += (pixelsYX * aYY2);
    pixels[3]  = (pixelsXY * aXY2);
    pixels[3] += (pixelsYY * aYY2);
}
