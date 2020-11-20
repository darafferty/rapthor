// Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

extern "C" {

//Run kernel_fft with global dim = {128*BatchSize}, local dim={128}

inline __device__ float2 operator + (float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator - (float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float2 operator * (float a, float2 b)
{
    return make_float2(a * b.x, a * b.y);
}

#define USE_CUDA_PI 1
#undef M_PI
#if USE_CUDA_PI
#include "math_constants.h"
#define M_PI CUDART_PI_F
#else
#define M_PI 0x1.921fb54442d18p+1f
#endif

#define mad(A,B,C) ((A)*(B)+(C))
#define mad24(A,B,C) ((A)*(B)+(C))
#define complexMul(a,b) (make_float2(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y)))
#define conj(a) (make_float2((a).x, -(a).y))
#define conjTransp(a) (make_float2(-(a).y, (a).x))

inline __device__ void fftKernel2S(float2 d1, float2 d2, int dir) {
    float2 c = (d1);
    (d1) = c + (d2);
    (d2) = c - (d2);
}

inline __device__ void fftKernel4(float2 *a, int dir) {
    fftKernel2S((a)[0], (a)[2], dir);
    fftKernel2S((a)[1], (a)[3], dir);
    fftKernel2S((a)[0], (a)[1], dir);
    (a)[3] = (dir)*(conjTransp((a)[3]));
    fftKernel2S((a)[2], (a)[3], dir);
    float2 c = (a)[1];
    (a)[1] = (a)[2];
    (a)[2] = c;
}

inline __device__ void bitreverse8(float2 *a) {
    float2 c;
    c = (a)[1];
    (a)[1] = (a)[4];
    (a)[4] = c;
    c = (a)[3];
    (a)[3] = (a)[6];
    (a)[6] = c;
}

inline __device__ void fftKernel8(float2 *a, int dir) {
	const float2 w1  = make_float2(0x1.6a09e6p-1f,  dir*0x1.6a09e6p-1f); 
	const float2 w3  = make_float2(-0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f); 
	fftKernel2S((a)[0], (a)[4], dir);
	fftKernel2S((a)[1], (a)[5], dir);
	fftKernel2S((a)[2], (a)[6], dir);
	fftKernel2S((a)[3], (a)[7], dir);
	(a)[5] = complexMul(w1, (a)[5]);
	(a)[6] = (dir)*(conjTransp((a)[6]));
	(a)[7] = complexMul(w3, (a)[7]);
	fftKernel2S((a)[0], (a)[2], dir);
	fftKernel2S((a)[1], (a)[3], dir);
	fftKernel2S((a)[4], (a)[6], dir);
	fftKernel2S((a)[5], (a)[7], dir);
	(a)[3] = (dir)*(conjTransp((a)[3]));
	(a)[7] = (dir)*(conjTransp((a)[7]));
	fftKernel2S((a)[0], (a)[1], dir);
	fftKernel2S((a)[2], (a)[3], dir);
	fftKernel2S((a)[4], (a)[5], dir);
	fftKernel2S((a)[6], (a)[7], dir);
	bitreverse8((a));
}

__global__ void kernel_fft(float2 *in, float2 *out, int dir)
{
    float2 *orig_out = out;
    __shared__ float2 tmp[32 * 32];
    __shared__ float sMem[1280];

    // FFT0
    {
        int i, j;
        int ii, jj, offset;
        float2 w;
        float ang, angf;
        float *lMemStore, *lMemLoad;
        float2 a[8];
        int lId = threadIdx.x;
        int groupId = blockIdx.x;
        ii = lId & 31;
        jj = lId >> 5;
        lMemStore = sMem + mad24( jj, 36, ii );
        offset = mad24( groupId, 32, jj);
        offset = mad24( offset, 32, ii );
        in += offset;
        out = tmp + mad24( jj, 32, ii );
        a[0] = in[0];
        a[1] = in[128];
        a[2] = in[256];
        a[3] = in[384];
        a[4] = in[512];
        a[5] = in[640];
        a[6] = in[768];
        a[7] = in[896];
        ii = lId & 3;
        jj = lId >> 2;
        lMemLoad  = sMem + mad24( jj, 36, ii);
        lMemStore[0] = a[0].x;
        lMemStore[144] = a[1].x;
        lMemStore[288] = a[2].x;
        lMemStore[432] = a[3].x;
        lMemStore[576] = a[4].x;
        lMemStore[720] = a[5].x;
        lMemStore[864] = a[6].x;
        lMemStore[1008] = a[7].x;
        __syncthreads();
        a[0].x = lMemLoad[0];
        a[1].x = lMemLoad[4];
        a[2].x = lMemLoad[8];
        a[3].x = lMemLoad[12];
        a[4].x = lMemLoad[16];
        a[5].x = lMemLoad[20];
        a[6].x = lMemLoad[24];
        a[7].x = lMemLoad[28];
        __syncthreads();
        lMemStore[0] = a[0].y;
        lMemStore[144] = a[1].y;
        lMemStore[288] = a[2].y;
        lMemStore[432] = a[3].y;
        lMemStore[576] = a[4].y;
        lMemStore[720] = a[5].y;
        lMemStore[864] = a[6].y;
        lMemStore[1008] = a[7].y;
        __syncthreads();
        a[0].y = lMemLoad[0];
        a[1].y = lMemLoad[4];
        a[2].y = lMemLoad[8];
        a[3].y = lMemLoad[12];
        a[4].y = lMemLoad[16];
        a[5].y = lMemLoad[20];
        a[6].y = lMemLoad[24];
        a[7].y = lMemLoad[28];
        __syncthreads();
        fftKernel8(a+0, dir);
        angf = (float) ii;
        ang = dir * ( 2.0f * M_PI * 1.0f / 32.0f ) * angf;
        w = make_float2(cos(ang), sin(ang));
        a[1] = complexMul(a[1], w);
        ang = dir * ( 2.0f * M_PI * 2.0f / 32.0f ) * angf;
        w = make_float2(cos(ang), sin(ang));
        a[2] = complexMul(a[2], w);
        ang = dir * ( 2.0f * M_PI * 3.0f / 32.0f ) * angf;
        w = make_float2(cos(ang), sin(ang));
        a[3] = complexMul(a[3], w);
        ang = dir * ( 2.0f * M_PI * 4.0f / 32.0f ) * angf;
        w = make_float2(cos(ang), sin(ang));
        a[4] = complexMul(a[4], w);
        ang = dir * ( 2.0f * M_PI * 5.0f / 32.0f ) * angf;
        w = make_float2(cos(ang), sin(ang));
        a[5] = complexMul(a[5], w);
        ang = dir * ( 2.0f * M_PI * 6.0f / 32.0f ) * angf;
        w = make_float2(cos(ang), sin(ang));
        a[6] = complexMul(a[6], w);
        ang = dir * ( 2.0f * M_PI * 7.0f / 32.0f ) * angf;
        w = make_float2(cos(ang), sin(ang));
        a[7] = complexMul(a[7], w);
        lMemStore = sMem + mad24(jj, 40, ii);
        j = ii;
        i = 0;
        i = mad24(jj, 40, i);
        lMemLoad = sMem + mad24(j, 5, i);
        lMemStore[0] = a[0].x;
        lMemStore[5] = a[1].x;
        lMemStore[10] = a[2].x;
        lMemStore[15] = a[3].x;
        lMemStore[20] = a[4].x;
        lMemStore[25] = a[5].x;
        lMemStore[30] = a[6].x;
        lMemStore[35] = a[7].x;
        __syncthreads();
        a[0].x = lMemLoad[0];
        a[1].x = lMemLoad[1];
        a[2].x = lMemLoad[2];
        a[3].x = lMemLoad[3];
        a[4].x = lMemLoad[20];
        a[5].x = lMemLoad[21];
        a[6].x = lMemLoad[22];
        a[7].x = lMemLoad[23];
        __syncthreads();
        lMemStore[0] = a[0].y;
        lMemStore[5] = a[1].y;
        lMemStore[10] = a[2].y;
        lMemStore[15] = a[3].y;
        lMemStore[20] = a[4].y;
        lMemStore[25] = a[5].y;
        lMemStore[30] = a[6].y;
        lMemStore[35] = a[7].y;
        __syncthreads();
        a[0].y = lMemLoad[0];
        a[1].y = lMemLoad[1];
        a[2].y = lMemLoad[2];
        a[3].y = lMemLoad[3];
        a[4].y = lMemLoad[20];
        a[5].y = lMemLoad[21];
        a[6].y = lMemLoad[22];
        a[7].y = lMemLoad[23];
        __syncthreads();
        fftKernel4(a+0, dir);
        fftKernel4(a+4, dir);
        lMemLoad  = sMem + mad24( jj, 36, ii );
        ii = lId & 31;
        jj = lId >> 5;
        lMemStore = sMem + mad24( jj,36, ii );
        lMemLoad[0] = a[0].x;
        lMemLoad[4] = a[4].x;
        lMemLoad[8] = a[1].x;
        lMemLoad[12] = a[5].x;
        lMemLoad[16] = a[2].x;
        lMemLoad[20] = a[6].x;
        lMemLoad[24] = a[3].x;
        lMemLoad[28] = a[7].x;
        __syncthreads();
        a[0].x = lMemStore[0];
        a[1].x = lMemStore[144];
        a[2].x = lMemStore[288];
        a[3].x = lMemStore[432];
        a[4].x = lMemStore[576];
        a[5].x = lMemStore[720];
        a[6].x = lMemStore[864];
        a[7].x = lMemStore[1008];
        __syncthreads();
        lMemLoad[0] = a[0].y;
        lMemLoad[4] = a[4].y;
        lMemLoad[8] = a[1].y;
        lMemLoad[12] = a[5].y;
        lMemLoad[16] = a[2].y;
        lMemLoad[20] = a[6].y;
        lMemLoad[24] = a[3].y;
        lMemLoad[28] = a[7].y;
        __syncthreads();
        a[0].y = lMemStore[0];
        a[1].y = lMemStore[144];
        a[2].y = lMemStore[288];
        a[3].y = lMemStore[432];
        a[4].y = lMemStore[576];
        a[5].y = lMemStore[720];
        a[6].y = lMemStore[864];
        a[7].y = lMemStore[1008];
        __syncthreads();
        out[0] = a[0];
        out[128] = a[1];
        out[256] = a[2];
        out[384] = a[3];
        out[512] = a[4];
        out[640] = a[5];
        out[768] = a[6];
        out[896] = a[7];
    }

    // FFT1
    out = orig_out;
    {
        int i, j, indexIn, indexOut, tid, xNum;
        float2 w;
        float ang;
        float *lMemStore, *lMemLoad;
        float2 a[8];
        int lId = threadIdx.x;
        int groupId = blockIdx.x;
        xNum = groupId >> 0;
        groupId = groupId & 0;
        indexIn = mad24(groupId, 32, xNum << 10);
        tid = groupId * 32;
        i = tid >> 5;
        j = tid & 31;
        indexOut = mad24(i, 1024, j + (xNum << 10));
        tid = lId;
        i = tid & 31;
        j = tid >> 5;
        in = tmp + lId;
        a[0] = in[0];
        a[1] = in[128];
        a[2] = in[256];
        a[3] = in[384];
        a[4] = in[512];
        a[5] = in[640];
        a[6] = in[768];
        a[7] = in[896];
        fftKernel8(a, dir);
        ang = dir*(2.0f*M_PI*1/32)*j;
        w = make_float2(cos(ang), sin(ang));
        a[1] = complexMul(a[1], w);
        ang = dir*(2.0f*M_PI*2/32)*j;
        w = make_float2(cos(ang), sin(ang));
        a[2] = complexMul(a[2], w);
        ang = dir*(2.0f*M_PI*3/32)*j;
        w = make_float2(cos(ang), sin(ang));
        a[3] = complexMul(a[3], w);
        ang = dir*(2.0f*M_PI*4/32)*j;
        w = make_float2(cos(ang), sin(ang));
        a[4] = complexMul(a[4], w);
        ang = dir*(2.0f*M_PI*5/32)*j;
        w = make_float2(cos(ang), sin(ang));
        a[5] = complexMul(a[5], w);
        ang = dir*(2.0f*M_PI*6/32)*j;
        w = make_float2(cos(ang), sin(ang));
        a[6] = complexMul(a[6], w);
        ang = dir*(2.0f*M_PI*7/32)*j;
        w = make_float2(cos(ang), sin(ang));
        a[7] = complexMul(a[7], w);
        indexIn = mad24(j, 256, i);
        lMemStore = sMem + tid;
        lMemLoad = sMem + indexIn;
        lMemStore[0] = a[0].x;
        lMemStore[128] = a[1].x;
        lMemStore[256] = a[2].x;
        lMemStore[384] = a[3].x;
        lMemStore[512] = a[4].x;
        lMemStore[640] = a[5].x;
        lMemStore[768] = a[6].x;
        lMemStore[896] = a[7].x;
        __syncthreads();
        a[0].x = lMemLoad[0];
        a[1].x = lMemLoad[32];
        a[2].x = lMemLoad[64];
        a[3].x = lMemLoad[96];
        a[4].x = lMemLoad[128];
        a[5].x = lMemLoad[160];
        a[6].x = lMemLoad[192];
        a[7].x = lMemLoad[224];
        __syncthreads();
        lMemStore[0] = a[0].y;
        lMemStore[128] = a[1].y;
        lMemStore[256] = a[2].y;
        lMemStore[384] = a[3].y;
        lMemStore[512] = a[4].y;
        lMemStore[640] = a[5].y;
        lMemStore[768] = a[6].y;
        lMemStore[896] = a[7].y;
        __syncthreads();
        a[0].y = lMemLoad[0];
        a[1].y = lMemLoad[32];
        a[2].y = lMemLoad[64];
        a[3].y = lMemLoad[96];
        a[4].y = lMemLoad[128];
        a[5].y = lMemLoad[160];
        a[6].y = lMemLoad[192];
        a[7].y = lMemLoad[224];
        __syncthreads();
        fftKernel4(a + 0, dir);
        fftKernel4(a + 4, dir);
        indexOut += mad24(j, 64, i);
        out += indexOut;
        out[0] = a[0];
        out[256] = a[1];
        out[512] = a[2];
        out[768] = a[3];
        out[32] = a[4];
        out[288] = a[5];
        out[544] = a[6];
        out[800] = a[7];
    }
}
}
