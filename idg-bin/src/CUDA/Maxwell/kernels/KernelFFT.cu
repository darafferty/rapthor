extern "C" {
//Run kernel fft0 with global dim = {128*BatchSize}, local dim={128}
//Run kernel fft1 with global dim = {128*BatchSize}, local dim={128}
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

#undef M_PI
#define M_PI 0x1.921fb54442d18p+1f

#define mad(A,B,C) ((A)*(B)+(C))
#define mad24(A,B,C) ((A)*(B)+(C))
#define complexMul(a,b) (make_float2(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y)))
#define conj(a) (make_float2((a).x, -(a).y))
#define conjTransp(a) (make_float2(-(a).y, (a).x))

#define fftKernel2(a,dir) \
{ \
    float2 c = (a)[0];    \
    (a)[0] = c + (a)[1];  \
    (a)[1] = c - (a)[1];  \
}

#define fftKernel2S(d1,d2,dir) \
{ \
    float2 c = (d1);   \
    (d1) = c + (d2);   \
    (d2) = c - (d2);   \
}

#define fftKernel4(a,dir) \
{ \
    fftKernel2S((a)[0], (a)[2], dir); \
    fftKernel2S((a)[1], (a)[3], dir); \
    fftKernel2S((a)[0], (a)[1], dir); \
    (a)[3] = (dir)*(conjTransp((a)[3])); \
    fftKernel2S((a)[2], (a)[3], dir); \
    float2 c = (a)[1]; \
    (a)[1] = (a)[2]; \
    (a)[2] = c; \
}

#define fftKernel4s(a0,a1,a2,a3,dir) \
{ \
    fftKernel2S((a0), (a2), dir); \
    fftKernel2S((a1), (a3), dir); \
    fftKernel2S((a0), (a1), dir); \
    (a3) = (dir)*(conjTransp((a3))); \
    fftKernel2S((a2), (a3), dir); \
    float2 c = (a1); \
    (a1) = (a2); \
    (a2) = c; \
}

#define bitreverse8(a) \
{ \
    float2 c; \
    c = (a)[1]; \
    (a)[1] = (a)[4]; \
    (a)[4] = c; \
    c = (a)[3]; \
    (a)[3] = (a)[6]; \
    (a)[6] = c; \
}

#define fftKernel8(a,dir) \
{ \
	const float2 w1  = make_float2(0x1.6a09e6p-1f,  dir*0x1.6a09e6p-1f);  \
	const float2 w3  = make_float2(-0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f);  \
	float2 c; \
	fftKernel2S((a)[0], (a)[4], dir); \
	fftKernel2S((a)[1], (a)[5], dir); \
	fftKernel2S((a)[2], (a)[6], dir); \
	fftKernel2S((a)[3], (a)[7], dir); \
	(a)[5] = complexMul(w1, (a)[5]); \
	(a)[6] = (dir)*(conjTransp((a)[6])); \
	(a)[7] = complexMul(w3, (a)[7]); \
	fftKernel2S((a)[0], (a)[2], dir); \
	fftKernel2S((a)[1], (a)[3], dir); \
	fftKernel2S((a)[4], (a)[6], dir); \
	fftKernel2S((a)[5], (a)[7], dir); \
	(a)[3] = (dir)*(conjTransp((a)[3])); \
	(a)[7] = (dir)*(conjTransp((a)[7])); \
	fftKernel2S((a)[0], (a)[1], dir); \
	fftKernel2S((a)[2], (a)[3], dir); \
	fftKernel2S((a)[4], (a)[5], dir); \
	fftKernel2S((a)[6], (a)[7], dir); \
	bitreverse8((a)); \
}

#define bitreverse4x4(a) \
{ \
	float2 c; \
	c = (a)[1];  (a)[1]  = (a)[4];  (a)[4]  = c; \
	c = (a)[2];  (a)[2]  = (a)[8];  (a)[8]  = c; \
	c = (a)[3];  (a)[3]  = (a)[12]; (a)[12] = c; \
	c = (a)[6];  (a)[6]  = (a)[9];  (a)[9]  = c; \
	c = (a)[7];  (a)[7]  = (a)[13]; (a)[13] = c; \
	c = (a)[11]; (a)[11] = (a)[14]; (a)[14] = c; \
}

#define fftKernel16(a,dir) \
{ \
    const float w0 = 0x1.d906bcp-1f; \
    const float w1 = 0x1.87de2ap-2f; \
    const float w2 = 0x1.6a09e6p-1f; \
    fftKernel4s((a)[0], (a)[4], (a)[8],  (a)[12], dir); \
    fftKernel4s((a)[1], (a)[5], (a)[9],  (a)[13], dir); \
    fftKernel4s((a)[2], (a)[6], (a)[10], (a)[14], dir); \
    fftKernel4s((a)[3], (a)[7], (a)[11], (a)[15], dir); \
    (a)[5]  = complexMul((a)[5], make_float2(w0, dir*w1)); \
    (a)[6]  = complexMul((a)[6], make_float2(w2, dir*w2)); \
    (a)[7]  = complexMul((a)[7], make_float2(w1, dir*w0)); \
    (a)[9]  = complexMul((a)[9], make_float2(w2, dir*w2)); \
    (a)[10] = (dir)*(conjTransp((a)[10])); \
    (a)[11] = complexMul((a)[11], make_float2(-w2, dir*w2)); \
    (a)[13] = complexMul((a)[13], make_float2(w1, dir*w0)); \
    (a)[14] = complexMul((a)[14], make_float2(-w2, dir*w2)); \
    (a)[15] = complexMul((a)[15], make_float2(-w0, dir*-w1)); \
    fftKernel4((a), dir); \
    fftKernel4((a) + 4, dir); \
    fftKernel4((a) + 8, dir); \
    fftKernel4((a) + 12, dir); \
    bitreverse4x4((a)); \
}

#define bitreverse32(a) \
{ \
    float2 c1, c2; \
    c1 = (a)[2];   (a)[2] = (a)[1];   c2 = (a)[4];   (a)[4] = c1;   c1 = (a)[8];   (a)[8] = c2;    c2 = (a)[16];  (a)[16] = c1;   (a)[1] = c2; \
    c1 = (a)[6];   (a)[6] = (a)[3];   c2 = (a)[12];  (a)[12] = c1;  c1 = (a)[24];  (a)[24] = c2;   c2 = (a)[17];  (a)[17] = c1;   (a)[3] = c2; \
    c1 = (a)[10];  (a)[10] = (a)[5];  c2 = (a)[20];  (a)[20] = c1;  c1 = (a)[9];   (a)[9] = c2;    c2 = (a)[18];  (a)[18] = c1;   (a)[5] = c2; \
    c1 = (a)[14];  (a)[14] = (a)[7];  c2 = (a)[28];  (a)[28] = c1;  c1 = (a)[25];  (a)[25] = c2;   c2 = (a)[19];  (a)[19] = c1;   (a)[7] = c2; \
    c1 = (a)[22];  (a)[22] = (a)[11]; c2 = (a)[13];  (a)[13] = c1;  c1 = (a)[26];  (a)[26] = c2;   c2 = (a)[21];  (a)[21] = c1;   (a)[11] = c2; \
    c1 = (a)[30];  (a)[30] = (a)[15]; c2 = (a)[29];  (a)[29] = c1;  c1 = (a)[27];  (a)[27] = c2;   c2 = (a)[23];  (a)[23] = c1;   (a)[15] = c2; \
}

#define fftKernel32(a,dir) \
{ \
    fftKernel2S((a)[0],  (a)[16], dir); \
    fftKernel2S((a)[1],  (a)[17], dir); \
    fftKernel2S((a)[2],  (a)[18], dir); \
    fftKernel2S((a)[3],  (a)[19], dir); \
    fftKernel2S((a)[4],  (a)[20], dir); \
    fftKernel2S((a)[5],  (a)[21], dir); \
    fftKernel2S((a)[6],  (a)[22], dir); \
    fftKernel2S((a)[7],  (a)[23], dir); \
    fftKernel2S((a)[8],  (a)[24], dir); \
    fftKernel2S((a)[9],  (a)[25], dir); \
    fftKernel2S((a)[10], (a)[26], dir); \
    fftKernel2S((a)[11], (a)[27], dir); \
    fftKernel2S((a)[12], (a)[28], dir); \
    fftKernel2S((a)[13], (a)[29], dir); \
    fftKernel2S((a)[14], (a)[30], dir); \
    fftKernel2S((a)[15], (a)[31], dir); \
    (a)[17] = complexMul((a)[17], make_float2(0x1.f6297cp-1f, dir*0x1.8f8b84p-3f)); \
    (a)[18] = complexMul((a)[18], make_float2(0x1.d906bcp-1f, dir*0x1.87de2ap-2f)); \
    (a)[19] = complexMul((a)[19], make_float2(0x1.a9b662p-1f, dir*0x1.1c73b4p-1f)); \
    (a)[20] = complexMul((a)[20], make_float2(0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f)); \
    (a)[21] = complexMul((a)[21], make_float2(0x1.1c73b4p-1f, dir*0x1.a9b662p-1f)); \
    (a)[22] = complexMul((a)[22], make_float2(0x1.87de2ap-2f, dir*0x1.d906bcp-1f)); \
    (a)[23] = complexMul((a)[23], make_float2(0x1.8f8b84p-3f, dir*0x1.f6297cp-1f)); \
    (a)[24] = complexMul((a)[24], make_float2(0x0p+0f, dir*0x1p+0f)); \
    (a)[25] = complexMul((a)[25], make_float2(-0x1.8f8b84p-3f, dir*0x1.f6297cp-1f)); \
    (a)[26] = complexMul((a)[26], make_float2(-0x1.87de2ap-2f, dir*0x1.d906bcp-1f)); \
    (a)[27] = complexMul((a)[27], make_float2(-0x1.1c73b4p-1f, dir*0x1.a9b662p-1f)); \
    (a)[28] = complexMul((a)[28], make_float2(-0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f)); \
    (a)[29] = complexMul((a)[29], make_float2(-0x1.a9b662p-1f, dir*0x1.1c73b4p-1f)); \
    (a)[30] = complexMul((a)[30], make_float2(-0x1.d906bcp-1f, dir*0x1.87de2ap-2f)); \
    (a)[31] = complexMul((a)[31], make_float2(-0x1.f6297cp-1f, dir*0x1.8f8b84p-3f)); \
    fftKernel16((a), dir); \
    fftKernel16((a) + 16, dir); \
    bitreverse32((a)); \
}

#if 0
__kernel void \
clFFT_1DTwistInterleaved(__global float2 *in, unsigned int startRow, unsigned int numCols, unsigned int N, unsigned int numRowsToProcess, int dir) \
{ \
   float2 a, w; \
   float ang; \
   unsigned int j; \
	unsigned int i = get_global_id(0); \
	unsigned int startIndex = i; \
	 \
	if(i < numCols) \
	{ \
	    for(j = 0; j < numRowsToProcess; j++) \
	    { \
	        a = in[startIndex]; \
	        ang = 2.0f * M_PI * dir * i * (startRow + j) / N; \
	        w = (float2)(cos(ang), sin(ang)); \
	        a = complexMul(a, w); \
	        in[startIndex] = a; \
	        startIndex += numCols; \
	    } \
	}	 \
} \

#endif
__global__ void kernel_fft(float2 *in, float2 *out, int dir, int S)
{
float2 *orig_in = in, *orig_out = out;
    __shared__ float2 tmp[32 * 32];
    __shared__ float sMem[1280];
{
    int i, j, r, indexIn, indexOut, index, tid, bNum, xNum, k, l;
    int s, ii, jj, offset;
    float2 w;
    float ang, angf, ang1;
    float *lMemStore, *lMemLoad;
    float2 a[8];
    int lId = threadIdx.x;
    int groupId = blockIdx.x;
        s = S & 31;
    ii = lId & 31;
    jj = lId >> 5;
    lMemStore = sMem + mad24( jj, 36, ii );
    offset = mad24( groupId, 32, jj);
    offset = mad24( offset, 32, ii );
        in += offset;
        //out += offset;
        out = tmp + mad24( jj, 32, ii );
#if 0
if((groupId == get_num_groups(0)-1) && s) {
    if( jj < s ) {
        a[0] = in[0];
    }
    jj += 4;
    if( jj < s ) {
        a[1] = in[128];
    }
    jj += 4;
    if( jj < s ) {
        a[2] = in[256];
    }
    jj += 4;
    if( jj < s ) {
        a[3] = in[384];
    }
    jj += 4;
    if( jj < s ) {
        a[4] = in[512];
    }
    jj += 4;
    if( jj < s ) {
        a[5] = in[640];
    }
    jj += 4;
    if( jj < s ) {
        a[6] = in[768];
    }
    jj += 4;
    if( jj < s ) {
        a[7] = in[896];
    }
}
 else {
#endif
        a[0] = in[0];
        a[1] = in[128];
        a[2] = in[256];
        a[3] = in[384];
        a[4] = in[512];
        a[5] = in[640];
        a[6] = in[768];
        a[7] = in[896];
//}
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
#if 0
if((groupId == get_num_groups(0)-1) && s) {
    if( jj < s ) {
        out[0] = a[0];
    }
    jj += 4;
    if( jj < s ) {
        out[128] = a[1];
    }
    jj += 4;
    if( jj < s ) {
        out[256] = a[2];
    }
    jj += 4;
    if( jj < s ) {
        out[384] = a[3];
    }
    jj += 4;
    if( jj < s ) {
        out[512] = a[4];
    }
    jj += 4;
    if( jj < s ) {
        out[640] = a[5];
    }
    jj += 4;
    if( jj < s ) {
        out[768] = a[6];
    }
    jj += 4;
    if( jj < s ) {
        out[896] = a[7];
    }
}
else {
#endif
        out[0] = a[0];
        out[128] = a[1];
        out[256] = a[2];
        out[384] = a[3];
        out[512] = a[4];
        out[640] = a[5];
        out[768] = a[6];
        out[896] = a[7];
//}
}
//__global__ void fft1(float2 *in, float2 *out, int dir, int S)
out = orig_out;
{
    int i, j, r, indexIn, indexOut, index, tid, bNum, xNum, k, l;
    int s, ii, jj, offset;
    float2 w;
    float ang, angf, ang1;
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
bNum = groupId;
tid = lId;
i = tid & 31;
j = tid >> 5;
//indexIn += mad24(j, 32, i);
//in += indexIn;
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
