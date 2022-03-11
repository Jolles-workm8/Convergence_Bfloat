#ifndef VECTOR_BFLOAT_HPP_
#define VECTOR_BFLOAT_HPP_

#include <immintrin.h>

#include <libxsmm.h>



#define mm512_shuffle_i16x4(a,imm8) _mm512_shufflelo_epi16 (_mm512_shufflehi_epi16(a,imm8), imm8)
 
#define FP32_TO_BF16_SPLIT(input,out0,out1,out2) \
{ \
    out0 = _mm512_maskz_mov_epi16(0xAAAAAAAA,_mm512_castps_si512(input)); \
    __m512 _temp = _mm512_sub_ps(input,_mm512_castsi512_ps(out0)); \
    out1 = _mm512_maskz_mov_epi16(0xAAAAAAAA,_mm512_castps_si512(_temp)); \
    _temp = _mm512_sub_ps(_temp,_mm512_castsi512_ps(out1)); \
    out2 = _mm512_maskz_mov_epi16(0xAAAAAAAA,_mm512_castps_si512(_temp)); \
}
 
void split_compress(float *input,libxsmm_bfloat16 *output0, libxsmm_bfloat16 *output1, libxsmm_bfloat16 *output2, const int size)
{
    static const unsigned short perm_idx_buffer[] = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,
                                       0x20|1,0x20|3,0x20|5,0x20|7,0x20|9,0x20|11,0x20|13,0x20|15,
                                       0x20|17,0x20|19,0x20|21,0x20|23,0x20|25,0x20|27,0x20|29,0x20|31};
 
    static const __m512i PERM_IDX = _mm512_loadu_si512(perm_idx_buffer);
 
   
    for(int i=0;i<size;i+=32)
    {
        __m512 fp_inputL = _mm512_loadu_ps(input + i);
        __m512 fp_inputH = _mm512_loadu_ps(input + 16 + i);

        __m512i inL[3];
        __m512i inH[3];

        FP32_TO_BF16_SPLIT(fp_inputL,inL[0],inL[1],inL[2]);
        FP32_TO_BF16_SPLIT(fp_inputH,inH[0],inH[1],inH[2]);

        __m512i in0 = _mm512_permutex2var_epi16 (inL[0],PERM_IDX, inH[0]);
        __m512i in1 = _mm512_permutex2var_epi16 (inL[1],PERM_IDX, inH[1]);
        __m512i in2 = _mm512_permutex2var_epi16 (inL[2],PERM_IDX, inH[2]);

        _mm512_storeu_si512(output0,in0);
        _mm512_storeu_si512(output1,in1);
        _mm512_storeu_si512(output2,in2);
        output0+=32;
        output1+=32;
        output2+=32;
    }
    
}
 
void split_vnni_rowmajor(float *input, const int input_width,
                         const int patch_width, const int patch_height,
                         libxsmm_bfloat16 *output0, libxsmm_bfloat16 *output1, libxsmm_bfloat16 *output2)
{
 
    for(int ih=0;ih<patch_height;ih+=32)
    for(int iw=0;iw<patch_width;iw+=16)
    {
        for(int i=0;i<32;i+=2)
        {
            __m512 fp_input_row0 = _mm512_loadu_ps(input + iw + (ih+i)*input_width);
            __m512 fp_input_row1 = _mm512_loadu_ps(input + iw + (ih+i+1)*input_width);
 
            __m512i in_row[2][3];
            __m512i vnni_out[3];
 
            FP32_TO_BF16_SPLIT(fp_input_row0,in_row[0][0],in_row[0][1],in_row[0][2]);
            FP32_TO_BF16_SPLIT(fp_input_row1,in_row[1][0],in_row[1][1],in_row[1][2]);
 
            in_row[0][0] = _mm512_srli_epi32 (in_row[0][0],16);
            in_row[0][1] = _mm512_srli_epi32 (in_row[0][1],16);
            in_row[0][2] = _mm512_srli_epi32 (in_row[0][2],16);
 
            vnni_out[0] = _mm512_or_si512 (in_row[0][0],in_row[1][0]);
            vnni_out[1] = _mm512_or_si512 (in_row[0][1],in_row[1][1]);
            vnni_out[2] = _mm512_or_si512 (in_row[0][2],in_row[1][2]);
 
            _mm512_storeu_si512(output0,vnni_out[0]);
            _mm512_storeu_si512(output1,vnni_out[1]);
            _mm512_storeu_si512(output2,vnni_out[2]);
            output0+=32;
            output1+=32;
            output2+=32;
        }
    }
}
 
static const unsigned short PERM_IDX_BUFFER[] = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,
                                   0x20|1,0x20|3,0x20|5,0x20|7,0x20|9,0x20|11,0x20|13,0x20|15,
                                   0x20|17,0x20|19,0x20|21,0x20|23,0x20|25,0x20|27,0x20|29,0x20|31};
 
template <int m_bct, int k_bct>
void split_compress_array(float *input, const int input_width,
                         libxsmm_bfloat16 (*output)[m_bct][k_bct][16][32])
{
    static const __m512i PERM_IDX = _mm512_loadu_si512(PERM_IDX_BUFFER);
 
    for(int imb=0;imb<m_bct;imb++)
    for(int ikb=0;ikb<k_bct;ikb++)
    {
        for(int i=0;i<16;i++)
        {
            __m512 fp_inputL = _mm512_loadu_ps(input + ikb*32 + (imb*16+i)*input_width);
            __m512 fp_inputH = _mm512_loadu_ps(input + ikb*32+16 + (imb*16+i)*input_width);
 
            __m512i inL[3];
            __m512i inH[3];
 
            FP32_TO_BF16_SPLIT(fp_inputL,inL[0],inL[1],inL[2]);
            FP32_TO_BF16_SPLIT(fp_inputH,inH[0],inH[1],inH[2]);
 
            __m512i in0 = _mm512_permutex2var_epi16 (inL[0],PERM_IDX, inH[0]);
            __m512i in1 = _mm512_permutex2var_epi16 (inL[1],PERM_IDX, inH[1]);
            __m512i in2 = _mm512_permutex2var_epi16 (inL[2],PERM_IDX, inH[2]);
 
            _mm512_storeu_si512(output[0][imb][ikb][i],in0);
            _mm512_storeu_si512(output[1][imb][ikb][i],in1);
            _mm512_storeu_si512(output[2][imb][ikb][i],in2);
        }
    }
}
 
template <int k_bct, int n_bct>
void split_vnni_rowmajor_array(float *input, const int input_width,
                         libxsmm_bfloat16 (*output)[k_bct][n_bct][16][32])
{
 
    for(int ikb=0;ikb<k_bct;ikb++)
    for(int inb=0;inb<n_bct;inb++)
    {
        for(int i=0;i<32;i+=2)
        {
            __m512 fp_input_row0 = _mm512_loadu_ps(input + inb*16 + (ikb*32+i)*input_width);
            __m512 fp_input_row1 = _mm512_loadu_ps(input + inb*16 + (ikb*32+i+1)*input_width);
 
            __m512i in_row[2][3];
            __m512i vnni_out[3];
 
            FP32_TO_BF16_SPLIT(fp_input_row0,in_row[0][0],in_row[0][1],in_row[0][2]);
            FP32_TO_BF16_SPLIT(fp_input_row1,in_row[1][0],in_row[1][1],in_row[1][2]);
 
            in_row[0][0] = _mm512_srli_epi32 (in_row[0][0],16);
            in_row[0][1] = _mm512_srli_epi32 (in_row[0][1],16);
            in_row[0][2] = _mm512_srli_epi32 (in_row[0][2],16);
 
            vnni_out[0] = _mm512_or_si512 (in_row[0][0],in_row[1][0]);
            vnni_out[1] = _mm512_or_si512 (in_row[0][1],in_row[1][1]);
            vnni_out[2] = _mm512_or_si512 (in_row[0][2],in_row[1][2]);
 
            _mm512_storeu_si512(output[0][ikb][inb][i>>1],vnni_out[0]);
            _mm512_storeu_si512(output[1][ikb][inb][i>>1],vnni_out[1]);
            _mm512_storeu_si512(output[2][ikb][inb][i>>1],vnni_out[2]);
        }
    }
}
 
// Split a large matrix (e.g. 512x512) into 4 quads (e.g. 256x256)
/*
    -----------
    | q0 | q1 |
    -----------
    | q2 | q3 |
    -----------
 
*/
void split_quadrants_avx(float *input, const int rows, const int cols, float *q0, float *q1, float *q2, float *q3)
{
    float *in0 = input;
    float *in1 = input + cols/2;
    float *in2 = input + cols*(rows/2);
    float *in3 = input + cols*(rows/2) + cols/2;
 
    for(int y=0;y<rows/2;y++)
    {
        for(int x=0;x<cols/2;x+=16)
        {
            __m512 vin0 = _mm512_load_ps(in0 + x + y*cols);
            __m512 vin1 = _mm512_load_ps(in1 + x + y*cols);
            __m512 vin2 = _mm512_load_ps(in2 + x + y*cols);
            __m512 vin3 = _mm512_load_ps(in3 + x + y*cols);
 
            _mm512_store_ps(q0,vin0);
            _mm512_store_ps(q1,vin1);
            _mm512_store_ps(q2,vin2);
            _mm512_store_ps(q3,vin3);
 
            q0+=16;q1+=16;q2+=16;q3+=16;
        }
    }
}

 
void join_quadrants_avx(float *output, const int rows, const int cols,
                        float *q0, float *q1, float *q2, float *q3)
{
    float *out0 = output;
    float *out1 = output + cols;
    float *out2 = output + (cols*2*rows);
    float *out3 = output + (cols*2*rows) + cols;
 
    for(int y=0;y<rows;y++)
    {
        for(int x=0;x<cols;x+=16)
        {
            __m512 in0 = _mm512_load_ps(q0 + x + y*cols);
            __m512 in1 = _mm512_load_ps(q1 + x + y*cols);
            __m512 in2 = _mm512_load_ps(q2 + x + y*cols);
            __m512 in3 = _mm512_load_ps(q3 + x + y*cols);
 
            _mm512_store_ps(out0,in0);
            _mm512_store_ps(out1,in1);
            _mm512_store_ps(out2,in2);
            _mm512_store_ps(out3,in3);
 
            out0+=16;out1+=16;out2+=16;out3+=16;
        }
        out0+=cols;
        out1+=cols;
        out2+=cols;
        out3+=cols;
    }
}

#endif