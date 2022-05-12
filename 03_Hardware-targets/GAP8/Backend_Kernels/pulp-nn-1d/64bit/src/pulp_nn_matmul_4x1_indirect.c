#include "rt/rt_api.h"
#include "utils.h"
#include "kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c)        __builtin_pulp_sdotusp4(a, b, c)
#define nn_round(out_shift)     (0x1 << (out_shift -1))
#define clip8(x)                __builtin_pulp_clipu_r(x, 255)
#define max4(a,b)  		    __builtin_pulp_maxu4(a,b)

uint8_t __attribute__ ((noinline)) *pulp_nn_matmul_4x1_uint8_indirect(
	int8_t * pWeight,
	uint8_t ** pInBuffer,
	uint16_t ch_out,
	const uint16_t  dim_kernel_y,
	const uint16_t  ch_in,
	uint16_t dim_ker,
	uint16_t bias_shift,
	uint16_t out_shift,
	int16_t out_mult,
  	int64_t * k,
  	int64_t * lambda,
	int8_t * bias,
	uint8_t * pOut,
	int flag_relu,
  	int flag_batch_norm
){
	const int8_t *  pBias = bias;
	int8_t  *pA = pWeight;
	uint16_t chan_left = ch_out & 0x3;
			
	v4s vecA;
  	v4s vecA2;
  	v4s vecA3;
  	v4s vecA4;
  	v4u vecB;


	for (int i = 0; i < ch_out>>2; i++){
		int8_t *pA2 = (pA + dim_ker);
    		int8_t *pA3 = (pA2 + dim_ker);
    		int8_t *pA4 = (pA3 + dim_ker);

		int bias1 = 0;
    		int bias2 = 0;
    		int bias3 = 0;
    		int bias4 = 0;

		if(bias!=NULL){
			bias1 = ((int) (*pBias++)  << bias_shift) + nn_round(out_shift);
      			bias2 = ((int) (*pBias++)  << bias_shift) + nn_round(out_shift);
			bias3 = ((int) (*pBias++)  << bias_shift) + nn_round(out_shift);
      			bias4 = ((int) (*pBias++)  << bias_shift) + nn_round(out_shift);
		}

		int sum1=bias1;
		int sum2=bias2;
		int sum3=bias3;
		int sum4=bias4;

		//uint16_t  col_cnt_im2col = dim_ker & 0x3;

		for(int j = 0; j <  dim_kernel_y; j++){
			uint8_t *pB=*(pInBuffer+j);
			uint16_t cnt = ch_in & 0x3;
			
			for(int k = 0; k < ch_in >> 2; k++){
				vecA  = * ( (v4s*) pA  );
      				vecA2 = * ( (v4s*) pA2 );
      				vecA3 = * ( (v4s*) pA3 );
      				vecA4 = * ( (v4s*) pA4 );
      				vecB  = * ( (v4u*) pB  );

				sum1 =  SumDotp (vecB,  vecA,  sum1 );
				sum2 =  SumDotp (vecB,  vecA2, sum2 );
				sum3 =  SumDotp (vecB,  vecA3, sum3 );
				sum4 =  SumDotp (vecB,  vecA4, sum4 );

      				pA  += 4;
      				pA2 += 4;
      				pA3 += 4;
      				pA4 += 4;
      				pB  += 4;
			}

			while(cnt){
				int8_t      inA  = *pA++;
      				int8_t      inA2 = *pA2++;
      				int8_t      inA3 = *pA3++;
      				int8_t      inA4 = *pA4++;
      				uint8_t     inB  = *pB++;
      				asm volatile("": : :"memory");
      				sum1 += inA  * inB;
      				sum2 += inA2 * inB;
      				sum3 += inA3 * inB;
      				sum4 += inA4 * inB;

      				cnt--;
			}

		}

		if (flag_batch_norm==1 && flag_relu==1){
      			*pOut = pulp_nn_bn_quant_u8(sum1, (int64_t) *k, (int64_t) *lambda, out_shift);
      			pOut++;
      			k++;
      			lambda++;

      			*pOut = pulp_nn_bn_quant_u8(sum2, (int64_t) *k, (int64_t) *lambda, out_shift);
      			pOut++;
      			k++;
      			lambda++;

      			*pOut = pulp_nn_bn_quant_u8(sum3, (int64_t) *k, (int64_t) *lambda, out_shift);
      			pOut++;
      			k++;
      			lambda++;

      			*pOut = pulp_nn_bn_quant_u8(sum4, (int64_t) *k, (int64_t) *lambda, out_shift);
      			pOut++;
      			k++;
      			lambda++;
		}
		else if(flag_relu==1){
			*pOut = pulp_nn_quant_u8(sum1, out_mult, out_shift);
        		pOut++;
        		*pOut = pulp_nn_quant_u8(sum2, out_mult, out_shift);
        		pOut++;
        		*pOut = pulp_nn_quant_u8(sum3, out_mult, out_shift);
        		pOut++;
        		*pOut = pulp_nn_quant_u8(sum4, out_mult, out_shift);
        		pOut++;
		}
		else{
			*pOut = (uint8_t) clip8(sum1 >> out_shift);
        		pOut++;
       	 		*pOut = (uint8_t) clip8(sum2 >> out_shift);
        		pOut++;
        		*pOut = (uint8_t) clip8(sum3 >> out_shift);
        		pOut++;
        		*pOut = (uint8_t) clip8(sum4 >> out_shift);
        		pOut++;
		}

		pA += 3 * dim_ker;
	}

	while(chan_left){

		int bias1 = 0;

		if(bias != NULL){
			bias1 = ((int) (*pBias++)  << bias_shift) + nn_round(out_shift);
		}

		int sum = bias1;

		for(int j = 0; j <  dim_kernel_y; j++){
			uint8_t *pB=*(pInBuffer+j);
			uint16_t cnt = ch_in & 0x3;
			
			for(int k = 0; k < ch_in >> 2; k++){
				vecA  = * ( (v4s*) pA  );
	
				sum =  SumDotp (vecB,  vecA,  sum );
	
      				pA  += 4;
      				pB  += 4;
			}	

			while(cnt){
				int8_t      inA  = *pA++;
      				uint8_t     inB  = *pB++;
      				asm volatile("": : :"memory");
      				sum += inA  * inB;

      				cnt--;
			}
		}

		if (flag_batch_norm==1 && flag_relu==1){
      			*pOut = pulp_nn_bn_quant_u8(sum, (int64_t) *k, (int64_t) *lambda, out_shift);
      			pOut++;
      			k++;
      			lambda++;
   		}
		else if(flag_relu==1){
			*pOut = pulp_nn_quant_u8(sum, out_mult, out_shift);
        		pOut++;
		}
		else{
			*pOut = (int8_t) clip8(sum >> out_shift);
        		pOut++;
		}

		chan_left--;
		
	}

	return pOut;


}








