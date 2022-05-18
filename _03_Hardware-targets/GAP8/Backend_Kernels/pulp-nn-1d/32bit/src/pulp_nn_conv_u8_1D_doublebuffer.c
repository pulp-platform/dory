#include "rt/rt_api.h"
#include "utils.h"
#include "kernels.h"
#include "mchan_test.h"
#include <stdbool.h>
#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c)        __builtin_pulp_sdotusp4(a, b, c)
#define nn_round(out_shift)     (0x1 << (out_shift -1))
#define clip8(x)                __builtin_pulp_clipu_r(x, 255)
#define max4(a,b)  		    __builtin_pulp_maxu4(a,b)


void __attribute__ ((noinline)) pulp_nn_convolution_1D_uint8_2buff(
	const uint8_t * pInBuffer,
  	const uint16_t  dim_in_y,
	const uint16_t  ch_in,
  	const int8_t *  pWeight,
	const uint16_t  ch_out,
  	const uint16_t  dim_kernel_y,
  	const uint16_t  padding_y_top,
  	const uint16_t  padding_y_bottom,
  	const uint16_t  stride_y,
  	const int8_t *  bias,
  	const uint16_t  bias_shift,
  	const uint16_t  out_shift,
	const int16_t  out_mult,
	uint8_t *       pOutBuffer,
  	const uint16_t  dim_out_y,
	uint8_t *       k,
  	uint8_t *       lambda,
	uint8_t *       pIm2ColBuffer1, 
	uint8_t *       pIm2ColBuffer2,
	const uint8_t * zero,
	const uint16_t	dil,
  	uint8_t *        pReserved,
	int             flag_relu,
  	int             flag_batch_norm
){
	uint16_t dilation = dil-1;

	bool bflag=0;
	bool first=1;

	int core_id = rt_core_id();

	unsigned int id = mchan_alloc();
	

	uint16_t i_out_y;
	int i=0, j=0;
	int dim_ker=ch_in*dim_kernel_y;
  	int Log2Core = log2(NUM_CORES);

	uint8_t * pIm2ColBase1 = pIm2ColBuffer1 + (2*core_id*dim_ker);
	uint8_t * pIm2ColBase2 = pIm2ColBuffer2 + (2*core_id*dim_ker);

	int chunk = (dim_out_y >> Log2Core) + ((dim_out_y & (NUM_CORES-1))!=0);

	int start_pixel, stop_pixel;
  	start_pixel = min(chunk *  core_id, dim_out_y);
  	stop_pixel = min(start_pixel+chunk, dim_out_y);

	uint8_t *pIm2Col = pIm2ColBase1;
	uint8_t *pIm2ColBase = pIm2ColBase1;

	uint8_t *pOut    = pOutBuffer + start_pixel * ch_out;

	for (i_out_y = start_pixel; i_out_y < stop_pixel+1; i_out_y++){

		//printf("Core %d entered cycle \n", core_id);

		//FIRST PART: LOADING IM2COL BUFFERS WITH DATA
		if(i_out_y != stop_pixel){		
			if(i_out_y < padding_y_top){
				for(i = i_out_y * stride_y - padding_y_top; i < i_out_y * stride_y - padding_y_top + (dim_kernel_y*(1+dilation)-dilation); i += (1+dilation)){
					if(i < 0 || i >= dim_in_y)
						pulp_zero_mem_dma(pIm2Col, ch_in, zero);
					else
						pulp_nn_im2col_uint8((uint8_t *) pInBuffer + (i * ch_in), pIm2Col, ch_in);

					pIm2Col += ch_in;
				}
			}
			else if(i_out_y < dim_out_y - padding_y_bottom){
				for(i = i_out_y * stride_y-padding_y_top; i < i_out_y * stride_y - padding_y_top + (dim_kernel_y*(1+dilation)-dilation); i += (1+dilation)){
					pulp_nn_im2col_uint8((uint8_t *) pInBuffer + (i * ch_in), pIm2Col, ch_in);
					pIm2Col += ch_in;
				}
			}
			else{
				for(i = i_out_y * stride_y - padding_y_top; i < i_out_y * stride_y - padding_y_top + (dim_kernel_y*(1+dilation)-dilation); i += (1+dilation)){
					if(i < 0 || i >= dim_in_y)
						pulp_zero_mem_dma(pIm2Col, ch_in, zero);
					else
						pulp_nn_im2col_uint8((uint8_t *) pInBuffer + (i * ch_in), pIm2Col, ch_in);

					pIm2Col += ch_in;
				}
			}
		}

		//SECOND PART: USING IM2COL BUFFERS FOR 4x2 CONVOLUTION
		if(pIm2Col==pIm2ColBase1+2*dim_ker || pIm2Col==pIm2ColBase2+2*dim_ker || i_out_y == stop_pixel){
  
			if(first){
				first--;
				mchan_barrier(id);
				pIm2Col = pIm2ColBase2;
			}
			else{
				pOut = pulp_nn_matmul_4x2_uint8(
					pWeight,
					pIm2ColBase,
					ch_out,
					dim_ker,
					bias_shift,
					out_shift,
					out_mult,
					k,
  					lambda,
					bias,
					pOut,
					flag_relu,
					flag_batch_norm
				);

				if(!bflag){
					pIm2Col = pIm2ColBase1;
					pIm2ColBase = pIm2ColBase2;
				}
				else{
					pIm2Col = pIm2ColBase2;
					pIm2ColBase = pIm2ColBase1;
				}

				bflag = !bflag;
				mchan_barrier(id);
			}

		}

	}
	//END OF CYCLE

	
	//LEFTOVER PIXEL

	if(((stop_pixel-start_pixel) & 1)!=0){
		pOut = pulp_nn_matmul_4x1_uint8(
			pWeight,
			pIm2ColBase,
			ch_out,
			dim_ker,
			bias_shift,
			out_shift,
			out_mult,
			k,
  			lambda,
			bias,
			pOut,
			flag_relu,
  			flag_batch_norm
		);
	}
	
	
	mchan_free(id);

	rt_team_barrier();

	return;
}
