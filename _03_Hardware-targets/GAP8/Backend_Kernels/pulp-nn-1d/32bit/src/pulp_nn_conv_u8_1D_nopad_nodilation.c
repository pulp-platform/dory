#include "rt/rt_api.h"
#include "utils.h"
#include "kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c)        __builtin_pulp_sdotusp4(a, b, c)
#define nn_round(out_shift)     (0x1 << (out_shift -1))
#define clip8(x)                __builtin_pulp_clipu_r(x, 255)
#define max4(a,b)  		    __builtin_pulp_maxu4(a,b)

void __attribute__ ((noinline)) pulp_nn_convolution_1D_uint8_nopad_nodilation(
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
	const int16_t   out_mult,
	uint8_t *       pOutBuffer,
  	const uint16_t  dim_out_y,
	int64_t *       k,
  	int64_t *       lambda,
	uint8_t *       pIm2ColBuffer, 
	const uint16_t	dil,
  	uint8_t *       pReserved,
	int             flag_relu,
  	int             flag_batch_norm
){
	int core_id = rt_core_id();

	uint16_t i_out_y;
	int i=0, j=0;
	int dim_ker=ch_in*dim_kernel_y;
  	int Log2Core = log2(NUM_CORES);

	int chunk = (dim_out_y >> Log2Core) + ((dim_out_y & (NUM_CORES-1))!=0);

	int start_pixel, stop_pixel;
  	start_pixel = min(chunk *  core_id, dim_out_y);
  	stop_pixel = min(start_pixel+chunk, dim_out_y);
	uint8_t *pOut    = pOutBuffer + start_pixel * ch_out;
int perf_cyc = 0;

	for(i_out_y = start_pixel; i_out_y < stop_pixel-1; i_out_y+=2){
		uint8_t *pIn=pInBuffer+(i_out_y*stride_y*ch_in);

		pOut = pulp_nn_matmul_4x2_uint8_nopad_nodilation(
			pWeight,
			pIn,
			ch_in,
			ch_out,
			stride_y,
			dim_ker,
			bias_shift,
			out_shift,
			out_mult,
			(int64_t *) k,
  			(int64_t *) lambda,
			bias,
			pOut,
			flag_relu,
  			flag_batch_norm
		);
	}

	//LEFTOVER OUTPUT PIXEL HERE
	if(i_out_y==stop_pixel-1){
		uint8_t *pIn=pInBuffer+(i_out_y*stride_y*ch_in);

		pOut = pulp_nn_matmul_4x1_uint8(
			pWeight,
			pIn,
			ch_out,
			dim_ker,
			bias_shift,
			out_shift,
			out_mult,
			(int64_t *) k,
  			(int64_t *) lambda,
			bias,
			pOut,
			flag_relu,
  			flag_batch_norm
		);
	}

	rt_team_barrier();
}


