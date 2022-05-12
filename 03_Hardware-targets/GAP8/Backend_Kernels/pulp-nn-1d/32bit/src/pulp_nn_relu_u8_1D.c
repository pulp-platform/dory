#include "rt/rt_api.h"
#include "utils.h"
#include "kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define max4(a,b) __builtin_pulp_max4(a,b)

void __attribute__((always_inline)) pulp_nn_relu_u8_1D(
  int8_t * data,
  uint16_t dim_im_in_y,
  uint16_t ch_im_in,
  uint8_t * out
) {
  int core_id = rt_core_id();
  int  Log2Core = log2(NUM_CORES );
  int chunck = (dim_im_in_y >> Log2Core ) + ((dim_im_in_y & (NUM_CORES-1))!=0);
  int start = min(chunck * core_id, dim_im_in_y);
  int stop = min(start + chunck, dim_im_in_y);
  uint8_t *pOut = out + start * ch_im_in;
  int8_t *pIn = data + start * ch_im_in;
  int dimension = (stop-start) * ch_im_in;
  v4s in;
  v4s in2;
  v4s mask =  (v4s) 0x00000000;
  for(int i=0; i < (dimension)>>2; i++)
  {
    in = *((v4s*) (pIn));
    *((v4s*) (pOut)) = max4(in,mask);
    pIn +=4;
    pOut +=4;
  }
  if (((dimension) & 0x3)!=0)
  {
    for(int i=0; i< (dimension & 0x3); i++)
    {
      if (*(pIn) < 0)
        *(pOut) = *(pIn) & 0x0;
      pIn+=1;
    }
  }
}
