%if config.api == "PULPNNConvolve":
void ${config.fn_name}(
                        uint8_t *pIn,
                        uint8_t *pIm2ColBuffer,
                        int8_t *pBias,
                        uint8_t *pOut,
                        int8_t *pWeight,
%if config.kernel.act_prec == '32bit':
                        int32_t *pKappa,
                        int32_t *pLambda,
%elif config.kernel.act_prec == '64bit':
                        int64_t *pKappa,
                        int64_t *pLambda,
%endif
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t dim_in_x,
                        uint16_t dim_in_y,
                        uint16_t ch_in,
                        uint16_t dim_out_x,
                        uint16_t dim_out_y,
                        uint16_t ch_out,
                        uint16_t dim_kernel_x,
                        uint16_t dim_kernel_y,
                        uint16_t padding_y_top,
                        uint16_t padding_y_bottom,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t stride_y,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
%elif config.api == "PULPNNConvolvePointwise":
void ${config.fn_name}(
                        uint8_t *pIn,
                        uint8_t *pIm2ColBuffer,
                        int8_t *pBias,
                        uint8_t *pOut,
                        int8_t *pWeight,
%if config.kernel.act_prec == '32bit':
                        int32_t *pKappa,
                        int32_t *pLambda,
%elif config.kernel.act_prec == '64bit':
                        int64_t *pKappa,
                        int64_t *pLambda,
%endif
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t dim_in_x,
                        uint16_t dim_in_y,
                        uint16_t ch_in,
                        uint16_t dim_out_x,
                        uint16_t dim_out_y,
                        uint16_t ch_out,
                        uint16_t dim_kernel_x,
                        uint16_t dim_kernel_y,
                        uint16_t padding_y_top,
                        uint16_t padding_y_bottom,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t stride_y,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
% elif config.api=="PULPNNMatMul":
uint8_t *${config.fn_name}(
                        uint8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
                        uint8_t *pOut2,
                        int8_t *pWeight,
%if config.kernel.act_prec == '32bit':
                        int32_t *pKappa,
                        int32_t *pLambda,
%elif config.kernel.act_prec == '64bit':
                        int64_t *pKappa,
                        int64_t *pLambda,
%endif
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t num_col_im2col,
                        uint16_t ch_out,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
% elif config.api=="PULPNNConvolveDepthwise":
void ${config.fn_name}(
                        uint8_t *pIn,
                        uint8_t *pIm2ColBuffer,
                        int8_t *pBias,
                        uint8_t *pOut,
                        int8_t *pWeight,
                        int8_t *pWtBuffer,
%if config.kernel.act_prec == '32bit':
                        int32_t *pKappa,
                        int32_t *pLambda,
%elif config.kernel.act_prec == '64bit':
                        int64_t *pKappa,
                        int64_t *pLambda,
%endif
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t dim_in_x,
                        uint16_t dim_in_y,
                        uint16_t ch_in,
                        uint16_t dim_out_x,
                        uint16_t dim_out_y,
                        uint16_t ch_out,
                        uint16_t dim_kernel_x,
                        uint16_t dim_kernel_y,
                        uint16_t padding_y_top,
                        uint16_t padding_y_bottom,
                        uint16_t padding_x_left,
                        uint16_t padding_x_right,
                        uint16_t stride_x,
                        uint16_t stride_y,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
%elif config.api=="PULPNNLinearNoQuant":
void ${config.fn_name}(
                        uint8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
                        int8_t *pWeight,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons);
%elif config.api=="PULPNNLinearQuant":
void ${config.fn_name}(
                        uint8_t *pIn,
                        int8_t *pBias,
                        uint8_t *pOut,
                        int8_t *pWeight,
%if config.kernel.act_prec == '32bit':
                        int32_t *pKappa,
                        int32_t *pLambda,
%elif config.kernel.act_prec == '64bit':
                        int64_t *pKappa,
                        int64_t *pLambda,
%endif
                        uint16_t out_mul,
                        uint16_t out_shift,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons,
                        uint8_t flag_relu,
                        uint8_t flag_batchnorm);
%elif config.api=="PULPNNMaxPool":
void ${config.fn_name}(
                        uint8_t * pIn,
                        uint8_t * pOut,
                        uint16_t  dim_im_in_x,
                        uint16_t  dim_im_in_y,
                        uint16_t  ch_im_in,
                        uint16_t  dim_im_out_x,
                        uint16_t  dim_im_out_y,
                        uint16_t  dim_kernel_x,
                        uint16_t  dim_kernel_y,
                        uint16_t  padding_t,
                        uint16_t  padding_b,
                        uint16_t  padding_l,
                        uint16_t  padding_r,
                        uint16_t  stride_x,
                        uint16_t  stride_y);
%elif config.api=="PULPNNAvgPool":
void ${config.fn_name}(
                        uint8_t * pIn,
                        uint8_t * pOut,
                        uint16_t  dim_im_in_x,
                        uint16_t  dim_im_in_y,
                        uint16_t  ch_im_in,
                        uint16_t  dim_im_out_x,
                        uint16_t  dim_im_out_y,
                        uint16_t  dim_kernel_x,
                        uint16_t  dim_kernel_y,
                        uint16_t  padding_t,
                        uint16_t  padding_b,
                        uint16_t  padding_l,
                        uint16_t  padding_r,
                        uint16_t  stride_x,
                        uint16_t  stride_y);
%elif config.api=="PULPNNAdd":
void ${config.fn_name}(
                        uint8_t * pIn1,
                        uint8_t * pIn2,
                        uint8_t * pOut,
                        uint16_t out_mult1,
                        uint16_t out_mult2,
                        uint16_t out_shift,
                        uint16_t dim_im_in_x,
                        uint16_t dim_im_in_y,
                        uint16_t ch_im_in);
%endif
