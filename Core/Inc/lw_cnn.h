#ifndef __LW_CNN__
#define __LW_CNN__

	#include "main.h"

	typedef enum
	{
		layer_1 = 1,
		layer_2 = 2,
		layer_3 = 3,
		linear_4 = 4
	}lw_layer_t;

	typedef enum
	{
		conv = 1,
		pool = 2,
	}lw_conv_pool_t;

	typedef struct
	{
		float head_len;
		float conv_layers; 
		float para_size;
		float image_size;

		float con_layer_input_dimension_1;
		float con_layer_output_dimension_1;
		float con_layer_kernel_1;
		float con_layer_pooling_1;

		float con_layer_input_dimension_2;
		float con_layer_output_dimension_2;
		float con_layer_kernel_2;
		float con_layer_pooling_2;

		float con_layer_input_dimension_3;
		float con_layer_output_dimension_3;
		float con_layer_kernel_3;
		float con_layer_pooling_3;

		float full_connect_len;
		float classification_out;
	}lw_head_t;


	float* lw_model(void);
	lw_head_t* lw_read_head(void);
	
	float* lw_read_image(lw_head_t* head);
	float* lw_read_signal(lw_head_t* head);

	float* lw_read_weight(lw_head_t* head, lw_layer_t layer);
	float* lw_read_bias(lw_head_t* head, lw_layer_t layer);

	float* lw_conv_computing(lw_head_t* head, float* input_images, lw_layer_t layer); 
	float* lw_relu_computing(lw_head_t* head, float* input_images, lw_layer_t layer);
	float* lw_pool_computing(lw_head_t* head, float* input_images, lw_layer_t layer);
	float* lw_line_computing(lw_head_t* head, float* input_images, lw_layer_t layer);

	float* lw_filter_conv_computing(lw_head_t* head, float* input_images, float* filter_weight, lw_layer_t layer);
	float* lw_channel_conv_computing(lw_head_t* head, float* channel_image, float* channel_weight, lw_layer_t layer);
	float* lw_channel_pool_computing(lw_head_t* head, float* channel_image, lw_layer_t layer);

	uint16_t lw_conv_pool_output_size(lw_head_t* head, lw_layer_t layer, lw_conv_pool_t conv_pool);
	uint16_t lw_conv_pool_input_size(lw_head_t* head, lw_layer_t layer, lw_conv_pool_t conv_pool);
	
	void lw_read_flash(float* read_buffer, uint32_t read_len, uint32_t seek);
	void lw_read_sd(float* read_buffer, uint32_t read_len, uint32_t seek);


#endif

