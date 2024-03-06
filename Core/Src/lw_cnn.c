#include "main.h"

#ifndef MCU
	char* file_name = "1_weight_bias_image.bin";
#endif

float* lw_model(void)
{
	lw_head_t* head = lw_read_head();		
	float* test_data = lw_read_signal(head);	
	float* stft_data = lw_stft_fft(head, test_data);	
	float* conv_out_1 = lw_conv_computing(head, stft_data, layer_1);	
	float* relu_out_1 = lw_relu_computing(head, conv_out_1, layer_1);	
	float* pool_out_1 = lw_pool_computing(head, relu_out_1, layer_1);
	float* conv_out_2 = lw_conv_computing(head, pool_out_1, layer_2);
	float* relu_out_2 = lw_relu_computing(head, conv_out_2, layer_2);
	float* pool_out_2 = lw_pool_computing(head, relu_out_2, layer_2);

	float* conv_out_3 = lw_conv_computing(head, pool_out_2, layer_3);
	float* relu_out_3 = lw_relu_computing(head, conv_out_3, layer_3);
	float* pool_out_3 = lw_pool_computing(head, relu_out_3, layer_3);

	float* linear_out = lw_line_computing(head, pool_out_3, linear_4);
	return linear_out;
}

lw_head_t* lw_read_head()
{
	lw_head_t* head = (lw_head_t*)calloc(sizeof(lw_head_t), 1);
	if (head != NULL)
	{
#ifdef MCU
		lw_read_flash((float*)head, sizeof(lw_head_t) / sizeof(float), 0);
#else
		lw_read_sd((float*)head, sizeof(lw_head_t) / sizeof(float), 0);
#endif
		printf("%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d\r\n", (uint32_t)head->con_layer_input_dimension_1, (uint32_t)head->con_layer_output_dimension_1, (uint32_t)head->con_layer_kernel_1, (uint32_t)head->con_layer_pooling_1,
			                                              (uint32_t)head->con_layer_input_dimension_2, (uint32_t)head->con_layer_output_dimension_2, (uint32_t)head->con_layer_kernel_2, (uint32_t)head->con_layer_pooling_2,
			                                              (uint32_t)head->con_layer_input_dimension_3, (uint32_t)head->con_layer_output_dimension_3, (uint32_t)head->con_layer_kernel_3, (uint32_t)head->con_layer_pooling_3);
	}
	else
	{
		printf("Calloc head fail!\r\n");
	}

	return head;
}

float* lw_read_image(lw_head_t* head)
{
	uint32_t seek = (uint32_t)(sizeof(float) * (head->head_len + head->para_size));

	float* image = (float*)calloc((uint32_t)(sizeof(float) * (head->image_size * head->image_size)), 1);

	if (image != NULL)
	{
#ifdef MCU
		lw_read_flash(image, head->image_size *head->image_size, seek);
#else 
		lw_read_sd(image, head->image_size *head->image_size, seek);
#endif 
	}
	else
	{
		printf("Calloc image fail!\r\n");
	}

	return image;
}


float* lw_read_signal(lw_head_t* head)
{
	uint32_t seek = (uint32_t)(sizeof(float) * (head->head_len + head->para_size));

	uint32_t signal_input_len = (head->image_size - 1) * ((head->image_size *2)/2) + (head->image_size *2);	
	float *signal = (float*)calloc(signal_input_len * sizeof(float), 1);	
	if (signal != NULL)
	{
#ifdef MCU
		lw_read_flash(signal, signal_input_len, seek);
#else 
		lw_read_sd(signal, head->image_size *head->image_size, seek);
#endif 
	}
	else
	{
		printf("Calloc image fail!\r\n");
	}

	return signal;
}

float* lw_read_weight(lw_head_t* head, lw_layer_t layer)
{
	uint32_t seek = 0;	uint32_t weight_size = 0;

	switch (layer)
	{
		case layer_1:
			weight_size = (uint32_t)(sizeof(float) * (head->con_layer_output_dimension_1 * head->con_layer_input_dimension_1 * head->con_layer_kernel_1 * head->con_layer_kernel_1));
			seek = (uint32_t)(sizeof(float) * head->head_len);
			break;

		case layer_2:
			weight_size = (uint32_t)(sizeof(float) * (head->con_layer_output_dimension_2 * head->con_layer_input_dimension_2 * head->con_layer_kernel_2 * head->con_layer_kernel_2));
			seek = (uint32_t)(sizeof(float) * (head->head_len + head->con_layer_output_dimension_1 * head->con_layer_input_dimension_1 * head->con_layer_kernel_1 * head->con_layer_kernel_1 + head->con_layer_output_dimension_1));
			break;

		case layer_3:
			weight_size = (uint32_t)(sizeof(float) * (head->con_layer_output_dimension_3 * head->con_layer_input_dimension_3 * head->con_layer_kernel_3 * head->con_layer_kernel_3));
			seek = (uint32_t)(sizeof(float) * (head->head_len + head->con_layer_output_dimension_1 * head->con_layer_input_dimension_1 * head->con_layer_kernel_1 * head->con_layer_kernel_1 + head->con_layer_output_dimension_1 +
				head->con_layer_output_dimension_2 * head->con_layer_input_dimension_2 * head->con_layer_kernel_2 * head->con_layer_kernel_2 + head->con_layer_output_dimension_2));
			break;

		case linear_4:
			weight_size = (uint32_t)(sizeof(float) * (head->full_connect_len * head->classification_out));
			seek = (uint32_t)(sizeof(float) * (head->head_len + head->con_layer_output_dimension_1 * head->con_layer_input_dimension_1 * head->con_layer_kernel_1 * head->con_layer_kernel_1 + head->con_layer_output_dimension_1 +
				head->con_layer_output_dimension_2 * head->con_layer_input_dimension_2 * head->con_layer_kernel_2 * head->con_layer_kernel_2 + head->con_layer_output_dimension_2 +
				head->con_layer_output_dimension_3 * head->con_layer_input_dimension_3 * head->con_layer_kernel_3 * head->con_layer_kernel_3 + head->con_layer_output_dimension_3));
			break;
	}

	float* weight_data = (float*)calloc(weight_size, 1);
	if (weight_data != NULL)
	{
#ifdef MCU
		lw_read_flash(weight_data, weight_size/sizeof(float), seek);
#else
		lw_read_sd(weight_data, weight_size/sizeof(float), seek);
#endif 
	}

	return weight_data;
}

float* lw_read_bias(lw_head_t* head, lw_layer_t layer)
{
	uint32_t seek = 0;	uint32_t bias_size = 0;	uint32_t wieght_size = 0;
	switch (layer)
	{
		case layer_1:
			bias_size = (uint32_t)(sizeof(float) * (head->con_layer_output_dimension_1));
			wieght_size = (uint32_t)(sizeof(float) * (head->con_layer_output_dimension_1 * head->con_layer_input_dimension_1 * head->con_layer_kernel_1 * head->con_layer_kernel_1));
			seek = wieght_size + (uint32_t)(sizeof(float) * head->head_len);
			break;

		case layer_2:
			bias_size = (uint32_t)(sizeof(float) * (head->con_layer_output_dimension_2));
			wieght_size = (uint32_t)(sizeof(float) * (head->con_layer_output_dimension_2 * head->con_layer_input_dimension_2 * head->con_layer_kernel_2 * head->con_layer_kernel_2));
			seek = wieght_size + (uint32_t)(sizeof(float) * (head->head_len + head->con_layer_output_dimension_1 * head->con_layer_input_dimension_1 * head->con_layer_kernel_1 * head->con_layer_kernel_1 + head->con_layer_output_dimension_1));
			break;

		case layer_3:
			bias_size = (uint32_t)(sizeof(float) * (head->con_layer_output_dimension_3));
			wieght_size = (uint32_t)(sizeof(float) * (head->con_layer_output_dimension_3 * head->con_layer_input_dimension_3 * head->con_layer_kernel_3 * head->con_layer_kernel_3));
			seek = wieght_size + (uint32_t)(sizeof(float) * (head->head_len + head->con_layer_output_dimension_1 * head->con_layer_input_dimension_1 * head->con_layer_kernel_1 * head->con_layer_kernel_1 + head->con_layer_output_dimension_1 +
			head->con_layer_output_dimension_2 * head->con_layer_input_dimension_2 * head->con_layer_kernel_2 * head->con_layer_kernel_2 + head->con_layer_output_dimension_2));
			break;

		case linear_4:
			bias_size = (uint32_t)(sizeof(float) * (head->classification_out));
			wieght_size = (uint32_t)(sizeof(float) * (head->full_connect_len * head->classification_out));
			seek = wieght_size + (uint32_t)(sizeof(float) * (head->head_len + head->con_layer_output_dimension_1 * head->con_layer_input_dimension_1 * head->con_layer_kernel_1 * head->con_layer_kernel_1 + head->con_layer_output_dimension_1 +
			head->con_layer_output_dimension_2 * head->con_layer_input_dimension_2 * head->con_layer_kernel_2 * head->con_layer_kernel_2 + head->con_layer_output_dimension_2 +
			head->con_layer_output_dimension_3 * head->con_layer_input_dimension_3 * head->con_layer_kernel_3 * head->con_layer_kernel_3 + head->con_layer_output_dimension_3));
			break;
	}

	float* bias = (float*)calloc(bias_size, 1);
	if (bias != NULL)
	{
#ifdef MCU
		lw_read_flash(bias, bias_size/sizeof(float), seek);
#else
		lw_read_sd(bias, bias_size/sizeof(float), seek);
#endif 
	}
	else
	{
		printf("Malloc bias_out fail!\r\n");
	}

	return bias;
}

float* lw_conv_computing(lw_head_t* head, float* input_images, lw_layer_t layer)
{
	uint16_t filter_nums = 0;	uint16_t channel_nums = 0;	uint8_t kernel_size = 0;
	switch (layer)
	{
	case layer_1:
		filter_nums = (uint32_t)head->con_layer_output_dimension_1;
		channel_nums = (uint32_t)head->con_layer_input_dimension_1;
		kernel_size = (uint8_t)head->con_layer_kernel_1;
		break;

	case layer_2:
		filter_nums = (uint32_t)head->con_layer_output_dimension_2;
		channel_nums = (uint32_t)head->con_layer_input_dimension_2;
		kernel_size = (uint8_t)head->con_layer_kernel_2;
		break;

	case layer_3:
		filter_nums = (uint32_t)head->con_layer_output_dimension_3;
		channel_nums = (uint32_t)head->con_layer_input_dimension_3;
		kernel_size = (uint8_t)head->con_layer_kernel_3;
		break;
	
	case linear_4:
		break;
	}

	float* conv_weights = lw_read_weight(head, layer);	float* bias = lw_read_bias(head, layer);
	uint32_t output_image_size = lw_conv_pool_output_size(head, layer, conv);
	uint32_t filters_images_size = (uint32_t)(sizeof(float) * filter_nums * output_image_size * output_image_size);
	float* output_filters_images = (float*)calloc(filters_images_size, 1);
	if (output_filters_images != NULL)
	{
		for (uint16_t i = 0; i < filter_nums; i++)		{
			uint32_t weight_size = (uint32_t)(sizeof(float) * channel_nums * kernel_size * kernel_size);			
			float* filter_weight = (float*)calloc(weight_size, 1);			
			memmove(filter_weight, conv_weights + (i * weight_size / sizeof(float)), weight_size);
			float* convd_filter_image = lw_filter_conv_computing(head, input_images, filter_weight, layer);
			for (uint32_t j = 0; j < output_image_size * output_image_size; j++)
			{
				convd_filter_image[j] += bias[i];
			}

			uint32_t image_mem_size = (uint32_t)(sizeof(float) * output_image_size * output_image_size);			
			memmove(output_filters_images + (i * image_mem_size / sizeof(float)), convd_filter_image, image_mem_size);
			free(filter_weight);
			free(convd_filter_image);
		}
		free(conv_weights);
		free(bias);
	}
	else
	{
		printf("Calloc output_filters_images fail!\r\n");
	}
	free(input_images);

	return output_filters_images;
}

float* lw_filter_conv_computing(lw_head_t* head, float* input_images, float* filter_weights, lw_layer_t layer)
{
	uint16_t channel_nums = 0;	uint8_t kernel_size = 0;
	switch (layer)
	{
	case layer_1:
		channel_nums = (uint16_t)head->con_layer_input_dimension_1;
		kernel_size = (uint8_t)head->con_layer_kernel_1;
		break;

	case layer_2:
		channel_nums = (uint16_t)head->con_layer_input_dimension_2;
		kernel_size = (uint8_t)head->con_layer_kernel_2;
		break;

	case layer_3:
		channel_nums = (uint16_t)head->con_layer_input_dimension_3;
		kernel_size = (uint8_t)head->con_layer_kernel_3;
		break;
	
	case linear_4:
		break;
	}

	uint32_t output_image_size = lw_conv_pool_output_size(head, layer, conv);	
	uint32_t output_filter_image_size = (uint32_t)(sizeof(float) * output_image_size * output_image_size);
	float* output_filter_image = (float*)calloc(output_filter_image_size, 1);	
	if (output_filter_image != NULL)
	{
		for (uint16_t i = 0; i < channel_nums; i++)
		{
			uint32_t weight_size = (uint32_t)(sizeof(float) * kernel_size * kernel_size);			
			float* channel_weight = (float*)calloc(weight_size, 1);			
			memmove(channel_weight, filter_weights + (i * weight_size / sizeof(float)), weight_size);

			uint32_t input_image_size = lw_conv_pool_input_size(head, layer, conv);			
			uint32_t channel_image_size = (uint32_t)(sizeof(float) * input_image_size * input_image_size);
			float* channel_image = (float*)calloc(channel_image_size, 1);			
			memmove(channel_image, input_images + (i * channel_image_size / sizeof(float)), channel_image_size);
			float* convd_channel_image = lw_channel_conv_computing(head, channel_image, channel_weight, layer);
			for (uint32_t j = 0; j < output_image_size * output_image_size; j++)
			{
				output_filter_image[j] = output_filter_image[j] + convd_channel_image[j];
			}
			free(convd_channel_image);
			free(channel_image);
			free(channel_weight);
		}
	}
	else
	{
		printf("Calloc output_filter_image fail!\r\n");
	}

	return output_filter_image;
}


float* lw_channel_conv_computing(lw_head_t* head, float* channel_image, float* channel_weight, lw_layer_t layer)
{
	uint8_t conv_kernel_size = 0;	uint32_t input_image_size = 0;	uint32_t output_image_size = 0;
	switch (layer)
	{
		case layer_1:
			conv_kernel_size = (uint32_t)head->con_layer_kernel_1;			
			input_image_size = (uint32_t)head->image_size;		
			output_image_size = lw_conv_pool_output_size(head, layer_1, conv);	
		break;

		case layer_2:
			conv_kernel_size = (uint32_t)head->con_layer_kernel_2;
			input_image_size = lw_conv_pool_input_size(head, layer_2, conv);
			output_image_size = lw_conv_pool_output_size(head, layer_2, conv);
			break;

		case layer_3:
			conv_kernel_size = (uint32_t)head->con_layer_kernel_3;
			input_image_size = lw_conv_pool_input_size(head, layer_3, conv);
			output_image_size = lw_conv_pool_output_size(head, layer_3, conv);
			break;
	}

	float* output_channel_image = (float*)calloc(sizeof(float) * output_image_size * output_image_size, 1);	if (output_channel_image != NULL)
	{
		for (uint32_t m = 0, n = 0; m < input_image_size * (input_image_size - conv_kernel_size + 1); m++)		
		{
			float conv_sum_value = 0;			uint32_t image_n = m;
			if ((image_n % input_image_size) < (input_image_size - conv_kernel_size + 1))			
				{
				for (uint32_t weight_k = 0; weight_k < (uint32_t)(conv_kernel_size * conv_kernel_size); weight_k++)				
				{
					conv_sum_value += channel_image[image_n] * channel_weight[weight_k];

					weight_k++;
					if (((weight_k % conv_kernel_size) != 0) || (weight_k < conv_kernel_size))					
					{
						image_n = image_n + 1;
					}
					else
					{
						image_n = image_n + input_image_size - conv_kernel_size + 1;
					}
					weight_k--;				
				}

				output_channel_image[n] = conv_sum_value;
				n++;
			}
		}
	}
	else
	{
		printf("Malloc output_channel_image fail!\r\n");
	}

	return output_channel_image;
}

float* lw_relu_computing(lw_head_t* head, float* input_images, lw_layer_t layer)
{
	uint16_t filter_nums = 0;
	switch (layer)
	{
	case layer_1:
		filter_nums = (uint32_t)head->con_layer_output_dimension_1;
		break;

	case layer_2:
		filter_nums = (uint32_t)head->con_layer_output_dimension_2;
		break;

	case layer_3:
		filter_nums = (uint32_t)head->con_layer_output_dimension_3;
		break;
	}

	uint32_t output_image_size = lw_conv_pool_output_size(head, layer, conv);
	for (uint16_t i = 0; i < filter_nums * output_image_size * output_image_size; i++)
	{
		if (input_images[i] > 0)
		{
			input_images[i] = input_images[i];
		}
		else
		{
			input_images[i] = 0;
		}
	}

	return input_images;
}


float* lw_pool_computing(lw_head_t* head, float* input_images, lw_layer_t layer)
{
	uint16_t channel_nums = 0;
	switch (layer)
	{
	case layer_1:
		channel_nums = (uint32_t)head->con_layer_output_dimension_1;
		break;

	case layer_2:
		channel_nums = (uint32_t)head->con_layer_output_dimension_2;
		break;

	case layer_3:
		channel_nums = (uint32_t)head->con_layer_output_dimension_3;
		break;
	}

	uint32_t output_image_size = lw_conv_pool_output_size(head, layer, pool);
	uint32_t filters_images_size = (uint32_t)(sizeof(float) * channel_nums * output_image_size * output_image_size);

	float* output_filters_images = (float*)calloc(filters_images_size, 1);
	if (output_filters_images != NULL)
	{
		uint32_t input_image_size = lw_conv_pool_input_size(head, layer, pool);		
		uint32_t channel_image_size = (uint32_t)(sizeof(float) * input_image_size * input_image_size);

		for (uint16_t i = 0; i < channel_nums; i++)
		{
			float* channel_image = (float*)calloc(channel_image_size, 1);			
			memmove(channel_image, input_images + (i * channel_image_size / sizeof(float)), channel_image_size);
			float* pooled_channel_image = lw_channel_pool_computing(head, channel_image, layer);

			uint32_t image_mem_size = (uint32_t)(sizeof(float) * output_image_size * output_image_size);		
			memmove(output_filters_images + (i * image_mem_size / sizeof(float)), pooled_channel_image, image_mem_size);
			free(pooled_channel_image);
			free(channel_image);
		}
	}
	else
	{
		printf("Malloc output_filters_images fail!\r\n");
	}

	free(input_images);

	return output_filters_images;
}

float* lw_channel_pool_computing(lw_head_t* head, float* channel_image, lw_layer_t layer)
{
	uint8_t pool_kernel_size = 0;	uint32_t input_image_size = lw_conv_pool_input_size(head, layer, pool);	
	uint32_t output_image_size = lw_conv_pool_output_size(head, layer, pool);
	switch (layer)
	{
		case layer_1:
			pool_kernel_size = (uint32_t)head->con_layer_pooling_1;			break;

		case layer_2:
			pool_kernel_size = (uint32_t)head->con_layer_pooling_2;
			break;

		case layer_3:
			pool_kernel_size = (uint32_t)head->con_layer_pooling_3;
			break;
	}

	uint16_t times = (uint16_t)(input_image_size / pool_kernel_size);
	float* output_pooled_channel_image = (float*)calloc(sizeof(float) * output_image_size * output_image_size, 1);	
	if (output_pooled_channel_image != NULL)
	{
		uint32_t t = 0;
		for (uint32_t m = 0; m < times; m++)
		{
			for (uint32_t n = 0; n < times; n++)
			{
				uint32_t image_n = m * pool_kernel_size * input_image_size + n * pool_kernel_size;
				float max_value = 0;			
				for (uint32_t k = 0; k < (uint32_t)(pool_kernel_size * pool_kernel_size); k++)			
				{

					if (channel_image[image_n] > max_value)
					{
						max_value = channel_image[image_n];
					}

					k++;					
					if (((k % pool_kernel_size) != 0) || (k < pool_kernel_size))					
					{
						image_n = image_n + 1;
					}
					else
					{
						image_n = image_n + input_image_size - pool_kernel_size + 1;				
					}
					k--;			
				}
				output_pooled_channel_image[t] = max_value;
				t++;
			}
		}

	}
	else
	{
		printf("Malloc output_pooled_channel_image fail!\r\n");
	}

	return output_pooled_channel_image;
}

float* lw_line_computing(lw_head_t* head, float* input_images, lw_layer_t layer)
{
	uint16_t row = (uint16_t)head->classification_out;	
	uint16_t col = (uint16_t)head->full_connect_len;;
	float* line_out = (float*)calloc(sizeof(float) * row, 1);
	if (line_out != NULL)
	{
		float* line_weights = lw_read_weight(head, layer);
		for (uint32_t i = 0; i < row; i++)	
		{
			float sum = 0;			for (uint32_t j = 0; j < col; j++)	
			{
				sum += line_weights[i * col + j] * input_images[j];
			}
			line_out[i] = sum;
		}
		free(input_images);

		float* bias = lw_read_bias(head, layer);
		free(head);

		for (uint32_t i = 0; i < row; i++)
		{
			line_out[i] += bias[i];
		}
		free(bias);
	}
	else
	{
		printf("Malloc line_out fail!\r\n");
	}

	return line_out;
}

uint16_t lw_conv_pool_output_size(lw_head_t* head, lw_layer_t layer, lw_conv_pool_t conv_pool)
{
	uint8_t conv_kernel_size = 0;
	uint8_t pool_kernel_size = 0;

	uint16_t output_image_size = (uint8_t)head->image_size;

	for (lw_layer_t i = (lw_layer_t)1; i < (lw_layer_t)(layer + 1); i++)
	{
		switch ((lw_layer_t)i)
		{
			case layer_1:
				conv_kernel_size = (uint8_t)head->con_layer_kernel_1;
				pool_kernel_size = (uint8_t)head->con_layer_pooling_1;
				break;

			case layer_2:
				conv_kernel_size = (uint8_t)head->con_layer_kernel_2;
				pool_kernel_size = (uint8_t)head->con_layer_pooling_2;
				break;

			case layer_3:
				conv_kernel_size = (uint8_t)head->con_layer_kernel_3;
				pool_kernel_size = (uint8_t)head->con_layer_pooling_3;
				break;
		}

		if (i != layer)		{
			output_image_size = (uint8_t)floor((output_image_size + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / (float)1 + 1);
			output_image_size = (uint8_t)floor((output_image_size + 2 * 0 - 1 * (pool_kernel_size - 1) - 1) / (float)pool_kernel_size + 1);
		}
		else
		{	
			if (conv_pool == conv)			{
				output_image_size = (uint8_t)floor((output_image_size + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / (float)1 + 1);
			}
			else			{
				output_image_size = (uint8_t)floor((output_image_size + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / (float)1 + 1);
				output_image_size = (uint8_t)floor((output_image_size + 2 * 0 - 1 * (pool_kernel_size - 1) - 1) / (float)pool_kernel_size + 1);
			}
		}
	}

	return output_image_size;
}

uint16_t lw_conv_pool_input_size(lw_head_t* head, lw_layer_t layer, lw_conv_pool_t conv_pool)
{
	uint8_t conv_kernel_size = 0;
	uint8_t pool_kernel_size = 0;

	uint16_t output_image_size = 0;
	if (conv_pool == conv) 
	{
		uint16_t image_size = 0;
		if (layer == layer_1)		{
			image_size = (uint16_t)head->image_size;
		}
		else		{
			image_size = (uint16_t)head->image_size;

			for (lw_layer_t i = (lw_layer_t)1; i < (lw_layer_t)(layer + 1 - 1); i++)
			{
				switch (i)
				{
					case layer_1:
						conv_kernel_size = (uint8_t)head->con_layer_kernel_1;
						pool_kernel_size = (uint8_t)head->con_layer_pooling_1;
						break;

					case layer_2:
						conv_kernel_size = (uint8_t)head->con_layer_kernel_2;
						pool_kernel_size = (uint8_t)head->con_layer_pooling_2;
						break;

					case layer_3:
						conv_kernel_size = (uint8_t)head->con_layer_kernel_3;
						pool_kernel_size = (uint8_t)head->con_layer_pooling_3;
						break;
				}

				image_size = (uint16_t)floor((image_size + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / (float)1 + 1);			
  			image_size = (uint16_t)floor((image_size + 2 * 0 - 1 * (pool_kernel_size - 1) - 1) / (float)pool_kernel_size + 1);			}
		}

		output_image_size = image_size;
	}
	else	{
		uint16_t image_size = (uint16_t)head->image_size;;
		for (lw_layer_t i = (lw_layer_t)1; i < (lw_layer_t)(layer + 1); i++)
		{
			switch (i)
			{
				case layer_1:
					conv_kernel_size = (uint8_t)head->con_layer_kernel_1;
					pool_kernel_size = (uint8_t)head->con_layer_pooling_1;
					break;

				case layer_2:
					conv_kernel_size = (uint8_t)head->con_layer_kernel_2;
					pool_kernel_size = (uint8_t)head->con_layer_pooling_2;
					break;

				case layer_3:
					conv_kernel_size = (uint8_t)head->con_layer_kernel_3;
					pool_kernel_size = (uint8_t)head->con_layer_pooling_3;
					break;
			}

			if (i == layer)		
			{
				image_size = (uint8_t)floor((image_size + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / (float)1 + 1);		
			}
			else
			{
				image_size = (uint8_t)floor((image_size + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / (float)1 + 1);
				image_size = (uint8_t)floor((image_size + 2 * 0 - 1 * (pool_kernel_size - 1) - 1) / (float)pool_kernel_size + 1);		
			}
		}

		output_image_size = image_size;
	}

	return output_image_size;
}

void lw_read_flash(float* read_buffer, uint32_t read_len, uint32_t seek)
{
	uint32_t flash_addr = (uint32_t)FLASH_DATA_BASE_ADDR + seek;
	for (uint32_t i = 0; i < read_len; i++)
	{
		read_buffer[i] = *(float*)(flash_addr);
		flash_addr += 4;	}
}

void lw_read_sd(float* read_buffer, uint32_t read_len, uint32_t seek)
{
	
}

