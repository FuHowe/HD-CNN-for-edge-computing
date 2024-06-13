#include "main.h"


float* lw_stft_fft(lw_head_t* head, float* stft_input)
{
		float pi = 3.1415926f;
	  uint32_t stft_input_len = (head->image_size - 1) * ((head->image_size *2)/2) + (head->image_size *2); 
		uint32_t window_len = head->image_size * 2; 
	  uint32_t shift_len = window_len/2; 
	
	  float* stft_output = (float*)calloc(head->image_size *sizeof(float) *head->image_size *sizeof(float), 1);


		arm_cfft_radix2_instance_f32 scfft; 
	  arm_cfft_radix2_init_f32(&scfft, window_len, 0, 1); 

    float *hamming_window = (float*)calloc(window_len * sizeof(float), 1); 
    for (uint32_t i = 0; i < window_len; i++)
        hamming_window[i] = (float)(0.5 -0.5 * cos(2 * pi * i / (window_len - 1)));

    for (uint32_t i = 0; i < (stft_input_len - window_len)/(shift_len) + 1; i++) 
    {
        float* window_data = (float*)calloc(window_len * sizeof(float), 1); 
        memmove(window_data, stft_input + i * shift_len, sizeof(float) * window_len); 
				float* hamming_data = (float*)calloc(2 * window_len * sizeof(float), 1); 
        for (uint32_t j = 0; j < window_len; j++) 
        {
					hamming_data[2*j] = window_data[j] * hamming_window[j];
        }

				arm_cfft_radix2_f32(&scfft, hamming_data);	
				float *fft_out = (float*)calloc(window_len * sizeof(float), 1);
   			arm_cmplx_mag_f32(hamming_data, fft_out, window_len);	
				
				for (uint32_t k = 0; k < window_len; k++) 
				{
					fft_out[k] = fft_out[k]/(window_len/2.0);
				}

        memmove(stft_output + i * window_len/2, fft_out, sizeof(float) * window_len/2); 
        free(window_data);
        free(hamming_data);
        free(fft_out);
    }

    free(hamming_window);
		
		return stft_output;
}

float* lw_stft_dft(lw_head_t* head, float* stft_input)
{
		float pi = 3.1415926f;
	  uint32_t stft_input_len = (head->image_size - 1) * ((head->image_size *2)/2) + (head->image_size *2); 
		uint32_t window_len = head->image_size *2; 
	  uint32_t shift_len = window_len/2; 
	
	  float* stft_output = (float*)calloc(head->image_size *sizeof(float) *head->image_size *sizeof(float), 1); 
	
    float *hamming_window = (float*)calloc(window_len * sizeof(float), 1); 
    for (uint32_t i = 0; i < window_len; i++)
        hamming_window[i] = (float)(0.5 -0.5 * cos(2 * pi * i / (window_len - 1)));

	
    for (uint32_t i = 0; i < (stft_input_len - window_len)/(shift_len) + 1; i++) 
    {
        float* window_data = (float*)calloc(window_len * sizeof(float), 1); 
        memmove(window_data, stft_input + i * shift_len, sizeof(float) * window_len); 

				float* hamming_data = (float*)calloc(window_len * sizeof(float), 1); 
        for (uint32_t j = 0; j < window_len; j++) 
        {
					hamming_data[j] = window_data[j] * hamming_window[j]; 
        }

				float* dft_out = (float*)calloc(window_len * sizeof(float), 1);
				for (uint32_t k = 0; k < window_len; k++)
				{
					float real = 0; 
					float imag = 0;
					for (uint32_t n = 0; n < window_len; n++)
					{
							real += (hamming_data[n] * cos(2 * pi * k * n / window_len));
							imag += (hamming_data[n] * sin(2 * pi * k * n / window_len));
					}
					dft_out[k] = sqrt(real * real + imag * imag)/(window_len/2); 
				}
				
				memmove(stft_output + i * window_len/2, dft_out, sizeof(float) * window_len/2); 

        free(window_data);
        free(hamming_data);
        free(dft_out);
    }

    free(hamming_window);
		
		return stft_output;
}
