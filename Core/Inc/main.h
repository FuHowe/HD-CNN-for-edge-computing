#ifndef __MIAN__
#define __MIAN__

	#include <stdio.h>
	#include <stdint.h>
	#include <stdlib.h>
	#include <math.h>
	#include <string.h>
	#include "lw_cnn.h"
	#include "lw_stft.h"
	#include "stm32h723xx.h" 
	#include "arm_math.h" 


	#define MCU
//  #define SD

#ifdef MCU
	#include "stm32h7xx_hal.h"
	#include "lw_init.h"
	#include "lw_usart.h"
#endif

#ifdef SD
	#include "stm32h7xx_hal.h"
	#include "lw_init.h"
	#include "lw_usart.h"
#endif

	#define FLASH_DATA_BASE_ADDR	0x08020000U 

#endif

