#ifndef __LW_USART_H
#define __LW_USART_H

	#include "main.h"

	#define USART1_BaudRate               115200
	
	#define	USART1_TX_PORT								GPIOA
	#define	USART1_RX_PORT								GPIOA
	#define USART1_RX_PIN									GPIO_PIN_10 
	#define USART1_TX_PIN									GPIO_PIN_9

	#define GPIO_USART1_TX_CLK_ENABLE     __HAL_RCC_GPIOA_CLK_ENABLE()
	#define GPIO_USART1_RX_CLK_ENABLE     __HAL_RCC_GPIOA_CLK_ENABLE()

	void lw_usart1_init(void);	

#endif





