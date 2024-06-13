import mycnn
import os
import numpy as np

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
    
    
    single = 1

    choice = 'CQU_1' 
    

    if single == 1:
        
        image_size = 32
        
        input_dim_1 = 1 
        output_dim_1 = 24 
        kernel_1 = 2 
        pooling_1 = 2 

        
        input_dim_2 = output_dim_1 
        output_dim_2 = 24 
        kernel_2 = 2 
        pooling_2 = 2 
    
        
        input_dim_3 = output_dim_2 
        output_dim_3 = 24 
        kernel_3 = 2 
        pooling_3 = 2 

        
        classification_out = 7 if choice == 'NTU' else 4 

        
        data_len = 10000 
        batch_size = 16 
        learning_rate = 0.01 
        epochs = 20 
    
        count = 1
        device = mycnn.lw_init() 
        _ = mycnn.lw_edge_computing(device, single, count, choice, image_size, data_len, batch_size, learning_rate, epochs, input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2, kernel_2, pooling_2, input_dim_3, output_dim_3, kernel_3, pooling_3, classification_out)
    else:
        
        image_size = 32 

        
        
        input_dim_1 = 1 
        output_dim_1 = 24 
        kernel_1 = 4 
        pooling_1 = 4 
            
        
        input_dim_2 = output_dim_1 
        output_dim_2 = 24 
        kernel_2 = 4 
        pooling_2 = 4 
    
        
        input_dim_3 = output_dim_2 
        output_dim_3 = 24 
        kernel_3 = 4 
        pooling_3 = 4 
    
        
        classification_out = 4 if choice == 'CQU' else 7
            
        
        data_len = 100000 
        batch_size = 4 
        learning_rate = 0.01 
        epochs = 20 

        
        dim_step = 8
        kernel_step = 1
        
        IMAGE_SIZE = list(np.uint32(np.arange(16, image_size + 16, 16)))  
        
        
        INPUT_DIM_1 = list(np.uint32(np.arange(1, input_dim_1 + 1, 1)))  
        OUTPUT_DIM_1 = list(np.uint32(np.arange(4, output_dim_1, dim_step)))  
        KERNEL_1 = list(np.uint32(np.arange(2, kernel_1 + kernel_step, kernel_step)))  
        POOLING_1 = list(np.uint32(np.arange(2, pooling_1 + kernel_step, kernel_step)))  
            
        
        INPUT_DIM_2 = OUTPUT_DIM_1  
        OUTPUT_DIM_2 = list(np.uint32(np.arange(4, output_dim_2, dim_step)))  
        KERNEL_2 = list(np.uint32(np.arange(2, kernel_2 + kernel_step, kernel_step)))  
        POOLING_2 = list(np.uint32(np.arange(2, pooling_2 + kernel_step, kernel_step)))  

        
        INPUT_DIM_3 = OUTPUT_DIM_2  
        OUTPUT_DIM_3 = list(np.uint32(np.arange(4, output_dim_3,  dim_step)))  
        KERNEL_3 = list(np.uint32(np.arange(2, kernel_3 + kernel_step, kernel_step)))  
        POOLING_3 = list(np.uint32(np.arange(2, pooling_3 + kernel_step, kernel_step)))  
        
        print(len(IMAGE_SIZE)*len(INPUT_DIM_1)*len(OUTPUT_DIM_1)*len(KERNEL_1)*len(POOLING_1)*len(OUTPUT_DIM_2)*len(KERNEL_2)*len(POOLING_2)*len(OUTPUT_DIM_3)*len(KERNEL_3)*len(POOLING_3))
        
        
        count = 1
        device = mycnn.lw_init() 
        for s in range(0, len(IMAGE_SIZE)):
            for i in range(0, len(INPUT_DIM_1)):
                for j in range(0, len(OUTPUT_DIM_1)):
                    for k in range(0, len(KERNEL_1)):
                        for l in range(0, len(POOLING_1)):
                            for m in range(0, len(OUTPUT_DIM_2)):
                                for n in range(0, len(KERNEL_2)):
                                    for o in range(0, len(POOLING_2)):
                                        for p in range(0, len(OUTPUT_DIM_3)):
                                            for q in range(0, len(KERNEL_3)):
                                                for r in range(0, len(POOLING_3)):
                                                    
                                                    if -1 == mycnn.lw_edge_computing(device, single, count, choice, IMAGE_SIZE[s], data_len, batch_size, learning_rate, epochs, INPUT_DIM_1[i], OUTPUT_DIM_1[j], KERNEL_1[k], POOLING_1[l], OUTPUT_DIM_1[j], OUTPUT_DIM_2[m], KERNEL_2[n], POOLING_2[o], OUTPUT_DIM_2[m], OUTPUT_DIM_3[p], KERNEL_3[q], POOLING_3[r], classification_out):
                                                        continue
                                                    else:
                                                        count += 1
                                                        print(count)

