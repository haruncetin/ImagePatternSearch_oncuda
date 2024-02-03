#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "include/lodepng.h"

uint16_t** read_png(char* filename, unsigned int* width, unsigned int* height)
{
    uint16_t** array;
    unsigned char* image;
    
    unsigned error = lodepng_decode_file(&image, width, height, filename, LCT_GREY, 16);
    if (error)
    {
        fprintf(stderr, "Png reading error: %s\n", lodepng_error_text(error));
        exit(1);
    }

    array = (uint16_t**)malloc(*width * sizeof(uint16_t*));
    for (int i = 0; i < *width; i++)
    {
        array[i] = (uint16_t*)malloc(*height * sizeof(uint16_t));
        for(int j = 0; j < *height; j++)
        {
            // Get the gray value of each pixel as a 16-bit integer
            array[i][j] = (image[j * *width * 2 + i * 2] << 8) + image[j * *width * 2 + i * 2 + 1];
        }
    }

    free(image);

    return array;
}

uint16_t* read_png_1d(char* filename, unsigned int* width, unsigned int* height) {
    uint16_t* array;
    unsigned char* image;
    
    unsigned error = lodepng_decode_file(&image, width, height, filename, LCT_GREY, 16);
    if (error)
    {
        fprintf(stderr, "Png reading error: %s\n", lodepng_error_text(error));
        exit(1);
    }

    array = (uint16_t*)malloc(*width * *height * sizeof(uint16_t));
    for (int i = 0; i < *width; i++)
    {
        for(int j = 0; j < *height; j++)
        {
            // Get the gray value of each pixel as a 16-bit integer
            array[j * *width + i] = (image[j * *width * 2 + i * 2] << 8) + image[j * *width * 2 + i * 2 + 1];
        }
    }

    free(image);

    return array;
}