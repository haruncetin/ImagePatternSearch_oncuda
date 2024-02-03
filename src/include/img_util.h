#ifndef IMG_UTIL_H
#define IMG_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint16_t** read_png(char* filename, unsigned int* width, unsigned int* height);
uint16_t* read_png_1d(char* filename, unsigned int* width, unsigned int* height);

#ifdef __cplusplus
}
#endif

#endif
