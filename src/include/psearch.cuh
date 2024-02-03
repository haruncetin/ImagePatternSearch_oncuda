#ifndef __PSEARCH_H__
#define __PSEARCH_H__

#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <cstdint>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "img_util.h"

#define _abs(x) ((x) < 0 ? -(x) : (x))

__global__ void psearch_kernel(uint16_t *I, unsigned int N, uint16_t *P, unsigned int K);

#endif
