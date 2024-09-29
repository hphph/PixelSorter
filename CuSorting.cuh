#pragma once

#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CImg.h"
#include "Sorting.h"

struct PixelSpan
{
	Pixel* pixels;
	int offset;
	int y;
	int length;
};

const int ThreadsPerBlock = 256;
__global__ void gpuBubbleSortPixels(PixelSpan* pixelRow, int channel);
__global__ void gpuOddEvenSortPixelRow(PixelSpan& pixelRow, int channel);
__global__ void gpuMergeSortPixelRow(PixelSpan& pixelRow, int channel);
__device__ void swapGpuPixels(Pixel& a, Pixel& b);
__device__ void merge(PixelSpan& pixelRow, int channel, int start, int mid, int end);
CImg<float> gpuPixelSortingBySpan(CImg<float>& input, float sensitivity, void (*sortingKernel)(PixelSpan* pixelRow, int channel));
CImg<float> gpuPixelSortingByPixelRow(CImg<float>& input, float sensitivity, void (*sortingKernel)(PixelSpan& pixelRow, int channel));