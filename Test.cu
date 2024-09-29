#pragma once

#include <cuda_runtime.h>
#include "Sorting.h"
#include "CuSorting.cuh"
#include <iostream>

int main()
{
	CImg<float> cpu_inputImage("input.bmp");
	CImg<float> gpu_inputImage("input.bmp");

	cpu_inputImage.resize_halfXY();
	cpu_inputImage.resize_halfXY();
	gpu_inputImage.resize_halfXY();
	gpu_inputImage.resize_halfXY();

	CImg<float> cpu_output = cpuPixelSorting(cpu_inputImage, 0.33, CPUQuickSortByChannel);
	//CImg<float> gpu_output = gpuPixelSortingBySpan(gpu_inputImage, 0.35, gpuBubbleSortPixels);
	CImg<float> gpu_output = gpuPixelSortingByPixelRow(gpu_inputImage, 0.33, gpuOddEvenSortPixelRow);
	CImg<float> mask = imageMask(cpu_inputImage, 0.33);
	cpu_output.save_bmp("cpu_out.bmp");
	gpu_output.save_bmp("gpu_out.bmp");
	CImgDisplay cpu_disp(cpu_output, "Posortowany obraz CPU");
	CImgDisplay gpu_disp(gpu_output, "Posortowany obraz GPU");
	CImgDisplay mask_disp(mask, "Maska");
    while (!cpu_disp.is_closed() && !mask_disp.is_closed() && !gpu_disp.is_closed()) 
	{
		cpu_disp.wait();
		gpu_disp.wait();
		mask_disp.wait();
    }
	return 0;
}