#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "CImg.h"

using namespace cimg_library;

struct Pixel
{
	float channel[3];
};



CImg<float> imageMask(CImg<float>& image, float sensitivity);
CImg<float> cpuPixelSorting(CImg<float>& input, float sensitivity, void (*sortByChannel)(std::vector<Pixel>& pixelRow, int channel));
void quickSortPixels(std::vector<Pixel>& pixelRow, int channel, int l, int r);
int quickSortPartition(std::vector<Pixel>& pixelRow, int channel, int l, int r);
void CPUBubbleSortByChannel(std::vector<Pixel>& pixelRow, int channel);
void CPUQuickSortByChannel(std::vector<Pixel>& pixelRow, int channel);
void swapPixels(Pixel& a, Pixel& b);
