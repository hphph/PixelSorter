#include "CuSorting.cuh"
#include <stdio.h>

__global__ void gpuBubbleSortPixels(PixelSpan* pixelSpans, int channel)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	PixelSpan ps = pixelSpans[idx];

	bool isSwapped = true;

	while (isSwapped)
	{
		isSwapped = false;
		for (int i = 1; i < ps.length; i++)
		{
			if (ps.pixels[i].channel[channel] < ps.pixels[i - 1].channel[channel])
			{
				isSwapped = true;
				swapGpuPixels(ps.pixels[i], ps.pixels[i - 1]);
			}
		}
	}
}

__global__ void gpuOddEvenSortPixelRow(PixelSpan& pixelRow, int channel)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	bool isEven = true;
	__shared__ bool isSortedOdd;
	__shared__ bool isSortedEven;

	isSortedOdd = false;
	isSortedEven = false;

	while (!isSortedOdd || !isSortedEven)
	{
		idx = threadIdx.x + blockIdx.x * blockDim.x;
		isSortedEven = true;
		isSortedOdd = true;
		if (isEven)
		{
			idx *= 2;
			if (idx + 1 < pixelRow.length && pixelRow.pixels[idx].channel[channel] > pixelRow.pixels[idx + 1].channel[channel])
			{
				isSortedEven = false;
				swapGpuPixels(pixelRow.pixels[idx], pixelRow.pixels[idx + 1]);
			}
		}
		else
		{
			idx = idx*2 + 1;
			if (idx + 1 < pixelRow.length && pixelRow.pixels[idx].channel[channel] > pixelRow.pixels[idx + 1].channel[channel])
			{
				isSortedOdd = false;
				swapGpuPixels(pixelRow.pixels[idx], pixelRow.pixels[idx + 1]);
			}
		}
		__syncthreads();
		isEven = !isEven;
	}
}

__global__ void gpuMergeSortPixelRow(PixelSpan& pixelRow, int channel)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int j = idx * 2;
	int maxLen = 2;
	int currentLayer = 1;

	while (currentLayer < pixelRow.length && j % (currentLayer*2) == 0 && j < pixelRow.length)
	{
		merge(pixelRow, channel, j, j + currentLayer, j - 1 + 2*currentLayer);
		currentLayer *= 2;
		__syncthreads();
	}
	__syncthreads();
	if (j == 0)
	{
		for (int i = 0; i < pixelRow.length; i++)
		{
			printf("%f ", pixelRow.pixels[i].channel[channel]);
		}
	}
}

__device__ void merge(PixelSpan& pixelRow, int channel, int start, int mid, int end)
{
	int start2 = mid;
	int index;
	Pixel value;

	if (pixelRow.pixels[mid - 1].channel[channel] <= pixelRow.pixels[mid].channel[channel])
	{
		return;
	}
	
	while (start < mid && start2 <= end)
	{
		
		if (pixelRow.pixels[start].channel[channel] <= pixelRow.pixels[start2].channel[channel])
		{
			start++;
		}
		else
		{
			value = pixelRow.pixels[start2];
			index = start2;
			
			while (index != start)
			{
				pixelRow.pixels[index] = pixelRow.pixels[index - 1];
				index--;
			}
			pixelRow.pixels[index] = value;

			start++;
			start2++;
			mid++;
		}

	}
	 
}

__device__ void swapGpuPixels(Pixel& a, Pixel& b)
{
	Pixel c = a;
	a = b;
	b = c;
}

CImg<float> gpuPixelSortingBySpan(CImg<float>& input, float sensitivity, void (*sortingKernel)(PixelSpan* pixelRow, int channel))
{
	input.RGBtoHSL();
	std::vector<Pixel> pixelRow;
	std::vector<PixelSpan> pixelSpans;
	int spanOffset = 0;
	cudaEvent_t sortStart, sortEnd;
	cudaEventCreate(&sortStart);
	cudaEventCreate(&sortEnd);
	cudaEventRecord(sortStart);

	for (int y = 0; y < input.height(); y++)
	{
		for (int x = 0; x < input.width(); x++)
		{
			if (input(x, y, 0, 2) < 1.0 - sensitivity && input(x, y, 0, 2) > sensitivity)
			{
				if (pixelRow.size() == 0)
				{
					spanOffset = x;
				}

				pixelRow.push_back({ input(x, y, 0, 0), input(x, y, 0, 1), input(x, y, 0, 2) });
			}
			else
			{
				if (pixelRow.size() > 0)
				{
					Pixel* pixels = (Pixel*)malloc(sizeof(Pixel) * pixelRow.size());
					std::copy(pixelRow.begin(), pixelRow.end(), pixels);
					pixelSpans.push_back({ pixels, spanOffset, y, (int)pixelRow.size() });
					pixelRow.clear();
				}
			}
		}
		if (pixelRow.size() > 0)
		{
			Pixel* pixels = new Pixel[pixelRow.size()];
			std::copy(pixelRow.begin(), pixelRow.end(), pixels);
			pixelSpans.push_back({ pixels, spanOffset, y, (int)pixelRow.size() });
			pixelRow.clear();
		}
	}

	PixelSpan* hostSpans = new PixelSpan[pixelSpans.size()];
	PixelSpan* hostSortedSpans = new PixelSpan[pixelSpans.size()];
	std::copy(pixelSpans.begin(), pixelSpans.end(), hostSortedSpans);
	for (int i = 0; i < pixelSpans.size(); i++)
	{
		hostSortedSpans[i].pixels = new Pixel[pixelSpans[i].length];
	}

	std::copy(pixelSpans.begin(), pixelSpans.end(), hostSpans);
	for (int i = 0; i < pixelSpans.size(); i++)
	{
		cudaMalloc(&(hostSpans[i].pixels), sizeof(Pixel) * pixelSpans[i].length);
		cudaMemcpy(hostSpans[i].pixels, pixelSpans[i].pixels, sizeof(Pixel) * pixelSpans[i].length, cudaMemcpyHostToDevice);
	}

	PixelSpan* deviceSpans;

	cudaMalloc(&deviceSpans, sizeof(PixelSpan) * pixelSpans.size());
	cudaMemcpy(deviceSpans, hostSpans, sizeof(PixelSpan) * pixelSpans.size(), cudaMemcpyHostToDevice);
	
	int blocksPerGrid = ((pixelSpans.size() - 1) / ThreadsPerBlock) + 1;
	
	sortingKernel<<<blocksPerGrid, ThreadsPerBlock>>>(deviceSpans, 2);

	cudaMemcpy(hostSpans, deviceSpans, sizeof(PixelSpan) * pixelSpans.size(), cudaMemcpyDeviceToHost);
	for (int i = 0; i < pixelSpans.size(); i++)
	{
		cudaMemcpy(hostSortedSpans[i].pixels, hostSpans[i].pixels, sizeof(Pixel) * pixelSpans[i].length, cudaMemcpyDeviceToHost);
	}


	for (int i = 0; i < pixelSpans.size(); i++)
	{
		for (int j = 0; j < hostSortedSpans[i].length; j++)
		{
			input(hostSortedSpans[i].offset + j, hostSortedSpans[i].y, 0, 0) = hostSortedSpans[i].pixels[j].channel[0];
			input(hostSortedSpans[i].offset + j, hostSortedSpans[i].y, 0, 1) = hostSortedSpans[i].pixels[j].channel[1];
			input(hostSortedSpans[i].offset + j, hostSortedSpans[i].y, 0, 2) = hostSortedSpans[i].pixels[j].channel[2];
		}
		delete hostSortedSpans[i].pixels;
	}
	cudaEventRecord(sortEnd);
	cudaEventSynchronize(sortEnd);
	float elapsedTime = 0.0;
	cudaEventElapsedTime(&elapsedTime, sortStart, sortEnd);
	std::cout << "Time to sort on GPU: " << elapsedTime << "ms" << std::endl;
	cudaEventDestroy(sortStart);
	cudaEventDestroy(sortEnd);
	for (int i = 0; i < pixelSpans.size(); i++)
	{
		cudaFree(hostSpans[i].pixels);
	}
	cudaFree(deviceSpans);

	for (PixelSpan s : pixelSpans)
	{
		delete s.pixels;
	}
	delete hostSpans;
	delete hostSortedSpans;

	input.HSLtoRGB();

	return input;
}

CImg<float> gpuPixelSortingByPixelRow(CImg<float>& input, float sensitivity, void (*sortingKernel)(PixelSpan& pixelRow, int channel))
{
	input.RGBtoHSL();
	std::vector<Pixel> pixelRow;
	std::vector<PixelSpan> pixelSpans;
	int spanOffset = 0;

	cudaEvent_t sortStart, sortEnd;
	cudaEventCreate(&sortStart);
	cudaEventCreate(&sortEnd);
	cudaEventRecord(sortStart);

	for (int y = 0; y < input.height(); y++)
	{
		for (int x = 0; x < input.width(); x++)
		{
			if (input(x, y, 0, 2) < 1.0 - sensitivity && input(x, y, 0, 2) > sensitivity)
			{
				if (pixelRow.size() == 0)
				{
					spanOffset = x;
				}

				pixelRow.push_back({ input(x, y, 0, 0), input(x, y, 0, 1), input(x, y, 0, 2) });
			}
			else
			{
				if (pixelRow.size() > 0)
				{
					Pixel* pixels = (Pixel*)malloc(sizeof(Pixel) * pixelRow.size());
					std::copy(pixelRow.begin(), pixelRow.end(), pixels);
					pixelSpans.push_back({ pixels, spanOffset, y, (int)pixelRow.size() });
					pixelRow.clear();
				}
			}
		}
		if (pixelRow.size() > 0)
		{
			Pixel* pixels = new Pixel[pixelRow.size()];
			std::copy(pixelRow.begin(), pixelRow.end(), pixels);
			pixelSpans.push_back({ pixels, spanOffset, y, (int)pixelRow.size() });
			pixelRow.clear();
		}
	}

	PixelSpan* hostSpans = new PixelSpan[pixelSpans.size()];
	PixelSpan* hostSortedSpans = new PixelSpan[pixelSpans.size()];
	std::copy(pixelSpans.begin(), pixelSpans.end(), hostSortedSpans);
	for (int i = 0; i < pixelSpans.size(); i++)
	{
		hostSortedSpans[i].pixels = new Pixel[pixelSpans[i].length];
	}

	std::copy(pixelSpans.begin(), pixelSpans.end(), hostSpans);
	for (int i = 0; i < pixelSpans.size(); i++)
	{
		cudaMalloc(&(hostSpans[i].pixels), sizeof(Pixel) * pixelSpans[i].length);
		cudaMemcpy(hostSpans[i].pixels, pixelSpans[i].pixels, sizeof(Pixel) * pixelSpans[i].length, cudaMemcpyHostToDevice);
	}


	PixelSpan* deviceSpans;

	cudaMalloc(&deviceSpans, sizeof(PixelSpan) * pixelSpans.size());
	cudaMemcpy(deviceSpans, hostSpans, sizeof(PixelSpan) * pixelSpans.size(), cudaMemcpyHostToDevice);


	for (int i = 0; i < pixelSpans.size(); i++)
	{
		int blocksPerGrid = ((pixelSpans[i].length - 1) / ThreadsPerBlock) + 1;
		sortingKernel<<<blocksPerGrid, ThreadsPerBlock>> >(deviceSpans[i], 2);
	}

	cudaMemcpy(hostSpans, deviceSpans, sizeof(PixelSpan) * pixelSpans.size(), cudaMemcpyDeviceToHost);
	for (int i = 0; i < pixelSpans.size(); i++)
	{
		cudaMemcpy(hostSortedSpans[i].pixels, hostSpans[i].pixels, sizeof(Pixel) * pixelSpans[i].length, cudaMemcpyDeviceToHost);
	}


	for (int i = 0; i < pixelSpans.size(); i++)
	{
		for (int j = 0; j < hostSpans[i].length; j++)
		{
			input(hostSpans[i].offset + j, hostSpans[i].y, 0, 0) = hostSortedSpans[i].pixels[j].channel[0];
			input(hostSpans[i].offset + j, hostSpans[i].y, 0, 1) = hostSortedSpans[i].pixels[j].channel[1];
			input(hostSpans[i].offset + j, hostSpans[i].y, 0, 2) = hostSortedSpans[i].pixels[j].channel[2];
		}
		delete hostSortedSpans[i].pixels;
	}

	cudaEventRecord(sortEnd);
	cudaEventSynchronize(sortEnd);
	float elapsedTime = 0.0;
	cudaEventElapsedTime(&elapsedTime, sortStart, sortEnd);
	std::cout << "Time to sort on GPU: " << elapsedTime << "ms" << std::endl;
	cudaEventDestroy(sortStart);
	cudaEventDestroy(sortEnd);

	for (int i = 0; i < pixelSpans.size(); i++)
	{
		cudaFree(hostSpans[i].pixels);
	}
	cudaFree(deviceSpans);

	for (PixelSpan s : pixelSpans)
	{
		delete s.pixels;
	}
	delete hostSpans;
	delete hostSortedSpans;

	input.HSLtoRGB();

	return input;
}