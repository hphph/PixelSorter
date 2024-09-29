#include "Sorting.h"
#include <memory>
#include <iostream>
#include <chrono>
#include <stack>


CImg<float> cpuPixelSorting(CImg<float>& input, float sensitivity, void (*sortByChannel)(std::vector<Pixel>& pixelRow, int channel))
{
	input.RGBtoHSL();
	std::vector<Pixel> pixelRow;
	Pixel currentPixel;
	const auto start = std::chrono::steady_clock::now();

	for (int y = 0; y < input.height(); y++)
	{
		for (int x = 0; x < input.width(); x++)
		{
			if (input(x, y, 0, 2) < 1.0 - sensitivity && input(x, y, 0, 2) > sensitivity)
			{
				pixelRow.push_back({ input(x, y, 0, 0), input(x, y, 0, 1), input(x, y, 0, 2) });
			}
			else if (pixelRow.size() > 0)
			{
				sortByChannel(pixelRow, 2);
				for (int i = 1; i <= pixelRow.size(); i++)
				{
					currentPixel = pixelRow[pixelRow.size() - i];
					input(x - i, y, 0, 0) = currentPixel.channel[0];
					input(x - i, y, 0, 1) = currentPixel.channel[1];
					input(x - i, y, 0, 2) = currentPixel.channel[2];
				}
				pixelRow.clear();
			}
		}
		if (pixelRow.size() > 0)
		{
			sortByChannel(pixelRow, 2);
			for (int i = 1; i <= pixelRow.size(); i++)
			{
				currentPixel = pixelRow[pixelRow.size() - i];
				input(input.width() - i, y, 0, 0) = currentPixel.channel[0];
				input(input.width() - i, y, 0, 1) = currentPixel.channel[1];
				input(input.width() - i, y, 0, 2) = currentPixel.channel[2];
			}
			pixelRow.clear();
		}
	}
	const auto end = std::chrono::steady_clock::now();
	const std::chrono::duration<double> diff = end - start;
	std::cout << "Time to sort on CPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << "ms" << std::endl;
	input.HSLtoRGB();
	return input;
}

CImg<float> imageMask(CImg<float>& image, float sensitivity)
{
	CImg<float> output(image.width(), image.height(), 1, 3);
	image.RGBtoHSL();
	for (int y = 0; y < image.height(); y++)
	{
		for (int x = 0; x < image.width(); x++)
		{
			if (image(x, y, 0, 2) < 1.0 - sensitivity && image(x, y, 0, 2) > sensitivity)
			{
				output(x, y, 0, 0) = 255;
				output(x, y, 0, 1) = 255;
				output(x, y, 0, 2) = 255;
			}
		}
	}

	return output;
}

void CPUBubbleSortByChannel(std::vector<Pixel>& pixelRow, int channel)
{
	bool isSwapped = true;
	if (channel < 0 || channel > 2)
	{
		std::cerr << "Value outside of channel range" << std::endl;
		return;
	}

	while (isSwapped)
	{
		isSwapped = false;
		for (int i = 1; i < pixelRow.size(); i++)
		{
			if (pixelRow[i].channel[channel] < pixelRow[i - 1].channel[channel])
			{
				isSwapped = true;
				swapPixels(pixelRow[i], pixelRow[i - 1]);
			}
		}
	}
}

void CPUQuickSortByChannel(std::vector<Pixel>& pixelRow, int channel)
{
	quickSortPixels(pixelRow, channel, 0, pixelRow.size() - 1);
}

void quickSortPixels(std::vector<Pixel>& pixelRow, int channel, int l, int r)
{
	std::stack<int> stack;
	stack.push(l);
	stack.push(r);

	int current_r, current_l, p;

	while (stack.size() > 0)
	{
		current_r = stack.top();
		stack.pop();
		current_l = stack.top();
		stack.pop();

		p = quickSortPartition(pixelRow, channel, current_l, current_r);
		if (p - 1 > current_l)
		{
			stack.push(current_l);
			stack.push(p - 1);
		}

		if (p + 1 < current_r)
		{
			stack.push(p + 1);
			stack.push(current_r);
		}
	}
}

int quickSortPartition(std::vector<Pixel>& pixelRow, int channel, int l, int r)
{
	float pivot = pixelRow[r].channel[channel];

	int i = l;

	for (int j = i; j <= r; j++)
	{
		if (pixelRow[j].channel[channel] < pivot)
		{
			swapPixels(pixelRow[j], pixelRow[i]);
			i++;
		}
	}
	swapPixels(pixelRow[r], pixelRow[i]);
	return i;
}

void swapPixels(Pixel& a, Pixel& b)
{
	Pixel c = a;
	a = b;
	b = c;
}
