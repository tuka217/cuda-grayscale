/*
    Copyright (C) <2009>  <Karl Phillip Buhr>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "kernel_gpu.h"

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <math.h>


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <input.png>" << " <output.png>" << std::endl;
        exit(1);
    }

	IplImage* input_image = NULL;
	input_image = cvLoadImage(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if(!input_image)
    {
        std::cout << "ERROR: Failed to load input image" << std::endl;
        return -1;
    }

	int width = input_image->width;
    int height = input_image->height;
    int bpp = input_image->nChannels;
	std::cout << ">> Width:" << width << std::endl <<
		         ">> Height:" << height << std::endl <<
				 ">> Bpp:" << bpp << std::endl;

#ifdef DEBUG
    std::cout << std::endl << ">>> Debugging Original data:" << std::endl;
    for (int i=0; i < width*height*bpp; i+=bpp)
    {
        if (!(i % (width*bpp)))
            std::cout << std::endl;

        std::cout << std::dec << "R:"<< (int) input_image->imageData[i] <<
                                 " G:" << (int) input_image->imageData[i+1] <<
                                 " B:" << (int) input_image->imageData[i+2] << "     ";
    }
    std::cout << std::endl << std::endl;
#endif

    float* imagem_cpu = new float[width * height * 4];
	for (int i = 0; i < width * height; i++)
	{
		imagem_cpu[i * 4 + 0] = (unsigned char)input_image->imageData[i * bpp + 0] / 255.f;
		imagem_cpu[i * 4 + 1] = (unsigned char)input_image->imageData[i * bpp + 1] / 255.f;
		imagem_cpu[i * 4 + 2] = (unsigned char)input_image->imageData[i * bpp + 2] / 255.f;
	}

	float* imagem_gpu = new float[width * height * 4];

	cudaMalloc((void **)(&imagem_gpu), (width * height * 4) * sizeof(float));
	cudaMemcpy(imagem_gpu, imagem_cpu, (width * height * 4) * sizeof(float), cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((int)ceil(double((width * height) / 256.0)));
															
	cuda_grayscale(imagem_gpu, width, height, grid, block);

	cudaMemcpy(imagem_cpu, imagem_gpu, (width * height * 4) * sizeof(float), cudaMemcpyDeviceToHost);

	char* buff = new char[width * height * bpp];
	for (int i = 0; i < (width * height); i++)
	{
		buff[i * bpp + 0] = (char)floor(imagem_cpu[i * 4 + 0] * 255.f);
		buff[i * bpp + 1] = (char)floor(imagem_cpu[i * 4 + 1] * 255.f);
		buff[i * bpp + 2] = (char)floor(imagem_cpu[i * 4 + 2] * 255.f);
	}

#ifdef DEBUG
    std::cout << std::endl << ">>> Debugging Output data:" << std::endl;
    for (int i=0; i < width*height*bpp; i+=bpp)
    {
        if (!(i % (width*bpp)))
            std::cout << std::endl;

        std::cout << std::dec << "R:"<< (int) buff[i] <<
                                 " G:" << (int) buff[i+1] <<
                                 " B:" << (int) buff[i+2] << "     ";
    }
    std::cout << std::endl << std::endl;
#endif

	IplImage* out_image = cvCreateImage( cvSize(width, height), input_image->depth, bpp);
	out_image->imageData = buff;

	if( !cvSaveImage(argv[2], out_image) )
    {
        std::cout << "ERROR: Failed to write image file" << std::endl;
    }

	cvReleaseImage(&input_image);
    cvReleaseImage(&out_image);

	return 0;
}

