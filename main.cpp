/*
 * Copyright (c) 2009, Karl Phillip Buhr
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the name of the Author nor the names of its contributors may be 
 *     used to endorse or promote products derived from this software without 
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

