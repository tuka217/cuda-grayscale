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
__global__ void grayscale(float4* imagem, int width, int height)
{
	const int i = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;

	if(i < width * height)
	{
		float v = 0.3 * imagem[i].x + 0.6 * imagem[i].y + 0.1 * imagem[i].z;
		imagem[i] = make_float4(v, v, v, 0);
	}
}


extern "C" void cuda_grayscale(float* imagem, int width, int height, dim3 blocks, dim3 block_size)
{
	grayscale <<< blocks, block_size >>> ((float4*)imagem, width, height);
}



