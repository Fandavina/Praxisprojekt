#include "LblLayer.cuh"
#include "DataLayer.h"

__global__ void getDiffData(float* targets, float* diffData, int label_count, int _batch_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= _batch_size)
		return;
	const int label_value = static_cast<int>(targets[idx]);
	diffData[idx * label_count + label_value] -= 1;
}

void LblLayer::copytoDevData(float* imagesfloat, int imageID) {
	int inputsize = inchannel*inheight*inwidth;
	error->checkError(cudaMemcpyAsync(ptrToOutData, &imagesfloat[imageID *  inputsize], sizeof(float) * inputsize, cudaMemcpyHostToDevice));
	error->checkError(cudaDeviceSynchronize());
}
void LblLayer::copytoHostData(float* &imagesfloat) {
	int inputsize = inchannel*inheight*inwidth;
	imagesfloat = new float[inputsize];
	error->checkError(cudaMemcpyAsync(&imagesfloat[0], ptrToOutData, sizeof(float) *inputsize, cudaMemcpyDeviceToHost));
}
void LblLayer::printGrad(int dimension) {
	Layer::printGrad(dimension, "Lbl ");
}
int LblLayer::getTypeId()
{
	return LayerID::LblLayer;
}
void LblLayer::copytoDevDiff(float*labelsfloat) {
	const float scalVal = 1.0f / static_cast<float>(baSize);
	float*lbl;
	cudaMalloc(&lbl, sizeof(float) * baSize);
	error->checkError(cudaMemcpyAsync(lbl, labelsfloat, sizeof(float)* baSize, cudaMemcpyHostToDevice));
	error->checkError(cudaDeviceSynchronize());
	error->checkError(cudaMemcpyAsync(&ptrToGradData[0], prevLayer->ptrToOutData, sizeof(float)* baSize* prevLayer->outchannel, cudaMemcpyDeviceToDevice));
	error->checkError(cudaDeviceSynchronize());
	getDiffData << <baSize, 1 >> > (lbl, ptrToGradData, prevLayer->outchannel, baSize);
	error->checkError(cudaDeviceSynchronize());
	cublasSscal_v2(handle->cublasHandle,baSize *prevLayer->outchannel, &scalVal, ptrToGradData, 1);
	ptrToOutData = ptrToGradData;
}
void LblLayer::copytoHostLabelwComp(float* &labelsfloat) {
	int outputsize = outchannel*outheight*outwidth;
	labelsfloat = new float[outputsize*baSize];
	error->checkError(cudaMemcpyAsync(labelsfloat, ptrToGradData, sizeof(float)* baSize* prevLayer->outchannel, cudaMemcpyDeviceToHost));
}