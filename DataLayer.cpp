#include "DataLayer.h"

void DataLayer::set(int channels, int width, int height, int batchSize) {
	Layer::set(batchSize, channels, height, width);
}


void DataLayer::copytoDevData(float* imagesfloat, int imageID) {
	error->checkError(cudaMemcpyAsync(ptrToOutData, &imagesfloat[imageID *  inputsize], sizeof(float) * inputsize*baSize, cudaMemcpyHostToDevice));
	error->checkError(cudaDeviceSynchronize());
}
void DataLayer::copytoHostData(float* &imagesfloat) {
	imagesfloat = new float[inputsize];
	error->checkError(cudaMemcpyAsync(&imagesfloat[0], ptrToOutData, sizeof(float) *inputsize*baSize, cudaMemcpyDeviceToHost));
}
void DataLayer::printDev(int dimension) {
	Layer::printDev(dimension, "Data ");
}
int DataLayer::getTypeId()
{
	return LayerID::DataLayer;
}