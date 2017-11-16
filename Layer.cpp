#include "Layer.h"

/*Init the PrevLayer*/
void Layer::initpre(Layer*preLayer) {
	prevLayer = preLayer;
	if(prevLayer!=nullptr)	prevLayer->nextLayer = this;
	if (prevLayer != nullptr && prevLayer->outchannel != 0)inchannel = prevLayer->outchannel;
	if (prevLayer != nullptr &&prevLayer->outheight != 0)inheight = prevLayer->outheight;
	if (prevLayer != nullptr &&prevLayer->outwidth != 0)inwidth = prevLayer->outwidth;
	if (prevLayer != nullptr)SrcTensor = &prevLayer->DstTensor;
}
/*Creates the TensorDescriptors for Cudnn Operation*/
void Layer::createDescr()
{
	if (DstTensor == NULL) error->checkError(cudnnCreateTensorDescriptor(&DstTensor));
}
/*Destroys the TensorDescriptors for Cudnn Operation*/
void Layer::destroyDescr()
{
	if (DstTensor != NULL) { error->checkError(cudnnDestroyTensorDescriptor(DstTensor)); DstTensor = NULL; }
}
/*Standard*/
Layer::Layer()
{
	handle = new Handle();
	createDescr();
}
/*Set Layer depending on output of previous Layer*/
void Layer::set(int batchsize)
{
	if (Layer::getTypeId() == Layer::Conv || Layer::getTypeId() == Layer::FullyConnectedLayer||Layer::getTypeId()==Layer::MaxPooling) {
		error->throwError("Not allowed to use this Set for this Kind of Layer" + Layer::getTypeId());
	}
	baSize = batchsize;
	if (prevLayer != nullptr && prevLayer->outchannel != 0)inchannel = prevLayer->outchannel;
	if (prevLayer != nullptr &&prevLayer->outheight != 0)inheight = prevLayer->outheight;
	if (prevLayer != nullptr &&prevLayer->outwidth != 0)inwidth = prevLayer->outwidth;
	outchannel = inchannel;
	outheight = inheight;
	outwidth = inwidth;
}
/*Set Layer*/
void Layer::set(int batchsize,int outc,int outh,int outw) {
	baSize = batchsize;
	outchannel = outc;
	outheight = outh;
	outwidth = outw;
	if(prevLayer!=nullptr && prevLayer->outchannel!=0)inchannel = prevLayer->outchannel;
	else { inchannel = outchannel; }
	if (prevLayer != nullptr &&prevLayer->outheight != 0)inheight = prevLayer->outheight;
	else { inheight = outheight; }
	if (prevLayer != nullptr &&prevLayer->outwidth != 0)inwidth = prevLayer->outwidth;
	else { inwidth = outwidth; }
	
}
/*Sets the TensorDescriptors for Cudnn Operation*/
void Layer::setDescr() {
	error->checkError(cudnnSetTensor4dDescriptor(DstTensor, tensorformat, dataType, baSize, outchannel, outheight, outwidth));
}

Layer::~Layer()
{
	destroyDescr();
}
void Layer::freeDevFwD() {
	//DEVICE MEM
	if (ptrToOutData != nullptr) {
		error->checkError(cudaFree(ptrToOutData));
		ptrToOutData = nullptr;
	}
}
void Layer::freeDevBwD() {
	//DEVICE MEM
	if (ptrToGradData != nullptr) {
		error->checkError(cudaFree(ptrToGradData));
		ptrToGradData = nullptr;
	}
}
void Layer::mallocDevFwD() {
	int outputsize = outchannel*outheight*outwidth;
	error->checkError(cudaMalloc(&ptrToOutData, sizeof(float) * baSize *outputsize));
}
void Layer::mallocDevBwD() {
	int inputsize = inchannel*inheight*inwidth;
	error->checkError(cudaMalloc(&ptrToGradData, sizeof(float) * baSize *inputsize));
}
void Layer::printDev(int dimension,std::string name) {
	int tensorsize = outchannel*outheight*outwidth;
	printptrDev(name+"Output", ptrToOutData, dimension, baSize *tensorsize);
}
void Layer::printGrad(int dimension, std::string name) {
	
	int tensorsize = inchannel*inheight*inwidth;
	printptrDev(name+"Grad", ptrToGradData, dimension, baSize *tensorsize);
}
void Layer::printptrDev(std::string name, float*devptr, int dimension, size_t size) {
	
	float* temp = new float[size];
	error->checkError(cudaMemcpy(temp, devptr, sizeof(float)*size, cudaMemcpyDeviceToHost));
	printHostptr(name, temp, dimension, size);
	delete[] temp;
}
void Layer::printHostptr(std::string name, float*ptr, int dimension, size_t size){
	std::cout << name << '\t' << '\t';

	for (int i = 0; i < size && i < dimension; i++) {
		std::cout << ptr[i] << " ";
	}
	std::cout << '\t' << "||";
	for (int i = (int)size - dimension; i < size && i >= dimension; i++) {
		std::cout << ptr[i] << " ";
	}
	std::cout << std::endl;
}
void Layer::printHost(std::string name, std::vector<float>hostdata, int dimension, size_t size) {
	std::cout << name << '\t' << '\t';

	for (int i = 0; i < size && i < dimension; i++) {
		std::cout << hostdata[i] << " ";
	}
	std::cout << '\t' << "||";
	for (int i = (int)size - dimension; i < size && i >= dimension; i++) {
		std::cout << hostdata[i] << " ";
	}
	std::cout << std::endl;
}
int Layer::getTypeId()
{
	return LayerID::ParentLayer;
}
