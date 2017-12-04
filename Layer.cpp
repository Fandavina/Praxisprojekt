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
	outputsize = outchannel*outheight*outwidth;
	inputsize = inchannel*inheight*inwidth;
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
	outputsize = outchannel*outheight*outwidth;
	inputsize = inchannel*inheight*inwidth;
	
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
	error->checkError(cudaMalloc(&ptrToOutData, sizeof(float) * baSize *outputsize));
}
void Layer::mallocDevBwD() {
	error->checkError(cudaMalloc(&ptrToGradData, sizeof(float) * baSize *inputsize));
}
void Layer::printDev(int dimension,std::string name) {
	printptrDev(name+"Output", ptrToOutData, dimension, baSize *outputsize);
}
void Layer::printGrad(int dimension, std::string name) {
	printptrDev(name + "Input", ptrToOutData, dimension, baSize*outputsize);
	if(nextLayer!=nullptr)printptrDev(name + "InputG", nextLayer->ptrToGradData, dimension, baSize*outputsize);
	printptrDev(name+"Grad", ptrToGradData, dimension, baSize *inputsize);
	std::cout << "_______" << std::endl;

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
void Layer::randomGenerator(size_t sizerand, size_t sizevec, std::vector<float>& res)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	res.resize(sizevec);

	// Xavier weight filling
	float wfc1 = sqrt(3.0f / (sizerand));
	std::uniform_real_distribution<> dActi(-wfc1, wfc1);
	for (int i = 0; i<sizevec; i++)	res[i]=(static_cast<float>(dActi(gen)));
	/*for (int i = 0; i < sizevec; i++)res[i] = RandomFloat(-1, 1);*/
}
float Layer::RandomFloat(float min, float max)
{
	// this  function assumes max > min, you may want 
	// more robust error checking for a non-debug build
	float random = ((float)rand()) / (float)RAND_MAX;

	// generate (in your case) a float between 0 and (4.5-.78)
	// then add .78, giving you a float between .78 and 4.5
	float range = max - min;
	return (random*range) + min;
}
int Layer::getTypeId()
{
	return LayerID::ParentLayer;
}
