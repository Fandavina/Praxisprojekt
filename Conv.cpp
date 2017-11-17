#include "Conv.h"
void ConvLayer::mallocDevFwD() {
	int outputsize = outchannel*outheight*outwidth;
	Layer::mallocDevFwD();
	error->checkError(cudaMalloc(&ptrToDevConv, sizeof(float) * pconv.size()));
	error->checkError(cudaMalloc(&ptrToDevBias, sizeof(float) * pbias.size()));

}
void ConvLayer::mallocDevBwD() {
	int inputsize = inchannel*inwidth*inheight;
	Layer::mallocDevBwD();
	error->checkError(cudaMalloc(&ptrToGradDevConv, sizeof(float) * pconv.size()));
	error->checkError(cudaMalloc(&ptrToGradDevBias, sizeof(float) * pbias.size()));

}
void ConvLayer::freeDevFwDData() {
	//DEVICE DATA
	if (ptrToDevConv != nullptr) {
		error->checkError(cudaFree(ptrToDevConv));
		ptrToDevConv = nullptr;
	}
	if (ptrToDevBias != nullptr) {
		error->checkError(cudaFree(ptrToDevBias));
		ptrToDevBias = nullptr;
	}
}
void ConvLayer::freeDevBwDData() {
	//DEVICE DATA
	if (ptrToGradDevConv != nullptr) {
		error->checkError(cudaFree(ptrToGradDevConv));
		ptrToGradDevConv = nullptr;
	}
	if (ptrToGradDevBias != nullptr) {
		error->checkError(cudaFree(ptrToGradDevBias));
		ptrToGradDevBias = nullptr;
	}
}
void ConvLayer::createDescr() {
	if (BiasTensorDescr == NULL)error->checkError(cudnnCreateTensorDescriptor(&BiasTensorDescr));
	if (FilterDescr == NULL)error->checkError(cudnnCreateFilterDescriptor(&FilterDescr));
	if (Descr == NULL)error->checkError(cudnnCreateConvolutionDescriptor(&Descr));
}
void ConvLayer::destroyDescr() {
	if (BiasTensorDescr == NULL) { error->checkError(cudnnDestroyTensorDescriptor(BiasTensorDescr)); BiasTensorDescr = NULL; }
	if (FilterDescr == NULL) { error->checkError(cudnnDestroyFilterDescriptor(FilterDescr)); FilterDescr = NULL; }
	if (Descr == NULL) { error->checkError(cudnnDestroyConvolutionDescriptor(Descr)); Descr = NULL; }
}
void ConvLayer::setDescr(size_t&workspace, bool setDataAlgo, bool withWorkspace) {
	Layer::setDescr();	
	workspace = (size_t)std::fmax(workspace, SetFwdConvolutionTensors(withWorkspace));
	error->checkError(cudnnSetTensor4dDescriptor(BiasTensorDescr, tensorformat, dataType, 1, outchannel, 1, 1));
	workspace = (size_t)std::fmax(workspace, SetBwdConvolutionTensors(setDataAlgo, withWorkspace));
}
size_t ConvLayer::SetFwdConvolutionTensors(bool withWorkspace) {
	size_t tempsize = 0;
	int stride = 1;
	int n = baSize;
	int c = inchannel;
	int h = inheight;
	int w = inwidth;

	error->checkError(cudnnSetFilter4dDescriptor(FilterDescr, dataType, tensorformat, outchannel, inchannel, kernel_size, kernel_size));

	error->checkError(cudnnSetConvolution2dDescriptor(Descr, 0, 0, stride,stride, 1, 1, convMode, dataType));

	error->checkError(cudnnGetConvolution2dForwardOutputDim(Descr, *SrcTensor, FilterDescr, &n, &c, &h, &w));


	if (withWorkspace) {
		error->checkError(cudnnGetConvolutionForwardAlgorithm(handle->cudnnHandle, *SrcTensor, FilterDescr, Descr, DstTensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &AlgoFwd));
	}
	else {
		error->checkError(cudnnGetConvolutionForwardAlgorithm(handle->cudnnHandle, *SrcTensor, FilterDescr, Descr, DstTensor, CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &AlgoFwd));
	}
	error->checkError(cudnnGetConvolutionForwardWorkspaceSize(handle->cudnnHandle, *SrcTensor, FilterDescr, Descr, DstTensor, AlgoFwd, &tempsize));

	return tempsize;
}
size_t ConvLayer::SetBwdConvolutionTensors( bool setDataAlgo, bool withWorkspace) {
	size_t sizeInBytes = 0, tmpsize = 0;
	if (withWorkspace) {
		error->checkError(cudnnGetConvolutionBackwardFilterAlgorithm(handle->cudnnHandle, *SrcTensor, DstTensor, Descr, FilterDescr,
			CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &AlgoBwd));
	}
	else {
		error->checkError(cudnnGetConvolutionBackwardFilterAlgorithm(handle->cudnnHandle, *SrcTensor, DstTensor, Descr, FilterDescr,
			CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &AlgoBwd));
	}

	error->checkError(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		handle->cudnnHandle, *SrcTensor, DstTensor, Descr, FilterDescr,
		AlgoBwd, &tmpsize));

	sizeInBytes = (size_t)std::fmax(sizeInBytes, tmpsize);

	// If backprop data algorithm was requested
	if (setDataAlgo)
	{
		dataAlgo = true;
		if (withWorkspace) {
			error->checkError(cudnnGetConvolutionBackwardDataAlgorithm(
				handle->cudnnHandle, FilterDescr, DstTensor, Descr, *SrcTensor,
				CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &AlgoDataBwd));
		}
		else {
			error->checkError(cudnnGetConvolutionBackwardDataAlgorithm(
				handle->cudnnHandle, FilterDescr, DstTensor, Descr, *SrcTensor,
				CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 0, &AlgoDataBwd));
		}

		error->checkError(cudnnGetConvolutionBackwardDataWorkspaceSize(
			handle->cudnnHandle, FilterDescr, DstTensor, Descr, *SrcTensor,
			AlgoDataBwd, &tmpsize));

		sizeInBytes = (size_t)std::fmax(sizeInBytes, tmpsize);
	}

	return sizeInBytes;
}
ConvLayer::ConvLayer() {
	createDescr();
}
ConvLayer::~ConvLayer() {
	destroyDescr();
}
ConvLayer::ConvLayer( int out_channels_, int kernel_size_, int batchsize) {
	createDescr();
	set( out_channels_, kernel_size_, batchsize);
}
void ConvLayer::set(int out_channels_, int kernel_size_,int batchsize) {
	kernel_size = kernel_size_;
	outchannel = out_channels_;
	pbias.resize(out_channels_);

	pconv.resize(prevLayer->inchannel * kernel_size_ * kernel_size_ * out_channels_); //Size = FilterDescr size
	Layer::set(batchsize, out_channels_, prevLayer->outheight - kernel_size_ + 1, prevLayer->outwidth - kernel_size_ + 1);
}
void ConvLayer::initLayer(const char *fileprefix) {
	std::string ssf = "";
	std::string ssbf = "";
	ssf = fileprefix;
	ssf += ".bin";
	ssbf = fileprefix;
	ssbf += ".bias.bin";

	FILE *fp = fopen(ssf.c_str(), "rb");
	if (!fp)
	{
		error->throwError("Keine gespeicherte ConvLayer unter " + ssf + "gefunden");
	}
	float *convtemp = new float[pconv.size()];
	fread(convtemp, sizeof(float), pconv.size(), fp);
	fclose(fp);
	for (unsigned int i = 0; i < pconv.size(); i++) {
		pconv[i] = convtemp[i];
	}

	fp = fopen(ssbf.c_str(), "rb");
	if (!fp)
	{
		error->throwError("Keine gespeicherte ConvLayer unter " + ssbf + "gefunden");
	}
	float *biastemp = new float[pbias.size()];
	fread(biastemp, sizeof(float), pbias.size(), fp);
	fclose(fp);
	for (unsigned int i = 0; i < pbias.size(); i++) {
		pbias[i] = biastemp[i];
	}
	fclose(fp);
}
void ConvLayer::initRandom() {
	if (pconv.size() == 0) {
		error->throwError("ConvInit failed. Reason: pconvsize =0");
	}
	if (pbias.size() == 0) {
		error->throwError("ConvInit failed. Reason: pbiassize =0");
	}
	// Create random network
	randomGenerator(kernel_size * kernel_size * inchannel, pconv.size(),pconv);
	randomGenerator(kernel_size * kernel_size * inchannel, pbias.size(),pbias);
}
void ConvLayer::saveLayer(const char *fileprefix) {
	std::string ssf = "";
	std::string ssbf = "";
	ssf = fileprefix;
	ssf += ".bin";
	ssbf = fileprefix;
	ssbf += ".bias.bin";
	// Write weights file
	FILE *fp = fopen(ssf.c_str(), "wb");
	if (!fp)
	{
		printf("ERROR: Cannot open file %s\n", ssf.c_str());
		exit(2);
	}
	for each(float var in pconv) {
		fwrite(&var, sizeof(float), 1, fp);
	}
	fclose(fp);

	// Write bias file
	fp = fopen(ssbf.c_str(), "wb");
	if (!fp)
	{
		printf("ERROR: Cannot open file %s\n", ssbf.c_str());
		exit(2);
	}
	for each(float var in pbias) {
		fwrite(&var, sizeof(float), 1, fp);
	}
	fclose(fp);
}
void ConvLayer::copyToDevFwD() {
	// Copy initial network to device
	error->checkError(cudaMemcpyAsync(ptrToDevConv, &pconv[0], sizeof(float) * pconv.size(), cudaMemcpyHostToDevice));
	error->checkError(cudaMemcpyAsync(ptrToDevBias, &pbias[0], sizeof(float) * pbias.size(), cudaMemcpyHostToDevice));
}
void ConvLayer::copyToHost() {
	error->checkError(cudaMemcpy(&pconv[0], ptrToDevConv, sizeof(float) * pconv.size(), cudaMemcpyDeviceToHost));
	error->checkError(cudaMemcpy(&pbias[0], ptrToDevBias, sizeof(float) * pbias.size(), cudaMemcpyDeviceToHost));
}
void ConvLayer::printDevCB(int dimension) {
	printptrDev("ConvData Device", ptrToDevConv, dimension, pconv.size());
	printptrDev("BiasData Device", ptrToDevBias, dimension, pbias.size());
}
void ConvLayer::printGradCB(int dimension) {
	printptrDev("ConvGradConv Device", ptrToGradDevConv, dimension, pconv.size());
	printptrDev("BiasGradData Device", ptrToGradDevBias, dimension, pbias.size());
}
void ConvLayer::printHostCB(int dimension) {
	printHost("ConvData Host", pconv, dimension, pconv.size());
	printHost("BiasData Host", pbias, dimension, pbias.size());
}
void ConvLayer::printDev(int dimension) {
	Layer::printDev(dimension, "Conv ");
}
void ConvLayer::printGrad(int dimension) {
	Layer::printGrad(dimension, "Conv ");
}
int ConvLayer::getTypeId()
{
	return LayerID::Conv;
}