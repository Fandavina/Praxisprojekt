#include "PropagationHelper.cuh"
__global__ void Fill (const float value,float *vec, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > size)
		return;

	vec[idx] = value;
	
}
PropagationHelper::PropagationHelper(int batchsize, int workspacesize, int workgpu)
{
	set(batchsize, workspacesize, workgpu);
}
void PropagationHelper::set(int batchsize, int workspacesize, int workgpu) {
	baSize = batchsize;
	workspaceSize = workspacesize;
	error->checkError(cudaSetDevice(workgpu));
	init();
}
void PropagationHelper::init() {
	error->checkError(cudaMalloc(&onevec, sizeof(float)* baSize));
	Fill << <baSize, 1 >> > (1.0f, onevec, baSize);
	error->checkError(cudaDeviceSynchronize());
	error->checkError(cudaMalloc(&workspace, workspaceSize));
	error->checkError(cudaDeviceSynchronize());
}

void PropagationHelper::convForward(ConvLayer* conv) {
	error->checkError(cudnnConvolutionForward(handle->cudnnHandle, &alpha,
		*conv->SrcTensor, conv->prevLayer->ptrToOutData, //x
		conv->FilterDescr, conv->ptrToDevConv,	//w
		conv->Descr, conv->AlgoFwd, workspace, workspaceSize, &beta,
		conv->DstTensor, conv->ptrToOutData));//y
	error->checkError(cudnnAddTensor(handle->cudnnHandle, &alpha,
		conv->BiasTensorDescr, conv->ptrToDevBias,//A
		&alpha, conv->DstTensor, conv->ptrToOutData));//C
}
void PropagationHelper::poolForward( MaxPoolLayer *pool) {
	error->checkError(cudnnPoolingForward(handle->cudnnHandle, pool->Descr, &alpha,
		*pool->SrcTensor, pool->prevLayer->ptrToOutData, &beta, //x
		pool->DstTensor, pool->ptrToOutData));//y
}
void PropagationHelper::softmaxForward(SoftMax *softmax)
{
	error->checkError(cudnnSoftmaxForward(handle->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha, *softmax->SrcTensor, softmax->prevLayer->ptrToOutData,//x
		&beta, softmax->DstTensor, softmax->ptrToOutData));//y
}
void PropagationHelper::lrnForward(LocalResponseNormalization *lrn)
{
	error->checkError(cudnnLRNCrossChannelForward(handle->cudnnHandle,
		lrn->Descr, CUDNN_LRN_CROSS_CHANNEL_DIM1,
		&alpha, *lrn->SrcTensor, lrn->prevLayer->ptrToOutData,//x
		&beta, lrn->DstTensor, lrn->ptrToOutData));//y
}
void PropagationHelper::activationForward(Activation *activation)
{
	error->checkError(cudnnActivationForward(handle->cudnnHandle,
		activation->Descr, &alpha,
		*activation->SrcTensor, activation->prevLayer->ptrToOutData,//x
		&beta, activation->DstTensor, activation->ptrToOutData));//y
}
void PropagationHelper::fullyConnectedForward(FullyConnectedLayer *full)
{
	error->checkError(cudaMemcpy(full->ptrToOutData, full->ptrToDevBias, full->outputsize * sizeof(float), cudaMemcpyDeviceToDevice));
	gemm(CUBLAS_OP_T, CUBLAS_OP_N,
		full->outputsize, baSize, full->inputsize,
		alpha, beta,
		full->ptrToDevNeuron, full->inputsize,
		full->prevLayer->ptrToOutData, full->inputsize,
		full->ptrToOutData, full->outputsize);
	error->checkError(cudaDeviceSynchronize());
	gemm(CUBLAS_OP_N, CUBLAS_OP_N,
		full->outputsize, baSize, 1,
		alpha, alpha,
		full->ptrToDevBias, full->outputsize,
		onevec, 1,
		full->ptrToOutData, full->outputsize);
	error->checkError(cudaDeviceSynchronize());
}

void PropagationHelper::softmaxBackward(SoftMax *softmax) {
	error->checkError(cudnnSoftmaxBackward(handle->cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha, softmax->DstTensor, softmax->ptrToOutData,		// y
		softmax->DstTensor, softmax->nextLayer->ptrToGradData,	//dy
		&beta, *softmax->SrcTensor, softmax->ptrToGradData));	//dx
	error->checkError(cudaDeviceSynchronize());
}
void PropagationHelper::fullyConnectedBackward(FullyConnectedLayer *full) {
	gemm(CUBLAS_OP_N, CUBLAS_OP_N,
		full->inputsize, baSize, full->outputsize, alpha, beta,
		full->ptrToDevNeuron, full->inputsize,
		full->nextLayer->ptrToGradData, full->outputsize,
		full->ptrToGradData, full->inputsize);
	error->checkError(cudaDeviceSynchronize());
}
void PropagationHelper::activationBackward(Activation *activation) {
	error->checkError(cudnnActivationBackward(handle->cudnnHandle, activation->Descr,
		&alpha, activation->DstTensor, activation->ptrToOutData,		// y
		activation->DstTensor, activation->nextLayer->ptrToGradData,	//dy
		*activation->SrcTensor, activation->prevLayer->ptrToOutData,	// x
		&beta, *activation->SrcTensor, activation->ptrToGradData));		//dx
	error->checkError(cudaDeviceSynchronize());
}
void PropagationHelper::lrnBackward(LocalResponseNormalization *lrn) {
	error->checkError(cudnnLRNCrossChannelBackward(handle->cudnnHandle, lrn->Descr, CUDNN_LRN_CROSS_CHANNEL_DIM1,
		&alpha, lrn->DstTensor, lrn->ptrToOutData,		// y
		lrn->DstTensor, lrn->nextLayer->ptrToGradData,	//dy
		*lrn->SrcTensor, lrn->prevLayer->ptrToOutData,	// x
		&beta, *lrn->SrcTensor, lrn->ptrToGradData));	//dx
	error->checkError(cudaDeviceSynchronize());
}
void PropagationHelper::poolBackward(MaxPoolLayer*pool) {
	error->checkError(cudnnPoolingBackward(handle->cudnnHandle, pool->Descr, &alpha,
		pool->DstTensor, pool->ptrToOutData,				// y
		pool->DstTensor, pool->nextLayer->ptrToGradData,	//dy
		*pool->SrcTensor, pool->prevLayer->ptrToOutData,	// x
		&beta, *pool->SrcTensor, pool->ptrToGradData));		//dx
	error->checkError(cudaDeviceSynchronize());
}
void PropagationHelper::convBackward(ConvLayer*conv) {
	if (conv->dataAlgo) error->checkError(cudnnConvolutionBackwardData(handle->cudnnHandle, &alpha,
		conv->FilterDescr, conv->ptrToDevConv,				// w
		conv->DstTensor, conv->nextLayer->ptrToGradData,	//dy
		conv->Descr, conv->AlgoDataBwd, workspace, workspaceSize,
		&beta, *conv->SrcTensor, conv->ptrToGradData));		//dx
	error->checkError(cudaDeviceSynchronize());
}

void PropagationHelper::UpdateWeightsConv(float learning_rate,ConvLayer* conv,bool output)
{
	float alphal = -learning_rate;
	float* dstData;
	error->checkError(cudnnConvolutionBackwardBias(handle->cudnnHandle, &alpha,
		conv->DstTensor, conv->nextLayer->ptrToGradData,		//dy
		&beta, conv->BiasTensorDescr, conv->ptrToGradDevBias));	//db

	error->checkError(cudnnConvolutionBackwardFilter(handle->cudnnHandle, &alpha,
		*conv->SrcTensor, conv->prevLayer->ptrToOutData, //x
		conv->DstTensor, conv->nextLayer->ptrToGradData,//dy
		conv->Descr, conv->AlgoBwd, workspace, workspaceSize,
		&beta, conv->FilterDescr, conv->ptrToGradDevConv));//dw
	if(output){
		conv->printDevCB(5);
		conv->printGradCB(5);
		std::cout << "____________________________________________________________________" << std::endl;
	}
	Saxpy(alphal, conv->pbias.size(), conv->ptrToGradDevBias, 1, conv->ptrToDevBias, 1);

	Saxpy(alphal,conv->pconv.size(), conv->ptrToGradDevConv, 1, conv->ptrToDevConv, 1);

	error->checkError(cudaDeviceSynchronize());
	if (output) {
		conv->printDevCB(5);
		std::cout << "__________________________________________________________________________________________________________________________________" << std::endl;
	}	
	error->checkError(cudaDeviceSynchronize());

}
void PropagationHelper::UpdateWeightsFull(float learning_rate, FullyConnectedLayer *full,bool output) {
	float alphal = -learning_rate;
	float* dstData;
	
	if(output){
		full->printDevNB(5);
		full->printGradCB(5);
		std::cout << "____________________________________________________________________" << std::endl;
	}
	gemm(CUBLAS_OP_N, CUBLAS_OP_T,
		full->inputsize, full->outputsize, baSize,
		alpha, beta,
		full->prevLayer->ptrToOutData, full->inputsize,//x
		full->nextLayer->ptrToGradData, full->outputsize,//dy
		full->ptrToGradDevNeuron, full->inputsize);
	
	error->checkError(cudaMalloc(&dstData, sizeof(float)* full->pneurons.size()));
	geam(CUBLAS_OP_N, full->inputsize, full->outputsize, alphal, alpha,
		dstData, full->inputsize,
		full->ptrToGradDevNeuron, full->inputsize,
		full->ptrToDevNeuron, full->inputsize);

	gemv(full->outputsize, baSize,
		full->nextLayer->ptrToGradData, full->outputsize,//dy
		onevec, 1,
		full->ptrToGradDevBias, 1);
	error->checkError(cudaDeviceSynchronize());
	error->checkError(cudaMalloc(&dstData, sizeof(float)* full->pbias.size()));
	geam(CUBLAS_OP_N, 1, full->outputsize,
		alphal, alpha,
		dstData, 1,
		full->ptrToGradDevBias, 1,
		full->ptrToDevBias, 1);

	if (output) {
		full->printDevNB(5);
		std::cout << "__________________________________________________________________________________________________________________________________" << std::endl;
	}
	error->checkError(cudaFree(dstData));
	error->checkError(cudaDeviceSynchronize());
}

void PropagationHelper::printDevptr(float*ptr, int size) {
	float* temp = new float[size];
	error->checkError(cudaMemcpyAsync(&temp[0], ptr, size*sizeof(float), cudaMemcpyDeviceToHost));
	error->checkError(cudaDeviceSynchronize());
	for (int i = 0; i < size; i++) {
		std::cout << temp[i] << " ";
	}
}
void PropagationHelper::Saxpy(float alphal,size_t size,float*x, int incx, float*y,int incy) {
	error->checkError(cublasSaxpy_v2(handle->cublasHandle, static_cast<int>(size),	&alphal,x, incx,y, incy));
}
void PropagationHelper::gemv(int m, int n, float*A, int lda, float*x, int intcx, float*y, int intcy) {
	error->checkError(cublasSgemv_v2(handle->cublasHandle, CUBLAS_OP_N, m, n, &alpha, A, lda, x, intcx, &beta, y, intcy));
	error->checkError(cudaDeviceSynchronize());
}
void PropagationHelper::gemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alphas, float betas, float*A, int lda, float*B, int ldb, float*C, int ldc) {
	error->checkError(cublasSgemm_v2(handle->cublasHandle, transa, transb, m, n, k, &alphas, A, lda, B, ldb, &betas, C, ldc));
	error->checkError(cudaDeviceSynchronize());
}
void PropagationHelper::gemm(cublasOperation_t transa, int m, int n, int k, float*A, int lda, float*B, int ldb, float*C, int ldc) {
	error->checkError(cublasSgemm_v2(handle->cublasHandle, transa, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
	error->checkError(cudaDeviceSynchronize());
}
void PropagationHelper::geam(cublasOperation_t transb, int m, int n, float a, float b, float*A, int lda, float*B, int ldb, float*C, int ldc) {
	error->checkError(cublasSgeam(handle->cublasHandle, CUBLAS_OP_N, transb, m, n, &a, A, lda, &b, B, ldb, C, ldc));
}