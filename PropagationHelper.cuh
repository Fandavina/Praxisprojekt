#pragma once
#include "LayerHandler.cuh"
class PropagationHelper {
public:
	void *workspace;
	float*onevec;
	PropagationHelper() { handle = new Handle(); };
	PropagationHelper(int batchsize, int workspacesize, int workgpu);
	void set(int batchsize, int workspacesize, int workgpu);
	void init();

	//FORWARD
	void convForward(ConvLayer *conv);
	void poolForward(MaxPoolLayer* pool);
	void softmaxForward(SoftMax* softmax);
	void lrnForward(LocalResponseNormalization* lrn);
	void activationForward(Activation * activation);
	void fullyConnectedForward(FullyConnectedLayer* full);
	//BACKWARD
	void softmaxBackward(SoftMax * softmax);
	void fullyConnectedBackward(FullyConnectedLayer * full);
	void activationBackward(Activation * activation);
	void lrnBackward(LocalResponseNormalization * lrn);
	void poolBackward(MaxPoolLayer * pool);
	void convBackward(ConvLayer * conv);
		
	//Update
	void UpdateWeightsConv(float learning_rate, ConvLayer * conv, bool output);

	void UpdateWeightsFull(float learning_rate, FullyConnectedLayer * full, bool output);

private:
	int baSize = 0; int workspaceSize = 0;
	float alpha = 1.0f, beta = 0.0f;
	ErrorHandler *error = &ErrorHandler::getInstance();
	Handle *handle;
	/*CUBLAS_OP_N;alpha=1;beta=0*/
	void gemv(int m, int n, float * A, int lda, float * x, int intcx, float * y, int intcy);

	void gemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alphas, float betas, float * A, int lda, float * B, int ldb, float * C, int ldc);
	/*transb=CUBLAS_OP_N;alpha=1;beta=0*/
	void gemm(cublasOperation_t transa, int m, int k, int n, float * A, int lda, float * B, int ldb, float * C, int ldc);
	/*transa=CUBLAS_OP_N*/
	void geam(cublasOperation_t transb, int m, int n, float a, float b, float * A, int lda, float * B, int ldb, float * C, int ldc);

	void printDevptr(float * ptr, int size);

	void Saxpy(float alphal, size_t size, float * x, int incx, float * y, int incy);



};