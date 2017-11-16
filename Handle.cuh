#pragma once
#include <cudnn.h>
#include <cuda.h>
#include <cublas.h>
#include "Error.h"
/*Cublas-&CudnnHandle*/
class Handle {
public:
	cudnnHandle_t cudnnHandle = NULL;
	cublasHandle_t cublasHandle = NULL;
	Handle() {
		if (cublasHandle == NULL)error->checkError(cublasCreate_v2(&cublasHandle));
		if (cudnnHandle == NULL) error->checkError(cudnnCreate(&cudnnHandle));
	}
	~Handle() {
		if (cublasHandle != NULL) {
			error->checkError(cublasDestroy_v2(cublasHandle)); 
			cublasHandle = NULL;
		}
		if (cudnnHandle != NULL) {
			error->checkError(cudnnDestroy(cudnnHandle));
			cudnnHandle = NULL;
		}
	}
private:
ErrorHandler *error = &ErrorHandler::getInstance();
	
};