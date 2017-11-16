#pragma once
#include <cudnn.h>
#include <cuda.h>
#include <cublas.h>
#include <time.h>
#include <Windows.h>
#include <direct.h>
#include <string>
/*Check for CudnnError and log them. ErrorHandle is a Singleton*/
class ErrorHandler {
public:
	static ErrorHandler& getInstance()
	{
		static ErrorHandler instance;
		return instance;
	};
	/*In Case status!=success throws status and log it*/
	void checkError(cudnnStatus_t status);
	/*In Case status!=success throws status and log it*/
	void checkError(cudaError status);
	/*In Case status!=success throws status and log it*/
	void checkError(cublasStatus_t status);
	/*Throws status and log it*/
	void throwError(std::string status);
	/*Log status*/
	void warning(std::string status);
private:
	std::string beginStr="";
	bool firsterror = true;
	const char * cublasGetErrorString(cublasStatus_t status);
	ErrorHandler();
	~ErrorHandler() {};
	void makelog(std::string status);
	std::string GetCurrentWorkingDir();
};