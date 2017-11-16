#include "Error.h"

void ErrorHandler::checkError(cudnnStatus_t status) {
	if (status != NULL) {
		std::string s = cudnnGetErrorString(status);
		makelog("ERROR "+s);
		throw status;
	}
}

void ErrorHandler::checkError(cudaError status) {
	if (status != NULL) {
		std::string s = cudaGetErrorString(status);
		makelog("ERROR "+s);
		throw status;
	}
}
void ErrorHandler::checkError(cublasStatus_t status) {
	if (status != NULL) {
		std::string s= cublasGetErrorString(status);
		makelog("ERROR "+s);
		throw status;
	}
}
void ErrorHandler::throwError(std::string status) {
	if (status != "") {
		makelog("WARNING "+status);
		throw status;
	}
}
void ErrorHandler::warning(std::string status) {
	if (status != "") {
		makelog("ERROR "+status);
	}
}
const char* ErrorHandler::cublasGetErrorString(cublasStatus_t status)
{
	switch (status)
	{
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "unknown error";
}

ErrorHandler::ErrorHandler() {
	time_t beginSession;
	time(&beginSession);
	beginStr = ctime(&beginSession);
	beginStr.erase(24);
	beginStr += "--------------------------------BEGIN SESSION-----------------------------";
}
void ErrorHandler::makelog(std::string status) {
	std::string errorstr = "";
	if (firsterror) {
		errorstr += beginStr;
	}
	std::string timestr = "";
	time_t timestamp ;
	time(&timestamp);
	timestr = ctime(&timestamp);
	timestr.erase(24);
	errorstr += "\n" + timestr;
	errorstr += "\t" + status+ "\n";
	
	FILE *File = NULL;
	std::string filename =GetCurrentWorkingDir()+"\\log.txt" ;
	if ((File = fopen(filename.c_str(), "a+")) == nullptr)return;
	for each(char c in errorstr) {
		fputc(c, File);
	}
	fclose(File);
}
std::string ErrorHandler::GetCurrentWorkingDir() {
	char buff[FILENAME_MAX];
	_getcwd(buff, FILENAME_MAX);
	std::string current_working_dir(buff);
	return current_working_dir;
}