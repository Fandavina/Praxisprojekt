#pragma once
#include <windows.h>
#include <stdlib.h> 
#include <iostream> 
#include "fstream" 
#include <vector>
class Image 
{ 
 public: 
  
     void ImageLoader(const char *Filename,std::string imagename);
     void ImageSaver(const char *Filename);

	 int label = -1;
	 std::string file;
	
    unsigned char *bits; 
	 std::vector<float> data;
     //BITMAPFILEHEADER 
     typedef struct 
     { 
     WORD bfType; 
     DWORD bfSize; 
     DWORD bfReserved; 
     DWORD bfOffBits; 
     }BITMAPFILEHEADER; 
  
     //BITMAPINFOHEADER 
     typedef struct 
     { 
     DWORD biSize; 
     LONG biWidth; 
     LONG biHeight; 
     WORD biPlanes; 
     WORD biBitCount; 
     DWORD biCompression; 
     DWORD biSizeImage; 
     LONG biXPelsPerMeter; 
     LONG biYPelsPerMeter; 
     DWORD biClrUsed; 
     DWORD biClrImportant; 
     }BITMAPINFOHEADER; 
    
     BITMAPFILEHEADER FileHeader; 
     BITMAPINFOHEADER InfoHeader; 
private:
	std::vector<float> copytoVec(char * bits, int bitsize);
	std::vector<float> copytoVec(unsigned char * bits, int bitsize);
}; 
