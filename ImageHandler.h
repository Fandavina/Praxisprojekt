#pragma once
#include <vector>
#include <map>
#include <stdlib.h> 
#include "Image.h"
#include <direct.h>
#include "Windows.h"
#include "Error.h"
class ImageHandler
{
public:
	ImageHandler();
	void loadImages(int mode);
	std::vector<Image> images;
	~ImageHandler();
private:
	std::string labeldatastr = "label.data";
	std::string GetCurrentWorkingDir(void);
	std::vector<std::string> ImageHandler::getFileNames(std::string filepath);
	std::map<std::string, int> getLabel(std::string filepath, int countlabel);
	std::vector<Image> getImages(std::string filepath, std::vector<std::string> filename);
	void matchImages(std::vector<Image>& images, std::map<std::string, int> Label);
	ErrorHandler *error = &ErrorHandler::getInstance();
};

