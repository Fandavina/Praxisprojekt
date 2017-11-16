#include "ImageHandler.h"



ImageHandler::ImageHandler()
{

}


ImageHandler::~ImageHandler()
{
}
std::string ImageHandler::GetCurrentWorkingDir(void) {
	char buff[FILENAME_MAX];
	_getcwd(buff, FILENAME_MAX);
	std::string current_working_dir(buff);
	return current_working_dir;
}

void  ImageHandler::loadImages(int mode) {
	std::string dir = GetCurrentWorkingDir();
	dir += "\\Image";

	if (mode == 1) {
		dir += "\\Training\\";
		std::vector<std::string> filename = getFileNames(dir);
		std::map<std::string, int> label = getLabel(dir, (int)filename.size());
		images = getImages(dir, filename);
		matchImages(images, label);
	}
	if (mode == 2) {
		dir += "\\Testing\\";
		std::vector<std::string> filename = getFileNames(dir);
		images = getImages(dir, filename);
	}

}
std::vector<std::string> ImageHandler::getFileNames(std::string filepath) {
	WIN32_FIND_DATA FindFileData;
	filepath += "*";
	HANDLE hFind;
	LARGE_INTEGER filesize;
	std::vector<std::string> filename;
	hFind = FindFirstFile(filepath.c_str(), &FindFileData);
	FindNextFile(hFind, &FindFileData);
	int temp;
	do
	{
		filesize.LowPart = FindFileData.nFileSizeLow;
		filesize.HighPart = FindFileData.nFileSizeHigh;
		std::string name = FindFileData.cFileName;
		int index = (int)name.find(".bmp", 0);
		if (filesize.QuadPart > 0 && index < name.size()) {
			filename.push_back(name);
		}
		temp = FindNextFile(hFind, &FindFileData);
	} while (temp != 0);
	return filename;
}

std::map<std::string, int> ImageHandler::getLabel(std::string filepath, int countlabel) {
	std::map<std::string, int> labelmap;
	FILE *fp;
	filepath += labeldatastr;
fp = fopen(filepath.c_str(), "r");
	std::string inhalt;
	std::string filename;
	int label;
	bool readfilename = true;
	char c; c = fgetc(fp);
	do {
		c = fgetc(fp);
		if (c == '\t') {
			readfilename = false;
			filename = inhalt;
			inhalt.clear();
		}
		else if (!readfilename) {
			label = c - '0';
			readfilename = true;
		}
		else if (c == '\n' || c == -1) {

			labelmap.insert_or_assign(filename, label);
		}
		else inhalt += c;
	} while (labelmap.size() < countlabel);
	return labelmap;
}
std::vector<Image> ImageHandler::getImages(std::string filepath, std::vector<std::string> filename) {
	std::vector<Image> images;
	for each(std::string name in filename) {
		Image temp;
		temp.ImageLoader((filepath + name).c_str(), name);
		images.push_back(temp);
	}
	return images;
}
void ImageHandler::matchImages(std::vector<Image> &images, std::map<std::string, int>Label) {

	for (int i = 0; i < images.size(); i++) {
		bool matched = false;
		for each (auto var in Label)
		{
			if (var.first==images[i].file) {
				images[i].label = var.second;
				matched = true;
				break;
			}
		}
		if (!matched)error->warning("No matching Label for " + images[i].file);
	}
}