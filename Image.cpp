#include "Image.h"


void Image::ImageLoader(const char *filename, std::string imagename)
{
	FILE             *fp;
	int              bitsize;
	int              infosize = 40;

	if ((fp = fopen(filename, "rb")) == NULL)
		return;
	fseek(fp, 0, SEEK_SET);
	if (fread(&FileHeader, sizeof(FileHeader), 1, fp) < 1)
	{
		fclose(fp);
		return;
	}

	if (FileHeader.bfType != 'MB')
	{
		fclose(fp);
		return;
	}
	fseek(fp, 14, SEEK_SET);
	if (fread(&InfoHeader, sizeof(InfoHeader), 1, fp) < 1)
	{
		fclose(fp);
		return;
	}
	bool upsideDown = false;
	if (InfoHeader.biHeight < 0) {
		upsideDown = true;
		InfoHeader.biHeight *= -1;
	}
	bitsize = InfoHeader.biSizeImage;
	if (bitsize == 0)InfoHeader.biWidth*abs(InfoHeader.biHeight)*(unsigned int)(InfoHeader.biBitCount / 8.0);

	bits = new unsigned char[bitsize];

	fseek(fp, 54, SEEK_SET);
	if (fread(bits, sizeof(unsigned char), bitsize, fp) == NULL)
	{
		fclose(fp);
		return;
	}
	if (upsideDown) {
		// switch down up
		unsigned char temp;
		for (unsigned int i = 0; i < bitsize; i++) {
			temp = bits[i];
			bits[i] = bits[bitsize - i];
			bits[bitsize - i] = temp;
		}
	}
	data = copytoVec(bits, bitsize);
	file = imagename;
	ImageSaver("B:\\Praxisprojekt\\Praxisprojekt\\Praxisprojekt\\Image\\Testing\\temp2.bmp");
	fclose(fp);
	return;
}


void Image::ImageSaver(const char *Filename)
{
	FILE *File = NULL;
	int bitsize = InfoHeader.biWidth*InfoHeader.biHeight*(unsigned int)(InfoHeader.biBitCount / 8.0);

	if ((File = fopen(Filename, "wb")) == nullptr)return;
	fwrite(&FileHeader, sizeof(BITMAPFILEHEADER), 1, File);
	fseek(File, 14, SEEK_SET);
	fwrite(&InfoHeader, sizeof(BITMAPINFOHEADER), 1, File);
	fseek(File, 54, SEEK_SET);
	fwrite(bits, sizeof(char), bitsize, File);
	fclose(File);
};

std::vector<float> Image::copytoVec(char* bits, int bitsize) {
	std::vector<float> vec;
	for (int i = 0; i < bitsize; i++) {
		vec.push_back((int)bits[i]);
	}
	return vec;
}
std::vector<float> Image::copytoVec(unsigned char* bits, int bitsize) {
	std::vector<float> vec;
	for (int i = 0; i < bitsize; i++) {
		vec.push_back((int)bits[i]);
	}
	return vec;
}