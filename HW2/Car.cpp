#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

#define RGB_COMPONENT_COLOR 255
#define N_THREADS 4
#define FREQ 10
#define MAX_SHIFTS 300

struct PPMPixel {
    int red;
    int green;
    int blue;
};

typedef struct{
    int x, y, all;
    PPMPixel * data;
} PPMImage;

void readPPM(const char *filename, PPMImage& img){
    std::ifstream file (filename);
    if (file){
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s!="P3") {std::cout<< "error in format"<<std::endl; exit(9);}
        file >> img.x >>img.y;
        file >>rgb_comp_color;
        img.all = img.x*img.y;
        std::cout << s << std::endl;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" <<img.all << std::endl;
        img.data = new PPMPixel[img.all];
        for (int i=0; i<img.all; i++){
            file >> img.data[i].red >>img.data[i].green >> img.data[i].blue;
        }

    }else{
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}

void writePPM(const char *filename, PPMImage & img){
    std::ofstream file (filename, std::ofstream::out);
    file << "P3"<<std::endl;
    file << img.x << " " << img.y << " "<< std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for(int i=0; i<img.all; i++){
        file << img.data[i].red << " " << img.data[i].green << " " << img.data[i].blue << (((i+1)%img.x ==0)? "\n" : " ");
    }
    file.close();
}

void shift_omp(PPMImage &img){
    
    double ** R = (double **)malloc(sizeof(double *)*img.x);
    double ** G = (double **)malloc(sizeof(double *)*img.x);
    double ** B = (double **)malloc(sizeof(double *)*img.x);

    for (int i = 0; i < img.x; ++i)
    {
        R[i] = (double *)malloc(img.y * sizeof(double));
	G[i] = (double *)malloc(img.y * sizeof(double));
        B[i] = (double *)malloc(img.y * sizeof(double));
    }
    
    int count = 0;
    for (int row = 0; row < img.y; ++row)
    {
        for (int column = 0; column < img.x; ++column)
	{
	    R[column][row] = img.data[count].red;
	    G[column][row] = img.data[count].green;
	    B[column][row] = img.data[count].blue;
	    ++count;
	}
    }
    
    for (int shift = 0; shift <= MAX_SHIFTS; ++shift)
    {
        #pragma omp parallel for shared(R, G, B, img)
        for (int row = 0; row < img.y; ++row)
        {
            for (int column = 0; column < img.x; ++column)
	    {
	        R[column][row] = R[(column + 1) % img.x][row];
	        G[column][row] = G[(column + 1) % img.x][row];
	        B[column][row] = B[(column + 1) % img.x][row];
	    }
        }

	count = 0;
	for (int row = 0; row < img.y; ++row)
        {
            for (int column = 0; column < img.x; ++column)
            {
                img.data[count].red = R[column][row];
                img.data[count].green = G[column][row];
                img.data[count].blue = B[column][row];
                ++count;
             }
        }

	if (shift % FREQ == 0)
	{
	    char name[sizeof "./pics/000.ppm"];
	    sprintf(name, "./pics/%d.ppm", MAX_SHIFTS - shift);
	    writePPM(name, img);
	}
    }

    free(R);
    free(G);
    free(B);
}

int main(int argc, char *argv[]){
    
    PPMImage image;
    
    double start, end;

    readPPM("car.ppm", image);
    
    start = omp_get_wtime();
    shift_omp(image);
    end = omp_get_wtime();

    printf("Time elapsed: %.5f\n", (double)(end - start));

    delete(image.data);

    return 0;
}
