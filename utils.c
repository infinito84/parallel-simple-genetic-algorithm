#include <stdlib.h>
#include <time.h>
#include <math.h>

void setSeed(){
    srand(time(NULL));
}

int randomInt(int max) {
    return rand() % max;
}

float randomDouble(double max){
    return (double)rand()/(double)(RAND_MAX/max);
}

double fitness(double x, double y){
    return 100.00 * sqrt(fabs(y - 0.01*x*x)) + 0.01*fabs(x+10);
}

int checkBoundaries(Cromosome *individual){
	return minX <= individual->x && individual->x <= maxX
	 	&& minY <= individual->y && individual->y <= maxY;
}

int compare (const void * a, const void * b){
	Cromosome c1 = *(Cromosome*)a;
	Cromosome c2 = *(Cromosome*)b;
  	return c1.fitness - c2.fitness;
}
