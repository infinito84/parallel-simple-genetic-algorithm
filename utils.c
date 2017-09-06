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
