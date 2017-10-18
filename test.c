#include <stdio.h>
#include <float.h>
#include "structures.c"
#include "constants.c"
#include "utils.c"
#include "cromosome.c"
#include "crossover.c"

int main(){
	setSeed();
	double count = 0;
	for(int i=0; i<10000; i++){
		count += randomDouble(1);
		printf("%.2f\n", randomBetween(minY, maxY));
	}
	printf("N: %.50f\n", count);
}

// Compile: gcc test.c -o bin/test -lm
// Execute: ./bin/test
