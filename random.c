#include <stdlib.h>
#include <time.h>

void setSeed(){
    srand(time(NULL));
}

int nextRandom(int max) {
    return rand() % max;
}
