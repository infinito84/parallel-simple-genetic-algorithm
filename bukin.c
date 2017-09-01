#include <math.h>

double fitness(double x, double y){
    return 100.00 * sqrt(fabs(y - 0.01*x*x)) + 0.01*fabs(x+10);
}
