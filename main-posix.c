#include <stdio.h>
#include <omp.h>
#include <float.h>
#include <pthread.h>
#include "bukin.c"
#include "random.c"
#include "cromosome.c"
#include "crossover.c"

#define THREADS 4
#define N 10000
#define N_PARENTS  N / 2

struct Cromosome *population;
struct Cromosome parents[N_PARENTS];
int generation = 0;
double minGlobal = DBL_MAX;
int minGeneration = 0;
struct Cromosome minCromosome;
int limit = 0;

void *selectParents(void *arg){
    int id = *(int*) arg;
    int chunk = N_PARENTS/THREADS;
    int start = id * chunk;
    int end = start + chunk;
    //printf("id: %d, limit: %d, start: %d, end: %d, parents: %d, chunk: %d\n",id, limit, start, end, N_PARENTS, chunk);
    do{
        int nRandom = nextRandom(limit);
        for(int j=0;j<N;j++){
            if(!population[j].selected && population[j].roulette >= nRandom){
                population[j].selected = 1;
                parents[start] = population[j];
                break;
            }
        }
        start++;
    }while(start < end);
}

void *crossoverParents(void *arg){
    int id = *(int*) arg;
    int chunk = N_PARENTS/THREADS;
    int start = id * chunk;
    int end = start + chunk;
    int child = start * 2;
    //printf("id: %d, limit: %d, start: %d, end: %d, parents: %d, chunk: %d\n",id, limit, start, end, N_PARENTS, chunk);
    do{
        crossover(&parents[start], &parents[start+1], &population[child++], &population[child++]);
        crossover(&parents[start], &parents[start+1], &population[child++], &population[child++]);
        start+=2;
    }while(start < end);
}


int main(){
    int id[THREADS];
    pthread_t threads[THREADS];
    setSeed();
    calcSizes();
    printf("\nMínimo local: %f\n", fitness(-10,1));
    // Se crea población inicial
    population = (struct Cromosome *)malloc(N * sizeof(struct Cromosome));
    for(int i=0;i<N;i++){
        population[i] = randomCromosome();
    }
    while(generation < 100){
        generation++;
        double min = DBL_MAX;
        double total = 0;
        struct Cromosome tempCromosome;
        for(int i=0;i<N;i++){
            total += population[i].fitness;
            if(population[i].fitness < min){
                min = population[i].fitness;
                tempCromosome = population[i];
            }
        }
        if(min < minGlobal){
            minGlobal = min;
            minGeneration = generation;
            minCromosome = tempCromosome;
        }
        printf("Generation #%d, min: %f, avg: %f, global(%d): %f\n", generation, min, total/N, minGeneration, minGlobal);

        // Se realiza cálculo de la ruleta
        double max = -DBL_MAX;
        for(int i=0;i<N;i++){
            if(population[i].fitness > max) max = population[i].fitness;
        }
        max = (max + fabs(min)) * 1000;
        limit = 0;
        for(int i=0;i<N;i++){
            population[i].roulette = max - population[i].fitness*1000 + limit;
            limit = population[i].roulette;
            //printf("Posición %d, value: %f\n", limit, population[i].fitness);
        }
        // Se seleccionan N_PARENTS
        limit++;
        for(int i=0; i <THREADS; i++){
            id[i] = i;
            pthread_create(&threads[i], NULL, selectParents, &id[i]);
        }
        for(int i=0; i <THREADS; i++) pthread_join(threads[i], NULL);

        // Se cruzan los padres los dos que están seguidos (se crea nueva generación)
        population = (struct Cromosome *)malloc(N * sizeof(struct Cromosome));
        for(int i=0; i <THREADS; i++){
            id[i] = i;
            pthread_create(&threads[i], NULL, crossoverParents, &id[i]);
        }
        for(int i=0; i <THREADS; i++) pthread_join(threads[i], NULL);
    }

    printf("Ganador: \n");
    showCromosome(&minCromosome);
}

// Compile: gcc main-posix.c -o bin/main-posix -lm -lpthread
// Execute: time ./bin/main-posix
// 2.778s
