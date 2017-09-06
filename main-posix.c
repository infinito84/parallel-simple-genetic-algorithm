#include <stdio.h>
#include <omp.h>
#include <float.h>
#include <pthread.h>
#include "structures.c"
#include "constants.c"
#include "utils.c"
#include "cromosome.c"
#include "crossover.c"

struct Cromosome *population;
struct Couple parents[N_COUPLES];
int generation = 0;
double minGlobal = DBL_MAX;
int minGeneration = 0;
struct Cromosome minCromosome;
double totalRoulette = 0;

void *selectParents(void *arg){
    int id = *(int*) arg;
    int chunk = N_COUPLES/THREADS;
    int start = id * chunk;
    int end = start + chunk;
    //printf("id: %d, limit: %F, start: %d, end: %d, parents: %d, chunk: %d\n",id, totalRoulette, start, end, N_COUPLES, chunk);
    do{
        double n1 = randomDouble(totalRoulette);
        double n2 = randomDouble(totalRoulette);
        for(int j=0;j<N;j++){
            if(population[j].roulette >= n1 && n1 != -1){
                parents[start].parent1 = population[j];
                n1 = -1;
            }
            if(population[j].roulette >= n2 && n2 != -1){
                parents[start].parent2 = population[j];
                n2 = -1;
            }
        }
        start++;
    }while(start < end);
}

void *crossoverParents(void *arg){
    int id = *(int*) arg;
    int chunk = N_COUPLES/THREADS;
    int start = id * chunk;
    int end = start + chunk;
    int child = start * 2;
    //printf("id: %d, limit: %d, start: %d, end: %d, parents: %d, chunk: %d\n",id, limit, start, end, N_COUPLES, chunk);
    do{
        crossover(&parents[start].parent1, &parents[start].parent2, &population[child++], &population[child++]);
        start++;
    }while(start < end);
}


int main(){
    int id[THREADS];
    pthread_t threads[THREADS];
    setSeed();
    calcSizes();
    // Se crea población inicial
    population = (struct Cromosome *)malloc(N * sizeof(struct Cromosome));
    for(int i=0;i<N;i++){
        population[i] = randomCromosome();
    }
    while(generation < GENERATIONS){
        generation++;
        double min = DBL_MAX, max = -DBL_MAX, total = 0;
        struct Cromosome tempCromosome;
        for(int i=0;i<N;i++){
            total += population[i].fitness;
            if(population[i].fitness < min){
                min = population[i].fitness;
                tempCromosome = population[i];
            }
            if(population[i].fitness > max){
                max = population[i].fitness;
            }
        }
        if(min < minGlobal){
            minGlobal = min;
            minGeneration = generation;
            minCromosome = tempCromosome;
        }
        printf("Generation #%d, min: %f, avg: %f, global(%d): %f\n", generation, min, total/N, minGeneration, minGlobal);

        // Se realiza cálculo de la ruleta (minimización)
        totalRoulette = 0;
        for(int i=0;i<N;i++){
            totalRoulette = max - min - population[i].fitness + totalRoulette;
            population[i].roulette = totalRoulette;
        }

        // Se seleccionan N_COUPLES
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
