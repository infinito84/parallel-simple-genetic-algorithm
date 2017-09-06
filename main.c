#include <stdio.h>
#include <float.h>
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

int main(){
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
        float totalRoulette = 0;
        for(int i=0;i<N;i++){
            totalRoulette = max - min - population[i].fitness + totalRoulette;
            population[i].roulette = totalRoulette;
        }

        // Se seleccionan N_COUPLES
        int i=0;
        while(i<N_COUPLES){
            double n1 = randomDouble(totalRoulette);
            double n2 = randomDouble(totalRoulette);
            if(n1 != n2){
                for(int j=0;j<N;j++){
                    if(population[j].roulette >= n1 && n1 != -1){
                        parents[i].parent1 = population[j];
                        n1 = -1;
                    }
                    if(population[j].roulette >= n2 && n2 != -1){
                        parents[i].parent2 = population[j];
                        n2 = -1;
                    }
                }
                i++;
            }
        }

        // Se cruzan los padres los dos que están seguidos (se crea nueva generación)
        population = (struct Cromosome *)malloc(N * sizeof(struct Cromosome));
        int child = 0;
        for(int i=0;i<N_COUPLES;i++){
            crossover(&parents[i].parent1, &parents[i].parent2, &population[child++], &population[child++]);
        }
    }

    printf("Ganador: \n");
    showCromosome(&minCromosome);
}

// Compile: gcc main.c -o bin/main -lm
// Execute: time ./bin/main
// 8.538
