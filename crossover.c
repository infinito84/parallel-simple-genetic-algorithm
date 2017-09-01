int mutation = 10; //among 100;

void binary2x(struct Cromosome *child){
    double xCount = 0;
    double x = 0;
    for(int i=adnSize - yBits - 2; i>=0; i--){
        if(child->adn[i]){
            x += pow(2, xCount);
        }
        xCount++;
    }
    child->x = x / 100 + minX;
}

void binary2y(struct Cromosome *child){
    int yCount = 0;
    double y = 0;
    for(int i=adnSize - 1; i>=0; i--){
        if(child->adn[i] && yCount <= yBits){
            y += pow(2, yCount);
        }
        yCount++;
    }
    child->y = y / 100 + minY;
}

void crossover(struct Cromosome *parent1, struct Cromosome *parent2, struct Cromosome *child1, struct Cromosome *child2){
    int bitSplitter = nextRandom(adnSize);
    int mutation1 = nextRandom(adnSize);
    int mutation2 = nextRandom(adnSize);
    int doMutation = nextRandom(100);
    child1->adn = (int *)malloc(adnSize * sizeof(int));
    child1->selected = 0;
    child2->adn = (int *)malloc(adnSize * sizeof(int));
    child2->selected = 0;
    //showCromosome(parent1);
    //showCromosome(parent2);
    //printf("bitSplitter: %d\n", bitSplitter);
    for(int i=0; i<adnSize; i++){
        if(i < bitSplitter){
            child1->adn[i] = parent1->adn[i];
            child2->adn[i] = parent2->adn[i];
        }
        else{
            child1->adn[i] = parent2->adn[i];
            child2->adn[i] = parent1->adn[i];
        }
        if(doMutation < mutation){
            if(i == mutation1){
                child1->adn[i] = !child1->adn[i];
            }
            if(i == mutation2){
                child2->adn[i] = !child2->adn[i];
            }
        }
    }
    binary2x(child1);
    binary2y(child1);
    binary2x(child2);
    binary2y(child2);
    child1->fitness = fitness(child1->x, child1->y);
    child2->fitness = fitness(child2->x, child2->y);
    //showCromosome(child1);
    //showCromosome(child2);
}
