struct Cromosome{
    double x;
    double y;
    int *adn;
    double fitness;
    double roulette;
};

struct Couple{
    struct Cromosome parent1;
    struct Cromosome parent2;
};
