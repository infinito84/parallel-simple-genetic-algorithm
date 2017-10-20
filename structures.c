typedef struct {
    double x;
    double y;
    int *adn;
    double fitness;
    double roulette;
	double before;
} Cromosome;

typedef struct {
    Cromosome parent1;
    Cromosome parent2;
} Couple;

typedef struct {
    Cromosome individual;
    int generation;
	int race;
} Optimal;

typedef struct {
    Cromosome *individuals;
	Couple *couples;
    int generation;
	int race;
	Optimal optimal;
} Population;
