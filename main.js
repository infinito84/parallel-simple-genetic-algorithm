class Generations {
  constructor (options = {}) {
    this.N = (options.N || 2000)
    this.N_COUPLES = this.N / 2
    this.GENERATIONS = options.GENERATIONS || 100
    this.MUTATIONS = options.MUTATIONS || 10
    this.population = []
    this.minGlobal = Number.MAX_VALUE
    this.minGeneration = 0
    this.minX = options.minX || -15
    this.maxX = options.maxX || 0
    this.minY = -3
    this.maxY = 3
    this.decimals = options.decimals || 2
    this.nDecimals = Math.pow(10, this.decimals)
    this.minGeneration = 0
    this.init()
  }

  init() {
    this.resetGeneration()
    this.getBits()
  }

  getBits() {
    this.xBits = 0
    this.yBits = 0
    this.xSize = (this.maxX - this.minX) * this.nDecimals
    this.ySize = (this.maxY - this.minY) * this.nDecimals
    let bits = 1
    while (this.xBits === 0 || this.yBits === 0) {
      let temp = Math.pow(2, bits)
      if(this.xBits === 0 && this.xSize < temp){
        this.xBits = bits
      }
      if(this.yBits === 0 && this.ySize < temp){
        this.yBits = bits
      }
      bits++
    }
    this.adnSize = this.xBits + this.yBits
    console.log("Bits: adn(%d), x(%d), y(%d)\n", this.adnSize, this.xBits, this.yBits);
  }

  getInitialPopulation() {
    return new Promise((resolve) => {
      let done = 0
      const getRandomCromosome = () => {
        this.population.push(this.getRandomCromosome())
        if (++done === this.N) resolve()
      }
      for (let i = 0; i < this.N; i++){
        setTimeout(getRandomCromosome.bind(this), 0)
      }
    })
  }

  getRandomCromosome() {
    let x = parseInt(Math.random() * (this.xSize + 1))
    let y = parseInt(Math.random() * (this.ySize + 1))
    let temp = {
      x: (x / this.nDecimals) + this.minX,
      y: (y / this.nDecimals) + this.minY,
      adn: new Array(this.adnSize)
    }
    temp.fitness = this.fitness(temp.x, temp.y)
    let yCount = 0
    for(let i = this.adnSize - 1; i >= 0; i--){
        if (yCount <= this.yBits){
            temp.adn[i] = parseInt(y % 2)
            y /= 2
            yCount++
        } else {
            temp.adn[i] = parseInt(x % 2)
            x /= 2
        }
    }
    // Calculate min and max to avoid iterating later
    this.checkBoundaries(temp)
    return temp
  }

  checkBoundaries(individual, generation = 0) {
    this.total += individual.fitness
    if (individual.fitness > this.max) this.max = individual.fitness
    if (individual.fitness < this.min) this.min = individual.fitness
    if (individual.fitness < this.minGlobal) {
      this.minCromosome = individual
      this.minGlobal = individual.fitness
      this.minGeneration = generation
    }
  }

  fitness (x, y) {
    return 100.00 * Math.sqrt(Math.abs(y - 0.01 * x * x)) + 0.01 * Math.abs(x + 10)
  }

  main() {
    this.getInitialPopulation()
    .then(this.interval.bind(this))
    .then(() => {
      console.log('Ganador: ', this.minCromosome, 'Generation:', this.minGeneration)
    })
    .catch(console.trace)
  }

  interval(generation = 0) {
    return new Promise((resolve) => {
      if (generation++ >= this.GENERATIONS) return resolve()
      console.log(`Generation ${generation}, min: ${this.min}, avg: ${this.total / this.N}, global(${this.minGeneration}): ${this.minGlobal}`)
      this.getRoulette()
      .then(this.getCouples.bind(this))
      .then((parents) => {
        this.resetGeneration()
        return this.crossOver(parents, generation)
      }).then(() => {
        return this.interval(generation)
      })
      .then(resolve)
    })
  }

  // Applies rulette rules in order
  getRoulette() {
    return new Promise((resolve) => {
      let n = -1
      this.totalRoulette = 0;
      while (++n < this.population.length) {
          let individual = this.population[n]
          this.totalRoulette += this.max - this.min - individual.fitness
          // TODO: Sort by roulette for binnary search
          individual.roulette = this.totalRoulette;
      }
      resolve()
    })
  }

  getCouples() {
    // Se seleccionan N_COUPLES
    return new Promise((resolve) => {
      let parents = []
      let i = 0
      const getCouple = () => {
        let n1 = Math.random() * this.totalRoulette
        let n2 = Math.random() * this.totalRoulette
        let couple = {}
        this.population.some((individual) => {
          if (individual.roulette >= n1 && !couple.parent1) {
            couple.parent1 = individual
          }
          // TODO: Check if it should be 'else' to avoid both parents to be the same individual
          if (individual.roulette >= n2 && !couple.parent2) {
            couple.parent2 = individual
          }
          // Break when both are defined
          return couple.parent1 && couple.parent2
        })
        parents.push(couple)
        if (parents.length === this.N_COUPLES) resolve(parents)
      }
      while (i++ < this.N_COUPLES) {
        setTimeout(getCouple.bind(this), 0)
      }
    })
  }

  resetGeneration() {
    this.population = []
    this.min = Number.MAX_VALUE
    this.max = Number.MIN_VALUE
    this.total = 0
  }

  binary2x(child) {
    let xCount = 0, x = 0
    for (let i = this.adnSize - this.yBits - 2; i >= 0; i--) {
      if (child.adn[i]) {
        x += Math.pow(2, xCount)
      }
      xCount++
    }
    child.x = x / this.nDecimals + this.minX
  }

  binary2y(child) {
    let yCount = 0, y = 0
    for (let i = this.adnSize - 1; i >= 0; i--) {
      if (child.adn[i] && yCount <= this.yBits){
          y += Math.pow(2, yCount)
      }
      yCount++
    }
    child.y = y / this.nDecimals + this.minY
  }

  crossOver(parents, generation) {
    return new Promise((resolve) => {
      let done = 0
      const crossOver = (couple) => {
        if (!couple || !couple.parent1 || !couple.parent2) return
        let bitSplitter = parseInt(Math.random() * this.adnSize)
        let mutation1 = parseInt(Math.random() * this.adnSize)
        let mutation2 = parseInt(Math.random() * this.adnSize)
        let doMutation = parseInt(Math.random() * 100)
        let child1 = {adn: new Array(this.adnSize)}
        let child2 = {adn: new Array(this.adnSize)}

        let i = -1
        // TODO: Parallelize
        for(let i = 0; i < this.adnSize; i++) {
          if (i < bitSplitter) {
            child1.adn[i] = couple.parent1.adn[i]
            child2.adn[i] = couple.parent2.adn[i]
          } else {
            child1.adn[i] = couple.parent2.adn[i]
            child2.adn[i] = couple.parent1.adn[i]
          }
          if (doMutation < this.MUTATION) {
            if (i === mutation1) {
                child1.adn[i] = !child1.adn[i]
            }
            if (i === mutation2) {
                child2.adn[i] = !child2.adn[i]
            }
          }
        }
        this.binary2x(child1)
        this.binary2y(child1)
        this.binary2x(child2)
        this.binary2y(child2)
        child1.fitness = this.fitness(child1.x, child1.y)
        child2.fitness = this.fitness(child2.x, child2.y)
        this.checkBoundaries(child1, generation)
        this.checkBoundaries(child2, generation)
        this.population.push(child1, child2)
      }
      let n = 0
      while (n++ < parents.length) {
        ((couple) => {
          setTimeout((() => {
            crossOver(couple)
            if (++done === parents.length) resolve()
          }).bind(this), 0)
        })(parents[n])
      }
    })
  }
}

const generations = new Generations()
generations.main()