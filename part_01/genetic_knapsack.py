import random
import numpy as np

class GeneticKnapsack:
    def __init__(
        self, weights, values, capacity,
        pop_size=60, generations=50,
        crossover_rate=0.8, mutation_rate=0.02,
        elitism_size=1, tournament_k=3,
        penalty_factor=10
    ):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.n = len(weights)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism_size
        self.tournament_k = tournament_k
        self.penalty = penalty_factor
        self.population = np.random.randint(0, 2, size=(pop_size, self.n))
        self.history = {'best': [], 'avg': [], 'worst': []}

    def _repair(self, indiv):
        while indiv.dot(self.weights) > self.capacity:
            ones = np.where(indiv == 1)[0]
            indiv[np.random.choice(ones)] = 0
        return indiv

    def _fitness(self, indiv):
        w = indiv.dot(self.weights)
        v = indiv.dot(self.values)
        return v if w <= self.capacity else v - self.penalty * (w - self.capacity)

    def _evaluate(self):
        return np.array([self._fitness(ind) for ind in self.population])

    def _select(self, fits):
        selected = []
        for _ in range(self.pop_size):
            contenders = np.random.choice(self.pop_size, self.tournament_k, replace=False)
            winner = contenders[np.argmax(fits[contenders])]
            selected.append(self.population[winner].copy())
        return np.array(selected)

    def _crossover(self, parents):
        children = []
        for i in range(0, self.pop_size, 2):
            p1, p2 = parents[i], parents[(i+1) % self.pop_size]
            if random.random() < self.crossover_rate:
                mask = np.random.rand(self.n) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            children.extend([c1, c2])
        return np.array(children[:self.pop_size])

    def _mutate(self, pop):
        for indiv in pop:
            for j in range(self.n):
                if random.random() < self.mutation_rate:
                    indiv[j] ^= 1
        return pop

    def run(self):
        for _ in range(self.generations):
            fits = self._evaluate()
            best, avg, worst = fits.max(), fits.mean(), fits.min()
            self.history['best'].append(best)
            self.history['avg'].append(avg)
            self.history['worst'].append(worst)
            elites = self.population[np.argsort(fits)[-self.elitism:]]
            selected = self._select(fits)
            children = self._crossover(selected)
            mutated = self._mutate(children)
            for i in range(self.pop_size):
                mutated[i] = self._repair(mutated[i])
            mutated[:self.elitism] = elites
            self.population = mutated

        fits = self._evaluate()
        idx = np.argmax(fits)
        indiv = self.population[idx]
        return {
            'individual': indiv,
            'weight': int(indiv.dot(self.weights)),
            'value': int(indiv.dot(self.values)),
            'history': self.history
        }
