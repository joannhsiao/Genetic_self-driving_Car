import random
import numpy as np

class genetic():
    def __init__(self, genes):
        self.gene_pool = genes
        self.group_num = genes.shape[0]
        self.len_gene = genes.shape[1]

    def reproduction(self, scores):
        max_score = max(scores)
        print("max score: ", max_score)

        # while len(new_population) < self.group_num:
        #     for i in range(self.group_num):
        #         if len(new_population) == self.group_num:
        #             return np.array(new_population)
        #         if random.random() < scores[i]/max_score:
        #             new_population.append(self.gene_pool[i])

        scores = [s + 1e-4 for s in scores]
        probs = [s/sum(scores) for s in scores]
        indices = random.choices(range(self.group_num), weights=probs, k=self.group_num)
        self.gene_pool = [self.gene_pool[i] for i in indices]
        self.gene_pool = np.array(self.gene_pool)

    def crossover(self, crossover_rate):
        # farther: 
        # x1 = x1 + sigma * (x1 - x2)
        # x2 = x2 - sigma * (x1 - x2)
        sigma = random.random()
        for i in range(int(self.group_num/2)):
            p1 = random.randint(0, self.group_num-1)
            p2 = random.randint(0, self.group_num-1)
            while p1 == p2:
                p2 = random.randint(0, self.group_num-1)
            rand = random.random()
            if rand > crossover_rate:
                self.gene_pool[p1] = self.gene_pool[p1] + sigma * (self.gene_pool[p1] - self.gene_pool[p2])
                self.gene_pool[p2] = self.gene_pool[p2] - sigma * (self.gene_pool[p1] - self.gene_pool[p2])

    def mutation(self, mutate_rate):
        # x = x + s * random_noise
        for i in range(self.group_num):
            for j in range(self.len_gene):
                rand = random.random()
                if rand > mutate_rate:
                    random_noise = random.uniform(-1, 1)
                    self.gene_pool[i][j] = self.gene_pool[i][j] + random_noise