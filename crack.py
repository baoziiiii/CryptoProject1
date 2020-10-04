from random import randint
from random import shuffle
from random import sample
from random import random
from collections import defaultdict
import re
import datetime
from collections import Counter
from collections import OrderedDict
from math import ceil
import sys
import nltk
import ssl
from ast import literal_eval
import numpy as np
from p_cipher import Permutation_Cipher

# pip install nltk numpy

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
output_file = 'log{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())

def fprint(line):
    with open(output_file, "a") as f:
        f.write(str(line)+'\n')
    print(line)


# ugram_score = {}
# ugram_score[' '] = 19
# ugram_score['a'] = 7
# ugram_score['b'] = 1
# ugram_score['c'] = 2
# ugram_score['d'] = 4
# ugram_score['e'] = 10
# ugram_score['f'] = 2
# ugram_score['g'] = 2
# ugram_score['h'] = 5
# ugram_score['i'] = 6
# ugram_score['j'] = 1
# ugram_score['k'] = 1
# ugram_score['l'] = 3
# ugram_score['m'] = 2
# ugram_score['n'] = 6
# ugram_score['o'] = 6
# ugram_score['p'] = 2
# ugram_score['q'] = 1
# ugram_score['r'] = 5
# ugram_score['s'] = 5
# ugram_score['t'] = 7
# ugram_score['u'] = 2
# ugram_score['v'] = 1
# ugram_score['w'] = 2
# ugram_score['x'] = 1
# ugram_score['y'] = 2
# ugram_score['z'] = 1

# gram_score = {
#     'eee':-5,
#     'e ': 2,
#     ' t':1,
#     'he':1,
#     'th':1,
#     ' a':1,
#     '   ':-10,
#     'ing':5,
#     's ':1,
#     '  ':-5,
#     ' th':5,
#     'the':5,
#     'he ':5,
#     'and':5
# }

class Individual:

    permutation_cipher = Permutation_Cipher()

    def __init__(self, crypt, key_length, gram_score = None, key = None, plaintext = None):
        self.fitness = 0
        self.crypt = crypt
        self.key_length = key_length
        self.gram_score = gram_score
        if key:
            self.key = key
        else:
            self.key = list(range(self.key_length))
            shuffle(self.key)
        self.plaintext = plaintext
        # self.expected_fitness = self.calExpectedFitness2(plaintext)
        self.calcFitness()

    def decrypt(self):
        plain = self.permutation_cipher.decrypt(self.crypt,self.permutation_cipher.get_decrypt_table(self.key))
        return plain

    def calcFitness(self):
        plain = self.decrypt()
        if self.plaintext == None:
            self.fitness = self.calFitness2(plain)
        else:
            self.fitness = self.calFitness3(plain)
        return self.fitness

    def calFitness2(self, plain):
        fitness = 0
        for i in range(len(plain)-8):
            p = plain[i:i+3] 
            for j in range(3,8):
                p += plain[i+j]
                if p in self.gram_score:
                    fitness +=  self.gram_score[p]
        return fitness
    
    def calFitness3(self, plain):
        fitness = 0
        K0 = 300
        for i in range(len(plain)):
            if plain[i] == self.plaintext[i]:
                pcf = self.permutation_cipher.frequency[plain[i]]
                if pcf  == 1:
                    fitness += 10000000
                elif pcf == 2:
                    fitness += 500000
                elif pcf == 3:
                    fitness += 100000 
                elif pcf == 4:
                    fitness += 10000
                elif pcf == 5:
                    fitness += 5000 
                elif pcf == 6:
                    fitness += 1000 
                else:
                    fitness += K0 
        return fitness

    # def calExpectedFitness(self, plain):
    #     fitness = 0
    #     for i in range(len(plain)-8):
    #         p = plain[i:i+4] 
    #         for j in range(4,8):
    #             p += plain[i+j]
    #             if p in self.gram_score:
    #                 fitness += self.gram_score[p]
    #     return fitness
    

    #  def calcFitnes4(self):
    #      self.fitness = 0
    #     plain = self.decrypt()
    #     length = len(plain)
    #     for i in range(length-1):
    #         ugram_score = 0
    #         bigram_score = 0
    #         tigram_score = 0
    #         K1 = 1
    #         K2 = 5
    #         K3 = 10
    #         ugram = plain[i]
    #         ugram_score = self.permutation_cipher.frequency[ugram]
    #         bigram = ugram + plain[i+1]
    #         if bigram in gram_score:
    #             bigram_score += gram_score[bigram]
    #         if i < length - 2:
    #             tigram = bigram + plain[i+2]
    #             if tigram in gram_score:
    #                 tigram_score += gram_score[tigram]
            
    #         self.fitness += K1 * ugram_score + K2 * bigram_score + K3 * tigram_score

    #     return self.fitness


class Population:
    
    def __init__(self, size, crypt, key_length, plaintext, gram_score):
        self.crypt = crypt
        self.key_length = key_length
        self.gram_score = gram_score
        self.plaintext = plaintext
        self.pop_size = size
        self.init()

    def init(self):
        self.individuals = np.array([Individual(self.crypt, self.key_length, gram_score= self.gram_score ,plaintext = self.plaintext) for i in range(self.pop_size)])
        self.individuals_dict = defaultdict(int)
        for individual in self.individuals:
            self.individuals_dict[tuple(individual.key)] += 1
        self.cal_fittest()

    def cal_fittest(self):
        self.fittest = self.individuals[self.get_fittest_index()]
        return self.fittest
    
    def get_fittest_index(self):
        max_index = -1
        _max = float('-inf')
        for i,ind in enumerate(self.individuals):
            if ind.fitness > _max:
                _max = ind.fitness
                max_index = i
        return max_index
    
    def get_least_fittest_index(self):
        min_index = -1
        _min = float('inf')
        for i,ind in enumerate(self.individuals):
            if ind.fitness < _min:
                _min = ind.fitness
                min_index = i
        return min_index

    def replace(self, replace_index, replace, revive_threshold):
        rk = tuple(replace.key)
        if rk in self.individuals_dict:
            self.individuals_dict[rk] += 1
            self.revive(rk, revive_threshold)
            return
        del self.individuals_dict[tuple(self.individuals[replace_index].key)]
        self.individuals[replace_index] = replace
        self.individuals_dict[rk] += 1
    
    # deprecated
    def revive(self, key, revive_threshold):
        if self.individuals_dict[key] > revive_threshold and False:
            fprint("revive...")
            self.init()
            # for i, individual in enumerate(self.individuals):
            #     if tuple(individual.key) == key:
            #         fprint("revive...")
            #         new_key = list(range(self.key_length))
            #         shuffle(new_key)
            #         nc = Individual(self.crypt, self.key_length ,key = new_key)
            #         nc.calcFitness()
            #         self.replace(i, nc, revive_threshold)
            #         return


class TranspositionGA:
    def __init__(self, crypt, key_length, plaintext, population_size, gram_score):
        self.crypt = crypt
        self.key_length = key_length
        self.plaintext = plaintext
        self.population_size = population_size
        self.gram_score = gram_score
        self.permutation_cipher = Permutation_Cipher()
        self.f_table = sorted(self.permutation_cipher.frequency.items())
        self.tmp_table = list(range(len(self.f_table)))
        base = 0
        for i in range(len(self.f_table)):
            self.f_table[i] = (self.f_table[i][0],self.f_table[i][1],base)
            base += self.f_table[i][1]

        self.init()
    
    def init(self):
        self.generationCount = 0
        fprint("Initializing...Please wait")
        self.population = Population(self.population_size, self.crypt, self.key_length,self.plaintext, self.gram_score)
        self.fittest = self.population.cal_fittest()
        fprint("Generation: {} Fittest: {}".format(self.generationCount,self.fittest.fitness))

    # selection based on the prob,  prob ~ fitness
    def selection(self):
        p = []
        offset = abs(self.population.individuals[self.population.get_least_fittest_index()].fitness)
        _sum = sum((i.fitness+offset)for i in self.population.individuals)
        for i in self.population.individuals:
            p.append((i.fitness+offset)/_sum)
        return np.random.choice(self.population.individuals,2, p = p,replace=False)

    def crossover(self):
        si1,si2 = self.selection()
        return self.crossover_i2(si1,si2)

    # two point crossover
    def crossover_i2(self, si1, si2):
        P = 106
        c1 = [0]*P 
        c2 = [0]*P
        p1 = si1.key
        p2 = si2.key
        tmp_set = set()
        L = list(range(P+1))
        s = sample(L,2)
        st,ed = min(s),max(s)

        for i in range(st,ed):
            c1[i] = p1[i]
            tmp_set.add(p1[i])

        i = 0
        k = 0
        while i < st and k < P:
            if p2[k] not in tmp_set:
                c1[i] = p2[k]
                tmp_set.add(p2[k])
                i += 1
            else:
                k += 1
        i = ed
        while i < P and k < P:
            if p2[k] not in tmp_set:
                c1[i] = p2[k]
                tmp_set.add(p2[k])
                i += 1
            else:
                k += 1

        tmp_set.clear()

        for i in range(st,ed):
            c2[i] = p2[i]
            tmp_set.add(p2[i])

        i = 0
        k = 0
        while i < st and k < P:
            if p1[k] not in tmp_set:
                c2[i] = p1[k]
                tmp_set.add(p1[k])
                i += 1
            else:
                k += 1
        i = ed
        while i < P and k < P:
            if p1[k] not in tmp_set:
                c2[i] = p1[k]
                tmp_set.add(p1[k])
                i += 1
            else:
                k += 1
        
        c_i1 = Individual(self.crypt, self.key_length , self.gram_score, key = c1, plaintext = self.plaintext)
        c_i1.calcFitness()
        c_i2 = Individual(self.crypt, self.key_length , self.gram_score, key = c2, plaintext = self.plaintext)
        c_i2.calcFitness()

        c_i_max= max(c_i1, c_i2, key=(lambda x:x.fitness))

        self.population.replace(self.population.get_least_fittest_index(), c_i_max, self.revive_threshold)

    # single point crossover
    def crossover_i1(self, si1, si2):
        P = 106
        c1 = [0]*P 
        c2 = [0]*P
        p1 = si1.key
        p2 = si2.key
        tmp_set = set()

        # r = randint(0,P-1)
        # for i in range(r):
        #     c1[i] = p1[i]
        #     tmp_set.add(p1[i])
        # i = 0
        # k = 0
        # while i < P - r and k < P:
        #     if p2[k] not in tmp_set:
        #         c1[i + r] = p2[k]
        #         tmp_set.add(p2[k])
        #         i += 1
        #     else:
        #         k += 1
        
        # tmp_set.clear()
        
        # r = randint(0,P-1)
        # for i in range(P-1,r-1,-1):
        #     c2[i] = p1[i]
        #     tmp_set.add(p1[i])
        # i = 1
        # k = P-1
        # while  i <= r and k >= 0:
        #     if p2[k] not in tmp_set:
        #         c2[r - i] = p2[k]
        #         tmp_set.add(p2[k])
        #         i += 1
        #     else:
        #         k -= 1

        r = randint(0,P-1)
        if r > P//2:
            for i in range(r):
                c1[i] = p1[i]
                tmp_set.add(p1[i])
            i = P - 1
            k = P - 1
            while i >= r and k >= 0:
                if p2[k] not in tmp_set:
                    c1[i] = p2[k]
                    tmp_set.add(p2[k])
                    i -= 1
                else:
                    k -= 1
            tmp_set.clear()


            for i in range(r):
                c2[i] = p2[i]
                tmp_set.add(p2[i])
            i = P - 1
            k = P - 1
            while i >= r and k >= 0:
                if p1[k] not in tmp_set:
                    c2[i] = p1[k]
                    tmp_set.add(p1[k])
                    i -= 1
                else:
                    k -= 1
        else:
            for i in range(P-1,r-1,-1):
                c1[i] = p1[i]
                tmp_set.add(p1[i])

            i = 0
            k = 0
            while i < r and k < P:
                if p2[k] not in tmp_set:
                    c1[i] = p2[k]
                    tmp_set.add(p2[k])
                    i += 1
                else:
                    k += 1

            tmp_set.clear()

            for i in range(P-1,r-1,-1):
                c2[i] = p2[i]
                tmp_set.add(p2[i])
            i = 0
            k = 0
            while i < r and k < P:
                if p1[k] not in tmp_set:
                    c2[i] = p1[k]
                    tmp_set.add(p1[k])
                    i += 1
                else:
                    k += 1

        # fprint("p1:{}\np2:{}\nc1:{}\nc2:{}".format(p1,p2,c1,c2))

        c_i1 = Individual(self.crypt, self.key_length , self.gram_score,  key = c1, plaintext = self.plaintext)
        c_i1.calcFitness()
        c_i2 = Individual(self.crypt, self.key_length , self.gram_score, key = c2, plaintext = self.plaintext)
        c_i2.calcFitness()

        c_i_max= max(c_i1, c_i2, key=(lambda x:x.fitness))

        self.population.replace(self.population.get_least_fittest_index(), c_i_max, self.revive_threshold)
    

    # randomly chosse two positions and swap : choose principal: first randomly choose from 1~27, then randomly choose the offset 
    def mutation(self):

        c1,c2 = self.selection()
        cc1 = list(c1.key)
        cc2 = list(c2.key)
        
        if self.fittest.fitness < 0:
            si1 = 0
            si2 = randint(1,len(self.f_table)-1)
        else:
            si1,si2 = sample(self.tmp_table,2)
        s1 = self.f_table[si1]
        s2 = self.f_table[si2]
        ss1 = s1[2] + randint(0,s1[1]-1)
        ss2 = s2[2] + randint(0,s2[1]-1)

        cc1[ss1], cc1[ss2] = cc1[ss2], cc1[ss1]

        if self.fittest.fitness < 0:
            si1 = 0
            si2 = randint(1,len(self.f_table)-1)
        else:
            si1,si2 = sample(self.tmp_table,2)
        s1 = self.f_table[si1]
        s2 = self.f_table[si2]
        ss1 = s1[2] + randint(0,s1[1]-1)
        ss2 = s2[2] + randint(0,s2[1]-1)
        cc2[ss1], cc2[ss2] = cc2[ss2], cc2[ss1]

        cc_i1 = Individual(self.crypt, self.key_length , self.gram_score, key = cc1, plaintext = self.plaintext)
        cc_i1.calcFitness()
        cc_i2 = Individual(self.crypt, self.key_length , self.gram_score, key = cc2, plaintext = self.plaintext)
        cc_i2.calcFitness()

        # if cc_i1.fitness > c_i_max.fitness or cc_i2.fitness > c_i_max.fitness:
        c_i_max= max(cc_i1, cc_i2, key=(lambda x:x.fitness))
        self.population.replace(self.population.get_least_fittest_index(), c_i_max, self.revive_threshold)

    def run(self, fitness_threshold, generation_limit, crossover_rate, mutation_rate):
        results = []
        self.revive_threshold = generation_limit//500
        while self.generationCount < generation_limit :        
            self.generationCount += 1
            for _ in range(3):
                if random() < crossover_rate:
                    self.crossover()
                    if random() < mutation_rate:
                        self.mutation()
            
            self.fittest = self.population.cal_fittest()
            # fprint("Generation: {} Progress: {}".format(self.generationCount,self.fittest.fitness/fitness_threshold))
            fprint("Generation: {} Fitness: {}\n Key: {}".format(self.generationCount,self.fittest.fitness, self.fittest.key))

            if self.fittest.fitness >= fitness_threshold:
                if results and tuple(self.fittest.key) == tuple(results[-1][2]):
                    continue
                r = (self.fittest.decrypt(),self.fittest.fitness,self.fittest.key)
                results.append(r)
                fprint("Solution Found:\n{}\nFitness:{} Key:{}".format(r[0],r[1],r[2]))
                switch = input("\nIs this solution bingo? Enter b to break the program. Otherwise press Enter to continue searching...")
                if switch == 'b':
                    break
                elif switch == 're': # restart
                    results.clear()
                    self.init()

        # for i,r in enumerate(results):
        #     fprint("{}.\n{}\nFitness:{} Key:{}".format(i,r[0],r[1],r[2]))

        best_guess = self.fittest.decrypt()
        return best_guess, self.fittest.fitness, results




# import gram from a text
def import_gram(text):
    ugram_score = {}
    gram_score = {}
    eg = nltk.everygrams(text,min_len=1,max_len=3)
    #compute frequency distribution for all the bigrams in the text
    fdist = nltk.FreqDist(eg)

    for k,v in fdist.items():
        k = ''.join(k)
        if len(k) == 1:
            ugram_score[k] = v
        elif 2 <= len(k) <= 3:
            gram_score[k] = v
    
    for k in range(2,100):
        gram_score[k*' '] = int(-(100*(k)))
    return ugram_score,gram_score

# import gram from word list
def import_word_gram():
    gram_score = {}
    with open('word.txt') as pf:
        words = pf.readlines()
    words = [word.strip() for word in words]
    wordstr = ' '.join(words)
    eg = nltk.everygrams(wordstr,min_len=4,max_len=8)
    fdist = nltk.FreqDist(eg)

    for k,v in fdist.items():
        k = ''.join(k)
        if k.count(' ') >= 2 or (k.count(' ')== 1 and k[0] != ' ' and k[-1] != ' '):
            continue
        gram_score[k] = ((15**len(k))//100000)*v+1
    for k in range(2,100):
        gram_score[k*' '] = int(-(100*k))
    for word in words:
        if len(word) <= 6:
            gram_score[' '+word+' '] = 100000
    # fprint(gram_score)
    return gram_score



with open('dictionary_test1.txt') as pf:
    dictionary_test1 = pf.readlines()
dictionary_test1 = [l.strip() for l in dictionary_test1]

# import gram from dictionary_test1.txt
def import_p_dict_gram(row_idx):
    gram_score = {}
    line = dictionary_test1[row_idx]
    eg = nltk.everygrams(line,min_len=6,max_len=10)

    fdist = nltk.FreqDist(eg)

    for k,v in fdist.items():
        k = ''.join(k)
        if k.count(' ') >= 2:
            continue
        gram_score[k] = ((20**len(k))//100000000)*v+1  # ...
    for i in range(2,50):
        gram_score[i*' '] = int(-(100**i))
    for word in line.split():
        if len(word) <= 8:
            gram_score[' '+word+' '] = 10000*2**len(word)  # ...
    # fprint(gram_score)
    return gram_score
    

def rand_sentence(N):
    with open('word.txt') as pf:
        words = pf.readlines()
    s = []
    for _ in range(N):
        i = randint(0,len(words)-1)
        s.append(words[i].strip())
    return " ".join(s)

def test1(crypt, gram_score = None):
    K = 106
    results = [0]*5
    for i in range(5):
        plain_to_check = dictionary_test1[i]
        if len(plain_to_check) < len(crypt):
            continue
        # expected_fitness = Individual(crypt, key_length = K ,gram_score = gram_score, key = key, plaintext = plain_to_check).calcFitness()
        # fprint("\nExpected fitness:{}\n".format(expected_fitness))

        d = TranspositionGA(crypt, K, plain_to_check , population_size = 10, gram_score = gram_score)
        _,fitness,_= d.run(generation_limit = 100, fitness_threshold = float('inf'),crossover_rate = 1, mutation_rate = 1 )
        results[i] = fitness
    
    s = sum(results)
    for i,f in enumerate(results):
        fprint("Plain{}\tConfidence:\t{}".format(i,f/s))

    fprint("\nMy Guess:\n{}".format(dictionary_test1[max(enumerate(results),key = lambda x:x[1])[0]]))

def test2(crypt, generation_limit = 100000 ):
    K = 106
    d = TranspositionGA(crypt, K, None , population_size = 40, gram_score = import_word_gram())

    # set generation_limit to the maximum attempts you want
    best_guess, fitness,_ = d.run(generation_limit = generation_limit, fitness_threshold = float('inf'),crossover_rate = 1, mutation_rate = 0.4 )
    fprint("\nMy Guess:\n{}".format(best_guess))


if __name__ == "__main__":
    generation_limit = 1000000
    argc = len(sys.argv) 
    p_flag = False
    input_filename = None
    for arg in sys.argv[1:]:
        if arg == '-p':
            p_flag = True
        elif arg.isdigit():
            generation_limit = int(arg)
        else:
            input_filename = arg

    if p_flag == True: # given-plaintext mode, key randomly generated, good for test
        if input_filename :
            with open(input_filename,'r') as af:
                plain = af.read()
            if len(plain.split()) < 50:
                fprint("Must have > 50 words")
                exit(0)
        else:
            plain = input("Enter some plain text below (must have > 50 words ):\n")
            while len(plain.split()) < 50:
                plain = input("Enter your plain text below (must have > 50 words ):\n")

        # plain = rand_sentence(500)

        plain = re.sub(r'[^a-z]',' ',plain.lower())
        plain = re.sub(r'\s+',' ',plain)
        pc = Permutation_Cipher()
        key = pc.rand_key()
        crypt = pc.encrypt(plain, pc.get_keytable(key))

        fprint("\nFormatted plain text({}):\n{}".format(len(plain.split()),plain))
        fprint("\nGenerated a random key:\n{}".format(key))
        fprint("\nCipher text encrypted by the key:\n{}\n".format(crypt))
        input("Press Enter to start cracking...")
        test1(crypt)

    else: # ciphertext only mode
        if input_filename :
            with open(input_filename) as input_file:
                crypt = input_file.readline()
        else:
            crypt = input("Enter ciphtertext:\n").strip()
        try:
            crypt = literal_eval(crypt)
            if type(crypt) != list:
                raise Exception()
        except Exception as e:
            print(e)
            fprint("Invalid ciphertext")
            exit(0)
        test2(crypt, generation_limit=generation_limit)





