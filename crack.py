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
from blist import sortedlist
from ast import literal_eval

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')


output_file = 'log{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())

f = open(output_file, "a")

def fprint(line):
    f.write(str(line)+'\n')
    print(line)


class Permutation_Cipher:

    '''
    1: ['b', 'j', 'k', 'q', 'v', 'x', 'z']
    2: ['c', 'f', 'g', 'm', 'p', 'u', 'w', 'y']
    3: ['l']
    4. ['d']
    5. ['h', 'r', 's']
    6. ['i', 'n', 'o']
    7. ['a', 't']
    10.['e']
    19.[' ']
    '''

    frequency = {}
    frequency[' '] = 19
    frequency['a'] = 7
    frequency['b'] = 1
    frequency['c'] = 2
    frequency['d'] = 4
    frequency['e'] = 10
    frequency['f'] = 2
    frequency['g'] = 2
    frequency['h'] = 5
    frequency['i'] = 6
    frequency['j'] = 1
    frequency['k'] = 1
    frequency['l'] = 3
    frequency['m'] = 2
    frequency['n'] = 6
    frequency['o'] = 6
    frequency['p'] = 2
    frequency['q'] = 1
    frequency['r'] = 5
    frequency['s'] = 5
    frequency['t'] = 7
    frequency['u'] = 2
    frequency['v'] = 1
    frequency['w'] = 2
    frequency['x'] = 1
    frequency['y'] = 2
    frequency['z'] = 1

    def __init__(self):
        pass

    def rand_key(self):
        keyspaces = list(range(106))
        shuffle(keyspaces)
        return keyspaces

    def get_keytable(self, key):
        kt = {}
        i = 0
        for c,f in sorted(self.frequency.items()):
            kt[c] = key[i:(i+f)]
            i += f
        return kt

    def encrypt(self, plain, encrypt_table):
        crypt = []
        for i,p in enumerate(plain):
            crypt.append(encrypt_table[p][i % len(encrypt_table[p])])
        return crypt

    def get_decrypt_table(self, key):
        t = []
        i = 0
        for c,f in sorted(self.frequency.items()):
            for _ in range(f):
                t.append(c)
                i += 1
        d = {}
        for i,k in enumerate(key):
            d[k] = t[i]
        return d

    def decrypt(self, crypt, decrypt_table):
        plain = "".join([decrypt_table[c] for c in crypt])
        return plain

ugram_score = {}
ugram_score[' '] = 19
ugram_score['a'] = 7
ugram_score['b'] = 1
ugram_score['c'] = 2
ugram_score['d'] = 4
ugram_score['e'] = 10
ugram_score['f'] = 2
ugram_score['g'] = 2
ugram_score['h'] = 5
ugram_score['i'] = 6
ugram_score['j'] = 1
ugram_score['k'] = 1
ugram_score['l'] = 3
ugram_score['m'] = 2
ugram_score['n'] = 6
ugram_score['o'] = 6
ugram_score['p'] = 2
ugram_score['q'] = 1
ugram_score['r'] = 5
ugram_score['s'] = 5
ugram_score['t'] = 7
ugram_score['u'] = 2
ugram_score['v'] = 1
ugram_score['w'] = 2
ugram_score['x'] = 1
ugram_score['y'] = 2
ugram_score['z'] = 1

gram_score = {
    'eee':-5,
    'e ': 2,
    ' t':1,
    'he':1,
    'th':1,
    ' a':1,
    '   ':-10,
    'ing':5,
    's ':1,
    '  ':-5,
    ' th':5,
    'the':5,
    'he ':5,
    'and':5
}



class Individual:

    permutation_cipher = Permutation_Cipher()

    def __init__(self, crypt, key_length,  key = None, plaintext = None):
        self.fitness = 0
        self.crypt = crypt
        self.key_length = key_length
        if key:
            self.key = key
        else:
            self.key = list(range(self.key_length))
            shuffle(self.key)
        self.plaintext = plaintext
        self.expected_fitness = self.calExpectedFitness2(plaintext)
        self.calcFitness()

    def decrypt(self):
        plain = self.permutation_cipher.decrypt(self.crypt,self.permutation_cipher.get_decrypt_table(self.key))
        return plain

    def calFitness2(self, plain):
        fitness = 0
        for i in range(len(plain)-8):
            p = plain[i:i+3] 
            for j in range(3,8):
                p += plain[i+j]
                if p in gram_score:
                    fitness += gram_score[p]
        return fitness


    def calExpectedFitness2(self, plain):
        K0 = 1
        K1 = 1
        K2 = 2
        K3 = 5
        fitness = 0
        for i in range(len(plain)-1):
            if plain[i] == self.plaintext[i]:
                fitness += K0 
            ugram_score = 0
            bigram_score = 0
            tigram_score = 0
            ugram = plain[i]
            ugram_score = self.permutation_cipher.frequency[ugram]
            bigram = ugram + plain[i+1]
            if bigram in gram_score:
                bigram_score += gram_score[bigram]
            if i < len(plain) - 2:
                tigram = bigram + plain[i+2]
                if tigram in gram_score:
                    tigram_score += gram_score[tigram]
            fitness += K1 * ugram_score + K2 * bigram_score + K3 * tigram_score
        return fitness

 # fitness:312305429
 #          60732059

    def calcFitness(self):
        plain = self.decrypt()
        self.fitness = self.calFitness2(plain)
        # if self.fitness > self.expected_fitness * 0.95:
        #     self.fitness = 0

        #     K0 = 300
        #     for i in range(len(plain)):
        #         if plain[i] == self.plaintext[i]:
        #             pcf = self.permutation_cipher.frequency[plain[i]]
        #             if pcf  == 1:
        #                 self.fitness += 1000000 
        #             elif pcf == 2:
        #                 self.fitness += 200000 
        #             elif pcf == 3:
        #                 self.fitness += 50000 
        #             elif pcf == 4:
        #                 self.fitness += 20000 
        #             elif pcf == 5:
        #                 self.fitness += 5000 
        #             elif pcf == 6:
        #                 self.fitness += 1000 
        #             else:
        #                 self.fitness += K0 
        # else:
        #     self.fitness = self.calExpectedFitness2(plain)

        return self.fitness

    #  def calcFitness(self):
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
    
    def __init__(self, size, crypt, key_length, plaintext):
        self.crypt = crypt
        self.key_length = key_length
        self.plaintext = plaintext
        self.pop_size = size
        self.init()
        self.fittest_rand_up = 1
        self.second_fittest_rand_up = 2

        
    def init(self):
        self.individuals = sortedlist([Individual(self.crypt, self.key_length, plaintext = self.plaintext) for i in range(self.pop_size)], key = lambda x:x.fitness)
        self.individuals_dict = defaultdict(int)
        for individual in self.individuals:
            self.individuals_dict[tuple(individual.key)] += 1
        self.fittest = 0

    def get_fittest(self):
        return self.individuals[-1]

    def get_fittest_rand(self):
        r = -1*randint(1,self.fittest_rand_up)
        max_fit = self.individuals[r]
        self.fittest = max_fit.fitness
        return max_fit

    def get_second_fittest_rand(self):
        r = -1*randint(2,self.second_fittest_rand_up)
        return self.individuals[r]
    
    def get_least_fittest_index(self):
        return 0
    
    # def calculate_fitness(self):
    #     for individual in self.individuals:
    #         individual.calcFitness()
    #     self.individuals.sort(key=(lambda x:x.fitness))

    def replace(self, replace_index, replace, revive_threshold):
        rk = tuple(replace.key)
        if rk in self.individuals_dict:
            self.individuals_dict[rk] += 1
            self.revive(rk, revive_threshold)
            return
        del self.individuals_dict[tuple(self.individuals[replace_index].key)]
        del self.individuals[replace_index] 
        self.individuals.add(replace)
        self.individuals_dict[rk] += 1


    def revive(self, key, revive_threshold):
        if self.individuals_dict[key] > revive_threshold and self.fittest < 20000:
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
    def __init__(self, crypt, key_length, plaintext, population_size):
        self.crypt = crypt
        self.key_length = key_length
        self.plaintext = plaintext
        self.population_size = population_size
        self.init()
        self.permutation_cipher = Permutation_Cipher()
        self.f_table = sorted(self.permutation_cipher.frequency.items())
        self.tmp_table = list(range(len(self.f_table)))
        base = 0
        for i in range(len(self.f_table)):
            self.f_table[i] = (self.f_table[i][0],self.f_table[i][1],base)
            base += self.f_table[i][1]
    
    def init(self):
        self.generationCount = 0
        fprint("Initializing...Please wait")

        self.population = Population(self.population_size, self.crypt, self.key_length,self.plaintext)
        # self.population.calculate_fitness()
        for i in range(self.population.pop_size):
            fprint(self.population.individuals[i].key)
        self.fittest = self.population.get_fittest_rand()
        self.secondFittest = self.population.get_second_fittest_rand()
        fprint("Generation: {} Fittest: {}".format(self.generationCount,self.population.fittest))

    def selection(self):
        

    def selection(self):
        # self.population.calculate_fitness()
        self.fittest = self.population.get_fittest_rand()
        self.secondFittest = self.population.get_second_fittest_rand()

    def crossover(self):
        P = 106
        c1 = [0]*P 
        c2 = [0]*P
        p1 = self.fittest.key
        p2 = self.secondFittest.key
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
        # while i <= r and k >= 0:
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

        fprint("r:{:03d}|\np1:{}|\np2:{}|\nc1:{}|\nc2:{}".format(r,p1,p2,c1,c2))

        c_i1 = Individual(self.crypt, self.key_length , key = c1, plaintext = self.plaintext)
        c_i1.calcFitness()
        c_i2 = Individual(self.crypt, self.key_length , key = c2, plaintext = self.plaintext)
        c_i2.calcFitness()

        c_i_max= max(c_i1, c_i2, key=(lambda x:x.fitness))

        self.population.replace(self.population.get_least_fittest_index(), c_i_max, self.revive_threshold)
        return c1,c2
    
    def mutation(self, c1,c2):
   # mutation
        cc1 = list(c1)
        cc2 = list(c2)
        
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

        cc_i1 = Individual(self.crypt, self.key_length , key = cc1, plaintext = self.plaintext)
        cc_i1.calcFitness()
        cc_i2 = Individual(self.crypt, self.key_length , key = cc2, plaintext = self.plaintext)
        cc_i2.calcFitness()

        # if cc_i1.fitness > c_i_max.fitness or cc_i2.fitness > c_i_max.fitness:
        c_i_max= max(cc_i1, cc_i2, key=(lambda x:x.fitness))
        self.population.replace(self.population.get_least_fittest_index(), c_i_max, self.revive_threshold)


    # get_fittest_offspring(self)

    #add_fittest_offspring(self)

    def run(self, fitness_threshold, generation_limit):
        result = []
        self.revive_threshold = generation_limit//500
        while self.generationCount < generation_limit :
            self.generationCount += 1
            self.selection()
            c1,c2 = self.crossover()
            if random() > 0.7:
                self.mutation(c1,c2)

            self.fittest = self.population.get_fittest()
            if self.fittest.fitness >= fitness_threshold:
                if result and tuple(self.fittest.key) == tuple(result[-1][2]):
                    continue
                r = (self.fittest.decrypt(),self.fittest.fitness,self.fittest.key)
                result.append(r)
                fprint("Solution Found:\n{}\nFitness:{} Key:{}".format(r[0],r[1],r[2]))
                switch = input("\nIs this solution bingo? Enter b to break the program. Otherwise press Enter to continue searching...")
                if switch == 'b':
                    return
                elif switch == 're':
                    result.clear()
                    self.init()

            fprint("Generation: {} Fittest: {}".format(self.generationCount,self.population.fittest))
        
        if not result:
            fprint("My Guess:{}\n ".format(self.fittest.decrypt()))
            fprint("Fitness:{} Key:{}".format(self.fittest.fitness,self.fittest.key))
        else:
            for i,r in enumerate(result):
                fprint("{}.\n{}\nFitness:{} Key:{}".format(i,r[0],r[1],r[2]))


def import_gram(text):
    eg = nltk.everygrams(text,min_len=1,max_len=3)
    #compute frequency distribution for all the bigrams in the text
    fdist = nltk.FreqDist(eg)
    ugram_score.clear()
    gram_score.clear()
    for k,v in fdist.items():
        k = ''.join(k)
        if len(k) == 1:
            ugram_score[k] = v
        elif 2 <= len(k) <= 3:
            gram_score[k] = v

    fprint(gram_score)
    
    for k in range(2,100):
        gram_score[k*' '] = int(-(10*(k)))


p_dicts = None
with open('p_dict.txt') as pf:
    p_dicts = pf.readlines()

def import_word_gram():
    ugram_score.clear()
    gram_score.clear()
    with open('word.txt') as pf:
        words = pf.readlines()
    words = [word.strip() for word in words]
    wordstr = ' '.join(words)
    eg = nltk.everygrams(wordstr,min_len=4,max_len=8)
    fdist = nltk.FreqDist(eg)

    for k,v in fdist.items():
        k = ''.join(k)
        gram_score[k] = ((10**len(k))//100000)*v+1
    for k in range(2,100):
        gram_score[k*' '] = int(-(1000*k))
    for word in words:
        if len(word) <= 6:
            gram_score[' '+word+' '] = 100000
    fprint(gram_score)


    
def rand_sentence(N):
    with open('word.txt') as pf:
        words = pf.readlines()
    s = []
    for _ in range(N):
        i = randint(0,len(words)-1)
        s.append(words[i].strip())
    return " ".join(s)

argc = len(sys.argv) 
d_flag = False
input_filename = None
if argc > 1:
    for arg in sys.argv[1:]:
        if arg == '-d':
            d_flag = True
        else:
            input_filename = arg

if d_flag == False:
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

    plain = re.sub(r'[^a-z]',' ',plain.lower())
    plain = re.sub(r'\s+',' ',plain)
    import_word_gram()

    fprint("\n\nFormatted plain text({}):\n{}".format(len(plain.split()),plain))
    pc = Permutation_Cipher()
    key = pc.rand_key()

    fprint("\nGenerated a random key:\n{}".format(key))
    crypt = pc.encrypt(plain, pc.get_keytable(key))
    fprint("\nEncrypted by the key:\n{}\n".format(crypt))

    K = 106

    i = Individual(crypt, key_length = K ,key = key, plaintext= plain)
    fitness_threshold = i.calcFitness()
    fprint("\nExpected fitness:{}\n".format(fitness_threshold))

    input("Press Enter to start cracking...")

    d = TranspositionGA(crypt, K, plain, 30)
    d.run(fitness_threshold-1, 100000)

    f.close()

else:
    if input_filename :
        with open(input_filename) as input_file:
            crypt = input_file.readline()
            key = input_file.readline()
    else:
        crypt = input("Enter crypt:\n").strip()
        key = input("Enter key:\n").strip()
    try:
        crypt = literal_eval(crypt)
        if type(crypt) != list:
            fprint("Invalid crypt.")
            exit(0)
        key = literal_eval(key)
        if type(key) != list and len(key) != 106:
            fprint("Invalid key.")
            exit(0)
        pc = Permutation_Cipher()
        plain = pc.decrypt(crypt, pc.get_decrypt_table(key))
        fprint("Your plaintext:\n{}\n".format(plain))
    except Exception as e:
        print(e)
        fprint("Invalid crypt.")
        exit(0)






