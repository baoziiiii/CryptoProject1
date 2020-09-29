from random import shuffle
from random import randint
from collections import Counter
from collections import OrderedDict
from math import ceil
import re
import datetime
import json


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




lines = []
with open("word.txt") as f:
    lines = f.readlines()

dictionary = []
MAX_LINES = 10000
for line in lines:
    if re.match("^[a-z]+$", line.strip()):
        dictionary.append(line.strip())
shuffle(dictionary)
dictionary = dictionary[:MAX_LINES]
dictionary_set = set(dictionary)

def rand_sentence(N):
    s = []
    for _ in range(N):
        i = randint(0,len(dictionary)-1)
        s.append(dictionary[i])
    return " ".join(s)

def rand_key():
    keyspaces = list(range(106))
    shuffle(keyspaces)
    return keyspaces

def get_keytable(key):
    kt = {}
    i = 0
    for c,f in sorted(frequency.items()):
        kt[c] = key[i:(i+f)]
        i += f
    return kt

def encrypt(plain, key):
    keytable = get_keytable(key)
    crypt = []
    for i,p in enumerate(plain):
        crypt.append(keytable[p][i % len(keytable[p])])
    return crypt


def get_decrypt_table(key):
    t = []
    i = 0
    for c,f in sorted(frequency.items()):
        for _ in range(f):
            t.append(c)
            i += 1
    d = {}
    for i,k in enumerate(key):
        d[k] = t[i]
    return d

def decrypt(crypt, decrypt_table):
    plain = "".join([decrypt_table[c] for c in crypt])
    return plain

def search_range(lb,hb):
    result = []
    for k,v in frequency.items():
        if lb <= v <= hb:
            result.append(k)
    return result


def crack2(crypt, N):

    range_list = [0 for _ in range(106)]
    range_list[1] = search_range(1,1)
    range_list[2] = search_range(1,3)
    range_list[3] = search_range(2,4)
    range_list[4] = search_range(3,5)
    range_list[5] = search_range(3,7)
    range_list[6] = search_range(4,10)
    range_list[7] = search_range(3,10)

    for i in range(8,len(range_list)):
        range_list[i] = search_range(7,20)


    crypt = list(map(str,crypt))
    set_of_crypt = set(crypt)


    crypt_cntr = Counter(crypt)
    
    d = {}
    for k,v in crypt_cntr.items():
        d[k] = ceil(v/len(crypt)*106)
    print(d)


    crypt_search = {}
    for ccs_k,ccs_v in crypt_cntr.items(): 
        crypt_search[ccs_k] = range_list[ceil(ccs_v/len(crypt)*106)]
    
    print(crypt_search)

    counter = init_Counter(frequency)
    plains = set()
    memo = {}

    def solve(C, c_set, counter, tmp):
        nonlocal memo
        nonlocal plains
        nonlocal N
        if tuple(tmp) in memo:
            return memo[tuple(tmp)]
        if len(c_set) == 0:
            p = "".join(tmp)
            if verify(p):
                fprintf(p)
                plains.add(p)
                N -= 1
                return True
            else:
                return False
        
        for c in list(c_set):
            if len(c_set) == C:
                print("[{}]".format(c))
            for ch in crypt_search[c]:
                if ch not in counter:
                    continue
                if len(c_set) == C:
                    print('[{},{}]'.format(ch,c))
                t = [ ch  if x == c else x for x in tmp ]
                if verify("".join(t)) == False:
                    continue
                decrease_Counter(counter,ch)
                c_set.remove(c)
                ret = solve(C,c_set,counter,t)
                memo[tuple(t)] = ret
                c_set.add(c)
                increase_Counter(counter,ch)
                if ret == True:
                    if N <= 0:
                        return True
        return False
    
    solve(len(set_of_crypt),set_of_crypt,counter,crypt)
    if plains:
        return list(plains)
    return None




sorted_frequency = OrderedDict(sorted(frequency.items(),key = lambda x:x[1]))



def crack(crypt,N):
    crypt = list(map(str,crypt))
    set_of_crypt = set(crypt)
    counter = init_Counter(frequency)
    plains = set()
    memo = {}
    def solve(C, c_set, counter, tmp):
        nonlocal memo
        nonlocal plains
        nonlocal N
        if tuple(tmp) in memo:
            return memo[tuple(tmp)]
        if len(c_set) == 0:
            p = "".join(tmp)
            if verify(p):
                fprintf(p)
                plains.add(p)
                N -= 1
                return True
            else:
                return False
        for ch in sorted_frequency.keys():
            if len(c_set) == C:
                print("[{}]".format(ch))
            if ch not in counter:
                continue
            for c in list(c_set):
                if len(c_set) == C:
                    print('[{},{}]'.format(ch,c))
                t = [ ch  if x == c else x for x in tmp ]
                if verify("".join(t)) == False:
                    continue
                decrease_Counter(counter,ch)
                c_set.remove(c)
                ret = solve(C,c_set,counter,t)
                memo[tuple(t)] = ret
                c_set.add(c)
                increase_Counter(counter,ch)
                if ret == True:
                    if N <= 0:
                        return True
        return False
    
    solve(len(set_of_crypt),set_of_crypt,counter,crypt)
    if plains:
        return list(plains)
    return None


def verify(_str):
    if _str.startswith(" ") or _str.endswith(" "):
        return False
    for word in _str.split(" "):
        if word.isalpha() and word not in dictionary_set:
            return False
    return True

def init_Counter(d):
    return Counter(d)


def increase_Counter(counter,key):
    counter[key] += 1

def decrease_Counter(counter,key):
    counter[key] -= 1
    if counter[key] == 0:
        del counter[key]

output = 'log{:%Y%m%d-%H%M%S}.txt'.format(datetime.datetime.now())
def fprintf(line,print_to_console=True):
    with open(output,"a") as f:
        f.write(line+'\n')
        if print_to_console:
            print(line)


n = int(input("How many words of plaintext you want to generate?\n"))
p = rand_sentence(n)
fprintf("How many words of plaintext you want to generate?\n{}".format(n),False)

fprintf("\nGenerated random sentence from dictionary:\n{}".format(p))

d = {}
for k,v in dict(Counter(p)).items():
    pf = ceil(v/len(p)*105)
    d[k] = pf

fprintf("\nFrequencies of generated sentence:\n{}".format(OrderedDict(sorted(d.items(), key=lambda t: t[0]))))

key = rand_key()
fprintf("\nGenerated random key:\n{}".format(key))
crypt = encrypt(p,key)
fprintf("\nCrypt:\n{}\n".format(crypt))

r = int(input("How many successful results you expect?\n"))
fprintf("How many successful results you expect?\n{}".format(r),False)
fprintf("\nCracking...")
fprintf("\nResults:\n{}".format(crack2(crypt,r)))