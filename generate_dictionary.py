import re
from random import shuffle
from random import randint

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

with open("p_dict.txt",'w') as f:
    for _ in range(5):
        f.write(rand_sentence(500)+'\n')
