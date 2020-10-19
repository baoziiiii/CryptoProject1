import re
from random import shuffle
from random import randint
import sys



if __name__ == "__main__":
    N = 500
    n_flag = False
    for arg in sys.argv[1:]:
        if arg[0] == '-':
            if arg[1] == 'n':
                n_flag = True
            else:
                exit(0)
        elif arg.isdigit() and n_flag == True:
            n_flag = False
            N = int(arg)

    lines = []
    with open(sys.argv[1]) as f:
        lines = f.readlines()

    dictionary = []
    MAX_LINES = 10000
    for line in lines:
        if re.match("^[a-z]+$", line.strip()):
            dictionary.append(line.strip())
    shuffle(dictionary)
    dictionary = dictionary[:MAX_LINES]
    dictionary_set = set(dictionary)

    def rand_sentence(n):
        s = []
        for _ in range(n):
            i = randint(0,len(dictionary)-1)
            s.append(dictionary[i])
        return " ".join(s)

    print(rand_sentence(N))


