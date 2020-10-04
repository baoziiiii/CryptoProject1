from random import shuffle

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
