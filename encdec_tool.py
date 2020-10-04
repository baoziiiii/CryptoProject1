from p_cipher import Permutation_Cipher
import sys
import re
from ast import literal_eval

enc = True
input_file = None

for arg in sys.argv:
    if arg[0] == '-':
        if arg[1] == 'd':
            enc = False
    else:
        input_file = arg

pc = Permutation_Cipher()

if enc:
    with open(input_file,'r') as f:
        plain = ' '.join(f.readlines())
    plain = re.sub(r'[^a-z]',' ',plain.lower())
    plain = re.sub(r'\s+',' ',plain)
    key = pc.rand_key()
    print("Generated a random key:{}".format(key))

    crypt = pc.encrypt(plain, pc.get_keytable(key))

    with open('cipher.txt','w') as f:
        f.write(str(crypt))
    
    print("Cipher text saved in cipher.txt")

else:
    with open(input_file,'r') as f:
        crypt = f.readline()
        key = f.readline()
    try:
        crypt = literal_eval(crypt)
        if type(crypt) != list:
            print("Invalid ciphtertext.")
            exit(0)
        key = literal_eval(key)
        if type(key) != list and len(key) != 106:
            print("Invalid key.")
            exit(0)
        pc = Permutation_Cipher()
        plain = pc.decrypt(crypt, pc.get_decrypt_table(key))
        print("Decrypted plaintext:\n{}\n".format(plain))
    except Exception as e:
        print("Invalid input to decrypt.")



