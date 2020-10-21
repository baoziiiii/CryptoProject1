from p_cipher import Permutation_Cipher
import sys
import re
from ast import literal_eval

enc = True
key = False
input_file = None
key_file = None

for arg in sys.argv:
    if arg[0] == '-':
        if arg[1] == 'd':
            enc = False
        elif arg[1] == 'k':
            key = True
        elif arg[1] == 'e':
            enc = True
        else:
            print("Unrecognized option.")
            exit(0)
    else:
        if key == True and key_file == None:
            key_file = arg
        else:
            input_file = arg

pc = Permutation_Cipher()

if key == True and key_file == None:
    print("Key file not specified,")
    exit(0)

if enc:
    with open(input_file,'r') as f:
        plain = ' '.join(f.readlines())
    plain = re.sub(r'[^a-z]',' ',plain.lower())
    plain = re.sub(r'\s+',' ',plain).strip()
    key = pc.rand_key()
    print("Generated a random key:{}".format(key))

    with open(key_file,'w+') as f:
        f.write(str(key))
    print("Key is saved to {}".format(key_file))

    crypt = pc.encrypt(plain, pc.get_keytable(key))
    print(crypt)
    with open('cipher.enc','w+') as f:
        f.write(",".join(list(map(str,crypt))))
        print("Cipher text saved in cipher.enc")
    

else:
    with open(input_file,'r') as f:
        crypt = f.readline()
    with open(key_file,'r') as f:
        key = f.readline()
    try:
        crypt = list(map(int,crypt.split(",")))
        key = literal_eval(key)
        if type(key) != list and len(key) != 106:
            print("Invalid key.")
            exit(0)
        pc = Permutation_Cipher()
        plain = pc.decrypt(crypt, pc.get_decrypt_table(key))
        print("Decrypted plaintext:\n{}\n".format(plain))
    except Exception as e:
        print("Invalid input to decrypt.")



