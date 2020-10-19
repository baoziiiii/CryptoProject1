# CryptoProject1

## Encryption and Decryption Tool
encdec_tool.py
+ -e encryption
+ -d decryption

### Usage
The following command will generate a random key(you must use -k to where to output the key) and use it to encrypt plain.txt. The cipher will be auto stored to 'cipher.enc'.
```
python3 encdec_tool.py -e plain.txt -k a.key
```

The following command will decrypt the cipher by key. 
```
python3 encdec_tool.py -d cipher.enc -k a.key
```

## Crack
crack.py
+ -p \<plaintexts file\> : Known plaintexts. Each line in the plaintexts file will be treated as a plaintext.
```
python3 crack.py test1.enc -p known_plaintexts.txt 
```
+ -w \<Words file\> : Known words. Each line in the words file will be treated as a word. 
```
python3 crack.py test2.enc -w words.txt
```

+ number : You can set the limit of attempts to the program. The number can be insert anywhere.
```
python3 crack.py test2.enc 1000 -w words.txt
```

+ -a Auto mode : 
This mode will take in a plaintext instead of cipher. The program will generate a random key & encrypt itself and try to crack it, which is convenient for test.
```
python3 crack.py -a test1_plain.txt -p known_plaintexts.txt
```
If you don't provide a filename after '-a', you will need to manually input plaintext.
```
python3 crack.py -a -p known_plaintexts.txt
```

### Usage

Test1 (Known plaintexts): 
If you already have a cipher:
```
python3 crack.py test1.enc -p known_plaintexts.txt
```

Otherwise you can generate a cipher manually and crack it.
```
sed -n '3,3p' known_plaintexts.txt > plain.txt  # extract line 3 from known plaintexts.
python3 encdec_tool.py -e plain.txt -k a.key  # encrypt it, autosaved to cipher.enc
python3 crack.py cipher.enc -p known_plaintexts.txt
```

Test2 (Known word dictionaries):
```
python3 crack.py test2.enc -w words.txt
```
Test2 (Set maximum attempts to 1000)
```
python3 crack.py test2.enc 1000 -w words.txt
```

Also I provide a way to generate a random plain from known words.
```
python3 plain_creator.py words.txt -n 500 > test2_plain.txt  # randomly choose 500 words from words.txt
python3 encdec_tool.py -e test2_plain.txt -k a.key  # encrypt test2_plain.txt, autosave the result to cipher.enc
cat cipher.enc > test2.enc  
python3 crack.py test2.enc 1000 -w words.txt
```
