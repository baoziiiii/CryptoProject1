# CryptoProject1

## Encryption and Decryption Tool(Optinal)
encdec_tool.py
+ -e encryption
+ -d decryption
+ -k key location
A key will be a permutation of 0 ~ 106, of which first 19 numbers are assigned to \<space\>,etc... An example of key will look like:
```
[26, 51, 25, 18, 101, 67, 48, 61, 0, 77, 31, 98, 75, 27, 78, 90, 13, 43, 97, 30, 71, 38, 46, 72, 68, 95, 91, 34, 16, 52, 104, 5, 56, 35, 69, 49, 47, 88, 17, 8, 86, 89, 100, 57, 54, 102, 96, 79, 66, 64, 62, 3, 80, 32, 7, 39, 10, 59, 45, 84, 1, 92, 70, 2, 14, 82, 21, 53, 28, 63, 33, 11, 93, 58, 6, 73, 83, 22, 55, 36, 23, 87, 60, 29, 85, 40, 74, 103, 99, 15, 65, 4, 105, 20, 19, 44, 42, 12, 37, 9, 50, 81, 94, 76, 41, 24]
```
### Usage
The following command will generate a random key(you must use -k to specify where to output the key) and use it to encrypt plain.txt. The cipher will be auto stored to 'cipher.enc'.
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
