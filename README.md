# CryptoProject1

## Encryption and Decryption Tool
encdec_tool.py
+ -e encryption
+ -d decryption

### Usage
The following command will generate a random key(print to the console) and use it to encrypt plain.txt. The cipher will be stored into 'cipher.txt'
```
python3 encdec_tool.py -e plain.txt  
```

The following command will decrypt by cipher and key in the cipher_and_key.txt. The format of cipher_and_key.txt is cipher in the first line, key in the second line. The decryption result will print to the console.
```
python3 encdec_tool.py -d cipher_and_key.txt
```

## Crack
crack.py
+ -p Mknown plain text mode. Give a program a plain text, it will generate a cipher by itself and try to crack it. Convenient for Test.
+ -T test1 mode, the program will use dictionary_test1.txt to take advantage.

### Usage
Interactive mode if doesn't provide an input file. Must provide a plain text longer than 50 words.
```
python3 crack.py -p
```
Test1: 
```
python3 crack.py -p test1_plain.txt -T
```
Or generate a cipher manually
```
sed -n '3,3p' dictionary_test1.txt > plain.txt  # extract line 3 of plaintext from dictionary
python3 encdec_tool.py -e plain.txt   # encrypt it, autosaved in cipher.txt
cat cipher.txt > test1.txt
python3 crack.py test1.txt -T 
```
Test2:
```
python3 crack.py test2.txt
```
Test2 (Set maximum attempts to 1000)
```
python3 crack.py test2.txt 1000
```
