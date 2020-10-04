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
+ -p known plain text mode. Convenient for test. Give a program a plain text, it will generate a cipher by itself and try to crack it.

### Usage
Interactive mode if doesn't provide an input file. Must provide a plain text longer than 50 words.
```
python3 crack.py -p
```
Test1:
```
python3 crack.py -p test1.txt
```
Test2:
```
python3 crack.py test2.txt
```
Test2 (Set maximum attempts to 1000)
```
python3 crack.py test2.txt 1000
```
