# TODO: create shell script for Problem 2


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uUQ79pKMa35182iDBmu1CxVqR7GZz4Xs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uUQ79pKMa35182iDBmu1CxVqR7GZz4Xs" -O p2_final.zip && rm -rf /tmp/cookies.txt

mv p2_final.zip ./p2/p2_final.zip
cd p2
unzip p2_final.zip
cd ..
python3 -m p2.inference $1 $2 $3


