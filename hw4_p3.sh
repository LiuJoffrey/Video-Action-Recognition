# TODO: create shell script for Problem 3


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NiesiUJOJb7hzkCD0ppHkPOtkufrSZVR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NiesiUJOJb7hzkCD0ppHkPOtkufrSZVR" -O p3_final.zip && rm -rf /tmp/cookies.txt

mv p3_final.zip ./p3/p3_final.zip
cd p3
unzip p3_final.zip
cd ..
python3 -m p3.inference $1 $2


