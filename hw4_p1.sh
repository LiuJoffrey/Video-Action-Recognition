# TODO: create shell script for Problem 1

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VAuxsIouf08KZbl_qHc0Mnbkx2HJ9l2N' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VAuxsIouf08KZbl_qHc0Mnbkx2HJ9l2N" -O p1_final.zip && rm -rf /tmp/cookies.txt

mv p1_final.zip ./p1/p1_final.zip
cd p1
unzip p1_final.zip
cd ..
python3 -m p1.inference $1 $2 $3


