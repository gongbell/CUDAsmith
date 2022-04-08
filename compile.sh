#/bin/sh

i=10001
#pass=0
sum=20000
while [ $i -le $sum ]
do 
        cp ./tg/$i.cu test.cu
        make clean >> ./compile/clean.log 2>&1
        ./replace.sh >> ./compile/replace.log 2>&1
        timeout 60 make  >> ./compile/tg/$i.log 2>&1
        if [ -f "./test" ];then
             echo $i.cu
             mv ./test ./bin/tg/$i.bin
        fi    
        i=$(($i+1))
done




