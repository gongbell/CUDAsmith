cp ./cuda_launcher.c.template ./cuda_launcher.cu
head -n1 test.cu|cut -f 1 --complement -d" "|awk '{COUNT = NF+1;print "sed -i s/PARAMS_COUNT/"COUNT"/g ./cuda_launcher.cu";}'|awk '{print $0;system($0)}'
head -n1 test.cu|cut -f 1 --complement -d" "|awk '{total="\\\"1.cu\\\"";for(i=1;i<=NF;i++){total=total ",\\\"" $i"\\\""; }print total;}'|awk '{print "sed -i s/PARAMS_LIST/"$0"/g ./cuda_launcher.cu";}'|awk '{system($0)}'
