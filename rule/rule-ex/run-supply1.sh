test_num=100;
for i in 15 25 35 45 65 95 125 ;  
do  
  for j in l m h ;
  do
    for k in "mp" "lpt";
    do
      python ruleTest.py --mode $k --part-num $i  --dist-type $j --test-num $test_num ;
    done
  done
done 
