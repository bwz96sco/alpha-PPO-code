test_num=20;
for i in 15 25;  
do  
  for j in l m;
  do
    for k in "wspt" "wmdd";
    do
      python ruleTest.py --mode $k --part-num $i  --dist-type $j --test-num $test_num ;
    done
  done
done 
