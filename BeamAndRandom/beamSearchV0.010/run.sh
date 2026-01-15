test_num=100

env_name="BSearch"

 
for part in 15 25 35 45 65 95 125;
do
  for dist_type in h;
  do
    python testPolicy.py --part-num $part --env-name $env_name --test-num $test_num --dist-type $dist_type
  done
done
