test_num=100

env_name="mach-ex"
part_num=35
dist_type='h'

for mode in "pure_policy" "K2";
do 
  for mach in 9 11 13 15 17 19;
  do
    python testPolicy.py --mode $mode --mach-num $mach --env-name $env_name --test-num $test_num --part-num $part_num --dist-type $dist_type
  done
done