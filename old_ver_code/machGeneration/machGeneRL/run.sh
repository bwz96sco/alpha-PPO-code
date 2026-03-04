test_num=100

env_name="mach-ex"
part_num=50
dist_type='h'

for mode in "pure_policy" "K2";
do 
  for mach in 10 12 14 16 18 20 22;
  do
    python testPolicy.py --mode $mode --mach-num $mach --env-name $env_name --test-num $test_num --part-num $part_num --dist-type $dist_type
  done
done