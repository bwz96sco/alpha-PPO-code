test_num=100

env_name="part-RL"
dist_type='h'

for policy in "pure_policy" "k2"
do
  # model 50-8
  for part in 20 30 40;
  do
    python testPolicy.py --model-part 50 --model-mach 8 --mode $policy --env-name $env_name --test-num $test_num --part-num $part --mach-num 8 --dist-type $dist_type
  done

  # model 50-32
  for part in 20 30 40;
  do
    python testPolicy.py --model-part 50 --model-mach 32 --mode $policy --env-name $env_name --test-num $test_num --part-num $part --mach-num 32 --dist-type $dist_type
  done

  # model 100-8
  for part in 20 30 40 50 60 70 80 90;
  do
    python testPolicy.py --model-part 100 --model-mach 8 --mode $policy --env-name $env_name --test-num $test_num --part-num $part --mach-num 8 --dist-type $dist_type
  done

  # model 100-32
  for part in 20 30 40 50 60 70 80 90;
  do
    python testPolicy.py --model-part 100 --model-mach 32 --mode $policy --env-name $env_name --test-num $test_num --part-num $part --mach-num 32 --dist-type $dist_type
  done
done
