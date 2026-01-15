test_num=100

env_name="part-RL"
dist_type='h'
#policy="pure_policy" 

for policy in "pure_policy" "k2"
do
  for model in 65 95 125 ;
  do
    for part in 15 25 35 45 65 95 125;
    do
      if [ $model -ge $part ]; then
        python testPolicy.py --model-part $model --mode $policy --env-name $env_name --test-num $test_num --part-num $part --dist-type $dist_type
      else
        echo "break"
        break
      fi
    done
  done
done

  