#!/bin/zsh
# this session's offline meta-RL study at rho=0.7, v=10 m/s (files prefixed mrs_)
cd "$(dirname "$0")"
R=0.7; V=10.0
for a in MT REP CTX FC FCX; do
  [ -f mrs_${a}_rho${R}_v${V}.pt ] || python3 mrs_meta_rl_study.py train $a $R $V > mrs_train_${a}.log 2>&1 &
done
wait
for a in D MT REP CTX FC FCX; do
  [ -f mrs_test_${a}_rho${R}_v${V}.npy ] || python3 mrs_meta_rl_study.py test $a $R $V > mrs_test_${a}.log 2>&1 &
done
wait
python3 mrs_meta_rl_study.py report $R $V > mrs_report_v10.txt 2>&1
echo STUDY_DONE >> mrs_report_v10.txt
