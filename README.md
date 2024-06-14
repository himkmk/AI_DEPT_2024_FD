## 2024 SELF FakeDetection
최신 인공지능 기반 생성 모델을 통해 생성된 이미지 및 영상에 대한 검출을 목표로 함

---
### 실행 환경
```
docker pull bluelati98/torch2/torch2.1.0-cuda11.8-cudnn8
```
---

### Simple Test Script
```
GPU=$1
EXP=$2

dirname="model_ckpt/exp_${EXP}"
mkdir -p -- "$dirname"
python3 -m tools.test \
                --gpus ${GPU} --exp ${EXP} --res 299 --model effb5 --ckpt_used 49 --batch 32 --manual_seed 42 \
                    | tee ${dirname}/test_log.txt
```
---
### Simple Train Script
```
GPU=$1
EXP=$2

dirname="model_ckpt/exp_${EXP}"
mkdir -p -- "$dirname"

 python3 -m tools.train_LFE \
                 --gpus ${GPU} --exp ${EXP} --res 256 --model effb5 --lr 0.002 --weight_decay 1e-5 --epoch 50 --batch 32 --log_freq 100 --manual_seed 42 --last_layer False \
                 --lfe_mode basic --lfe_unnormalize_mode none --lfe_add_global_skip False \
                 --lfe_rrdb_bic_ckpt YOUR_RRDBPATH
                 --lfe_rrdb_real_ckpt YOUR_REALRRDBPATH
                     | tee -a ${dirname}/log.txt



```
### Contact
Please contact via himkmk@naver.com for any inquiries
