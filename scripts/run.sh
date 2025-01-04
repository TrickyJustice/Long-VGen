version=512 ##1024, 512, 256
seed=123
name=dynamicrafter_512_seed${seed}

ckpt=checkpoints/dynamicrafter_512_v1/model.ckpt
config=configs/inference_512_v1.0.yaml

prompt_dir=prompts/512/
res_dir="results"

if [ "$1" == "256" ]; then
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width 512 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 48 \
--frame_stride ${FS}
else
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 48 \
--frame_stride 24 \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae
fi


## multi-cond CFG: the <unconditional_guidance_scale> is s_txt, <cfg_img> is s_img
#--multiple_cond_cfg --cfg_img 7.5
#--loop