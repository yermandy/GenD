P=amdgpudeadline,amdgpu
#! CLIP
# seeds=(0 1 2 3 4)

# for seed in ${seeds[@]}; do
#     sh run_job.sh -p $P wacv-Baseline-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-LN+L2-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-LN+L2+UnAl-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     #! noaug 
#     sh run_job.sh -p $P wacv-LN-noaug-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
# done

#! CLIP+components without LN
# seeds=(0 1 2 3 4)

# for seed in ${seeds[@]}; do
#     sh run_job.sh -p $P wacv-NoLN-L2-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-NoLN-L2+UnAl-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
# done

#! PE
# seeds=(0 1 2 3 4)

# for seed in ${seeds[@]}; do
    # sh run_job.sh -p $P wacv-PE_L-Baseline-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
    # sh run_job.sh -p $P wacv-PE_L-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
    # sh run_job.sh -p $P wacv-PE_L-LN+L2-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True 
    # sh run_job.sh -p $P wacv-PE_L-LN+L2+UnAl-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True 
    #! experimental
    # sh run_job.sh -p $P wacv-PE_L-LN+L2+UA-U1.0-A0.5-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True 
#     sh run_job.sh -p $P wacv-PE_L-LN+L2+UA-U1.0-A0.1-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True 
#     sh run_job.sh -p $P wacv-PE_L-LN+L2+UA-U0.5-A0.0-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True 
#     sh run_job.sh -p $P wacv-PE_L-LN+L2+UA-U1.0-A0.0-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True 
#     #! noaug
#     sh run_job.sh -p $P wacv-PE_L-LN-noaug-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
# done

#! DINOv3
# seeds=(0 1 2 3 4)

# for seed in ${seeds[@]}; do
#     sh run_job.sh -p $P wacv-DINOv3L-Baseline-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-DINOv3L-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-DINOv3L-LN+L2-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True 
#     sh run_job.sh -p $P wacv-DINOv3L-LN+L2+UnAl-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True 
    # sh run_job.sh -p $P wacv-DINOv3L-LN+L2+UA-U0.5-A0.5-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-DINOv3L-LN+L2+UA-U0.1-A0.5-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
    #! noaug
#     sh run_job.sh -p $P wacv-DINOv3L-LN-noaug-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
# done

#! Paired / Unpaired
# Seeds from 00 to 19
# seeds=(00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19)

# for seed in ${seeds[@]}; do
#     sh run_job.sh -p $P wacv-paired-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-unpaired-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-unpaired-PE_L-LN+L2-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-paired-PE_L-LN+L2-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
# done

# for seed in ${seeds[@]}; do
#     sh run_job.sh -p $P wacv-paired-CDFv2-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-unpaired-CDFv2-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
# done

#? For now only seeds 0-5
# seeds=(00 01 02 03 04 05)
# for seed in ${seeds[@]}; do
#     sh run_job.sh -p $P wacv-paired-FAVC-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-unpaired-FAVC-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
# done

# seeds=(00 01 02 03 04 05)
# for seed in ${seeds[@]}; do
#     sh run_job.sh -p $P wacv-paired-FAVC-PE_L-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_job.sh -p $P wacv-unpaired-FAVC-PE_L-LN-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
# done

#! JPEG compression robustness
# jpeg_qualities=(100 80 60 40 20 10)
# jpeg_qualities=(100)

# for q in ${jpeg_qualities[@]}; do
#     sh run_test_effort.sh Effort-jpeg${q} --test_augmentations.jpeg_quality "[${q},${q}]" --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_test_forada.sh ForAda-jpeg${q} --test_augmentations.jpeg_quality "[${q},${q}]" --throw_exception_if_run_exists False --remove_if_run_exists True
# done

# seeds=(0 1 2 3 4)
# jpeg_qualities=(100 80 60 40 20 10)

# for seed in ${seeds[@]}; do
#     for q in ${jpeg_qualities[@]}; do
#         sh run_job.sh -p amdgpufast,amdgpudeadline wacv-LN+L2+UnAl-seed${seed}-jpeg${q} --test --from_exp wacv-LN+L2+UnAl-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#         sh run_job.sh -p amdgpufast,amdgpudeadline wacv-PE_L-LN+L2-seed${seed}-jpeg${q} --test --from_exp wacv-PE_L-LN+L2-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     done
# done

#! Gaussian blur tests
# blur_kernel_sizes=(0 5 7 11 13 19)
# blur_sigmas=(0.0 0.5 1.0 1.5 2.0 3.0)
# blur_kernel_sizes=(0)
# blur_sigmas=(0.0)

# for i in ${!blur_kernel_sizes[@]}; do
#     k=${blur_kernel_sizes[$i]}
#     s=${blur_sigmas[$i]}
#     sh run_test_effort.sh Effort-blur-${k}-${s} --test_augmentations.gaussian_blur_kernel_size ${k} --test_augmentations.gaussian_blur_sigma "[${s},${s}]" --test_augmentations.gaussian_blur_prob 1.0 --throw_exception_if_run_exists False --remove_if_run_exists True
#     sh run_test_forada.sh ForAda-blur-${k}-${s} --test_augmentations.gaussian_blur_kernel_size ${k} --test_augmentations.gaussian_blur_sigma "[${s},${s}]" --test_augmentations.gaussian_blur_prob 1.0 --throw_exception_if_run_exists False --remove_if_run_exists True
# done

# seeds=(0 1 2 3 4)
# for seed in ${seeds[@]}; do
#     for i in ${!blur_kernel_sizes[@]}; do
#         k=${blur_kernel_sizes[$i]}
#         s=${blur_sigmas[$i]}
#         sh run_job.sh -p amdgpufast,amdgpudeadline wacv-PE_L-LN+L2-seed${seed}-blur-${k}-${s} --test --from_exp wacv-PE_L-LN+L2-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#     done
# done

#! Resize tests
# resize=(224 112 64)
# interpolation=(0 1 2 3 4 5) # 0=NEAREST, 1=BILINEAR, 2=BICUBIC, 3=BOX, 4=HAMMING, 5=LANCZOS

# for r in ${resize[@]}; do
#     for i in ${interpolation[@]}; do
#         sh run_test_effort.sh Effort-resize-${r}-${i} --test_augmentations.resize ${r} --test_augmentations.resize_interpolation ${i} --throw_exception_if_run_exists False --remove_if_run_exists True
#         sh run_test_forada.sh ForAda-resize-${r}-${i} --test_augmentations.resize ${r} --test_augmentations.resize_interpolation ${i} --throw_exception_if_run_exists False --remove_if_run_exists True
#     done
# done

# seeds=(0 1 2 3 4)

# for r in ${resize[@]}; do
#     for i in ${interpolation[@]}; do
#         for seed in ${seeds[@]}; do
#             sh run_job.sh -p amdgpufast,amdgpudeadline wacv-PE_L-LN+L2-seed${seed}-resize-${r}-${i} --test --from_exp wacv-PE_L-LN+L2-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#         done
#     done
# done


#! DEBUG

# sh run_test_effort.sh Effort-blur-0-0.0-tmp --test_augmentations.gaussian_blur_kernel_size 0 --test_augmentations.gaussian_blur_sigma "[0.0, 0.0]" --test_augmentations.gaussian_blur_prob 1.0 --throw_exception_if_run_exists False --remove_if_run_exists True
# sh run_test_forada.sh ForAda-blur-0-0.0-tmp --test_augmentations.gaussian_blur_kernel_size 0 --test_augmentations.gaussian_blur_sigma "[0.0, 0.0]" --test_augmentations.gaussian_blur_prob 1.0 --throw_exception_if_run_exists False --remove_if_run_exists True    

# sh run_test_effort.sh Effort-jpeg100-tmp --test_augmentations.jpeg_quality "[100,100]" --throw_exception_if_run_exists False --remove_if_run_exists True
# sh run_test_forada.sh ForAda-jpeg100-tmp --test_augmentations.jpeg_quality "[100,100]" --throw_exception_if_run_exists False --remove_if_run_exists True


#! Uniformity-alignment α, β hyperparameter sweep
# seeds=(0)
# alphas=(0.0 0.1 0.5 1.0 5.0)
# betas=(0.0 0.1 0.5 1.0 5.0)

# for seed in ${seeds[@]}; do
#     for alpha in ${alphas[@]}; do
#         for beta in ${betas[@]}; do
#             sh run_job.sh -p $P wacv-PE_L-LN+L2+UnAl-sweep-A${alpha}-U${beta}-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#             sh run_job.sh -p $P wacv-DINOv3L-LN+L2+UnAl-sweep-A${alpha}-U${beta}-seed${seed} --throw_exception_if_run_exists False --remove_if_run_exists True
#         done
#     done
# done