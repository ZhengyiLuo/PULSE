mkdir sample_data
mkdir -p output output/HumanoidIm/ output/HumanoidIm/phc_3 output/HumanoidIm/phc_comp_3 output/HumanoidIm/pulse_vae_iclr
gdown https://drive.google.com/uc?id=1bLp4SNIZROMB7Sxgt0Mh4-4BLOPGV9_U -O  sample_data/ # filtered shapes from AMASS
gdown https://drive.google.com/uc?id=1arpCsue3Knqttj75Nt9Mwo32TKC4TYDx -O  sample_data/ # all shapes from AMASS
gdown https://drive.google.com/uc?id=1fFauJE0W0nJfihUvjViq9OzmFfHo_rq0 -O  sample_data/ # sample standing neutral data.
gdown https://drive.google.com/uc?id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc -O  sample_data/ # amass_occlusion_v3
gdown https://drive.google.com/uc?id=1BDUJ3nlub9tv1fF0UMANVUryJo1h5-lo -O  sample_data/ # amass_isaac_simple_run_upright_slim
gdown https://drive.google.com/uc?id=1ztyljPCzeRwQEJqtlME90gZwMXLhGTOQ -O  output/HumanoidIm/pulse_vae_iclr/
gdown https://drive.google.com/uc?id=1S7_9LesLjfsFYqi4Ps6Sjzyuyun0Oaxi -O  output/HumanoidIm/pulse_vae_x/
gdown https://drive.google.com/uc?id=1JbK9Vzo1bEY8Pig6D92yAUv8l-1rKWo3 -O  output/HumanoidIm/phc_comp_3/
gdown https://drive.google.com/uc?id=1pS1bRUbKFDp6o6ZJ9XSFaBlXv6_PrhNc -O  output/HumanoidIm/phc_3/
