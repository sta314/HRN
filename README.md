# HRN
This repository is a fork of HRN work to be used for face normalization. It focuses on acquiring the frontal view after all of processing is done. Following information is only for our local project.

### Media folder information

After execution, output folder in the media folder will contain the results. It will record the face normalization image of each frame, restored angle of the face and reconstructed video with the normalized face.

### Dataset seperation
media_l_* contains long videos in the dataset (t2$)
media_s_* contains short videos in the dataset (t2$)
Each divided as 2 parts

### Execution

Execution will be done as follows:

`conda activate hrn_facenorm`
`CUDA_VISIBLE_DEVICES="0" bash -c "python custom_demo_mancrop.py --media_dir media_l_p1"`
`CUDA_VISIBLE_DEVICES="0" bash -c "python custom_demo_mancrop.py --media_dir media_l_p2"`
`CUDA_VISIBLE_DEVICES="1" bash -c "python custom_demo_mancrop.py --media_dir media_l_p3"`
`CUDA_VISIBLE_DEVICES="1" bash -c "python custom_demo_mancrop.py --media_dir media_l_p4"`

then, when needed:

`conda activate hrn_facenorm`
`CUDA_VISIBLE_DEVICES="0" bash -c "python custom_demo_mancrop.py --media_dir media_s_p1"`
`CUDA_VISIBLE_DEVICES="0" bash -c "python custom_demo_mancrop.py --media_dir media_s_p2"`
`CUDA_VISIBLE_DEVICES="1" bash -c "python custom_demo_mancrop.py --media_dir media_s_p3"`
`CUDA_VISIBLE_DEVICES="1" bash -c "python custom_demo_mancrop.py --media_dir media_s_p4"`

