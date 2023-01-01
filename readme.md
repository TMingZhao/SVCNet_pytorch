This is the pytorch implementation of SVCNet for point cloud optimiztion

## environment config
```bash
git clone https://github.com/TMingZhao/SVCNet_pytorch
cd SVCNet_pytorch

# conda environment
conda env create -f environment.yml
conda activate ENV-SVCNet

# compile
cd sampling
python setup.py install
cd ..

```
## dataset ##

dataset1 provided by [PUNet](https://github.com/yulequan/PU-Net) for denoising training and testing, download it [here](https://drive.google.com/file/d/1R21MD1O6q8E7ANui8FR0MaABkKc30PG4/view?usp=sharing).

dataset2 provided by [ECNet](https://github.com/yulequan/EC-Net) for denoising training and testing.

dataset3 provided by [PUGAN](https://github.com/liruihui/PU-GAN) for upsampling testing, download it [here](https://drive.google.com/open?id=1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC).



## train and test ##
```
# train code
python train.py 

## run demo
# You can denoise the point cloud with [denoisingX] function or upsample the point cloud with [upsamplingX] function

# for denoising
python demo.py --task denoise --ply_path data/denoise_data/star_30000n20e-3.ply --iteration 3
# for upsampling
python demo.py --task upsample --ply_path data/upsample_data/11509_Panda_v42048n20e-3.ply --iteration 3
```

