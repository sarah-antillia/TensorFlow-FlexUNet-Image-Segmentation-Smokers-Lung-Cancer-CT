<h2>TensorFlow-FlexUNet-Image-Segmentation-Smokers-Lung-Cancer-CT (2026/11/01)</h2>
<h3>Smokers-Lung-Cancer-CT: AI Generated Pseudo Masks Segmentation Challenge</h3>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Smokers-Lung-Cancer-CT </b>
 based on
our <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">TensorFlowFlexUNet Model</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>) and a PNG
 <a href="https://drive.google.com/file/d/14S_u-FMWpAKnRyzHo1Js1I2KUFEwgGc5/view?usp=sharing">
Smokers-Lung-Cancer-CT-ImageMask-Dataset.zip</a> (<a href="https://creativecommons.org/publicdomain/zero/1.0/">
CC0: Public Domain</a>)
which was derived by us from the Kaggle  
<br><br>
<a href="https://www.kaggle.com/datasets/leelanaveenkumar/lung-cancer-ct-slice-images-metadatasynthetic">
Lung Cancer : CT Slice Images, Metadata(Synthetic)
</a><br>
by Leela Naveen Kumar.
<hr>
<b>Actual Image Segmentation for Smokers-Lung-Cancer-CT Images of 512x512 pixels</b><br>
As shown below, the inferred masks resemble the ground-truth masks. <br>
<br>
<table>
<tr>
<th width="320" height="auto">Input: image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/images/hflipped_Patient (554).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/masks/hflipped_Patient (554).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test_output/hflipped_Patient (554).png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/images/hflipped_Patient (1003).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/masks/hflipped_Patient (1003).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test_output/hflipped_Patient (1003).png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/images/vflipped_Patient (780).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/masks/vflipped_Patient (780).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test_output/vflipped_Patient (780).png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from the following Kaggle:<br><br>
<a href="https://www.kaggle.com/datasets/leelanaveenkumar/lung-cancer-ct-slice-images-metadatasynthetic">
Lung Cancer : CT Slice Images, Metadata(Synthetic)
</a><br>
<b>Lung CT Scans with Metadata for Smoking and Lung Cancer Classification</b><br>
by Leela Naveen Kumar.
<br><br>
The following explanation (excerpt) was taken the website above.
<br><br>
<b>About Dataset</b><br>
This dataset contains 1,097 anonymized CT scan images of lungs, originally sourced from the 
<a href="https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset">
<b>IQ-OTH/NCCD Lung Cancer Dataset</b> 
</a>
(Hamdallah, 2020), 
which is released under the <b>CC0: Public Domain</b> license.
<br><br>
The dataset has been enhanced with <b>synthetically generated metadata</b> to support research in:<br>
<ul>
<li>Lung cancer classification using deep learning</li>
<li>The impact of smoking status on cancer diagnosis</li>
<li>Metadata-integrated CNN model training and explainability</li>
</ul>
<br>
<b>Citation</b><br>
<pre>
 Guduru, L. N. K. (2025). Lung CT Scans with Metadata for Smoking and Lung Cancer Classification [Dataset]. Kaggle.
 Source images from: Hamdallah, A. (2020). IQ-OTH/NCCD Lung Cancer Dataset. Kaggle.
</pre>
<br>
<b>License</b><br>
Images: Public Domain – IQ-OTH/NCCD Dataset on Kaggle<br>
Metadata: Created by the uploader. Licensed under CC BY 4.0<br>
<br>
<br>
<h3>
2. Smokers-Lung-Cancer-CT ImageMask Dataset
</h3>
<h3>2.1 Download ImageMask Dataset</h3>
 If you would like to train this Smokers-Lung-Cancer-CT Segmentation model,
 please download the dataset from Google Drive  
 <a href="https://drive.google.com/file/d/14S_u-FMWpAKnRyzHo1Js1I2KUFEwgGc5/view?usp=sharing">
Smokers-Lung-Cancer-CT-ImageMask-Dataset.zip</a>. 
Expand the downloaded ImageMaskDataset and put it under the <b>./dataset</b> folder.
<br>
<pre>
./dataset
└─Smokers-Lung-Cancer-CT
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
         ├─images
         └─masks
</pre>
<br>
<b>Smokers-Lung-Cancer-CT Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/Smokers-Lung-Cancer-CT_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images in the training and valid datasets is not large enough to use for the
 training set of our segmentation model.
<br>
<h3>2.2 Derivation of ImageMask Dataset</h3>
The folder structure of the original dataset  
</a> is as follows, but it contains no annotation (mask) files.<br>
<pre>
./lung-cancer-smoking-dataset
  ├─Images
  └─metadata
</pre>
<b>Step 1</b><br>
We generated pseudo masks corresponding to the Lung-Cancer-CT images in <b>Images</b> folder 
by applying a segmentation (inference) method of the pretrained model 
<a href="TensorFlow-FlexUNet-Image-Segmentation-Lung-Cancer-CT">
TensorFlow-FlexUNet-Image-Segmentation-Lung-Cancer-CT
</a> to the master images.
<br><br>
<b>Step 2</b><br>
We generated an augmented ImageMask dataset from the pseudo masks and the master images 
by applying flipping augmentation tools to all pairs of the images and masks.
<br><br>
<b>Step 3</b><br>
We generated a new pretrained FlexUNet segmentation model by using the augmented ImageMask dataset derived at Step 2.
<br><br>
<b>Step 4</b><br>
We generated a new pseudo masks to the master images by applying a segmentation method of the new 
pretrained FlexUNet model obtained at Step 3.
<br><br>
We iteratively repeated Steps 2 to 4 two times to generate our curated Smokers-Lung-Cancer-CT-ImageMask-Dataset.
<br><br>
<h3>2.3 Train Sample Images and Masks</h3>
<b>Train_sample_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_sample_masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3. Train TensorFlowFlexUNet Model
</h3>
 We trained the Smokers-Lung-Cancer-CT TensorFlowFlexUNet model using the following
<a href="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
This runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>
<b>Model parameters</b><br>
Defined a small <b>base_filters=16 </b> and large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers=8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
; You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 2
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.03
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_focal_dice_loss" and <a href="./src/dice_coef_multiclass.py">"dice_coef_hybrid"</a>.<br>
<pre>
[model]
loss           = "categorical_focal_dice_loss"
metrics        = ["dice_coef_hybrid"]
hybrid_alpha   = 1.4
hybrid_beta    = 0.6
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled the learning_rate_reducer callback and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with the patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specified RGB color map dict for Smokers-Lung-Cancer-CT 2 classes.<br>
<pre>
[mask]
mask_datatype= "categorized"
mask_file_format = ".png"
; Smokers-Lung-Cancer-CT RGB color map dict for 1+1 classes.
;      Background: black,  Cancer:dark_red
rgb_map = {(0,0,0):0,(210,20,20):1,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch change, the inference procedure can be called
 for 6 images in the <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
<!--
As shown below, early in the model training, the predicted masks from our FlexUNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
-->
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 16,17,18)</b><br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 34,35,36)</b><br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 36.<br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/asset/train_console_output_at_epoch36.png" width="1024" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4. Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT</b> folder,
and run the following bat file to evaluate the TensorFlowUNet model for Smokers-Lung-Cancer-CT.<br>
<pre>
>./2.evaluate.bat
</pre>
This runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/asset/evaluate_console_output_at_epoch36.png" width="1024" height="auto">
<br><br>Image-Segmentation-Smokers-Lung-Cancer-CT

<a href="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_focal_dice_loss) on this Smokers-Lung-Cancer-CT/test was very low, 
but dice_coef_hybrid was not high, as shown below.
<br>
<pre>
categorical_focal_dice_loss,0.002
dice_coef_hybrid,0.83
</pre>
<br>
<h3>
5. Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT</b> folder
and run the following bat file to infer segmentation regions for images using the trained TensorFlowUNet model for Smokers-Lung-Cancer-CT.<br>
<pre>
>./3.infer.bat
</pre>
This runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Smokers-Lung-Cancer-CT Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks except for the first case.<br>
<br>
<table>
<tr>
<th width="320" height="auto">Image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/images/hflipped_Patient (538).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/masks/hflipped_Patient (538).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test_output/hflipped_Patient (538).png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/images/hflipped_Patient (585).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/masks/hflipped_Patient (585).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test_output/hflipped_Patient (585).png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/images/hflipped_Patient (1003).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/masks/hflipped_Patient (1003).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test_output/hflipped_Patient (1003).png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/images/Patient (1009).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/masks/Patient (1009).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test_output/Patient (1009).png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/images/vflipped_Patient (780).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/masks/vflipped_Patient (780).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test_output/vflipped_Patient (780).png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/images/vflipped_Patient (1008).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test/masks/vflipped_Patient (1008).png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Smokers-Lung-Cancer-CT/mini_test_output/vflipped_Patient (1008).png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Lung cancer segmentation dataset with Lung-RADS class</b><br>
Nam, Diana; Panina, Alexandra; Pak, Alexandr <br>
(2024),  Mendeley Data, V1, doi: 10.17632/5rr22hgzwr.1 <br>
<a href ="https://data.mendeley.com/datasets/5rr22hgzwr/1">
  https://data.mendeley.com/datasets/5rr22hgzwr/1
</a>
<br><br>
<b>2. Lung Tumor Image Segmentation from Computer Tomography Images Using MobileNetV2 and Transfer Learning</b><br>
Zainab Riaz, Bangul Khan, Saad Abdullah, Samiullah Khan, Md Shohidul Islam<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10451633/" >
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10451633/
</a>
<br><br>
<b>3. NLSTseg: A Pixel-level Lung Cancer Dataset Based on NLST LDCT Images</b><br>
Kun-Hui Chen, Yi-Hui Lin, Shawn Wu, Nai-Wen Shih, Hsing-Chen Meng, Yen-Yu Lin, Chun-Rong Huang & Jing-Wen Huang <br>
<a href ="https://www.nature.com/articles/s41597-025-05742-x">
https://www.nature.com/articles/s41597-025-05742-x
</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Lung-Cancer-CT </b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Lung-Cancer-CT">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Lung-Cancer-CT
</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br><br>
