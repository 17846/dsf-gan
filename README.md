<h1 style="text-align:center">Dual-Style Feature Fusion GANs for High-Recognizability Few-Shot Handwritten Chinese Character Generation</h1>



​	In this paper, we propose a novel method for generating high-recognizability handwritten Chinese characters using a limited number of style samples. By employing **dual-style feature fusion** within Generative Adversarial Networks (GANs), our approach effectively addresses the challenges of poor style resemblance and limited diversity in existing handwritten samples. The global features extracted from the exemplar samples encode style characteristics and character-specific information, while detailed features captured through contrastive learning highlight unique writing idiosyncrasies. An adaptive fusion strategy judiciously integrates these features to synthesize target characters. Additionally, an outline loss function imposes edge constraints, improving the structural fidelity of the generated glyphs. Experimental results demonstrate that our model significantly outperforms state-of-the-art methods across various metrics, achieving a Recognizable Accuracy of 93.76% even with as few as four style samples. Our contributions lie in the integration of global and detailed features, the introduction of an outline loss function, and the demonstration of high-quality handwritten character generation from minimal style exemplars.

​	The figure shows handwritten characters from six different authors along with the generation results from DSF-GAN. Here, "GT" refers to the target character to be generated, while "Generate" refers to the result produced by DSF-GAN.

![](.\img\img1.png)





# Requirement

To ensure the code runs correctly, the following packages are required:

* python 3.10

* Install [pytorch](https://pytorch.org/get-started/previous-versions/) with appropriate CUDA version, e.g.

  ```bash
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
  ```

* hydra

* tqdm

* pickle

* apex

  Apex is an accelerator for NVIDIA GPU computing. You can refer to [here ](https://blog.csdn.net/weixin_45914748/article/details/139864002)for the install.

# Training

## 1. dataset

The dataset used for training is mainly adapted from [CASIA-HWDB-1.1](http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html). We put the characters by the same writer into the same directory. The folder name represents the writer and the file name represents the character. We render the template characters from [Source-Han-Sans](https://github.com/adobe-fonts/source-han-sans). You can download the dataset [here](https://pan.baidu.com/s/1HcgWjjFXVuhyKKQ2smSQHA?pwd=7pw2) and extract it.

You can also build the dataset by yourself for customization. The directory structure is as follows:

```markdown
dataset
├── script
│   ├── writer_1
│   │   ├── character_1.png
│   │   ├── character_2.png
│   │   └── ...
│   ├── writer_2
│   │   ├── character_1.png
│   │   ├── character_2.png
│   │   └── ...
│   └── ...
└── template
    ├── character_1.png
    ├── character_2.png
    └── ...
```

## 2. Pre-Train OCR model

​	Before the formal training, it is necessary to pre-train the handwritten Chinese character recognition model. First, you need to adjust the `dataset_path` in the `hccr_hccr_train.py` file, and set the appropriate `batch_size` and `num_workers` in the `DataLoader`. Then, start the model training by running `python hccr_train.py`.

## 3. Train DSF-GANs 

​		First, you need to set `dataset_path` and `ocr_path` in the `config/training.yaml` file. The `ocr_path` refers to the pre-trained model obtained in Step 2. In this file, you can also configure the learning rates for the generator and discriminator, as well as the loss function weights and other training settings.

* To train using `DataParallel`

  ```bash
  python training.py
  ```

* To train using  `DistributedDataParallel`

  ```bash
  python training_multi.py
  ```

  ```python
  # set environ in training_multi.py
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '29501'
  os.environ['WORLD_SIZE'] = '2'
  os.environ['RANK'] = '0'
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
  ```

## 3. Inference

​	Through inference, you can use specified handwritten samples to generate the desired characters. First, you need to make some settings in the `config/inference.yaml` file:

* `model_path`: Path to the trained generator model
* `reference_path`: Path to the style samples
* `reference_count`: The number of style samples used for generating each character
* `target_text`: The target text to be generated
* `template_path`: Path to the target character

Once the configuration is complete, execute the following command to perform inference:

```bash
python inference.py
```

We have provided a pre-trained model that has undergone 15,000 iterations. Click [here](https://pan.baidu.com/s/1sQ8rlxokR7NpGyuQOD498A?pwd=ecvr) to download it.

## 4. Test

​		Through testing, select the characters from the `test` directory in the dataset as style samples to generate a complete set of template characters. First, you need to configure the settings in the `config/test.yaml` file:

* `model_path`: your model path
* `checkpoint_path`: save file path
* `dataset_path`: your dataset path
* `reference_count`:  the number of reference sample
* `pre`: the result image concatenate the resource image

Once the configuration is complete, execute the following command to perform test:

```bash
python test.py
```

## 5. Evaluate

### 5.1 RA

* pre-train `densenet121` in `evaluate/densenet_train.py`
* test generate sample in `test_ra.py`

### 5.2 SSIM

```bash
python evaluate/SSIM.py
```

### 5.3 IS and FID

```bash
python evaluate/is_fid_pytorch.py
```

