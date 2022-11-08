# Music Genre Classification


## Project process:
  - Read audio data preprocessing process (pair with a blog)
  - Re-implement in Pytorch
      - data pipeline
      - model building
      - finetuning pretrained models
      - model checkpointing
      - model saving
      - model deployment
  - Experiments design (define which architectures to train)
      - architecture exps
        - backbone + linear (resnet, inception, efficientnet)
        - backbone + prcnn
        - backbone + transformer
      - preprocessing exps
        - raw signal
        - spectrogram
        - melspectrogram
        - mfcc
      - data augmentation exps
      - with distillation
  - Implement model training library (config structure)
      - data
        - data path
        - preprocessing type
        - preprocessing parameters
        - augment or not
      - model
        - backbone
        - head
      - training
        - batch size
        - shuffle buffer size
        - epochs
        - checkpoint path
        - initial epoch
        - optimizer
        - lr
  - Do experiments and make report  
 [Nov]
  - Deploy the best model to api
  - Realtime inferencing with django channel
  - Data augmentation to simulate realtime audio  
 [Dec]
  - Build custom dataset
  - Make api endpoints better, more secure and add more features
  - Make training api better
