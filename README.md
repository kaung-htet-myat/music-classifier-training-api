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
        - backbone + linear
        - backbone + prcnn
        - backbone + transformer
      - preprocessing exps
        - raw signal
        - spectrogram
        - melspectrogram
        - mfcc
      - data augmentation exps
  - Implement model training library
  - Do experiments and make report
  - Deploy the best model to api
  - Realtime inferencing with django channel
  - Data augmentation to simulate realtime audio
  - Build custom dataset
  - Make api endpoints better, more secure and add more features
