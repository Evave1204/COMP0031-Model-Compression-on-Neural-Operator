import wandb
run = wandb.init()
artifact = run.use_artifact('ucl-neural-operator/training/model-fno-darcy-16-resolution-2025-02-05-19-55:v0', type='model')
artifact_dir = artifact.download()

run = wandb.init()
artifact = run.use_artifact('ucl-neural-operator/training/model-gino-carcfd-32-resolution-2025-02-12-18-47:v0', type='model')
artifact_dir = artifact.download()

run = wandb.init()
artifact = run.use_artifact('ucl-neural-operator/training/model-codano-darcy-16-resolution-2025-02-11-21-13:v0', type='model')
artifact_dir = artifact.download()

run = wandb.init()
artifact = run.use_artifact('ucl-neural-operator/training/model-fno-darcy-16-resolution-2025-02-05-18-13:v0', type='model')
artifact_dir = artifact.download()

run = wandb.init()
artifact = run.use_artifact('ucl-neural-operator/training/model-fno-darcy-16-resolution-2025-02-05-16-07:v0', type='model')
artifact_dir = artifact.download()