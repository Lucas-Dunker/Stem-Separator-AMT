# Imports and Setup
from common import utils
import torch
from StemSeparationModel import StemSeparationModel

# Whether to train the model or load a pre-trained model
TRAIN_MODEL = False

if __name__ == '__main__':

    # Prepare Data & Define Model
    from StepSeparationProcess import prepareModel

    train_folder = "~/.nussl/tutorial/train"
    val_folder = "~/.nussl/tutorial/valid"
    test_folder = "~/.nussl/tutorial/test"

    model, train_data, train_dataloader, val_dataloader, test_data = prepareModel(train_folder, val_folder, test_folder)


    # Train Model
    from StepSeparationProcess import trainModel
    NUM_EPOCHS = 1
    EPOCH_LENGTH = 10

    model.verbose = False
    utils.logger()

    if (TRAIN_MODEL):
        trainModel(model, NUM_EPOCHS, EPOCH_LENGTH, train_data, train_dataloader, val_dataloader)

    # Alternatively, you can load a pre-trained model
    checkpoint_path = "stem-separation/checkpoints/200-epochs/best.model.pth"

    if not TRAIN_MODEL:
        model = torch.load(checkpoint_path, weights_only=False)


    # Deploy Model
    from StemSeparationDeployment import deployModel
    deployModel(model, test_data)


    # Evaluate Model
    from StemSeparationEvaluation import evaluateModel
    NUM_EVALUATION_ITEMS = len(test_data)

    evaluation_df = evaluateModel(model, test_data, NUM_EVALUATION_ITEMS, "stem-separation/checkpoints/eval_df.csv")