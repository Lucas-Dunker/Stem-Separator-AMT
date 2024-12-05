# Imports and Setup
from common import utils
import torch
from StemSeparationModel import StemSeparationModel # Necessary import to load pre-trained model file

# Decide whether to train the model or load a pre-trained model
TRAIN_MODEL = False


if __name__ == '__main__':
    """
    Our main driver function for our stem separation process.

    This function will:
    1. Load in our desired training data, separating it into training, validation, and testing sets
    2. Train the model (if TRAIN_MODEL is True), or alternatively load pre-trained model weights
    3. Separate vocal audio from the input mixture audio via model deployment
    4. Evaluate our model, saving the resulting scores and figures to disk
    """

    # Prepare Data & Define Model
    from StepSeparationProcess import prepareModel

    train_folder = "./stem-separation/dataset/train"
    val_folder = "./stem-separation/dataset/valid"
    test_folder = "./stem-separation/dataset/test"

    model, train_data, train_dataloader, val_dataloader, test_data = prepareModel(train_folder, val_folder, test_folder)


    # Train Model
    from StepSeparationProcess import trainModel
    NUM_EPOCHS = 200
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

    # Evaluate Model & Plot Results
    from StemSeparationEvaluation import evaluateModel
    NUM_EVALUATION_ITEMS = len(test_data)
    eval_path = "stem-separation/metrics/eval_df.csv"
    evaluation_df = evaluateModel(model, test_data, NUM_EVALUATION_ITEMS, eval_path)
