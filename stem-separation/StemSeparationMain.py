# Imports and Setup
from common import utils
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

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

    trainModel(model, NUM_EPOCHS, EPOCH_LENGTH, train_data, train_dataloader, val_dataloader)

    # Alternatively, you can load a pre-trained model
    checkpoint_path = "checkpoints/200-epochs/best.model.pth"
    # model = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)


    # Deploy Model
    from StemSeparationDeployment import deployModel
    deployModel(model, test_data)


    # Evaluate Model
    from StemSeparationEvaluation import evaluateModel
    NUM_EVALUATION_ITEMS = len(test_data)

    evaluation_df = evaluateModel(model, test_data, NUM_EVALUATION_ITEMS, "checkpoints/eval_df.csv")