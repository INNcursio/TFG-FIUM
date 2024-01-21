import subprocess

def run_cellpose_command():
    command = [
        "python", "-m", "cellpose",
        "--train",
        "--use_gpu",
        "--dir", "D:\MRC\TRAIN-3-ER",
        "--test_dir", "D:\MRC\TEST-3-ER",
        "--mask_filter", "_masks",
        "--pretrained_model", "None",
        "--chan", "0",
        "--chan2", "0",
        "--n_epochs", "200",
        "--batch_size", "25",
        "--min_train_masks", "2",
        "--diam_mean", "20",
        "--verbose"
    ]

    # Ejecutar el comando
    subprocess.run(command)
#",
if __name__ == "__main__":
    run_cellpose_command()
