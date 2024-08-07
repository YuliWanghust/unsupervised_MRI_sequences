# Pre-training
# import schedule
# import time
# import os

# def run_training():
#     sizes = [128, 64]
#     for size in sizes:
#         cmd = f"python -m pretrain.simsiam.train -c pathmnist -epochs 20 -aug diskeep -fout simclr-9class-simsiam-{size}-84 -bsize {size}"
#         #cmd = f"python -m pretrain.simclr.train -c pathmnist -epochs 50 -aug diskeep -fout simclr-9class-simclr-{size} -bsize {size}"
#         os.system(cmd)
#         print(f"Training for size {size} completed.")
#     print("Next training batch will start in 8 hours.")

# # Run training immediately upon script start
# run_training()

# # Schedule subsequent runs every 8 hours
# schedule.every(8).hours.do(run_training)

# if __name__ == "__main__":
#     print("Training scheduler started.")
#     while True:
#         schedule.run_pending()
#         time.sleep(1)  # sleep for 1 second to minimize CPU usage

# ## Fine-tuning
# import subprocess
# import os

# # # List of sample sizes
# # sample_sizes = [150, 300, 1500, 15000]

# # # List of layer numbers
# # layer_numbers = [64, 128, 256, 512, 1024, 2048]

# # List of sample sizes
# sample_sizes = [300]

# # List of layer numbers
# layer_numbers = [512, 1024, 2048]

# # Path template for the file to be removed
# file_template = r"C:\Users\z004yxbu\Downloads\Project\simclr-MedMNIST\downstream\resnet\models\simclr-9class-simsiam-{layer_number}.ckpt"

# # Command template for subprocess
# command_template = "python -m downstream.resnet.train -c pathmnist -samples {samples} -epochs 50 -fin {fin} -fout {fout}"

# # Iterate over each sample size
# for samples in sample_sizes:
#     # Iterate over each layer number
#     for layer_number in layer_numbers:
#         # File and command configuration for each layer model
#         file_path = file_template.format(layer_number=layer_number)
#         if os.path.exists(file_path):
#             os.remove(file_path)  # Remove the file if it exists
        
#         fin_fout = f"simclr-9class-simsiam-{layer_number}"
#         command = command_template.format(samples=samples, fin=fin_fout, fout=fin_fout)
#         print(command)
#         subprocess.run(command, shell=True)

import subprocess
import os

# List of sample sizes and layer numbers for different scenarios
sample_size_layers = [
    #(300, [1024, 2048]),
    #(30000, [64, 128, 256, 512, 1024, 2048]),
    ([150, 300, 1500, 15000, 30000], [64, 128, 256, 512])
]

# Path template for the file to be removed
file_template = r"C:\Users\z004yxbu\Downloads\Project\simclr-MedMNIST\downstream\resnet\models\simclr-9class-simclr-{layer_number}.ckpt"

# Command template for subprocess
command_template = "python -m downstream.resnet.train -c pathmnist -samples {samples} -epochs 50 -fin {fin} -fout {fout}"

# Iterate over each scenario
for sample_sizes, layer_numbers in sample_size_layers:
    # Ensure sample_sizes is a list
    if isinstance(sample_sizes, int):
        sample_sizes = [sample_sizes]

    # Iterate over each sample size
    for samples in sample_sizes:
        # Ensure layer_numbers is a list
        if not isinstance(layer_numbers, list):
            layer_numbers = [layer_numbers]

        # Iterate over each layer number
        for layer_number in layer_numbers:
            # File and command configuration for each layer model
            file_path = file_template.format(layer_number=layer_number)
            if os.path.exists(file_path):
                os.remove(file_path)  # Remove the file if it exists
            
            fin_fout = f"simclr-9class-simclr-{layer_number}"
            command = command_template.format(samples=samples, fin=fin_fout, fout=fin_fout)
            print(command)
            subprocess.run(command, shell=True)