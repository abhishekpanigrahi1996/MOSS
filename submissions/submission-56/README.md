The notebook ICL_assoc_recall_moss_2025.ipynb contains shell commands that will run inference on training checkpoints from four different model sizes: tiny (212K parameters), small (701K parameters), medium (2.42M parameters), and big (10.7M parameters). These checkpoints are in this google drive folder: [ pretrained checkpoints](https://drive.google.com/drive/folders/1KALcTkduq9uFj7CJ6CaGBvAgsJuXGn58?usp=drive_link).

Go to that link for zip files of the model checkpoints. Right click on the `ortho_haar_ckpts` in Google Drive, and then click `Organize`, then `Add Shortcut`. In the `All locations` tab, choose `MyDrive` as the place to put the shortcut.

This must be done since we are unsure how to get Google Colab to pull from a Google Drive that is not the user's. We couldn't add the checkpoints to the anonymous Github repo since they are ~8 GB.

Then make sure to run the first two code cells that mount your Google drive to Colab and unzip the folders containing the checkpoints.

From that point on, you can hit `Run all` and the notebook will run inference on these checkpoints and generate the figures seen in the paper.

The next cell containing ```!git clone https://github.com/anon4mossreview/moss_2025.git``` clones a repo that contains the code that generates the figures. Look here if you'd like to modify any figures.

