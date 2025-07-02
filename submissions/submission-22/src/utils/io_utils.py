import csv
import os


def create_csv_files(n, root_dir):
    """
    Create csv files for storing singular values during training.

    Parameters
    ----------
    n : int
        The number of files to create.

    Notes
    -----
    This function deletes existing files with the same name, so be careful when using it.
    """
    for i in range(n):
        filename = root_dir + "/" + f"singular_values_{i}.csv"
        # print("Save singular spectra to", filename)

        os.makedirs(
            os.path.dirname(filename), exist_ok=True
        )  # Create directories if they don't exist

        if os.path.exists(filename):
            os.remove(filename)  # Delete the file if it exists
        if not os.path.exists(filename):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["iteration"] + [f"sv_{j}" for j in range(10)]
                )  # Assuming 10 singular values per file


def append_singular_values(n, singular_value_vectors, iteration, root_dir):
    """
    Append singular values to CSV files.

    Parameters
    ----------
    n : int
        The number of CSV files to append to.
    singular_value_vectors : list of torch.Tensor
        A list of vectors containing singular values to be recorded.
    iteration : int
        The current iteration number to be recorded with each set of singular values.

    Notes
    -----
    Each file is named "singular_values_{i}.csv" where `i` ranges from 0 to n-1.
    The singular values are appended to the files along with the iteration number.
    """

    for i in range(len(singular_value_vectors)):
        # print("write to", root_dir + "/" + f"singular_values_{i}.csv")
        filename = root_dir + "/" + f"singular_values_{i}.csv"
        # print(singular_value_vectors[i].tolist())
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([iteration] + singular_value_vectors[i].tolist())
