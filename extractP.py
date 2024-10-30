import h5py
import numpy as np
import time

def extract_and_save_data(file_path, output_file):
    # Open the CGNS file
    start_time = time.time()
    with h5py.File(file_path, 'r') as cgns_file:
        # Verify paths
        print("Verifying paths...")
        list_all_paths(cgns_file)
        
        # Define the paths to the required data
        P = "/Base/Air Body/Solution00001/meanPressureMonitor/ data"
        
        # Check if paths exist before accessing
        if P in cgns_file:
            # Extract data from the CGNS file
            P = np.array(cgns_file[P])
            print(P)

            # Save the data to a CSV file
            np.savetxt(output_file, np.column_stack((P)),
                       delimiter=',', header='P', comments='')
            print(f"Data saved to '{output_file}'.")
        else:
            print("One or more specified paths do not exist in the CGNS file.")

    end_time = time.time()
    print(f"Compilation running time: {end_time - start_time} seconds")

def list_all_paths(group, path=''):
    """Recursively list all paths in the CGNS file."""
    for key in group.keys():
        item = group[key]
        item_path = f"{path}/{key}"
        print(item_path)
        if isinstance(item, h5py.Group):
            list_all_paths(item, item_path)

if __name__ == "__main__":
    file_path = "/home/adahdah/ID2_Surface19.cgns"
    output_file = "P_datad19.csv"

    # Step 1: Extract and save the data
    extract_and_save_data(file_path, output_file)
