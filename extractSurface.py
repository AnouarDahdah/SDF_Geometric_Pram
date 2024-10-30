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
        X = "/Base/Air Body/GridCoordinates/CoordinateX/ data"
        Y = "/Base/Air Body/GridCoordinates/CoordinateY/ data"
        Z = "/Base/Air Body/GridCoordinates/CoordinateZ/ data"

        # Check if paths exist before accessing
        if X in cgns_file and Y in cgns_file and Z in cgns_file:
            # Extract data from the CGNS file
            X = np.array(cgns_file[X])
            Y = np.array(cgns_file[Y])
            Z = np.array(cgns_file[Z])

            # Save the data to a CSV file
            np.savetxt(output_file, np.column_stack((X, Y, Z)),
                       delimiter=',', header='X,Y,Z', comments='')
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
    
    file_path= "/home/adahdah/ID2_Surface20.cgns"

    output_file = "1test.csv"
    extract_and_save_data(file_path, output_file)