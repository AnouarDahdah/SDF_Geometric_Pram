import h5py
import numpy as np
import time

def extract_and_save_data(file_path, output_file):
    # Start the timer
    start_time = time.time()

    # Open the CGNS file
    with h5py.File(file_path, 'r') as cgns_file:
        # Verify paths
        print("Verifying paths...")
        list_all_paths(cgns_file)
        
        # Define the paths to the required data
        meanVxMonitorPath = "/Base/Air/Solution00001/meanVxMonitor/ data"
        meanVyMonitorPath = "/Base/Air/Solution00001/meanVyMonitor/ data"
        meanVzMonitorPath = "/Base/Air/Solution00001/meanVzMonitor/ data"

        # Check if paths exist before accessing
        if meanVxMonitorPath in cgns_file and meanVyMonitorPath in cgns_file and meanVzMonitorPath in cgns_file:
            # Extract data from the CGNS file
            meanVxMonitor_data = np.array(cgns_file[meanVxMonitorPath])
            meanVyMonitor_data = np.array(cgns_file[meanVyMonitorPath])
            meanVzMonitor_data = np.array(cgns_file[meanVzMonitorPath])

            # Save the data to a CSV file
            np.savetxt(output_file, np.column_stack((meanVxMonitor_data, meanVyMonitor_data, meanVzMonitor_data)),
                       delimiter=',', header='meanVxMonitor,meanVyMonitor,meanVzMonitor', comments='')
            print(f"Data saved to '{output_file}'.")
        else:
            print("One or more specified paths do not exist in the CGNS file.")

    # Stop the timer and print the time taken
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
    file_path = "/mnt/c/users/DELL/Downloads/ID2_Volume.cgns"
    output_file = "monitor_data.csv"
    extract_and_save_data(file_path, output_file)
