import os
import numpy as np
import glob
import multiprocessing as mp
from config import j

def apply_function_and_write_to_file(lines, output_file, time, num_entries):
    # Apply your desired function to each line and write to the output file
    if os.path.exists(output_file):
        print(f'{output_file} already exists')
        return
    with open(output_file, 'a') as f:
        count = 0
        f.write("ITEM: TIMESTEP\n")
        f.write('%d\n' % int(time))
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write('%d\n' % 1024800)
        f.write("ITEM: BOX BOUNDS pp ss pp\n")
        f.write("-800 1400\n")
        f.write("-62 62\n")
        f.write("-56 56\n")
        f.write("ITEM: ATOMS id type x y z vx vy vz count\n")
        for line in lines:
            tmp = np.zeros(8)
            num = line.lstrip(" ").rstrip('\n').split(" ")
            for j in range(8):
                tmp[j] = float(num[j])
            if abs(tmp[2]) < 61 and tmp[1] < 1300:
                 f.write(
                    '%d %d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n' % (tmp[0], 15, tmp[1], tmp[2], 0,
                                                                                  tmp[4], tmp[5], tmp[6], tmp[3]))
                 count += 1
        print(f'time: {time} total: {len(lines)}, written: {count}')
        f.close()


def read_and_process_file(input_file):
    with open(input_file, 'r') as f:
        # Skip the first 3 lines
        for _ in range(3):
            next(f)

        # Read the fourth line and split it into 3 parts
        fourth_line = next(f).strip()
        part1, part2, part3 = fourth_line.split()

        while True:
            # Create the output file name
            output_file = j(f'Dump/slice_{part1}.dump')

            # Get the number of lines to process based on part2

            num_lines_to_process = int(part2)

            # Read the next n lines
            lines_to_process = [next(f).strip() for _ in range(num_lines_to_process)]


            # Process the lines and write to the output file
            apply_function_and_write_to_file(lines_to_process, output_file, part1, part2)

            try:
                # Read the next line with 3 parts (part1, part2, part3)
                next_line = next(f).strip()
                part1, part2, part3 = next_line.split()
            except StopIteration:
                # If there are no more lines, exit the loop
                break

# Replace 'input_file.txt' with the path to your input file
ls = glob.glob(j('MNF/*.mnf'))

with mp.Pool() as pool:
    pool.map(read_and_process_file, ls)
