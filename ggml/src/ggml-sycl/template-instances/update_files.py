import sys, os


def main(input_files):
    # output_file = f"{input_file}.new"
    for input_file in input_files:

        output_file = input_file
        res = []

        with open(input_file, "r") as f:
            for line in f.readlines():
                #include "../fattn-vec.cuh"
                if "#include" in line and 'cuh' in line:
                    print(line)
                    line = line.replace('.cuh', '.hpp')

                res.append(line)

        with open(output_file, "w") as f:
            f.write("".join(res))
            print(f"save to {output_file}")

if __name__=="__main__":
    main(sys.argv[1:])