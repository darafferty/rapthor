import os
import re

def get_kernel_string(filename):
    # Helper function to recursively get a parent directory
    def get_parent_dir(dirname, level=1):
        if (level == 0):
            return dirname
        else:
            parentdir = os.path.abspath(os.path.join(dirname, os.pardir))
            return get_parent_dir(parentdir, level -1)

    # All the directories to look for kernel sources and header files
    prefixes = []
    dirname_kernel = os.path.dirname(filename) # e.g. idg-lib/src/CUDA/common/kernels
    prefixes.append(dirname_kernel)
    dirname_src = get_parent_dir(dirname_kernel, 3) # e.g. idg-lib/src
    prefixes.append(dirname_src)

    # Helper function to recursively get file contents with local includes
    def add_file(filename, level=1):
        result = [""]
        for prefix in prefixes:
            try:
                with open(f"{prefix}/{filename}", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        # Match lines where a local header file is included
                        if line.startswith('#include "'):
                            # Extract the name of the header file
                            m = re.findall(r'"(.*?)"', line)

                            # If a valid filename was found, add it recursively
                            if len(m):
                                header_file = m[0]
                                padding = "*" * level
                                result += f"/{padding} BEGIN INLINE {header_file} {padding}/\n"
                                result += add_file(header_file, level + 1)
                                result += "\n"
                                result += (
                                    f"/{padding} END INLINE {header_file} {padding}/\n"
                                )
                        else:
                            result += [line]
                break
            except FileNotFoundError:
                # It is ok if a file is not found, it might exists in another prefix
                pass
        return result

    # Start gathering all the source lines
    filename_kernel = os.path.basename(filename) # e.g. KernelGridder.cu
    source_lines = add_file(filename_kernel)

    # Return the result as a string
    return "".join(source_lines)