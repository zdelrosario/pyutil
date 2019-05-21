#!/usr/bin/python
import sys, os, re
from shutil import copy

## Helper functions
def line_image(s):
    """Parses a line for \includegraphics{} call
    Usage
        modified_line, filename, image_dir = line_image(s)
    Arguments
        s = input string
    Returns
        modified_line = modified string; flattens any sibling directories
                        in \includegraphics{} argument
        filename      = filename of \includegraphics{} argument with
                        directories stripped. Equal to None if line
                        does not contain \includegraphics{}
        image_dir     = directory of image file. Equal to None if line
                        does not contain \includegraphics{}
    """
    if (s.find("\includegraphics") != -1):
        ##
        ind_start = s.rfind("{") + 1
        ind_end   = s.rfind("}")
        filename_full = s[ind_start:ind_end]
        ## Detect sibling filepath
        ind_stem = filename_full.rfind("/") + 1
        if (ind_stem != -1):
            filename = filename_full[ind_stem:]
        else:
            filename = filename_full

        modified_line = s[:ind_start] + filename + s[ind_end:]
        ## Construct image_dir
        image_dir = filename_full[:ind_stem]
        if len(image_dir) == 0: # Catch root directory case
            image_dir = "."

        return modified_line, filename, image_dir
    else:
        return s, None, None

def flatten_includegraphics(input_line, target_dir):
    """Flattens a \includegraphics{} call
    Usage
        modified_line = line_image(s)
    Arguments
        input_line = input string
        target_dir = output director for flattened document
    Returns
        modified_line = modified string; flattens any sibling directories
                        in \includegraphics{} argument
    Post-conditions
        image argument to \includegraphics{} copied to target_dir
    """
    modified_line, filename, image_dir = line_image(input_line)

    if not (filename is None):
        ## Find all matching images
        file_matches = \
            [f for f in os.listdir(image_dir) if not (re.search(filename + "\.", f))]
        ## Copy matching images
        for match in file_matches:
            copy(image_dir + match, target_dir)

    return modified_line

## Primary function
def flatten_tex(filename_in, target_dir):
    """
    Usage
        flatten_tex(f)
    Arguments
        filename_in = name of tex file to read
        target_dir  = target directory
    """
    ## Construct output filename
    ind_filename = filename_in.rfind("/")
    if (ind_filename != -1):
        filename_out = target_dir + filename_in[ind_filename+1:]
    else:
        filename_out = target_dir + filename_in

    ## Parse input file line-by-line; copy to output
    with open(filename_in, mode = "r") as f_read:
        with open(filename_out, mode = "w") as f_out:
            for line_read in f_read:
                ## Cascade all the flatten functions

                ## \includegraphics parse
                line1 = flatten_includegraphics(line_read, target_dir)

                line_fin = line1
                ## Write the modified line
                f_out.write(line_fin)

if __name__ == "__main__":
    ## Debug
    s0 = "Foo bar\n"
    s1 = "\includegraphics[width=0.75\textwidth]{../images/filename}\n"
    s2 = "\includegraphics[width=0.75\textwidth]{filename}\n"
    s3 = "\includegraphics[width=0.75\textwidth]{./filename}\n"
