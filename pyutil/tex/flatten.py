#!/usr/bin/env python3
import argparse, fnmatch, os, re
from shutil import copy

# SETUP
##################################################
parser = argparse.ArgumentParser(description="Flatten a target latex file")
parser.add_argument(
    "filename",
    type=str,
    help="Input filename",
)
parser.add_argument(
    "-d",
    "--directory",
    type=str,
    help="Target (output) directory; defaults to original file's directory",
    default=None
)

parser.add_argument(
    "-s",
    "--suffix",
    type=str,
    help="Suffix for new tex file",
    default="-flat"
)

## Parse arguments
args = parser.parse_args()

dir_base = os.path.abspath(os.path.dirname(args.filename))
if args.directory is None:
    dir_target = dir_base
else:
    dir_target = os.path.abspath(args.directory)

filename_in = os.path.abspath(args.filename)
filename_out = os.path.join(
    dir_target,
    re.sub(r"\.tex", args.suffix + ".tex", os.path.basename(filename_in))
)

# Helper functions
# --------------------------------------------------

# Find files at a path
def find_file(pattern, path):
    r"""Search for files at a given path

    Args:
        pattern (str): Filename pattern to match
        path (str): Path to search

    Returns:
        list: Filename matches

    Examples:
        >>> import os
        >>> find_file("foo.txt", os.path.abspath("."))

    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

# Execute
##################################################
with open(filename_in, 'r') as file_in:
    with open(filename_out, 'w') as file_out:

        ## Image directory name; to be replaced if \graphicspath found
        dirname = "."

        for line in file_in.readlines():

            ## Replace the graphicspath with local directory
            if not (re.search(r"^\\graphicspath", line) is None):
                dirname = re.search("\{.*\}", line).group(0)
                dirname = re.sub("[\{\}]", "", dirname)

                modline = re.sub("\{.*\}", "{{.}}", line)
                file_out.write(modline)

            ## Copy all images
            elif not (re.search("includegraphics", line) is None):
                ## Find the image
                imagename = re.search("\{.*\}", line).group(0)
                imagename = re.sub("[\{\}]", "", imagename)

                image_matches = find_file(
                    imagename + ".*",
                    os.path.abspath(os.path.join(dir_base, dirname))
                )

                ## Copy the first match
                copy(image_matches[0], dir_target)

                ## Echo the original line
                file_out.write(line)

            ## Copy imported listings
            elif not (re.search(r"^\\lstinputlisting", line) is None):
                ## Find the code file
                codename = re.search("\{.*\}", line).group(0)
                codename = re.sub("[\{\}]", "", codename)
                codeabspath = os.path.abspath(
                    os.path.join(dir_base, codename)
                )

                ## Copy the sole match
                copy(codeabspath, dir_target)

                ## Replace original absolute path with new relative path
                codeline = re.sub("\{.*\}", "{" + os.path.basename(codeabspath) + "}", line)
                file_out.write(codeline)

            ## Echo an unmatched line
            else:
                file_out.write(line)
