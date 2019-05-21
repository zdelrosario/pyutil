from pyutil.tex import flatten_tex
import sys

if len(sys.argv) == 1:
    print("Usage:")
    print("    (Usage string goes here...)")
else:
    input_filename = sys.argv[1]
    output_dir     = sys.argv[2]

    flatten_tex(input_filename, output_dir)
