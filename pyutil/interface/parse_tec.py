# Tecplot .dat parser

def dat_parse(filename):
    """Feature-poor parsing script for tecplot .dat files
    Usage
        Data = dat_parse(filename)
    Arguments
        filename = string filename of .dat file
    Returns
        Data = dictionary of parsed data
    """
    with open(filename,'r') as f:
        # Read header
        title_str = f.readline()
        var_str = f.readline()
        title = title_str.split("=")[1].strip().strip('"')
        Var = var_str.split("=")[1].split(",")
        Var = [v.strip().strip('"') for v in Var]
        # Read remaining lines
        Data = {}
        mode = 0 # 0=looking for ZONE, 1=grabbing points
        count= 0
        n = 0
        # Main loop
        for line in f:
            # Detect a ZONE definition
            if ('ZONE' in line) and (mode==0):
                items = line.split(",")
                zone_dict = {it.split("=")[0].strip():it.split("=")[1].strip().strip('"') \
                             for it in items}
                # Find number of nodes
                n = int(zone_dict['NODES'])
                # Set mode to grab data
                mode = 1; count = 0; X = []
            # Iterate to grab all nodes
            elif (mode==1):
                #
                x_list = line.split()
                X.append([float(x) for x in x_list])
                count += 1
                # End condition
                if count == n:
                    # Record data
                    zone_dict['POINTS'] = X
                    Data[zone_dict['ZONE T']] = zone_dict
                    # Reset mode
                    mode = 0
    return Data

if __name__ == "__main__":
    filename = "surface_grid.dat"
    res = dat_parse(filename)