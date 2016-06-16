"""Recipe for positioning figures in pyplot
"""
# matplotlib import block
import matplotlib
# choose the backend to handle window geometry
matplotlib.use("Qt4Agg")
# Import pyplot
import matplotlib.pyplot as plt
# Plot settings
offset = [(0,500),(700,500),(1400,500)] # window locations


# Example details
import numpy as np
import pyutil.plotting as ut

# Example plots
C = ut.linspecer(len(offset))

for i in range(len(offset)):
    # Generate some fake data
    X = np.linspace(-1,1)
    Y = np.random.random(len(X))
    # Plot
    plt.figure()
    plt.plot(X,Y,color=C[i])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data Set {}'.format(i+1))
    # Set plot location on screen
    manager = plt.get_current_fig_manager()
    # Grab window dimensions
    x,y,dx,dy = manager.window.geometry().getRect()
    # Set window positions, maintain dimensions
    manager.window.setGeometry(offset[i][0],offset[i][1],dx,dy)

# Show all plots
plt.show()