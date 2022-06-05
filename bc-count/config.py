# global variables
cell_type        = 'red'             # red, white or platelets
input_shape      = (188, 188, 3)
output_shape     = (100, 100, 1)
padding          = [200, 100]
if cell_type == 'red':
    output_directory = 'output/rbc'
elif cell_type == 'white':
    output_directory = 'output/wbc'
elif cell_type == 'platelets':
    output_directory = 'output/platelets'
else:
    print('Invalid blood cell type!\n')
