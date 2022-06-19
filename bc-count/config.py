##############################################
#                                            #
#         Project configuration file         #
#                                            #
# Author: Amine Neggazi                      #
# Email: neggazimedlamine@gmail/com          #
# Nick: nemo256                              #
#                                            #
# Please read bc-count/LICENSE               #
#                                            #
##############################################

''' 
Cell type can be wither rbc, wbc or plt
which stands for:
    rbc --> Red blood cells
    wbc --> White blood cells
    plt --> Platelets
'''
cell_type  = 'rbc'             # rbc, wbc or plt
model_type = 'segnet'

if model_type == 'do-u-net':
    model_name   = cell_type
    input_shape  = (188, 188, 3)
    output_shape = (100, 100, 1)
else:
    model_name   = cell_type + '_segnet'
    input_shape  = (128, 128, 3)
    output_shape = (128, 128, 1)

padding = [200, 100]
if cell_type == 'rbc':
    output_directory = 'output/rbc'
elif cell_type == 'wbc':
    output_directory = 'output/wbc'
elif cell_type == 'plt':
    output_directory = 'output/plt'
else:
    print('Invalid blood cell type!\n')
