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
cell_type  = 'wbc'             # rbc, wbc or plt
model_type = 'do_unet'          # do_unet or segnet

if model_type == 'do_unet':
    model_name   = cell_type
    input_shape  = (188, 188, 3)
    output_shape = (100, 100, 1)
else:
    model_name   = cell_type + '_segnet'
    input_shape  = (128, 128, 3)
    output_shape = (128, 128, 1)

padding = [200, 100]
output_directory = 'output/' + model_type + '/' + cell_type

if not cell_type in ['rbc', 'wbc', 'plt']:
    print('Invalid cell type!')
