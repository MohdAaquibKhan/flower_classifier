import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_train_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'pet_images'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--a', type = int, default = 2, help = 'path to the folder of pet images')
    parser.add_argument('--b', type = int, default = 3, help = 'CNN Model Architecture classifier')
    parser.add_argument('--c', type = int, default = 1, help = 'Text File with Dog Names')
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    inargs = parser.parse_args()
    print("Argument 1:", inargs.a)
    print("Argument 2:", inargs.b)
    print("Argument 3:", inargs.c)
    return parser.parse_args()
