import torch
import math

class NetworkOptions():
    def __init__(self):
        ### General ###
        #TO SET
        self.experiment_name = 'superDWI_fixnet'   # Directory name to be stored in the checkpoint directory. **General**
        self.model_name = 'best_model.pth'  # Working model name - will be stored within the experiment name directory found within the ** General **
                                            # checkpoints directory. 
        self.continue_training = False      # Whether to continue training using existing model weights

        self.inference=False
        self.perform_inference=False

        ### Training Options ###
        self.lr = 1e-4                      # Learning rate - set to this value post learning rate warmup. 
        self.warmup_factor = 1e-2              # When the network is warming up the effective learning rate is set to warmup_factor*lr
        self.warmup_iter = 10000             # Number of iterations after which the network stops warming up.
        self.batch_size = 256               # Training batch_size.
        
        self.val_freq = 10000                 # How often (iterations) to run the validation loop inside the training loop
        self.val_iters = 10                 # How many iterations of validation data to loop through when the validation loop is called.
        
        # Stopping Conditions
        self.epochs = 10                   # Maxium number of epochs
        self.iteration_limit = math.inf      # Maximum number of iterations (training updates)
        self.lr_decay_limit = math.inf      # The number of times to decay the learning rate 
        self.lr_decay_factor = 1            # The learning rate decay factor

        self.device = torch.device("cuda:0") # if torch.cuda.is_available() else "cpu") # **General**
        self.train_workers = 10            # Number of workers used for the training dataloader.
        self.val_workers = 10 # Number of workers used for the validation dataloader. 

        #Early Stopping Parameters
        self.early_stopping = False          # Whether to include early stopping in the network.
        self.early_stopping_threshold = math.inf  # When early_stopping_counter reaches this value training will stop. This counter is updated every validation loop, therefore
                                            # training will be stopped due to early stopping when the network hasn't improved for early_stopping_threshold validation loops. 

        ### Network ###
        self.deep_reg = 0.25                # The deep regularisation parameter. If learn_lambda = True this is only the initial value. 
        self.learn_lambda = True            # Whether to optimise lambda within the network.
        self.fixel_lambda = 0               # kappa.

        self.init_type = 'xavier'               # {'normal', 'xavier', 'kaiming', 'orthogonal'}
        self.activation = 'relu'           # {'relu', 'tanh', 'sigmoid', 'leaky_relu', 'prelu'}

        ### Data ###
        # TO SET
        self.data_dir = '/homes/andrey/E_ResearchData/HCP_New_Download'  
        self.train_subject_list = ['100307', '100408', 
                                   '101915', '103414', '103818', '105115',
                                   '106016', '110411', '111312', '111716', '113619', '115320',
                                   '117122', '118730', '118932', '120111', '122317', '123117',
                                   '124422', '125525', '126325', '127933', '128632', '129028',
                                   '130013', '130316', '133928', '135932', '136833', '138534',
                                   '139637', '148335', '149337', '149539', '151223', '151526',
                                   '151627', '153025', '156637', '159340', '160123', '161731',
                                   '162733', '163129', '176542', '178950', '188347', '189450',
                                   '190031', '192540'
                                   ]

        #TO SET
        self.val_subject_list = ['196750', '198451', 
                                 '199655', '201111', '208226', '211417',
                                 '211720', '212318', '214423', '221319', '239944', '245333',
                                 '280739', '298051', '366446', '397760', '414229', '499566',
                                 '654754', '672756', '751348', '756055', '792564', '856766',
                                 '857263', '899885'
                                 ]

        self.diffusion_dir = 'Diffusion'
        self.shell_number = 2
        self.data_file = 'normalised_data_denoised.nii.gz'
        self.dwi_number = 108
        self.dwi_folder_name = 'undersampled_fod_b0_18_b1000_90'

        ### Inference ###
        # TO SET
        # self.test_subject_list = 

        print(self.__dict__)
