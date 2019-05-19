
import imgaug # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa

#### 
class Config(object):
    def __init__(self):
        self.seed    = 5
        self.init_lr = 1.0e-4
        self.lr_steps   = 30  # decrease at every n-th epoch 
        self.train_batch_size = 8
        self.infer_batch_size = 8
        self.nr_epochs  = 60
        self.nr_classes = 3
        
        # nr of processes for parallel processing input
        self.nr_procs_train = 4 
        self.nr_procs_valid = 4 

        self.nr_fold = 5
        self.fold_idx = 4
        self.cross_valid = False

        self.load_network  = False
        self.save_net_path = ""

        self.data_size  = [1024, 1024]
        self.input_size = [512, 512]

        #
        self.dataset = 'prostate_manual'
        # v1.0.3.0 test classifying cancer only
        self.logging = True # for debug run only
        self.log_path = '/media/vqdang/Data_2/dang/output/NUCLEI-ENHANCE/%s/' % self.dataset
        # self.log_path = '/mnt/dang/output/NUCLEI-ENHANCE/%s/' % self.dataset
        self.chkpts_prefix = 'model'
        self.model_name = 'v1.0.0.1'
        self.log_dir =  self.log_path + self.model_name

    def train_augmentors(self):
        shape_augs = [
            iaa.Scale({
                    "height": 1024, 
                    "width" : 1024}
            ),
        ]
        #
        input_augs = [
            iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # gaussian blur with random sigma
                        iaa.MedianBlur(k=(3, 5)), # median with random kernel sizes
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                        ]),
            iaa.Sequential([
                iaa.Add((-26, 26)),
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.LinearContrast((0.75, 1.25), per_channel=1.0),
            ], random_order=True),
        ]   
        return shape_augs, input_augs

    ####
    def infer_augmentors(self):
        shape_augs = [
            iaa.Scale({
                    "height": 1024, 
                    "width" : 1024}
            ),
        ]
        return shape_augs, None

############################################################################