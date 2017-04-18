import tensorflow as tf


def get_flags(config_file=None):
    """
	:return: This function return an constructer with all the variable parsed
	from the command line or the config file given the config file
	
	"""
    
    # Initialize all the variable with flags, used gflags here
    
    FLAGS = tf.app.flags.FLAGS
    
    # Flags governing the check point, data and eval directory
    tf.app.flags.DEFINE_string('eval_dir', './tensor_record/',
                               """Directory where to write event logs.""")
    tf.app.flags.DEFINE_string('checkpoint_dir', './models',
                               """Directory where to read model checkpoints.""")
    tf.app.flags.DEFINE_string('dataset_dir', './Dataset',
                               """Directory where to read model checkpoints.""")
    tf.app.flags.DEFINE_string('load_ckpt_path',
                               './tensor_record/tmp/tmp_12464.ckpt',
                               """ckpt path to be loaded""")
    
    # Flags governing the frequency of the eval.
    tf.app.flags.DEFINE_integer('eval_interval_iter_soft', 50,
                                """How often to run the soft eval.""")
    tf.app.flags.DEFINE_integer('eval_interval_iter_hard', 500,
                                """How often to run the soft eval.""")
    tf.app.flags.DEFINE_boolean('run_once', False,
                                """Whether to run eval only once.""")
    
    # Flags governing the data used for the eval.
    tf.app.flags.DEFINE_integer('num_eval_examples_hard', 50000,
                                """Number of examples to run. Note that the eval
	                            """)
    tf.app.flags.DEFINE_integer('num_eval_examples_soft', 128,
                                """Number of examples to run. Note that the eval
	                            """)
    
    # Flags governing the architecture
    tf.app.flags.DEFINE_string('structure_string', '1-4-16-64',
                               """ Fine to coarse structure""")
    
    tf.app.flags.DEFINE_string('subset', 'validation',
                               """Either 'validation' or 'train'.""")
    
    # Flags governing the train and test dataset partition
    tf.app.flags.DEFINE_string('data_split_string_train', 'S1-S5-S0-S6-S7-S8',
                               """ Train Data Subjects """)
    tf.app.flags.DEFINE_string('data_split_string_train_2d',
                               '2D',
                               """ Train Data 2D Subjects """)
    tf.app.flags.DEFINE_string('data_split_string_test', 'S9',
                               """ Test Data Subjects """)
    
    # Flags governing GPU utilization
    tf.app.flags.DEFINE_string('gpu_string', '0-1-2-3',
                               """ Availabele GPU indxes""")
    
    # Flags governing learning, output, and peprocessing parameters
    tf.app.flags.DEFINE_integer('batch_size', 32,
                                """batch size for the learning""")
    tf.app.flags.DEFINE_integer('volume_res', 64,
                                """Volume Resolution of the output""")
    tf.app.flags.DEFINE_integer('num_joints', 14,
                                """Total number of joints tobe estimated""")
    tf.app.flags.DEFINE_integer('mul_factor', 1,
                                """factor to be mult with Z data to conv to cm""")
    tf.app.flags.DEFINE_float('sigma', 1,
                              """sigma value for probability distribution in
	                            pixels""")
    tf.app.flags.DEFINE_integer('image_res', 256,
                                """input image resolution""")
    tf.app.flags.DEFINE_integer('image_c', 3,
                                """input image channels""")
    tf.app.flags.DEFINE_float('learning_rate', 5e-4,
                              """Define Learning rate""")
    tf.app.flags.DEFINE_float('joint_prob_max', 1,
                              """This parameter basically boost the value of
	                          joint's location probability which results in
	                          the boosting of the value of loss, so faster
	                          convergence maybe""")
    tf.app.flags.DEFINE_boolean('train_2d', False, """This will tell if it
                                should be trained on 2D or 3D""")
    
    # GAN Parameters
    tf.app.flags.DEFINE_float('const_1', 5e-4,
                              """Define Learning rate for KL Divergence""")
    tf.app.flags.DEFINE_float('const_2', 1e-4,
                              """Define Learning rate for reconstruction""")
    
    return FLAGS
