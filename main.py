import os, sys, time
import hashlib
import coadapt
import experiment_configs as cfg
import json

def main(config):

    # Create foldr in which to save results
    folder = config['data_folder']
    #generate random hash string - unique identifier if we start
    # multiple experiments at the same time
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
    file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id
    config['data_folder_experiment'] = file_str

    # Create experiment folder
    if not os.path.exists(file_str):
      os.makedirs(file_str)

    # Store config
    with open(os.path.join(file_str, 'config.json'), 'w') as fd:
            fd.write(json.dumps(config,indent=2))

    co = coadapt.Coadaptation(config)
    co.run()



if __name__ == "__main__":
    # We assume we call the program only with the name of the config we want to run
    # nothing too complex
    if len(sys.argv) > 1:
        config = cfg.config_dict[sys.argv[1]]
    else:
        config = cfg.config_dict['base']
    main(config)
