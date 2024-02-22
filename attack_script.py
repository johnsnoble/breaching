import torch
import breaching.breaching as breaching
from torchvision import models
import logging, sys
import base64
import zipfile
import os, shutil

AttackStatistics = breaching.attacks.attack_info.AttackStatistics
AttackProgress = breaching.attacks.attack_info.AttackProgress
AttackParameters = breaching.attacks.attack_info.AttackParameters

def construct_cfg(attack_params: AttackParameters):
    cfg = None
    
    match attack_params.attack:
        case 'invertinggradients':
            cfg = breaching.get_config()
            cfg.case.data.partition="unique-class"
            # default case.model=ResNet18
        case 'analytic':
            cfg = breaching.get_config(overrides=["attack=analytic", "case.model=linear"])
            cfg.case.data.partition="balanced"
            cfg.case.data.default_clients = 50
            cfg.case.user.num_data_points = 256 # User batch size 
        case 'rgap':
            cfg = breaching.get_config(overrides=["attack=rgap", "case.model=cnn6", "case.user.provide_labels=True"])
            cfg.case.user.num_data_points = 1
        case 'april_analytic':
            cfg = breaching.get_config(overrides=["attack=april_analytic", "case.model=vit_small_april"])
            cfg.case.data.partition="unique-class"
            cfg.case.user.num_data_points = 1
            cfg.case.server.pretrained = True
            cfg.case.user.provide_labels = False
        case 'deepleakage':
            cfg = breaching.get_config(overrides=["attack=deepleakage", "case.model=ConvNet"])
            cfg.case.data.partition="unique-class"
            cfg.case.user.provide_labels=False
        case 'modern':
            cfg = breaching.get_config(overrides=["attack=modern"])
            cfg.case.data.partition="unique-class"
            cfg.attack.regularization.deep_inversion.scale=1e-4
        case 'fishing_for_user_data':
            cfg = breaching.get_config(overrides=["case/server=malicious-fishing", "attack=clsattack", "case/user=multiuser_aggregate"])
            cfg.case.user.user_range = [0, 1]
            cfg.case.data.partition = "random" # This is the average case
            cfg.case.user.num_data_points = 256
            cfg.case.data.default_clients = 32
            cfg.case.user.provide_labels = True # Mostly out of convenience
            cfg.case.server.target_cls_idx = 0 # Which class to attack?
           
            
    #setup all customisable parameters
    if attack_params != None:
        if cfg.case.model != attack_params.model:
            err_msg = f"model for given attack does not match. 
                            Requested model {attack_params.model}. 
                            Attack model {cfg.case.model}. 
                            Attack {attack_params.attack}"
            raise TypeError(err_msg)
        # cfg.case.model = attack_params.model
        if attack_params.datasetStructure == "CSV":
            cfg.case.data.name = "CustomCsv"
        elif attack_params.datasetStructure == "Foldered":
            cfg.case.data.name = "CustomFolders"
        else:
            cfg = breaching.get_config(overrides = ["case/data=CIFAR10"])
        cfg.case.data.path = 'dataset'
        cfg.case.data.size = attack_params.datasetSize
        cfg.case.data.classes = attack_params.numClasses
        cfg.case.data.batch_size = attack_params.batchSize
        cfg.attack.restarts.num_trials = attack_params.numRestarts
        cfg.attack.optim.step_size = attack_params.stepSize
        cfg.attack.optim.max_iterations = attack_params.maxIterations
        cfg.attack.optim.callback = attack_params.callbackInterval
        
    return cfg
            

def setup_attack(attack_params:AttackParameters=None, cfg=None, torch_model=None):
    
    print(f'~~~[Attack Params]~~~ {attack_params}')
    
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if cfg == None:
        cfg = breaching.get_config()

    if torch_model is None:
        torch_model = buildUploadedModel(attack_params.model, attack_params.ptFilePath)
        
    # unzipped_directory = attack_params.zipFilePath.split('.')[0]
    print(os.listdir())
    if (os.path.exists('dataset')):
        shutil.rmtree('dataset')
    with zipfile.ZipFile(attack_params.zipFilePath, 'r') as zip_ref:
        zip_ref.extractall('./dataset')
    print(os.listdir('dataset'))
    
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    print(setup)

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
    logger = logging.getLogger() 

    cfg = construct_cfg(attack_params)
    
    print(cfg)

    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup, prebuilt_model=torch_model)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)


    if torch_model is not None:
        model = torch_model
    
    if not check_image_size(model, cfg.case.data.shape):
        raise ValueError("Mismatched dimensions")
    
    return cfg, setup, user, server, attacker, model, loss_fn

def perform_attack(cfg, setup, user, server, attacker, model, loss_fn, response):
    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)
    breaching.utils.overview(server, user, attacker)

    user.plot(true_user_data, saveFile="true_data")
    print("reconstructing attack")
    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun, response=response)
    user.plot(reconstructed_user_data, saveFile="reconstructed_data")
    return reconstructed_user_data, true_user_data, server_payload
    
def get_metrics(reconstructed_user_data, true_user_data, server_payload, server, cfg, setup, response):
    metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload], 
                                     server.model, order_batch=True, compute_full_iip=False, 
                                     cfg_case=cfg.case, setup=setup, compute_lpips=False)
    print(metrics)
    # stats = AttackStatistics(MSE=0, SSIM=0, PSNR=0)
    stats = AttackStatistics(MSE=metrics.get('mse', 0), SSIM=0, PSNR=metrics.get('psnr', 0))
    token, channel = response

    with open("./reconstructed_data.png", 'rb') as image_file:
        image_data_rec = image_file.read()
    base64_reconstructed = base64.b64encode(image_data_rec).decode('utf-8')
    
    with open("./true_data.png", 'rb') as image_file:
        image_data_true = image_file.read()
    base64_true = base64.b64encode(image_data_true).decode('utf-8')

    iterations = cfg.attack.optim.max_iterations
    restarts = cfg.attack.restarts.num_trials
    channel.put(token, AttackProgress(current_iteration=iterations, 
                                      current_restart=restarts,
                                      max_iterations=iterations,
                                      max_restarts=restarts,
                                      statistics=stats, 
                                      true_image=base64_true,
                                      reconstructed_image=base64_reconstructed))
    return metrics
    
def check_image_size(model, shape):
    return True

def buildUploadedModel(model_type, state_dict_path):
    model = None
    if model_type == "ResNet-18":
        model = models.resnet18()
    
    if not model:
        raise TypeError("given model type did not match any of the options")
    
    try:
        model.load_state_dict(torch.load(state_dict_path))
    except RuntimeError as r:
        print(f'''Runtime error loading torch model from file:
{r}
Model is loaded from default values.
''')
    except FileNotFoundError as f:
        print(f'''Runtime error loading torch model from file:
{f}
Model is loaded from default values.
''')
    model.eval()
    return model
