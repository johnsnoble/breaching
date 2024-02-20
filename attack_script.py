import torch
import breaching.breaching as breaching
from torchvision import models
import logging, sys
import base64

AttackStatistics = breaching.attacks.attack_info.AttackStatistics
AttackProgress = breaching.attacks.attack_info.AttackProgress
AttackParameters = breaching.attacks.attack_info.AttackParameters

def setup_attack(attack_params:AttackParameters=None, cfg=None, torch_model=None):
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if cfg == None:
        cfg = breaching.get_config()
    print(cfg)

    if torch_model is None:
        torch_model = buildUploadedModel(attack_params.model, attack_params.ptFilePath)

    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    print(setup)

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
    logger = logging.getLogger() 

    #setup all customisable parameters
    if attack_params != None:
        cfg.case.model = attack_params.model
        if attack_params.datasetStructure == "csv":
            cfg.case.data.name = "CustomCsv"
        elif attack_params.datasetStructure == "folders":
            cfg.case.data.name = "CustomFolders"
        else:
            cfg = breaching.get_config(overrides = ["case/data=CIFAR10"])
        cfg.case.data.path = attack_params.csvPath
        cfg.case.data.size = attack_params.datasetSize
        cfg.case.data.classes = attack_params.numClasses
        cfg.case.data.batch_size = attack_params.batchSize
        cfg.attack.restarts.num_trials = attack_params.numRestarts
        cfg.attack.optim.step_size = attack_params.stepSize
        cfg.attack.optim.max_iterations = attack_params.maxIterations
        cfg.attack.optim.callback = attack_params.callbackInterval

    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
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
                                     cfg_case=cfg.case, setup=setup, compute_lpips=True)
    stats = AttackStatistics(MSE=metrics.get('mse', 0), SSIM=metrics.get('ssim', 0), PSNR=metrics.get('psnr', 0))
    token, channel = response

    image_data = None
    with open("../reconstructed_data.png", 'rb') as image_file:
        image_data = image_file.read()
    base64_encoded_data = base64.b64encode(image_data).decode('utf-8')

    iterations = cfg.attack.optim.max_iterations
    channel.put(token, AttackProgress(current_iteration=iterations, max_iterations=iterations,
                                      statistics=stats, reconstructed_image=base64_encoded_data))
    return metrics
    
def check_image_size(model, shape):
    return True

def buildUploadedModel(model_type, state_dict_path):
    model = None
    if model_type == "resnet18":
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
