import configparser


def set_config_image(render_path, mask_path, output_path, testing_path):
    config = configparser.ConfigParser()
    with open("cookies.ini", "w+") as ini:
        config.add_section('paths')
        config.set('paths', 'image', render_path)
        config.set('paths', 'mask', mask_path)
        config.set('paths', 'output', output_path)
        config.set('paths', 'testing', testing_path)
        config.write(ini)


def set_config_model(image_path, label_path, batch_size, num_experiments, pre_train_path):
    config = configparser.ConfigParser()
    with open("model_cookies.ini", "w+") as ini:
        config.add_section('paths')
        config.set('paths', 'training', image_path)
        config.set('paths', 'label', label_path)
        config.set('paths', 'pretrain', pre_train_path)
        config.add_section('values')
        config.set('values', 'batch', str(batch_size))
        config.set('values', 'experiments', str(num_experiments))
        config.write(ini)


def get_config_image():
    config = configparser.ConfigParser()
    config.read('cookies.ini')
    images_path = config['paths']['image']
    mask_path = config['paths']['mask']
    output_path = config['paths']['output']
    testing_path = config['paths']['testing']
    return images_path, mask_path, output_path, testing_path


def get_config_model():
    config = configparser.ConfigParser()
    config.read('model_cookies.ini')
    image_path = config['paths']['training']
    label_path = config['paths']['label']
    pre_train_path = config['paths']['pretrain']
    batch_size = config['values']['batch']
    num_experiments = config['values']['experiments']
    return image_path, label_path, pre_train_path, batch_size, num_experiments


def config_setup():
    config = configparser.ConfigParser()
    with open("cookies.ini", "r+") as ini:
        if len(ini.readlines()) == 0:
            config.add_section('paths')
            config.set('paths', 'image', '')
            config.set('paths', 'mask', '')
            config.set('paths', 'output', '')
            config.set('paths', 'testing', '')
            config.write(ini)
    with open("model_cookies.ini", "r+") as modelini:
        if len(modelini.readlines()) == 0:
            config.add_section('paths')
            config.set('paths', 'training', '')
            config.set('paths', 'label', '')
            config.set('paths', 'pretrain', '')
            config.add_section('values')
            config.set('values', 'batch', '5')
            config.set('values', 'experiments', '20')
            config.write(ini)
