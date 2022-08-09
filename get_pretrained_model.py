import gdown
import os


def download_weights(model_name, url, ):
    output_dir = 'pretrained_{}'.format(model_name.upper())
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output = output_dir + '/{}_pretrained'.format(model_name)
    info = 'Downloading pretrained tb model'
    print(info)
    gdown.download(url, output)


if __name__ == '__main__':
    tb_model = 'https://drive.google.com/uc?id=1f1uIPCJzX-F160Gl9lD7beAyFieBfiIE'
    

    download_weights('tb_model', tb_model)
