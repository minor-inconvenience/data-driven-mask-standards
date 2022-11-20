# Messing around to see if face_alignment works: https://github.com/1adrianb/face-alignment/blob/master/examples/demo.ipynb

# had to install face_alignment (pip install face_alignment)
# had to update numpy (pip install numpy --upgrade)
# had to update pillow
# i think something about urllib vs urllib3
# ok no it's because python 3.6 does weird openssl things so no longer certified so u have to
# pip install certifi --upgrade
# ended up using dodgy work around
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# okay nevermind uninstall torch
# then do pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
# i also uninstalled and reinstalled numpy
# and found a random file to delete


if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())




    """ 
    potentially necessary dodgy ssl things
    import certifi
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context  # sketchy ssl workaround
    """
