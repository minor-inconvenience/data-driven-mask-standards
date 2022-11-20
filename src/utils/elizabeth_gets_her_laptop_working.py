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
    import os
    import torch
    print(torch.cuda.is_available())
    for k, v in os.environ.items():
        print(f'{k}={v}')
    #torch.cuda.set_device(torch.cuda.device('cuda:1'))  #https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    print(torch.cuda.current_device())
    print(torch.cuda.memory_snapshot)
    torch.cuda.can_device_access_peer




    #so i thought i had to do this, but after some super sleuthing
    with torch.cuda.device('cuda:1'):  # from the forums, this looks like it actually sets aside 500mb of gpu0 (i only have one gpu but in case that mattered)
        torch.cuda.empty_cache()  # if you call this without the with statement, it initialises gpu0 which is just the integrated intel one https://github.com/pytorch/pytorch/issues/25752
    # turns out that putting
    # nvidia-smi -L    into the anaconda prompt returned this
    # GPU 0: NVIDIA GeForce RTX 2060 with Max-Q Design (UUID: GPU-a2a3a7ef-ef84-f2a6-7bd7-2fae1e0823c4)
    # so it only sees one GPU for cuda and that's the one i need so actually it's all good
    """ 
    potentially necessary dodgy ssl things
    import certifi
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context  # sketchy ssl workaround
    """
