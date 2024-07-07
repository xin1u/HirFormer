# HirFormer: Dynamic High Resolution Transformer for Large-Scale Image Shadow Removal

## 🥇 Winner solution on the NTIRE 2024 Image Shadow Removal Challenge

Our team (LUMOS) wins the [New Trends in Image Restoration and Enhancement workshop and associated challenges in conjunction with CVPR 2024] https://cvlai.net/ntire/2024/NTIRE2024awards_certificates.pdf)!

This is our transformer-based shadow removal model designed for the NTIRE 2024 Image Shadow Removal Challenge. It is specifically tailored for effectively removing shadows in large-scale images.

## Directory of the code package

In order to easily replicate our competition results, you need to download the compressed file "LUMOS_shadowSubmit.zip" that we have sent. Once extracted, the directory structure of the uncompressed files will be as follows:

```powershell
--LUMOS_shadowSubmit
    --ckpt
    --datasets
    --networks
    --ntire_24_sh_rem_final_test_inp
    --utils
    --readme.md
    --TEST.py
```

The folder "ntire_24_sh_rem_final_test_inp" contains the test set of 75 shadow images for the competition. If you have additional test images, you can directly replace them.

## Testing the shadow removal effectiveness of HirFormer

You can easily reproduce our submission results by using the following command. Please make sure you have a PyTorch environment set up and navigate to the "LUMOS_shadowSubmit" directory. If your environment does not have the required Python modules installed, you will need to install them using pip.

``` powershell
python TEST.py  --eval_in_path  ./ntire_24_sh_rem_final_test_inp/  --result_path  ./running_result/
```
The directory "./ntire_24_sh_rem_final_test_inp/" contains the images that need shadow removal. The directory "./running_result/" stores the resulting images after performing shadow removal. Additionally, the directory "./running_result/log_file/test.txt" contains the log file for printing program execution results, allowing you to monitor the progress at any time.

If you have another dataset of shadow images, feel free to replace "./ntire_24_sh_rem_final_test_inp/" with the path to your shadow image directory, and modify "./running_result/" to the desired directory for storing the restored images.

Note: It is important to ensure that both of these paths for modification end with "/" to avoid unnecessary errors.

## Contact

Please feel free to contact us if there is any question(luxion@mail.ustc.edu.cn).
