# Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models

> Changhyun Choi, Sungha Kim, H. Jin Kim

**Abstract** Recently, it has been shown that investing computing resources in searching for good initial noise for a text-to-image diffusion model helps improve performance. However, previous studies required external models to evaluate the resulting images, which is impossible on GPUs with small VRAM. For these reasons, we apply Best-of-N inference-time scaling to algorithms that optimize the initial noise of a diffusion model without external models across multiple datasets and backbones. We demonstrate that inference-time scaling for text-to-image diffusion models in this setting quickly reaches a performance plateau, and a relatively small number of optimization steps suffices to achieve the maximum achievable performance with each algorithm.


**Running the code**
* All experiments can be run through RUNME.ipynb. 

    * If you run the code in Google Colab, follow "Install essentials & requirements (for Google Colab)" part in the ipynb file.    
    * If you run the code in your own workspace, follow "Install essentials & requirements (for local workspace)" part in the ipynb file.
    

* Requirements are in environment.yaml file.

* Feel free to contact us if there are any questions.

* This code is built on [the official implementation code of Self-Cross guidance](https://github.com/mengtang-lab/selfcross-guidance).