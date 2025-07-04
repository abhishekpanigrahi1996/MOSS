# **The Necessity for Intervention Fidelity: Unintended Side Effects When Steering LLMs**

*Authors: Jonas B. Raedler, Weiyue Li, Manasvi Goyal, Alyssa Mia Taliotis, Siddharth Swaroop, Weiwei Pan*

---
This is the notebook that accompanies our paper, published at the *Methods and Opportunities at Small Scale (MOSS)* workshop, held at ICML 2025 (Vancouver, Canada).

### **Please Read:**

**Instructions**
1. This notebook makes use of the `transformer_lens` library. The pretrained models that it uses require a lot of memory, so you'll have to change the runtime environment of this notebook: **Change the runtime type from CPU to v2-8 TPU**. This is available in Google Colab free tier (I've tested it successfully --- there might be usage limits after using it for a while, but it should be usable again after a while).

2. Furthermore, Google Colab doesn't seem to have the `transformer_lens` library pre-installed. Try to execute the imports cell - if it fails, **uncomment the `!pip install` command, install the package, and then rerun the imports**.

3. This notebook uses the `google/gemma-2-2b` family of models. Using them requires a huggingface token (which is free). You can get one here: https://huggingface.co/docs/hub/en/security-tokens. Provide your access token in the field `HUGGINGFACE_TOKEN` below.

4. Last but not least, this notebook uses the StereoSet dataset. You can get the data from here: https://github.com/moinnadeem/StereoSet/blob/master/data/dev.json. Download the dev.json file and upload it to this session (on the left under "Files", select "Upload to session storage").


**Further Information**

In the interest of time, this notebook does not run our experiments on the entire dataset (this would take significantly longer than only 3 hours). Instead, we run our experiments on a subset of the data. From the four available bias types, we chose two: "gender" and "race". We also run our experiments on only one dataset (intersentence), use only 30 samples per bias type, and steer at 3 layers (5, 11, 17) instead of every second layer.
These decisions were made to make it possible to run this notebook in 1.5 hours, while still covering the entire experiment spectrum as holistically as possible.

We aimed for a time of 1.5 hours, as one run of this notebook only executes the experiments for either the base or the fine-tuned model. If desired, the reviewer can change the model and rerun the notebook to get the other set of results.

All our decisions can be altered, if desired, though we recommend to keep a wide range of layers (i.e., don't pick 1, 2, 3). We specify these decision in the first cell and they can be altered there.


