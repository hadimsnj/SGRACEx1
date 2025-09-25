SGRACE demo 

To use the demo:

1. Check lines in notebook demo_sgrace.ipynb

import sys

sys.path.insert(1, '/home/xilinx/jupyter_notebooks/sgrace_lib')

This lines indicate where the sgrace hardware library files are stored. 

This library takes of all the interfacing and control of the hardware accelerator. 

Therefore only minimum modifications are needed in the model itself to make use of the accelerator. 

Create a directory like '/home/xilinx/jupyter_notebooks/sgrace_lib' and save sgrace.py and config.py in it.


2. sgrace.py loads a FPGA bit file called gat_all_unsigned.bit. 

Make sure you move the provided  bit and hwh files to the overlays directory with the correct name.

3. If the notebook is set to training=0 only inference will be run. Set training to 1 to perform training as well. 

The notebook is set to load model parameters from a file "models/model_photo_8bit.ptx". 

Make sure you store the provided model to the "models" directory.

4. The notebook uses neighborLoader to load subgraphs and perform inference on them. Accuracy is around 90% with this data set and configuration.  

In demo_sgrace there is a setting full_graph that when set to 1 uses standard loader to process the whole graph together while setting at 0 uses neighborloader.
Different datasets are possible from planetoid for full_graph 1 and Amazon for full_graph 0. 

5. The hardware libary makes it easy to use the hardware since the only key steps are:

 5.1

 Import the hardware layers:

 import config

 from sgrace import init_SGRACE,GATConv_SGRACE, Relu_SGRACE

 config.py contains important hardware settings like using attention (i.e GAT) or not (i.e. GCN). The quantization target for on-device training (i.e 8-bit downto 1-bit) etc.

 5.2

 Use in the notebook init_sgrace to initialize the hardware that will load the bit file and set some registers.

 5.3

 replace software layers with hardware layers like:

 self.att1 = GATConv(dataset.num_node_features, hidden_channels)

 with

 self.att2 = GATConv_SGRACE(dataset.num_node_features, hidden_channels,head_count,dropout=0.1, alpha=0.2, concat=False)


 

