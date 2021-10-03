# ATLAS-collective-AE
This is an autoencoder implemented using pytorch. It builds upon previous work done by Eric Wulff, Erik Wallin, Jessica Lastow and Honey Gupta. See for example: https://github.com/Autoencoders-compression-anomaly/AE-Compression-pytorch

The autoencoder compresses jet four-momentum data in two steps. First, single jets are compressed into three-dimensional latent space representations (based on the work done by Eric Wulff in his master's thesis). Secondly, groups of jet representations are concatenated and collectively compressed further. The information is then decoded, and evaluation metrics examined.

The data used in this example comes from the ATLAS Open Data release of 2020. It can be downloaded from the following site: http://opendata.cern.ch/record/15007
