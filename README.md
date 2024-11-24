# TurNuP4 - turnover number prediction--Quaternary

## anaconda
```
conda creat -n turnup4 python=3.11.4
activate turnup4

chmod +x setup.sh       
./setup.sh    
```

## data & model

```
  https://zenodo.org/records/13982490

download  data/  encoder/  save_model/  into the root folder of project and run the following commands
unzip encoder.zip
mv encoder AE
unzip save_model.zip
unzip data.zip

finally it would be like :
.
├── AE
│   ├── encoder
├── data
├── save_model
...

```

### for main.sh
```            
./main.sh              
```






## Citation 

Our work was first presented in the bibm for [Revolutionizing Enzyme Turnover Predictions with Non-Binary Reaction Fingerprints and AI]

Index Terms—turnover number, differential reaction fingerprints (DRFPs), kinetically uncharacterized enzyme, pre-training,
deep learning
