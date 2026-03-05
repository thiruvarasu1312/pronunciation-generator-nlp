# Seq2Seq Pronunciation Generator using Attention

This project builds a sequence-to-sequence neural network with attention to convert English word spellings into phoneme pronunciations.

The model is trained using the CMU Pronouncing Dictionary from NLTK.

## Technologies Used
- Python
- PyTorch
- NLTK
- Sequence-to-Sequence Model
- Attention Mechanism

## Dataset
CMU Pronouncing Dictionary (available through NLTK)

## Model Architecture
Encoder-Decoder GRU with Attention

## Steps
1. Load CMU Pronouncing Dictionary
2. Prepare spelling → phoneme pairs
3. Build character vocabulary
4. Train Seq2Seq model with attention
5. Predict pronunciation of new words

## Example

cat → K AE T  
dog → D AO G  

## Run the Project

Install dependencies

pip install -r requirements.txt

Run the program

python pronunciation_model.py
