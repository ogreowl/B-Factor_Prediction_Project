<h3>Abstract</h3>
In this project, we aimed to predict the B-Factor, a measure of atomic displacement, of amino acids within‬
‭ a protein sequence using deep learning approaches. We tested various parameters and architectures of‬
‭ linear models, recurrent neural networks, long-short term models, and transformers in order to‬
‭ systematically measure the performance of these approaches for this prediction task. For each model, we‬
‭ measured the performance of models using amino acid sequences raw, as well as the embeddings of‬
‭ sequence positions using ProBERT. We found that, across all models, ProBERT embeddings produce a‬
‭ significant improvement in their predictive power. We then scaled up our experiments, implementing a‬
‭ customized LSTM and Transformer model to predict B-Factors on a training dataset of 60,000 protein‬
‭ sequences. On this larger scale, we were able to match state-of-the-art performance with an 81% Pearson‬
‭ Correlation Coefficient, and found that positional embeddings allowed our models to make predictions‬
‭ with higher accuracy and more efficiency.


<h3>Scripts & Files</h3>
--> Deep_Learning_B-Factor_Prediction.ipynb .. a systematic testing on a small subset of our data.<br>
--> b_factor_prediction_transformer.ipynb .. our final model, trained on 60k protein sequences<br>
--> src .. helper scripts to modularize our work<br>
--> report.pdf .. our full report of our results, with selected data points & visualizations explained and analyzed<br>
--> images .. unexhaustive screenshots on our results


‭
